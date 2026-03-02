/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op/scan/duckdb_scan_executor.hpp"

#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task_queue.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/memory/common.hpp>

#include <iostream>
#include <mutex>

namespace sirius::op::scan {

duckdb_scan_executor::duckdb_scan_executor(parallel::task_executor_config config,
                                           cucascade::memory::memory_reservation_manager* mem_mgr)
  : itask_executor(std::make_unique<parallel::task_queue>(config.num_threads), config),
    _mem_mgr(mem_mgr)
{
}

duckdb_scan_executor::~duckdb_scan_executor() = default;

void duckdb_scan_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue->push(std::move(task));
}

void duckdb_scan_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

void duckdb_scan_executor::set_completion_handler(
  sirius::pipeline::completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

void duckdb_scan_executor::cache_scan_results_for_query(const std::string& query)
{
  if (!_caching_enabled) { return; }
  std::hash<std::string> hash_fn;
  auto new_query_hash = hash_fn(query);
  if (new_query_hash == _query_hash) {
    SIRIUS_LOG_INFO("Scan results for query already cached, preloading: {}", query);
    return;
  }
  SIRIUS_LOG_INFO("Caching scan results for query: {}", query);
  _query_hash = new_query_hash;
  _cache.clear();
}

void duckdb_scan_executor::set_scan_caching_enabled(bool enabled)
{
  _caching_enabled = enabled;
  SIRIUS_LOG_INFO("Scan caching {}", enabled ? "enabled" : "disabled");
}

void duckdb_scan_executor::prepare_cache_for_scan_operators(
  const std::vector<sirius::op::sirius_physical_operator*>& scan_operators)
{
  if (!_caching_enabled) { return; }

  std::lock_guard<std::mutex> lock(_cache_mutex);
  _preload_mode = !_cache.empty();

  if (!_preload_mode) {
    for (auto* op : scan_operators) {
      auto operator_id    = op->get_pipeline()->get_pipeline_id();
      _cache[operator_id] = std::make_unique<cache_entry>();  // Create empty entry
    }
  } else {
    // In PRELOAD mode: verify all operator IDs are present in the cache
    for (auto* op : scan_operators) {
      auto operator_id = op->get_pipeline()->get_pipeline_id();
      auto iter        = _cache.find(operator_id);
      if (iter == _cache.end()) {
        SIRIUS_LOG_ERROR("Cache entry not found for operator {} in PRELOAD mode", operator_id);
        throw std::runtime_error("Cache entry not found for operator " +
                                 std::to_string(operator_id) + " in PRELOAD mode");
      }
      iter->second->batch_index = 0;  // Reset batch index for PRELOAD mode
    }
  }
}

void duckdb_scan_executor::submit_scan_request()
{
  // Note: This method is no longer needed with the new architecture
  // but kept for potential future use
}

std::unique_ptr<op::operator_data> duckdb_scan_executor::get_scan_output(
  pipeline::sirius_pipeline_itask* task, rmm::cuda_stream_view stream)
{
  if (!_caching_enabled) {
    return task->compute_task(stream);
  } else {
    auto pipe_id = task->get_pipeline_id();
    std::lock_guard<std::mutex> lock(_cache_mutex);
    auto& entry = _cache.at(pipe_id);
    if (!entry) { throw std::runtime_error("Scan results for query not cached"); }
    if (_preload_mode) {
      if (entry->batch_index >= entry->batches.size()) {
        throw std::runtime_error("Scan results for query not cached");
      }
      auto batches = entry->batches[entry->batch_index++];
      std::vector<std::shared_ptr<cucascade::data_batch>> cloned_batches;
      cloned_batches.reserve(batches.size());
      for (auto& b : batches) {
        cloned_batches.push_back(b->clone(::sirius::get_next_batch_id(), stream));
      }
      return std::make_unique<op::operator_data>(std::move(cloned_batches));
    } else {
      auto scan_output = task->compute_task(stream);
      std::vector<std::shared_ptr<cucascade::data_batch>> cloned_batches;
      cloned_batches.reserve(scan_output->get_data_batches().size());
      for (auto& b : scan_output->get_data_batches()) {
        cloned_batches.push_back(b->clone(::sirius::get_next_batch_id(), stream));
      }
      entry->batches.push_back(std::move(cloned_batches));
      return scan_output;
    }
  }
}

void duckdb_scan_executor::worker_loop(int worker_id)
{
  while (true) {
    if (!_running.load()) { break; }

    auto task = _task_queue->pull();
    if (!task) { break; }

    // Cast to scan task
    auto* scan_task = dynamic_cast<sirius::op::scan::duckdb_scan_task*>(task.get());
    if (!scan_task) {
      SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast task to duckdb_scan_task");
      continue;
    }

    // Make host memory reservation and set it on the local state
    // todo (amin): fix this later, and make the reservation in the executor.
    auto* pipeline_task = dynamic_cast<pipeline::sirius_pipeline_itask*>(task.get());
    if (pipeline_task && pipeline_task->is<parquet_scan_task>()) {
      auto bytes_needed = pipeline_task->get_estimated_reservation_size();
      auto reservation  = _mem_mgr->request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, bytes_needed);
      if (!reservation) {
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to acquire host memory reservation");
        if (_completion_handler) {
          _completion_handler->report_error(
            std::make_exception_ptr(std::runtime_error("Failed to acquire memory reservation")));
        }
        continue;
      }
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
            pipeline_task->local_state())) {
        local_state->set_reservation(std::move(reservation));
      } else {
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to cast local state for task");
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for task");
        if (_completion_handler) {
          _completion_handler->report_error(
            std::make_exception_ptr(std::runtime_error("Failed to cast local state")));
        }
        continue;
      }
    }

    auto stream = cudf::get_default_stream();

    try {
      auto consumers   = scan_task->get_output_consumers();
      auto output_data = get_scan_output(scan_task, stream);
      scan_task->publish_output(*output_data, stream);
      task.reset();
      if (_task_creator && !(_completion_handler && _completion_handler->is_completed())) {
        for (auto* consumer : consumers) {
          _task_creator->schedule(consumer);
        }
      }
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("DuckDB Scan Executor: Error executing task: {}", e.what());
      /// Fatal error
      if (_completion_handler) { _completion_handler->report_error(std::make_exception_ptr(e)); }
    } catch (...) {
      SIRIUS_LOG_ERROR("DuckDB Scan Executor: Unknown error executing task");
      /// Fatal error
      if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
    }
  }
}

}  // namespace sirius::op::scan
