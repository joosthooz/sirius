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

#include "creator/task_creator.hpp"

#include "log/logging.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>

#include <optional>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(parallel::task_executor_config gpu_executor_config,
                           parallel::task_executor_config scan_executor_config,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
                           const cucascade::memory::system_topology_info* sys_topology)
  : _mem_res_mgr(mem_res_mgr)
{
  // Create the scan executor with memory manager for host allocations
  _scan_executor =
    std::make_unique<sirius::op::scan::duckdb_scan_executor>(scan_executor_config, &mem_res_mgr);

  auto gpu_spaces = mem_res_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Initialize GPU pipeline executors for each available GPU
  for (auto* space : gpu_spaces) {
    auto config   = gpu_executor_config;
    int device_id = space->get_device_id();
    if (sys_topology) {
      auto it = std::find_if(sys_topology->gpus.begin(),
                             sys_topology->gpus.end(),
                             [device_id](const cucascade::memory::gpu_topology_info& dev) {
                               return dev.id == device_id;
                             });

      if (it != sys_topology->gpus.end()) { config.cpu_affinity_list = it->cpu_cores; }
    }
    _gpu_executors.emplace(device_id,
                           std::make_unique<pipeline::gpu_pipeline_executor>(
                             config, const_cast<cucascade::memory::memory_space*>(space)));
  }

  // Set task creator reference on executors
  _scan_executor->set_task_creator(this);
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_task_creator(this);
  }
}

task_creator::~task_creator() = default;

void task_creator::set_client_context(::duckdb::ClientContext& client_context)
{
  _client_context = std::addressof(client_context);
  _thread_context = std::make_unique<duckdb::ThreadContext>(client_context);
  _execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_context, *_thread_context, nullptr);
}

void task_creator::reset()
{
  // Clear the scan operator global state map for the new query
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _scan_operator_global_state_map.clear();
  _gpu_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

[[nodiscard]] sirius::op::scan::duckdb_scan_executor& task_creator::get_scan_executor() noexcept
{
  return *_scan_executor;
}

[[nodiscard]] const sirius::op::scan::duckdb_scan_executor& task_creator::get_scan_executor()
  const noexcept
{
  return *_scan_executor;
}

void task_creator::prepare_for_query(duckdb::shared_ptr<planner::query> query)
{
  // Drain leftover tasks from previous query
  _scan_executor->drain_leftover_tasks();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_leftover_tasks();
  }

  auto scans = query->get_scan_operators();
  _scan_executor->prepare_cache_for_scan_operators(scans);

  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  while (!_priority_scans.empty()) {
    _priority_scans.pop();
  }
  for (auto* scan : scans) {
    if (auto* tscan = dynamic_cast<op::sirius_physical_duckdb_scan*>(scan)) {
      _priority_scans.push(tscan);
    } else {
      SIRIUS_LOG_ERROR("Failed to cast scan to sirius_physical_duckdb_scan");
      continue;
    }
  }
}

std::future<void> task_creator::start_query()
{
  // Create a new completion handler for this query
  _completion_handler      = std::make_unique<pipeline::completion_handler>();
  std::future<void> future = _completion_handler->get_awaitable();

  // Set completion handler on all executors
  _scan_executor->set_completion_handler(_completion_handler.get());
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_completion_handler(_completion_handler.get());
  }

  schedule_next_scan_tasks();

  return future;
}

void task_creator::schedule_next_scan_tasks()
{
  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  if (!_priority_scans.empty()) {
    auto* scan_op = _priority_scans.front();
    _priority_scans.pop();
    for (auto i = 0; i != _scan_executor->get_num_threads(); ++i) {
      schedule(scan_op);
    }
  }
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }
  auto hint = node->get_next_task_hint();

  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.has_value() &&
             hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    return get_operator_for_next_task(hint.value().producer);
  }
  return nullptr;
}

void task_creator::schedule(op::sirius_physical_operator* node, int device_id)
{
  if (node == nullptr) {
    SIRIUS_LOG_WARN("Task Creator: schedule() called with nullptr node");
    return;
  }

  try {
    create_and_schedule_task(node, device_id);
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Task Creator: Exception during task creation: {}", e.what());
    throw;
  }
}

void task_creator::schedule(std::unique_ptr<sirius::parallel::itask> task, int device_id)
{
  if (task->is<sirius::op::scan::duckdb_scan_task>()) {
    _scan_executor->schedule(std::move(task));
  } else {
    auto gpu_task = dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(task.get());
    _gpu_executors.at(device_id)->schedule(std::move(task));
  }
}

void task_creator::create_and_schedule_task(op::sirius_physical_operator* node, int device_id)
{
  // Find the operator to create a task for based on hints
  node = get_operator_for_next_task(node);
  if (node == nullptr) { return; }

  // Get what we need to create the task
  auto pipeline = node->get_pipeline();
  std::vector<cucascade::shared_data_repository*> destination_data_repositories;
  auto next_port_after_sink = pipeline->get_sink()->get_next_port_after_sink();
  for (auto& [next_op, port_id] : next_port_after_sink) {
    destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
  }

  // scheduling scan task
  if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
    // Check to see if you need to create a new global state for this scan operator
    size_t operator_id = node->get_operator_id();
    {
      std::lock_guard<std::mutex> lock(_global_state_mutex);
      auto it = _scan_operator_global_state_map.find(operator_id);
      if (it == _scan_operator_global_state_map.end()) {
        // If not found, create new global state and store it in the map
        auto scan_task_global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
          pipeline, *this, *_client_context, &node->Cast<op::sirius_physical_duckdb_scan>());
        _scan_operator_global_state_map[operator_id] = scan_task_global_state;
      }
    }

    auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
      *_scan_operator_global_state_map[operator_id], *_execution_context);
    if (destination_data_repositories.empty()) {
      throw std::runtime_error("No destination data repositories provided for scan task creation.");
    }
    auto scan_task = std::make_unique<op::scan::duckdb_scan_task>(
      get_next_task_id(),
      destination_data_repositories[0],  // WSM amin TODO: is this correct? there probably
                                         // needs to be multiple possible destination data
                                         // repositories
      std::move(scan_task_local_state),
      _scan_operator_global_state_map[operator_id]);
    pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically
                                    // with the task creation
    schedule(std::move(scan_task));
    // scheduling pipeline task
  } else {
    // need to exhaust input batches until all ports are empty
    while (!node->all_ports_empty()) {
      auto input_data = node->get_next_task_input_data();
      if (!input_data.has_value()) { break; }
      pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically with the
                                      // task creation

      // Check to see if you need to create a new global state for this operator
      size_t operator_id = node->get_operator_id();
      {
        std::lock_guard<std::mutex> lock(_global_state_mutex);
        auto it = _gpu_operator_global_state_map.find(operator_id);
        if (it == _gpu_operator_global_state_map.end()) {
          // If not found, create new global state and store it in the map
          auto gpu_pipeline_task_global_state =
            std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
          _gpu_operator_global_state_map[operator_id] = gpu_pipeline_task_global_state;
        }
      }

      auto local_state =
        std::make_unique<pipeline::gpu_pipeline_task_local_state>(input_data.value());
      auto task =
        std::make_unique<pipeline::gpu_pipeline_task>(get_next_task_id(),
                                                      destination_data_repositories,
                                                      std::move(local_state),
                                                      _gpu_operator_global_state_map[operator_id]);
      schedule(std::move(task));
    }
  }
}

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
