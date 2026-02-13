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

#include "pipeline/gpu_pipeline_executor.hpp"

#include "creator/task_creator.hpp"
#include "cucascade/memory/stream_pool.hpp"
#include "cuda_runtime_api.h"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_queue.hpp"

#include <rmm/cuda_device.hpp>

#include <util/stream_check_wrapper.hpp>

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(const parallel::task_executor_config& config,
                                             cucascade::memory::memory_space* mem_space)
  : itask_executor(std::make_unique<gpu_pipeline_queue>(config.num_threads), config),
    _stream_pool(rmm::cuda_device_id{mem_space->get_device_id()}, config.num_threads),
    _memory_space(mem_space)
{
}

void gpu_pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue->push(std::move(task));
}

void gpu_pipeline_executor::worker_loop(int worker_id)
{
  rmm::cuda_set_device_raii set_device_guard(rmm::cuda_device_id{_memory_space->get_device_id()});
  sirius::util::enable_log_on_default_stream();

  while (true) {
    if (!_running.load()) { break; }

    auto task = _task_queue->pull();
    if (!task) { break; }

    auto* gpu_task = cast_to_gpu_pipeline_task(task.get());
    if (!gpu_task) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast task to gpu_pipeline_task");
      continue;
    }

    // Get task information
    auto bytes_needs      = gpu_task->get_estimated_reservation_size();
    auto output_consumers = gpu_task->get_output_consumers();
    auto* pipeline        = gpu_task->get_pipeline();

    // Acquire memory reservation
    auto reservation = _memory_space->make_reservation(bytes_needs);
    if (!reservation) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task");
      if (_completion_handler) {
        _completion_handler->report_error(
          std::make_exception_ptr(std::runtime_error("Failed to acquire memory reservation")));
      }
      continue;
    }

    // Set reservation on local state
    if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_itask_local_state*>(
          gpu_task->local_state())) {
      local_state->set_reservation(std::move(reservation));
    } else {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast local state for task");
      if (_completion_handler) {
        _completion_handler->report_error(
          std::make_exception_ptr(std::runtime_error("Failed to cast local state")));
      }
      continue;
    }

    // Acquire stream
    auto exc_stream = _stream_pool.acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

    // Execute the task
    try {
      task->execute(exc_stream);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Error executing task: {}", e.what());
      if (_completion_handler) { _completion_handler->report_error(std::make_exception_ptr(e)); }
      continue;
    } catch (...) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Unknown error executing task");
      if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
      continue;
    }
    task.reset();

    // Check if query is complete BEFORE scheduling downstream tasks.
    // mark_completed() signals the future that engine.execute() is waiting on,
    // which may destroy the engine and its operators. We must not schedule
    // tasks that reference those operators after signaling completion.
    bool query_complete = false;
    if (_completion_handler && pipeline) {
      auto sink = pipeline->get_sink();
      if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
        query_complete = pipeline->is_pipeline_finished();
      }
    }

    if (!query_complete && _task_creator) {
      for (auto* consumer : output_consumers) {
        _task_creator->schedule(consumer);
      }
    }

    if (query_complete && _completion_handler) { _completion_handler->mark_completed(); }
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

void gpu_pipeline_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

void gpu_pipeline_executor::set_completion_handler(completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

}  // namespace pipeline
}  // namespace sirius
