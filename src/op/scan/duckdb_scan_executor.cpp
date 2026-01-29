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

// sirius
#include <op/scan/duckdb_scan_executor.hpp>
#include <pipeline/pipeline_executor.hpp>
#include <pipeline/task_request.hpp>

// standard library
#include <mutex>

namespace sirius::op::scan {

void duckdb_scan_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _total_tasks.fetch_add(1, std::memory_order_relaxed);
  _task_queue->push(std::move(task));
}

void duckdb_scan_executor::start()
{
  itask_executor::start();
  _manager_thread = std::thread(&duckdb_scan_executor::manager_loop, this);
}

void duckdb_scan_executor::stop() { itask_executor::stop(); }

void duckdb_scan_executor::wait()
{
  std::unique_lock<std::mutex> lock(_finish_mutex);
  _finish_cv.wait(lock, [&]() {
    return _total_tasks.load(std::memory_order_relaxed) ==
           _finished_tasks.load(std::memory_order_relaxed);
  });
}

void duckdb_scan_executor::worker_loop(int32_t worker_id)
{
  while (_running.load()) {
    auto task = _task_queue->pull();
    if (task == nullptr) {
      // Task queue is closed or empty
      break;
    }
    try {
      task->execute();
    } catch (const std::exception& e) {
      on_task_error(worker_id, std::move(task), e);
    }
    // Update counters and notify
    {
      std::unique_lock<std::mutex> lock(_finish_mutex);
      _finished_tasks.fetch_add(1, std::memory_order_relaxed);
      if (_finished_tasks.load(std::memory_order_relaxed) ==
          _total_tasks.load(std::memory_order_relaxed)) {
        _finish_cv.notify_one();
      }
    }
  }
}

void duckdb_scan_executor::set_pipeline_executor(pipeline::pipeline_executor* pipeline_exec)
{
  _pipeline_exec = pipeline_exec;
}

void duckdb_scan_executor::manager_loop()
{
  while (_running) {
    std::unique_lock<std::mutex> lock(_task_count_mutex);
    _req_count_cv.wait(
      lock, [this]() { return _num_active_requests < _config.num_threads || !_running.load(); });
    if (!_running.load()) { break; }
    _num_active_requests++;
    lock.unlock();
    _pipeline_exec->submit_task_request(std::make_unique<pipeline::task_request>(-1, true));
  }
}

}  // namespace sirius::op::scan
