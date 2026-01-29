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

#pragma once

// sirius
#include "op/scan/duckdb_scan_task_queue.hpp"
#include "parallel/task_executor.hpp"

namespace sirius {

namespace pipeline {
class pipeline_executor;  // Forward declaration
}

namespace op::scan {

//===----------------------------------------------------------------------===//
// DuckDB Scan Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class extends the generic itask_executor simply by instantiating it with a
 * duckdb_scan_task_queue.
 *
 */
class duckdb_scan_executor : public sirius::parallel::itask_executor {
 public:
  //===----------Constructor----------===//
  explicit duckdb_scan_executor(sirius::parallel::task_executor_config config)
    : sirius::parallel::itask_executor(std::make_unique<duckdb_scan_task_queue>(config.num_threads),
                                       config)
  {
  }

  void start() override;
  void stop() override;

  void set_pipeline_executor(pipeline::pipeline_executor* pipeline_exec);

  //===----------Methods----------===//
  /**
   * @brief Schedule a new task for execution.
   *
   * @param task The task to be scheduled.
   */
  void schedule(std::unique_ptr<sirius::parallel::itask> task) override;

  /**
   * @brief Wait for all scheduled tasks to complete.
   */
  void wait();

  /**
   * @brief Worker thread loop.
   *
   * @param worker_id The ID of the worker thread.
   */
  void worker_loop(int32_t worker_id) override;

  /**
   * @brief Manager loop to consume task from local buffer and dispatch to the thread pool
   */
  void manager_loop();

  /**
   * @brief Get the number of threads in the thread pool for this executor.
   *
   * @return The number of threads in the thread pool for this executor.
   */
  [[nodiscard]] int32_t get_num_threads() const { return _config.num_threads; }

  //===----------Fields----------===//
 private:
  std::thread _manager_thread;
  std::mutex _task_count_mutex;
  std::size_t _num_active_requests{0};
  std::condition_variable _req_count_cv;
  std::unique_ptr<std::thread> _gpu_pipeline_executor_manager_thread;
  pipeline::pipeline_executor* _pipeline_exec;

  std::atomic<uint64_t> _total_tasks    = 0;  ///< The total number of scheduled tasks
  std::atomic<uint64_t> _finished_tasks = 0;  ///< The total number of finished tasks
  std::mutex _finish_mutex;                   ///< Mutex to protect condition variable
  std::condition_variable _finish_cv;         ///< Condition variable to signal task completion
};

}  // namespace op::scan
}  // namespace sirius
