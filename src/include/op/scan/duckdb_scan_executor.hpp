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

#include "op/scan/duckdb_scan_task.hpp"
#include "parallel/config.hpp"
#include "parallel/task_executor.hpp"

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <memory>
#include <string>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::creator {
class task_creator;
}  // namespace sirius::creator

namespace sirius::pipeline {
class completion_handler;
}  // namespace sirius::pipeline

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// DuckDB Scan Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class inherits from itask_executor and manages threads dedicated to
 * executing DuckDB scan tasks.
 */
class duckdb_scan_executor : public sirius::parallel::itask_executor {
 public:
  /**
   * @brief Constructs a new duckdb_scan_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, etc.)
   * @param mem_mgr Pointer to the memory reservation manager for host allocations
   */
  explicit duckdb_scan_executor(parallel::task_executor_config config,
                                cucascade::memory::memory_reservation_manager* mem_mgr);

  /**
   * @brief Destructor for the duckdb_scan_executor.
   */
  ~duckdb_scan_executor();

  // Non-copyable and non-movable
  duckdb_scan_executor(const duckdb_scan_executor&)            = delete;
  duckdb_scan_executor& operator=(const duckdb_scan_executor&) = delete;
  duckdb_scan_executor(duckdb_scan_executor&&)                 = delete;
  duckdb_scan_executor& operator=(duckdb_scan_executor&&)      = delete;

  /**
   * @brief Schedule a new task for execution.
   *
   * @param task The task to be scheduled.
   */
  void schedule(std::unique_ptr<sirius::parallel::itask> task) override;

  /**
   * @brief Get the number of threads in the thread pool for this executor.
   *
   * @return The number of threads in the thread pool for this executor.
   */
  [[nodiscard]] int32_t get_num_threads() const { return _config.num_threads; }

  /**
   * @brief Set the task creator for scheduling output consumers
   *
   * @param task_creator Pointer to the task creator
   */
  void set_task_creator(sirius::creator::task_creator* task_creator);

  /**
   * @brief Set the completion handler for query completion signaling
   *
   * @param handler Pointer to the completion handler
   */
  void set_completion_handler(sirius::pipeline::completion_handler* handler) noexcept;

  /**
   * @brief Cache scan results for the given query
   *
   * @param query The query string to cache results for
   */
  void cache_scan_results_for_query(const std::string& query);

  /**
   * @brief Enable or disable scan result caching
   *
   * @param enabled True to enable caching, false to disable
   */
  void set_scan_caching_enabled(bool enabled);

  /**
   * @brief Check if scan result caching is enabled
   *
   * @return True if caching is enabled, false otherwise
   */
  [[nodiscard]] bool is_scan_caching_enabled() const noexcept { return _caching_enabled; }

  /**
   * @brief Prepare cache for scan operators
   *
   * In CACHE mode: ensures cache is empty and creates entries for each operator's ID
   * In PRELOAD mode: verifies all operator IDs are present in the cache
   *
   * @param scan_operators Vector of scan operators to prepare cache for
   */
  void prepare_cache_for_scan_operators(
    const std::vector<sirius::op::sirius_physical_operator*>& scan_operators);

 private:
  /**
   * @brief Worker loop override for scan-specific task execution
   *
   * @param worker_id The ID of the worker thread
   */
  void worker_loop(int worker_id) override;

  /**
   * @brief Submit a scan task request to pipeline_executor
   */
  void submit_scan_request();

  op::operator_data get_scan_output(op::scan::duckdb_scan_task* task, rmm::cuda_stream_view stream);

  struct cache_entry {
    std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> batches;
    std::size_t batch_index{0};
  };

  std::mutex _cache_mutex;
  std::unordered_map<size_t, std::unique_ptr<cache_entry>> _cache;
  std::size_t _query_hash{0};
  bool _caching_enabled{false};
  bool _preload_mode{false};

  cucascade::memory::memory_reservation_manager* _mem_mgr{nullptr};
  sirius::creator::task_creator* _task_creator{nullptr};
  sirius::pipeline::completion_handler* _completion_handler{nullptr};
};

}  // namespace sirius::op::scan
