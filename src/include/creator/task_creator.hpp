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

#include "duckdb/main/client_context.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/config.hpp"
#include "parallel/task.hpp"
#include "pipeline/completion_handler.hpp"
#include "planner/query.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

namespace sirius::pipeline {
class gpu_pipeline_executor;
class gpu_pipeline_task_global_state;
}  // namespace sirius::pipeline

namespace sirius::op {
class sirius_physical_duckdb_scan;
}  // namespace sirius::op

namespace sirius::op::scan {
class duckdb_scan_executor;
class duckdb_scan_task_global_state;
}  // namespace sirius::op::scan

namespace sirius::creator {

// WSM TODO: remove this once task_creation_info is removed
// /**
//  * @brief Contains information needed to create a task.
//  *
//  * This class holds a reference to a sirius physical operator node and its associated
//  * pipeline, which together provide the context needed for task creation.
//  */
// class task_creation_info {
//  public:
//   task_creation_info(sirius::op::sirius_physical_operator* node,
//                      duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline)
//     : _node(node), _pipeline(std::move(pipeline))
//   {
//     if (!_pipeline) {
//       return;  // Skip port setup if no pipeline provided
//     }
//     // get next port after sink and then get the data repository from the port
//     auto next_port_after_sink = _pipeline->get_sink()->get_next_port_after_sink();
//     for (auto& [next_op, port_id] : next_port_after_sink) {
//       destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
//     }
//     if (_node->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
//       auto& first_operator = _pipeline->get_inner_operators()[0].get();
//       destination_data_repositories.push_back(first_operator.get_port("scan")->repo);
//     }
//     if (_pipeline->get_sink()->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
//       destination_data_repositories.push_back(_node->get_port("final")->repo);
//     }
//   };
//   ~task_creation_info() = default;
//   sirius::op::sirius_physical_operator* _node;
//   std::vector<cucascade::shared_data_repository*> destination_data_repositories;
//   duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> _pipeline;
// };

// /**
//  * @brief A thread-safe queue for managing task creation requests.
//  *
//  * This queue allows multiple producers to push task creation info and multiple
//  * consumers to pull tasks for processing. It supports open/close semantics to
//  * control when the queue accepts and returns tasks.
//  */
// class task_creation_queue {
//  public:
//   /**
//    * @brief Construct a new task_creation_queue object.
//    *
//    * @param num_threads The number of worker threads that will consume from this queue.
//    *                    Used to send sentinel values when closing the queue.
//    */
//   task_creation_queue(size_t num_threads);

//   /**
//    * @brief Opens the task queue to start accepting and returning tasks.
//    */
//   void open();

//   /**
//    * @brief Closes the task queue from accepting new tasks or returning tasks.
//    *
//    * This method wakes up all threads blocked in pull() by pushing nullptr sentinels.
//    */
//   void close();

//   /**
//    * @brief Push a new task creation info to be scheduled.
//    *
//    * @param info The task creation info to be scheduled.
//    * @throws sirius::runtime_error If the scheduler is not currently accepting requests.
//    */
//   void push(std::unique_ptr<task_creation_info> info);

//   /**
//    * @brief Pull a task to execute.
//    *
//    * This is a blocking call that waits for a task to become available. If the queue
//    * is closed and empty, it returns nullptr to signal that no more tasks will arrive.
//    *
//    * @return A unique pointer to the task creation info if available, nullptr if the
//    *         queue is closed and empty.
//    */
//   std::unique_ptr<task_creation_info> pull();

//   /**
//    * @brief Check if the queue is currently open.
//    *
//    * @return true if the queue is open, false otherwise.
//    */
//   bool is_open() const { return _is_open.load(std::memory_order_acquire); }

//  private:
//   size_t _num_threads;
//   duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<task_creation_info>> _queue;
//   std::atomic<bool> _is_open{false};  ///< Whether the queue is open for pushing/pulling tasks
// };

// WSM TODO: update these comments

/**
 * @brief Manages the creation and scheduling of GPU pipeline tasks.
 *
 * The task_creator is responsible for creating tasks from GPU pipelines and scheduling
 * them for execution. It uses hints from operators to determine the next tasks to create
 * and directly schedules them on the appropriate executor (GPU or scan). It also manages
 * the GPU executors and scan executors.
 *
 * Usage:
 *   1. Construct with executor configurations, memory reservation manager, and optional system
 * topology.
 *   2. Call set_client_context().
 *   3. Call prepare_for_query() to set up for a query.
 *   4. Call start_query() to begin execution and get a completion future.
 *   5. Call reset() between queries to clean up state.
 */
class task_creator {
 public:
  /**
   * @brief Construct a new task_creator.
   *
   * @param task_creator_config Configuration for the task creator
   * @param gpu_executor_config Configuration for the GPU pipeline executor thread pool
   * @param scan_executor_config Configuration for the scan executor thread pool
   * @param mem_res_mgr Reference to the memory reservation manager
   * @param sys_topology Optional system topology info for CPU affinity
   */
  task_creator(parallel::task_executor_config gpu_executor_config,
               parallel::task_executor_config scan_executor_config,
               memory::sirius_memory_reservation_manager& mem_res_mgr,
               const cucascade::memory::system_topology_info* sys_topology = nullptr);

  /**
   * @brief Destructor that ensures executors are stopped.
   */
  virtual ~task_creator();

  // Non-copyable and movable
  task_creator(const task_creator&)            = delete;
  task_creator& operator=(const task_creator&) = delete;
  task_creator(task_creator&&)                 = delete;
  task_creator& operator=(task_creator&&)      = delete;

  /// \brief sets client context needed for task creation
  void set_client_context(::duckdb::ClientContext& client_context);

  /// \brief clean-up query bound resources and prepare the task creator for next query
  void reset();

  /**
   * @brief Schedule a task for the given operator.
   *
   * This method directly creates the task and schedules it on the appropriate executor.
   *
   * @param node The operator node to schedule a task for.
   */
  virtual void schedule(op::sirius_physical_operator* node, int device_id = 0);

  /**
   * @brief Schedules a task for execution on the appropriate executor
   *
   * Routes tasks to either the scan executor or GPU executors based on task type.
   *
   * @param task The task to schedule
   */
  void schedule(std::unique_ptr<parallel::itask> task, int device_id = 0);

  /**
   * @brief Get the next task id.
   *
   * @return uint64_t The next task id.
   */
  uint64_t get_next_task_id();

  /**
   * @brief Get the scan executor reference
   *
   * @return Reference to the duckdb scan executor
   */
  [[nodiscard]] sirius::op::scan::duckdb_scan_executor& get_scan_executor() noexcept;

  [[nodiscard]] const sirius::op::scan::duckdb_scan_executor& get_scan_executor() const noexcept;

  /**
   * @brief Prepare for query execution
   *
   * Sets up the scan operators and prepares the executors.
   *
   * @param query The query to prepare for
   */
  void prepare_for_query(duckdb::shared_ptr<planner::query> query);

  /**
   * @brief Start query execution and return a future for completion.
   *
   * Sets up the completion handler and returns a future that will be satisfied
   * when the query completes or errors. Note: prepare_for_query must be called
   * before this method.
   *
   * @return A future that will be satisfied when the query completes.
   */
  std::future<void> start_query();

 protected:
  /**
   * @brief Find the operator for which to create the next task based on operator hints.
   *
   * This method queries the given node for a hint about what task to create next.
   *
   * @param node The operator node to get the next task hint from.
   * @return The operator node that should be scheduled next, or nullptr if no task should be
   * scheduled.
   */
  op::sirius_physical_operator* get_operator_for_next_task(op::sirius_physical_operator* node);

  /**
   * @brief Create and schedule a task for the given operator.
   *
   * @param node The operator node to create a task for.
   */
  void create_and_schedule_task(op::sirius_physical_operator* node, int device_id);

  /**
   * @brief Schedule the next batch of scan tasks from the priority queue
   */
  void schedule_next_scan_tasks();

  ::duckdb::ClientContext* _client_context;
  sirius::memory::sirius_memory_reservation_manager& _mem_res_mgr;
  std::atomic<uint64_t> _task_id{0};

  // Executor management
  std::unique_ptr<sirius::op::scan::duckdb_scan_executor> _scan_executor;
  std::unordered_map<int, std::unique_ptr<pipeline::gpu_pipeline_executor>> _gpu_executors;

  // Query management
  std::unique_ptr<pipeline::completion_handler> _completion_handler;
  std::mutex _priority_scans_mutex;
  std::queue<op::sirius_physical_duckdb_scan*> _priority_scans;

  // Map of operator ID to global state for scan operators
  std::map<size_t, std::shared_ptr<op::scan::duckdb_scan_task_global_state>>
    _scan_operator_global_state_map;
  std::map<size_t, std::shared_ptr<pipeline::gpu_pipeline_task_global_state>>
    _gpu_operator_global_state_map;
  std::unique_ptr<duckdb::ThreadContext> _thread_context;
  std::unique_ptr<duckdb::ExecutionContext> _execution_context;
  std::mutex _global_state_mutex;  // Protect concurrent access to the map
};

}  // namespace sirius::creator
