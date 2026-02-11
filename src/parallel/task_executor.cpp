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

#include "parallel/task_executor.hpp"

#include "log/logging.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <pthread.h>

#include <latch>

namespace sirius {
namespace parallel {

void itask_executor::start() { start(nullptr); }

void itask_executor::start(absl::AnyInvocable<void() noexcept> per_thread_init)
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();

  // Set up CPU affinity if specified
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (!_config.cpu_affinity_list.empty()) {
    for (int cpu_id : _config.cpu_affinity_list) {
      CPU_SET(cpu_id, &cpuset);
    }
  }

  // Create latch if per-thread initialization is requested
  std::unique_ptr<std::latch> init_latch;
  if (per_thread_init) { init_latch = std::make_unique<std::latch>(_config.num_threads); }

  auto* init_fn_ptr = per_thread_init ? &per_thread_init : nullptr;
  auto* latch_ptr   = init_latch.get();

  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    auto& t = _threads.emplace_back([this, i, init_fn_ptr, latch_ptr]() {
      // Run per-thread initialization if provided
      if (init_fn_ptr) {
        (*init_fn_ptr)();
        latch_ptr->count_down();
      }
      worker_loop(i);
    });

    // Set thread name if specified
    if (!_config.thread_name_prefix.empty()) {
      std::string thread_name = _config.thread_name_prefix + "_" + std::to_string(i);
      pthread_setname_np(t.native_handle(), thread_name.c_str());
    }

    // Set CPU affinity if specified
    if (!_config.cpu_affinity_list.empty()) {
      pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
  }

  // Wait for all threads to complete initialization
  if (init_latch) { init_latch->wait(); }
}

void itask_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  for (auto& thread : _threads) {
    if (thread.joinable()) { thread.join(); }
  }
  _threads.clear();
}

void itask_executor::schedule(std::unique_ptr<itask> task) { _task_queue->push(std::move(task)); }

void itask_executor::on_start() { _task_queue->open(); }

void itask_executor::on_stop() { _task_queue->close(); }

void itask_executor::worker_loop(int worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto task = _task_queue->pull();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      task->execute(cudf::get_default_stream());
    } catch (const std::exception& e) {
      if (_config.retry_on_error) {
        SIRIUS_LOG_ERROR("itask_executor::worker_loop(): Error executing task, retrying: {}",
                         e.what());
        schedule(std::move(task));
      } else {
        SIRIUS_LOG_ERROR(
          "itask_executor::worker_loop(): Error executing task, stopping executor: {}", e.what());
        stop();
      }
    } catch (...) {
      SIRIUS_LOG_ERROR(
        "itask_executor::worker_loop(): Unknown error executing task, stopping executor");
      stop();
    }
  }
}

}  // namespace parallel
}  // namespace sirius
