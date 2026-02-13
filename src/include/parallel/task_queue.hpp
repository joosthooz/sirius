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

#include "task.hpp"

#include <blockingconcurrentqueue.h>

#include <atomic>
#include <memory>

namespace sirius {
namespace parallel {

/**
 * Interface for concrete task queues for customized scheduling policies.
 */
class itask_queue {
 public:
  virtual ~itask_queue() = default;

  // Open the queue and start accepting new tasks.
  virtual void open() = 0;

  // Close the queue and stop processing new tasks.
  virtual void close() = 0;

  // Add a task to the queue.
  virtual void push(std::unique_ptr<itask> task) = 0;

  // Pull a task from the queue. Wait until a task available or the queue is closed.
  virtual std::unique_ptr<itask> pull() = 0;
};

/**
 * @brief A concurrent task queue implementing itask_queue.
 *
 * Wraps moodycamel's BlockingConcurrentQueue (from DuckDB's blockingconcurrentqueue.h).
 * Used by both scan and GPU pipeline executors.
 */
class task_queue : public itask_queue {
 public:
  explicit task_queue(size_t num_threads) : _num_threads(num_threads), _queue() {}
  ~task_queue() override = default;

  void open() override
  {
    _is_open.store(true, std::memory_order_release);
    std::unique_ptr<itask> task;
    while (_queue.try_dequeue(task)) {
      // Drain any remaining items (including nullptr sentinels) from previous close()
    }
  }

  void close() override
  {
    _is_open.store(false, std::memory_order_release);
    for (size_t i = 0; i < _num_threads; ++i) {
      _queue.enqueue(nullptr);
    }
  }

  void push(std::unique_ptr<itask> task) override { _queue.enqueue(std::move(task)); }

  std::unique_ptr<itask> pull() override
  {
    std::unique_ptr<itask> task;
    while (true) {
      if (_queue.try_dequeue(task)) { return task; }
      if (!_is_open.load(std::memory_order_acquire)) { return nullptr; }
      _queue.wait_dequeue(task);
      if (task) { return task; }
    }
  }

 private:
  size_t _num_threads;
  std::atomic<bool> _is_open{false};
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<itask>> _queue;
};

}  // namespace parallel
}  // namespace sirius
