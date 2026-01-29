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

#include "pipeline/gpu_pipeline_queue.hpp"
#include "pipeline/pipeline_executor.hpp"

#include <absl/cleanup/cleanup.h>

#include <memory>

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(sirius::parallel::task_executor_config config,
                                             const cucascade::memory::memory_space* mem_space,
                                             pipeline_executor* pipeline_exec)
  : itask_executor(std::make_unique<gpu_pipeline_queue>(config.num_threads), std::move(config)),
    _memory_space_view(mem_space),
    _pipeline_exec(pipeline_exec)
{
}

void gpu_pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue->push(std::move(task));
}

void gpu_pipeline_executor::on_start() { _task_queue->open(); }

void gpu_pipeline_executor::on_stop() { _task_queue->close(); }

void gpu_pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    auto& t = _threads.emplace_back(&gpu_pipeline_executor::worker_loop, this, i);
    if (!_config.cpu_affinity_list.empty()) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      for (int core_id : _config.cpu_affinity_list) {
        CPU_SET(core_id, &cpuset);
      }
      pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
  }
  _gpu_pipeline_executor_manager_thread =
    std::make_unique<std::thread>(&gpu_pipeline_executor::manager_loop, this);
}

void gpu_pipeline_executor::stop()
{
  std::unique_lock<std::mutex> lock(_task_count_mutex);
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  lock.unlock();
  _req_count_cv.notify_all();
  on_stop();
  for (auto& thread : _threads) {
    if (thread.joinable()) { thread.join(); }
  }
  if (_gpu_pipeline_executor_manager_thread->joinable()) {
    _gpu_pipeline_executor_manager_thread->join();
  }
  _gpu_pipeline_executor_manager_thread.reset();
  _threads.clear();
}

void gpu_pipeline_executor::worker_loop(int worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto task    = _task_queue->pull();
    auto cleanup = absl::MakeCleanup([this]() {
      std::unique_lock<std::mutex> lock(_task_count_mutex);
      _num_active_requests--;
      lock.unlock();
      _req_count_cv.notify_one();
    });
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      // TODO:
      // if reservation hasn't been made, request reservation (blocking)
      // set stream reservation
      task->execute();
      // reset memory resource
    } catch (const std::exception& e) {
      on_task_error(worker_id, std::move(task), e);
    }
  }
}

void gpu_pipeline_executor::submit_task_request(std::unique_ptr<task_request> request)
{
  _pipeline_exec->submit_task_request(std::move(request));
}

void gpu_pipeline_executor::manager_loop()
{
  while (_running) {
    std::unique_lock<std::mutex> lock(_task_count_mutex);
    _req_count_cv.wait(
      lock, [this]() { return _num_active_requests < _config.num_threads || !_running.load(); });
    if (!_running.load()) { break; }
    _num_active_requests++;
    lock.unlock();
    submit_task_request(std::make_unique<task_request>(_memory_space_view->get_device_id(), false));
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

}  // namespace pipeline
}  // namespace sirius
