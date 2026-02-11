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

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"
#include "parallel/config.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "scan/test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/device_memory_resource.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kReservationBytes = 20 * 1024 * 1024;
constexpr std::size_t kAllocationBytes  = 10 * 1024 * 1024;

// Helper function to create a simple test data batch with two int64 columns
std::shared_ptr<cucascade::data_batch> create_test_batch(cucascade::memory::memory_space& space,
                                                         uint64_t batch_id,
                                                         int64_t start_value,
                                                         int64_t num_rows)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = cudf::get_default_stream();

  // Create two simple int64 columns
  std::vector<int64_t> col0_vals(num_rows);
  std::vector<int64_t> col1_vals(num_rows);
  for (int64_t i = 0; i < num_rows; ++i) {
    col0_vals[i] = start_value + i;
    col1_vals[i] = start_value + i * 2;
  }

  auto col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                        static_cast<cudf::size_type>(num_rows),
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        mr);
  cudaMemcpy(col0->mutable_view().data<int64_t>(),
             col0_vals.data(),
             sizeof(int64_t) * col0_vals.size(),
             cudaMemcpyHostToDevice);

  auto col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                        static_cast<cudf::size_type>(num_rows),
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        mr);
  cudaMemcpy(col1->mutable_view().data<int64_t>(),
             col1_vals.data(),
             sizeof(int64_t) * col1_vals.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col0));
  cols.push_back(std::move(col1));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(std::move(table), space);
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(gpu_repr));
}

class test_gpu_pipeline_task_global_state
  : public sirius::pipeline::gpu_pipeline_task_global_state {
 public:
  test_gpu_pipeline_task_global_state() : gpu_pipeline_task_global_state(nullptr) {}

  void add_error(std::string message)
  {
    std::cerr << message << std::endl;
    error_count.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(error_mutex);
    errors.push_back(std::move(message));
  }

  std::atomic<int> executed_count{0};
  std::atomic<int> error_count{0};
  std::mutex error_mutex;
  std::vector<std::string> errors;

  std::mutex memory_mutex;
  std::vector<std::size_t> memory_consumption;
};

class test_gpu_pipeline_task_local_state : public sirius::pipeline::gpu_pipeline_task_local_state {
 public:
  using sirius::pipeline::gpu_pipeline_task_local_state::gpu_pipeline_task_local_state;
};

class sirius_pipeline_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  sirius_pipeline_task(uint64_t task_id,
                       std::unique_ptr<test_gpu_pipeline_task_local_state> local_state,
                       std::shared_ptr<test_gpu_pipeline_task_global_state> global_state)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<test_gpu_pipeline_task_global_state>();
    auto& local  = _local_state->cast<test_gpu_pipeline_task_local_state>();

    auto reservation = local.release_reservation();
    if (!reservation) {
      global.add_error("Missing GPU memory reservation for task.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    auto& mem_space = reservation->get_memory_space();
    auto* allocator =
      reservation->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    if (!allocator) {
      global.add_error("Missing reservation-aware allocator for GPU memory space.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (!allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
      global.add_error("Failed to attach reservation to stream tracker.");
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    void* allocation = nullptr;
    try {
      allocation = allocator->allocate(stream, kAllocationBytes);
    } catch (const std::exception& e) {
      global.add_error(std::string("GPU allocation failed: ") + e.what());
      allocator->reset_stream_reservation(stream);
      global.executed_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    allocator->deallocate(stream, allocation, kAllocationBytes);

    auto consumed_bytes = mem_space.get_total_reserved_memory();
    {
      std::lock_guard<std::mutex> lock(global.memory_mutex);
      global.memory_consumption.push_back(consumed_bytes);
    }

    allocator->reset_stream_reservation(stream);
    global.executed_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::size_t get_estimated_reservation_size() const override { return kReservationBytes; }

  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }
};

}  // namespace

TEST_CASE("GPU pipeline executor schedules and executes GPU tasks", "[gpu_pipeline_executor]")
{
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  try {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(256 * 1024 * 1024)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_host_capacity(1 * 1024 * 1024 * 1024)
      .use_host_per_gpu()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_host(0.75);
    auto space_configs = builder.build();
    manager =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  } catch (const std::exception& e) {
    WARN("Skipping test due to insufficient GPUs: " << e.what());
    return;
  }

  auto* mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (!mem_space) {
    WARN("Skipping test because no GPU memory space is available.");
    return;
  }

  sirius::parallel::task_executor_config config;
  config.num_threads        = 2;
  config.thread_name_prefix = "gpu-pipeline-test";

  sirius::pipeline::gpu_pipeline_executor executor(config, mem_space);
  auto global_state = std::make_shared<test_gpu_pipeline_task_global_state>();

  const int num_tasks = 10;

  executor.start();

  for (int i = 0; i < num_tasks; ++i) {
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
    batches.push_back(create_test_batch(*mem_space, i, i * 100, 100));

    auto local_state = std::make_unique<test_gpu_pipeline_task_local_state>(std::move(batches));
    auto task = std::make_unique<sirius_pipeline_task>(i, std::move(local_state), global_state);
    executor.schedule(std::move(task));
  }

  // Wait for all tasks to complete
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(20);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      executor.stop();
      FAIL("Timed out waiting for GPU pipeline tasks to complete.");
    }
  }

  executor.stop();

  if (global_state->error_count.load(std::memory_order_relaxed) > 0) {
    std::lock_guard<std::mutex> lock(global_state->error_mutex);
    for (const auto& error : global_state->errors) {
      INFO(error);
    }
  }

  REQUIRE(global_state->error_count.load(std::memory_order_relaxed) == 0);
  REQUIRE(global_state->executed_count.load(std::memory_order_relaxed) == num_tasks);

  {
    std::lock_guard<std::mutex> lock(global_state->memory_mutex);
    REQUIRE(global_state->memory_consumption.size() == static_cast<size_t>(num_tasks));
    for (auto consumed_bytes : global_state->memory_consumption) {
      REQUIRE(consumed_bytes >= kReservationBytes);
    }
  }
}
