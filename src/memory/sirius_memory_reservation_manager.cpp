
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

#include "memory/sirius_memory_reservation_manager.hpp"

#include "cucascade/memory/common.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <spdlog/spdlog.h>

namespace sirius {
namespace memory {

sirius_memory_reservation_manager::sirius_memory_reservation_manager(
  const std::vector<cucascade::memory::memory_space_config>& configs)
  : cucascade::memory::memory_reservation_manager(configs)
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error("At least one GPU memory space must be configured");
  }
  std::for_each(gpu_spaces.begin(), gpu_spaces.end(), [](const auto* space) {
    auto* device_mr = space->get_default_allocator();
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{space->get_device_id()}};
    cudf::set_current_device_resource(device_mr);
  });
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  std::for_each(gpu_spaces.begin(), gpu_spaces.end(), [](const auto* space) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{space->get_device_id()}};
    cudf::reset_current_device_resource_ref();
  });
}

void sirius_memory_reservation_manager::log_peak_memory_stats() const
{
  for (const auto* space : this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    auto* adaptor = dynamic_cast<cucascade::memory::reservation_aware_resource_adaptor*>(
      space->get_default_allocator());
    if (adaptor) {
      auto peak_bytes = adaptor->get_peak_total_allocated_bytes();
      spdlog::info("Peak device memory (GPU {}): {} bytes ({:.2f} MiB)",
                   space->get_device_id(),
                   peak_bytes,
                   static_cast<double>(peak_bytes) / (1024.0 * 1024.0));
    }
  }

  for (const auto* space : this->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST)) {
    auto* host_mr = dynamic_cast<cucascade::memory::fixed_size_host_memory_resource*>(
      space->get_default_allocator());
    if (host_mr) {
      auto peak_bytes = host_mr->get_peak_total_allocated_bytes();
      spdlog::info("Peak host memory (node {}): {} bytes ({:.2f} MiB)",
                   space->get_device_id(),
                   peak_bytes,
                   static_cast<double>(peak_bytes) / (1024.0 * 1024.0));
    }
  }
}

}  // namespace memory
}  // namespace sirius
