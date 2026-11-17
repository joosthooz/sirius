/*
 * Copyright 2026, Sirius Contributors.
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

#include "data/sliced_view_guards.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace sirius {

bool has_row_offset(cudf::column_view const& col) noexcept
{
  if (col.offset() != 0) { return true; }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    if (has_row_offset(col.child(i))) { return true; }
  }
  return false;
}

bool has_row_offset(cudf::table_view const& table) noexcept
{
  return std::any_of(
    table.begin(), table.end(), [](cudf::column_view const& col) { return has_row_offset(col); });
}

namespace {

using registry_type = cucascade::representation_converter_registry;

/// Which sources a guard refuses: every sliced one, or only those crossing devices.
enum class guard_scope { always, cross_device };

/// Wrap the builtin converter to @p Target so it throws for a sliced GPU source instead of
/// copying the wrong bytes. @p builtins holds the unwrapped builtin converters.
template <typename Target>
cucascade::representation_converter_fn guarded(std::shared_ptr<registry_type const> builtins,
                                               char const* conversion,
                                               guard_scope scope)
{
  return [builtins = std::move(builtins), conversion, scope](
           cucascade::idata_representation const& source,
           cucascade::memory::memory_space const* target_memory_space,
           ::cuda::stream_ref stream,
           cucascade::memory::reservation* reservation)
           -> std::unique_ptr<cucascade::idata_representation> {
    bool const checked = scope == guard_scope::always ||
                         source.get_device_id() != target_memory_space->get_device_id();
    if (checked &&
        has_row_offset(source.cast<cucascade::gpu_table_representation>().get_table_view())) {
      CUCASCADE_FAIL(std::string{conversion} +
                     ": a sliced GPU table (non-zero row offset) is not supported");
    }
    if (reservation != nullptr) { return builtins->convert<Target>(source, *reservation, stream); }
    return builtins->convert<Target>(source, target_memory_space, stream);
  };
}

}  // namespace

void register_sliced_view_guards(registry_type& registry)
{
  auto builtins = std::make_shared<registry_type>();
  cucascade::register_builtin_converters(*builtins);
  std::shared_ptr<registry_type const> const unwrapped = std::move(builtins);

  using cucascade::gpu_table_representation;
  // register_converter throws on duplicate keys, so drop each builtin registration first.
  registry.unregister_converter<gpu_table_representation, cucascade::host_data_representation>();
  registry.register_converter<gpu_table_representation, cucascade::host_data_representation>(
    guarded<cucascade::host_data_representation>(unwrapped, "GPU->HOST", guard_scope::always));
  registry.unregister_converter<gpu_table_representation, cucascade::disk_data_representation>();
  registry.register_converter<gpu_table_representation, cucascade::disk_data_representation>(
    guarded<cucascade::disk_data_representation>(unwrapped, "GPU->DISK", guard_scope::always));
  // A same-device GPU->GPU conversion is a clone, which deep-copies the view and so compacts it.
  registry.unregister_converter<gpu_table_representation, gpu_table_representation>();
  registry.register_converter<gpu_table_representation, gpu_table_representation>(
    guarded<gpu_table_representation>(unwrapped, "GPU->GPU", guard_scope::cross_device));
}

}  // namespace sirius
