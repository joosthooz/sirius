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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/representation_converter.hpp>

namespace sirius {

/// True if @p col or any of its children starts at a non-zero row offset, i.e. views a slice.
[[nodiscard]] bool has_row_offset(cudf::column_view const& col) noexcept;

/// True if any column of @p table views a slice (see has_row_offset(cudf::column_view const&)).
[[nodiscard]] bool has_row_offset(cudf::table_view const& table) noexcept;

/**
 * @brief Make the cucascade builtin converters that cannot read a sliced GPU table throw.
 *
 * The builtin GPU->HOST (fast), GPU->DISK, and cross-device GPU->GPU converters copy every column
 * buffer from its first byte, so they assume a table that is not a slice. Given one they copy the
 * wrong bytes, guarded only by an assert that release builds compile out. This re-registers those
 * three pairs with wrappers that throw cucascade::logic_error for a source whose table has a
 * non-zero row offset, and otherwise forward to the builtin.
 *
 * Call after cucascade::register_builtin_converters(). A converter registered for one of these
 * pairs afterwards, such as the chunked GPU->HOST spill converter, replaces its guard and must
 * handle slices itself.
 */
void register_sliced_view_guards(cucascade::representation_converter_registry& registry);

}  // namespace sirius
