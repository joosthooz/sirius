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

#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace sirius::transparent {

/**
 * @brief Turn every MARK join whose marker only feeds the FILTER directly above it into a SEMI
 * join.
 *
 * Such a FILTER keeps the rows whose marker is TRUE, which are exactly the rows a SEMI join keeps;
 * rows with a FALSE or NULL marker are dropped by both. DuckDB converts this shape itself, except
 * when the marker was still referenced at the time it checked: with delim_join_as_cte the marker
 * of `x IN (subquery)` is an output of the materialized CTE until the CTE filter pusher moves the
 * filter into the CTE definition, so TPC-H q20 keeps a MARK join that the GPU cannot use as a
 * dynamic-filter source.
 *
 * Accepts plans with positional or binding column references. Only joins on plain equality
 * conditions whose FILTER has a projection map without the marker are converted, so no operator
 * above the FILTER can read the dropped marker column.
 *
 * @return The number of MARK joins converted.
 */
std::size_t convert_filtered_mark_joins(duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace sirius::transparent
