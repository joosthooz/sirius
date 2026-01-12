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

// sirius
#include <helper/utils.hpp>
#include <memory/host_table_utils.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/vector_size.hpp>

// cudf
#include <cudf/types.hpp>

// cucascade
#include <data/cpu_data_representation.hpp>
#include <memory/fixed_size_host_memory_resource.hpp>

// standard library
#include <memory>
#include <vector>

namespace sirius::op::result {

class host_table_chunk_reader {
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  struct column_reader {
    size_t size{0};
    size_t null_count{0};
    ::sirius::memory::multiple_blocks_allocation_accessor<uint8_t> data_accessor;
    ::sirius::memory::multiple_blocks_allocation_accessor<uint8_t> mask_accessor;
    ::sirius::memory::multiple_blocks_allocation_accessor<int64_t> offset_accessor;

    column_reader(metadata_node const& node,
                  std::unique_ptr<multiple_blocks_allocation> const& allocation);
    void copy_mask_to_validity(duckdb::ValidityMask& validity,
                               size_t row_offset,
                               size_t count,
                               std::unique_ptr<multiple_blocks_allocation> const& allocation);
    void copy_fixed_width(duckdb::Vector& vector,
                          size_t row_offset,
                          size_t count,
                          std::unique_ptr<multiple_blocks_allocation> const& allocation);
    void copy_string(duckdb::Vector& vector,
                     size_t row_offset,
                     size_t count,
                     std::unique_ptr<multiple_blocks_allocation> const& allocation);
  };

 public:
  host_table_chunk_reader(cucascade::host_table_representation const& host_table,
                          duckdb::vector<duckdb::LogicalType> const& types);
  ~host_table_chunk_reader() = default;
  bool get_next_chunk(duckdb::DataChunk& chunk);
  size_t calculate_num_chunks()
  {
    return utils::ceil_div(total_rows, static_cast<size_t>(STANDARD_VECTOR_SIZE));
  }

 private:
  std::unique_ptr<multiple_blocks_allocation> const& allocation;
  duckdb::vector<duckdb::LogicalType> types;
  size_t total_rows{0};
  size_t row_offset{0};
  std::vector<column_reader> column_readers;
};

}  // namespace sirius::op::result
