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

// sirius
#include <helper/utils.hpp>
#include <memory/host_table_utils.hpp>
#include <result/host_table_chunk_reader.hpp>

// duckdb
#include <duckdb/common/vector_size.hpp>

// standard library
#include <algorithm>
#include <cstring>
#include <vector>

namespace sirius::op::result {

host_table_chunk_reader::column_reader::column_reader(
  metadata_node const& node, std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  size       = static_cast<size_t>(node.size);
  null_count = static_cast<size_t>(node.null_count);
  if (node.null_mask_offset < 0) { null_count = 0; }

  data_accessor.initialize(static_cast<size_t>(node.data_offset), allocation);

  if (null_count > 0) {
    mask_accessor.initialize(static_cast<size_t>(node.null_mask_offset), allocation);
  }

  if (node.type.id() == cudf::type_id::STRING) {
    if (node.children.size() != 1) {
      throw std::runtime_error(
        "[host_table_chunk_reader::column_reader::initialize_accessors] STRING type must have one "
        "child node for offsets.");
    }
    offset_accessor.initialize(node.children[0].data_offset, allocation);
  }
}

void host_table_chunk_reader::column_reader::copy_mask_to_validity(
  duckdb::ValidityMask& validity,
  size_t row_offset,
  size_t count,
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(row_offset + count <= static_cast<size_t>(size));
  assert(utils::mod_8(row_offset) == 0);  // Must be byte-aligned start

  // Initialize validity mask
  validity.Initialize(count);

  auto* validity_ptr       = reinterpret_cast<uint8_t*>(validity.GetData());
  auto const bytes_to_copy = utils::ceil_div_8(count);
  mask_accessor.memcpy_to(allocation, validity_ptr, bytes_to_copy);
}

void host_table_chunk_reader::column_reader::copy_fixed_width(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() != duckdb::PhysicalType::VARCHAR);
  assert(row_offset + count <= static_cast<size_t>(size));

  // We are copying into a flat vector
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  // Do the data copy
  auto const type_size =
    static_cast<size_t>(duckdb::GetTypeIdSize(vector.GetType().InternalType()));
  auto* dest_ptr = duckdb::FlatVector::GetData<uint8_t>(vector);
  data_accessor.memcpy_to(allocation, dest_ptr, count * type_size);

  // Do the validity mask copy, if necessary
  if (null_count != 0) {
    auto& validity = duckdb::FlatVector::Validity(vector);
    copy_mask_to_validity(validity, row_offset, count, allocation);
  }
}

void host_table_chunk_reader::column_reader::copy_string(
  duckdb::Vector& vector,
  size_t row_offset,
  size_t count,
  std::unique_ptr<multiple_blocks_allocation> const& allocation)
{
  assert(vector.GetType().InternalType() == duckdb::PhysicalType::VARCHAR);
  assert(row_offset + count <= static_cast<size_t>(size));

  // We are copying into a flat vector
  vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);

  // Allocate duckdb buffer and bulk copy data into it
  auto const start_offset     = offset_accessor.get_current(allocation);
  auto const end_offset       = offset_accessor.get(row_offset + count, allocation);
  auto const total_data_bytes = end_offset - start_offset;
  auto str_buffer             = duckdb::make_buffer<duckdb::VectorBuffer>(total_data_bytes);
  auto str_buffer_ptr         = str_buffer->GetData();
  data_accessor.memcpy_to(allocation, str_buffer_ptr, total_data_bytes);

  // Set the null mask, if necessary
  if (null_count != 0) {
    auto& validity = duckdb::FlatVector::Validity(vector);
    copy_mask_to_validity(validity, row_offset, count, allocation);
  }

  // Construct each string
  auto strings = duckdb::FlatVector::GetData<duckdb::string_t>(vector);
  auto start   = start_offset;
  for (size_t row = 0; row < count; ++row) {
    offset_accessor.advance();
    auto end = offset_accessor.get_current(allocation);
    if (!duckdb::FlatVector::IsNull(vector, row)) {
      auto d_ptr   = str_buffer_ptr + (start - start_offset);
      auto str_len = end - start;
      strings[row] = duckdb::string_t(reinterpret_cast<char*>(d_ptr), str_len);
    }
    start = end;
  }

  duckdb::StringVector::AddBuffer(vector, str_buffer);
}

host_table_chunk_reader::host_table_chunk_reader(
  cucascade::host_table_representation const& host_table,
  duckdb::vector<duckdb::LogicalType> const& types_p)
  : allocation(host_table.get_host_table()->allocation), types(types_p)
{
  // Unpack metadata
  auto metadata_nodes = sirius::unpack_metadata_to_nodes(host_table.get_host_table()->metadata);

  if (metadata_nodes.size() != types.size()) {
    throw std::runtime_error(
      "[host_table_chunk_reader] Metadata column count does not match expected column count.");
  }

  // Initialize column readers
  for (size_t col_idx = 0; col_idx < metadata_nodes.size(); ++col_idx) {
    if (col_idx == 0) {
      total_rows = static_cast<size_t>(metadata_nodes[col_idx].size);
      if (total_rows < 0) {
        throw std::runtime_error("[host_table_chunk_reader] Negative total rows in first column.");
      }
    } else if (metadata_nodes[col_idx].size != total_rows) {
      throw std::runtime_error(
        "[host_table_chunk_reader] Metadata column size mismatch across columns.");
    }

    // For the time being, we do not handle HUGEINT, as cudf does not support it
    if (types[col_idx] == duckdb::LogicalType::HUGEINT) {
      throw std::runtime_error(
        "[host_table_chunk_reader] HUGEINT type is not currently supported.");
    }

    column_readers.emplace_back(metadata_nodes[col_idx], allocation);
  }
}

bool host_table_chunk_reader::get_next_chunk(duckdb::DataChunk& chunk)
{
  if (row_offset >= total_rows) {
    chunk.SetCardinality(0);
    return false;
  }

  // Initialize the chunk
  auto const remaining = total_rows - row_offset;
  auto const count     = std::min(remaining, static_cast<size_t>(STANDARD_VECTOR_SIZE));
  /// TODO: pass in client context allocator?
  chunk.Initialize(duckdb::Allocator::DefaultAllocator(), types, count);

  // Copy each column into the chunk
  for (size_t col_idx = 0; col_idx < column_readers.size(); ++col_idx) {
    auto& vec = chunk.data[col_idx];
    if (vec.GetType().InternalType() == duckdb::PhysicalType::VARCHAR) {
      column_readers[col_idx].copy_string(vec, row_offset, count, allocation);
    } else {
      column_readers[col_idx].copy_fixed_width(vec, row_offset, count, allocation);
    }
  }

  chunk.SetCardinality(static_cast<duckdb::idx_t>(count));
  row_offset += count;

  return true;
}
}  // namespace sirius::op::result
