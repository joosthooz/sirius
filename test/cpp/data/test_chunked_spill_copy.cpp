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
#include "data/chunked_spill_copy.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "data/sliced_view_guards.hpp"
#include "data/spill_chunked_converters.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using sirius::spill::chunked_copy_batcher;
using sirius::spill::copy_op;

namespace {

/// Fake source/destination addresses; the batcher never dereferences them.
void* fake_ptr(std::uintptr_t v) { return reinterpret_cast<void*>(v); }

struct captured_chunk {
  std::vector<copy_op> ops;
  std::size_t bytes;
};

std::vector<captured_chunk> run_batcher(std::size_t chunk_bytes,
                                        const std::vector<std::size_t>& op_sizes)
{
  std::vector<captured_chunk> chunks;
  chunked_copy_batcher batcher(chunk_bytes, [&](std::span<const copy_op> ops) {
    captured_chunk c;
    c.ops.assign(ops.begin(), ops.end());
    c.bytes =
      std::accumulate(ops.begin(), ops.end(), std::size_t{0}, [](std::size_t a, const copy_op& op) {
        return a + op.size;
      });
    chunks.push_back(std::move(c));
  });
  std::uintptr_t addr = 0x1000;
  for (auto size : op_sizes) {
    batcher.add(fake_ptr(addr), fake_ptr(addr + 0x100000000ull), size);
    addr += size;
  }
  batcher.flush_pending();
  return chunks;
}

}  // namespace

TEST_CASE("chunked_copy_batcher flushes at chunk boundaries", "[chunked_spill_copy]")
{
  auto chunks = run_batcher(8, {4, 4, 4});
  REQUIRE(chunks.size() == 2);
  REQUIRE(chunks[0].bytes == 8);
  REQUIRE(chunks[1].bytes == 4);
}

TEST_CASE("chunked_copy_batcher splits ops larger than the chunk", "[chunked_spill_copy]")
{
  // No submission, and no op within one, may exceed the chunk size.
  auto chunks = run_batcher(8, {28});
  REQUIRE(chunks.size() == 4);  // 8 + 8 + 8 + 4
  std::size_t total = 0;
  for (auto const& c : chunks) {
    REQUIRE(c.bytes <= 8);
    for (auto const& op : c.ops) {
      REQUIRE(op.size <= 8);
    }
    total += c.bytes;
  }
  REQUIRE(total == 28);
  REQUIRE(chunks.back().bytes == 4);
}

TEST_CASE("chunked_copy_batcher split pieces are contiguous", "[chunked_spill_copy]")
{
  // Split pieces must tile the original [dst, dst+size) range in order.
  std::vector<copy_op> all_ops;
  chunked_copy_batcher batcher(10, [&](std::span<const copy_op> ops) {
    all_ops.insert(all_ops.end(), ops.begin(), ops.end());
  });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 25);
  batcher.flush_pending();

  REQUIRE(all_ops.size() == 3);  // 10 + 10 + 5
  std::uintptr_t expected_dst = 0x1000;
  std::uintptr_t expected_src = 0x2000;
  for (auto const& op : all_ops) {
    REQUIRE(reinterpret_cast<std::uintptr_t>(op.dst) == expected_dst);
    REQUIRE(reinterpret_cast<std::uintptr_t>(op.src) == expected_src);
    expected_dst += op.size;
    expected_src += op.size;
  }
  REQUIRE(expected_dst == 0x1000 + 25);
}

TEST_CASE("chunked_copy_batcher with chunk_bytes 0 flushes once", "[chunked_spill_copy]")
{
  auto chunks = run_batcher(0, {4, 4, 4, 4});
  REQUIRE(chunks.size() == 1);
  REQUIRE(chunks[0].bytes == 16);
}

TEST_CASE("chunked_copy_batcher ignores empty and null ops", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 0);
  batcher.add(nullptr, fake_ptr(0x2000), 8);
  batcher.add(fake_ptr(0x1000), nullptr, 8);
  batcher.flush_pending();
  REQUIRE(flushes == 0);
  REQUIRE(batcher.bytes_added() == 0);
  REQUIRE(batcher.chunks_flushed() == 0);
}

TEST_CASE("chunked_copy_batcher reports stats", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 20);
  batcher.flush_pending();
  REQUIRE(batcher.bytes_added() == 20);
  REQUIRE(batcher.chunks_flushed() == flushes);
  REQUIRE(batcher.chunks_flushed() == 3);  // 8 + 8 + 4
  REQUIRE(batcher.largest_submission_bytes() == 8);
}

TEST_CASE("chunked_copy_batcher flush_pending is idempotent", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 4);
  batcher.flush_pending();
  batcher.flush_pending();
  REQUIRE(flushes == 1);
}

TEST_CASE("register_chunked_spill_converters replaces the GPU->HOST pair", "[chunked_spill_copy]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());

  sirius::spill::register_chunked_spill_converters(registry, 1ull << 20);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());
  // The HOST->GPU restore direction stays on the builtin converter.
  REQUIRE(
    registry
      .has_converter<cucascade::host_data_representation, cucascade::gpu_table_representation>());

  // Registering again must not throw on the duplicate key.
  REQUIRE_NOTHROW(sirius::spill::register_chunked_spill_converters(registry, 2ull << 20));
}

TEST_CASE("register_chunked_spill_converters with 0 keeps the builtin converter",
          "[chunked_spill_copy]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  sirius::spill::register_chunked_spill_converters(registry, 0);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());
}

namespace {

struct spill_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager =
    sirius::test::operator_utils::initialize_memory_manager();
  cucascade::memory::memory_space* gpu_space =
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  cucascade::memory::memory_space* host_space =
    manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
};

spill_env& env()
{
  sirius::test::operator_utils::ensure_converter_registry();
  static spill_env e;
  return e;
}

constexpr cudf::size_type kParentRows = 1000;
constexpr cudf::size_type kNullBegin  = 100;
constexpr cudf::size_type kNullEnd    = 140;

/// A view batch of rows [@p begin, @p end) of a table holding rows 0..999 as BIGINT, rows
/// [100, 140) NULL, plus the same values as strings.
std::shared_ptr<cucascade::data_batch> make_sliced_batch(cudf::size_type begin,
                                                         cudf::size_type end,
                                                         rmm::cuda_stream_view stream)
{
  cudf::numeric_scalar<int64_t> const zero{0, true, stream};
  auto ints = cudf::sequence(kParentRows, zero, stream);
  auto mask = cudf::create_null_mask(kParentRows, cudf::mask_state::ALL_VALID, stream);
  cudf::set_null_mask(
    static_cast<cudf::bitmask_type*>(mask.data()), kNullBegin, kNullEnd, false, stream);
  auto strings = cudf::strings::from_integers(ints->view(), stream);
  ints->set_null_mask(std::move(mask), kNullEnd - kNullBegin);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(ints));
  columns.push_back(std::move(strings));
  auto parent = std::make_shared<cudf::table>(std::move(columns));

  auto const slice = cudf::slice(parent->view(), {begin, end}, stream)[0];
  return sirius::make_data_batch_from_view(slice,
                                           parent,
                                           parent->alloc_size(),
                                           *env().gpu_space,
                                           stream,
                                           sirius::telemetry::batch_telemetry_info{});
}

/// Require that @p batch, on the GPU, holds exactly rows [@p begin, @p end) of the parent.
void require_slice_rows(cucascade::data_batch& batch,
                        cudf::size_type begin,
                        cudf::size_type end,
                        rmm::cuda_stream_view stream)
{
  auto ro         = batch.to_read_only();
  auto const view = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  auto const rows = static_cast<std::size_t>(end - begin);
  REQUIRE(view.num_rows() == end - begin);

  auto const download = [&](cudf::column_view const& col) {
    std::vector<int64_t> out(rows);
    REQUIRE(cudaMemcpyAsync(out.data(),
                            col.data<int64_t>(),
                            rows * sizeof(int64_t),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    return out;
  };
  auto const values        = download(view.column(0));
  auto const parsed_column = cudf::strings::to_integers(
    cudf::strings_column_view{view.column(1)}, cudf::data_type{cudf::type_id::INT64}, stream);
  auto const parsed = download(parsed_column->view());
  auto const nulls  = cudf::is_null(view.column(0), stream);
  std::vector<uint8_t> is_null(rows);
  REQUIRE(
    cudaMemcpyAsync(
      is_null.data(), nulls->view().data<bool>(), rows, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();

  for (std::size_t i = 0; i < rows; ++i) {
    auto const row = static_cast<int64_t>(begin) + static_cast<int64_t>(i);
    INFO("row " << row);
    REQUIRE(parsed[i] == row);
    REQUIRE(static_cast<bool>(is_null[i]) == (row >= kNullBegin && row < kNullEnd));
    if (!is_null[i]) { REQUIRE(values[i] == row); }
  }
}

// An offset that is neither element- nor bitmask-word-aligned in bytes.
constexpr cudf::size_type kSliceBegin = 37;
constexpr cudf::size_type kSliceEnd   = 637;

}  // namespace

TEST_CASE("chunked spill of a sliced view round-trips the slice's rows", "[chunked_spill_copy]")
{
  rmm::cuda_stream owned_stream;
  auto const stream = owned_stream.view();
  auto batch        = make_sliced_batch(kSliceBegin, kSliceEnd, stream);
  // After env(): setting up the memory manager may replace the registry instance.
  auto const& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, env().host_space, stream);
    REQUIRE(mut.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
    mut.convert_to<cucascade::gpu_table_representation>(registry, env().gpu_space, stream);
  }
  require_slice_rows(*batch, kSliceBegin, kSliceEnd, stream);
}

TEST_CASE("guarded builtin converters refuse a sliced GPU table", "[chunked_spill_copy]")
{
  rmm::cuda_stream owned_stream;
  auto const stream = owned_stream.view();
  // The builtins plus their guards, without the chunked converter that replaces the GPU->HOST
  // guard in the Sirius registry.
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  sirius::register_sliced_view_guards(registry);

  SECTION("a slice that does not start at row 0 throws and stays on the GPU")
  {
    auto batch = make_sliced_batch(kSliceBegin, kSliceEnd, stream);
    {
      auto mut = batch->to_mutable();
      REQUIRE_THROWS_AS(
        mut.convert_to<cucascade::host_data_representation>(registry, env().host_space, stream),
        cucascade::logic_error);
      REQUIRE(mut.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
    }
    require_slice_rows(*batch, kSliceBegin, kSliceEnd, stream);
  }

  SECTION("an unsliced table converts")
  {
    auto batch = make_sliced_batch(0, kParentRows, stream);
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, env().host_space, stream);
      REQUIRE(mut.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
      mut.convert_to<cucascade::gpu_table_representation>(registry, env().gpu_space, stream);
    }
    require_slice_rows(*batch, 0, kParentRows, stream);
  }

  SECTION("a same-device GPU->GPU conversion of a slice is a compacting clone")
  {
    auto batch = make_sliced_batch(kSliceBegin, kSliceEnd, stream);
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(registry, env().gpu_space, stream);
    }
    require_slice_rows(*batch, kSliceBegin, kSliceEnd, stream);
  }
}
