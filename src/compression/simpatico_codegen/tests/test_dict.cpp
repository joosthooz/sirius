// Round trips for the fused Dict operator: both key-lookup modes (bitmap rank and binary
// search), every value width, its codes and keys chained into other operators, the decline
// above the key cap, and the on-disk format.

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using simpatico::compress_with_plan;
using simpatico::compressed_table;
using simpatico::decompress;

template <typename T>
std::unique_ptr<cudf::table> make_table(cudf::type_id id, std::vector<T> const& host)
{
  auto col = cudf::make_fixed_width_column(
    cudf::data_type{id}, static_cast<cudf::size_type>(host.size()), cudf::mask_state::UNALLOCATED);
  if (!host.empty()) {
    cuda_check(cudaMemcpy(
      col->mutable_view().head<T>(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

// `n` rows drawn from `keys` in a scrambled order.
template <typename T>
std::vector<T> draw(std::vector<T> const& keys, int n)
{
  std::vector<T> out(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r)
    out[static_cast<std::size_t>(r)] =
      keys[static_cast<std::size_t>((static_cast<std::uint64_t>(r) * 2654435761u) % keys.size())];
  return out;
}

compressed_table roundtrip(cudf::table_view input, std::string const& dsl, char const* label)
{
  std::fprintf(stderr, "[dict] %s\n", label);
  auto ct = compress_with_plan(
    input, dsl, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  auto out = decompress(ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  expect(out != nullptr && out->num_columns() == input.num_columns(),
         (std::string(label) + ": decompress shape").c_str());
  for (int i = 0; i < input.num_columns(); ++i)
    expect(columns_equal_any(input.column(i), out->view().column(i), cudf::get_default_stream()),
           (std::string(label) + ": column mismatch").c_str());
  return ct;
}

constexpr char const* kDictBitpack =
  "input -> dict -> codes, keys\n"
  "dict.codes -> bitpack\n";

constexpr int kRows = 100'000;

void test_widths()
{
  // Spans under 2^24 take the bitmap path; a small span keeps the bitmap in shared memory.
  std::vector<std::int32_t> i32;
  for (int k = 0; k < 50; ++k)
    i32.push_back(k * 997 - 20'000);
  roundtrip(make_table(cudf::type_id::INT32, draw(i32, kRows))->view(), kDictBitpack, "i32_bitmap");

  // A span past the shared-memory bitmap marks a global one.
  std::vector<std::int32_t> wide;
  for (int k = 0; k < 100; ++k)
    wide.push_back(k * 150'001);
  roundtrip(
    make_table(cudf::type_id::INT32, draw(wide, kRows))->view(), kDictBitpack, "i32_global_bitmap");

  // A span past 2^24 takes the sort path and binary-searches the keys.
  std::vector<std::int64_t> i64;
  for (int k = 0; k < 300; ++k)
    i64.push_back((k - 150) * std::int64_t{1'000'000'007} * 1000);
  roundtrip(make_table(cudf::type_id::INT64, draw(i64, kRows))->view(), kDictBitpack, "i64_search");

  std::vector<std::int16_t> i16{-32768, -5, 0, 7, 32767};
  roundtrip(make_table(cudf::type_id::INT16, draw(i16, kRows))->view(), kDictBitpack, "i16");

  std::vector<std::int8_t> i8{-128, -1, 3, 127};
  roundtrip(make_table(cudf::type_id::INT8, draw(i8, kRows))->view(), kDictBitpack, "i8");

  std::vector<std::uint32_t> u32{0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
  roundtrip(make_table(cudf::type_id::UINT32, draw(u32, kRows))->view(), kDictBitpack, "u32");

  // Floats are keyed by their bit patterns, so -0.0 and 0.0 stay distinct.
  std::vector<double> f64{0.0, -0.0, 0.01, 0.02, 0.05, 0.1, -3.5, 1e300};
  roundtrip(make_table(cudf::type_id::FLOAT64, draw(f64, kRows))->view(), kDictBitpack, "f64");

  std::vector<float> f32{0.0f, 1.5f, -2.25f, 1e30f};
  roundtrip(make_table(cudf::type_id::FLOAT32, draw(f32, kRows))->view(), kDictBitpack, "f32");

  std::vector<std::int32_t> days;
  for (int k = 0; k < 2526; ++k)
    days.push_back(8035 + k);
  roundtrip(
    make_table(cudf::type_id::TIMESTAMP_DAYS, draw(days, kRows))->view(), kDictBitpack, "date");

  std::vector<std::int64_t> cents;
  for (int k = 0; k < 1000; ++k)
    cents.push_back(90'000 + k * 37);
  auto t   = make_table(cudf::type_id::INT64, draw(cents, kRows));
  auto col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::DECIMAL64, -2},
                                            kRows,
                                            std::move(*t->get_column(0).release().data),
                                            rmm::device_buffer{},
                                            0);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  roundtrip(cudf::table(std::move(cols)).view(), kDictBitpack, "decimal64");

  // A single key; and a chunk tail shorter than a full chunk.
  roundtrip(make_table(cudf::type_id::INT32, std::vector<std::int32_t>(5000, 42))->view(),
            kDictBitpack,
            "single_key");
}

void test_cascades()
{
  std::vector<std::int64_t> keys;
  for (int k = 0; k < 4000; ++k)
    keys.push_back(1'000'000 + k * 3);
  auto t = make_table(cudf::type_id::INT64, draw(keys, kRows));

  roundtrip(t->view(), "input -> dict -> codes, keys\n", "raw_codes");
  roundtrip(t->view(),
            "input -> dict -> codes, keys\n"
            "dict.codes -> bitpack\n"
            "dict.keys -> delta -> differences\n"
            "dict.keys.differences -> bitpack\n",
            "keys_delta_bitpack");
  // A Dict below another op roots its own region over that op's stored stream.
  roundtrip(make_table(cudf::type_id::INT32, std::vector<std::int32_t>(4096, 7))->view(),
            "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed, chunk_divisors\n"
            "bitpack.chunk_count -> dict -> codes, keys\n"
            "bitpack.chunk_count.codes -> dict -> codes, keys\n",
            "nested_dict");
  roundtrip(t->view(),
            "input -> for -> deltas, references\n"
            "for.references -> delta -> differences\n"
            "for.references.differences -> dict -> codes, keys\n",
            "dict_below_delta");
  roundtrip(t->view(),
            "input -> dict -> codes, keys\n"
            "dict.codes -> rle -> values, runs\n"
            "dict.codes.values -> bitpack\n"
            "dict.codes.runs -> bitpack\n",
            "codes_rle");
}

void test_declines()
{
  // Every value distinct: past the half-the-rows key cap, so the encode declines.
  std::vector<std::int32_t> distinct(4096);
  for (int r = 0; r < 4096; ++r)
    distinct[static_cast<std::size_t>(r)] = r * 3;
  auto t     = make_table(cudf::type_id::INT32, distinct);
  bool threw = false;
  try {
    compress_with_plan(t->view(),
                       kDictBitpack,
                       cudf::get_default_stream(),
                       rmm::mr::get_current_device_resource_ref());
  } catch (std::exception const&) {
    threw = true;
  }
  expect(threw, "dict over the key cap should decline");

  roundtrip(make_table(cudf::type_id::INT32, std::vector<std::int32_t>{})->view(),
            kDictBitpack,
            "zero_rows");
}

void test_io()
{
  std::vector<std::int32_t> keys;
  for (int k = 0; k < 64; ++k)
    keys.push_back(k * k);
  auto t  = make_table(cudf::type_id::INT32, draw(keys, kRows));
  auto ct = roundtrip(t->view(), kDictBitpack, "io_source");

  std::string const path = "/tmp/simpatico_test_dict.spc";
  auto const err         = simpatico::write_compressed_table(ct, path);
  expect(err.empty(), ("dict io: write failed: " + err).c_str());
  std::string read_err;
  auto back = simpatico::read_compressed_table(
    path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &read_err);
  expect(read_err.empty(), ("dict io: read failed: " + read_err).c_str());
  auto out =
    decompress(back, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  expect(out != nullptr && out->num_columns() == 1, "dict io: decompress shape");
  expect(columns_equal_any(t->view().column(0), out->view().column(0), cudf::get_default_stream()),
         "dict io: column mismatch");
  std::remove(path.c_str());
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_dict: cudaSetDevice failed\n");
    return 1;
  }
  try {
    test_widths();
    test_cascades();
    test_declines();
    test_io();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_dict: FAILED: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "test_dict: OK\n");
  return 0;
}
