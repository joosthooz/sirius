// SPDX-License-Identifier: Apache-2.0
//
// Ad hoc bench/correctness harness for the OnPair-GPU decode port
// (src/operators/onpair_compressor.cu). Loads dump_onpair.rs "ONP4" binary
// dumps (Vortex CPU onpair_compress output on real SF1000 TPC-H string
// columns), round-trip-checks the decoded strings, and times full-pipeline
// and isolated-decode-kernel throughput the same way the FSST-GPU comparison
// did (CUDA-event timed, warmup excluded).
//
// v4: onpair_compressed_representation is chars-only now (see
// onpair_representation.hpp) — it never sees row offsets. This harness plays
// the role FSST's own str_split.offsets/chars split already plays in
// production: it loads the dump's out-of-band ground-truth row offsets and
// round-trips them through Sirius's *real* str_split+bitpack machinery
// (simpatico::compress_with_plan / compressed_table::decompress with plan
// "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed" — the
// literal cascade FSST's own plans use for str_split.offsets), then zips the
// resulting offsets column with the OnPair-decoded chars buffer into a
// STRING column for the correctness check, exactly as the plan interpreter
// would combine str_split's two children.
//
// Usage: bench_onpair_gpu <dump_dir> [iters]
//   dump_dir must contain <col>.onpair.bin + <col>.strings.bin pairs for the
//   six columns produced by vortex-src/vortex-cuda/examples/dump_onpair.rs.

#include "api/simpatico_codegen.hpp"
#include "operators/onpair_representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}

// Builds an INT32 cudf column from a plain host vector of row offsets and
// compresses it via Sirius's real bitpack cascade — the same one FSST's own
// plans use for str_split.offsets. Returns the compressed_table (caller
// decompresses/describes it as needed); this stands in for "compress the
// original string column's row offsets," decoupled entirely from whatever
// codec (fsst, onpair, ...) compressed the chars stream.
simpatico::compressed_table compress_offsets_via_bitpack(std::vector<std::int32_t> const& offsets,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, static_cast<cudf::size_type>(offsets.size()),
    cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpyAsync(col->mutable_view().head<std::int32_t>(),
                  offsets.data(),
                  offsets.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  cudf::table table(std::move(cols));

  return simpatico::compress_with_plan(
    table.view(), "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n", stream, mr);
}

std::vector<std::string> read_strings_dump(std::string const& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  auto read_u64 = [&]() {
    std::uint64_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };
  std::uint64_t n_rows = read_u64();
  std::vector<std::uint64_t> lens(n_rows);
  for (auto& l : lens) l = read_u64();
  std::vector<std::string> out(n_rows);
  for (std::uint64_t i = 0; i < n_rows; ++i) {
    out[i].resize(lens[i]);
    f.read(out[i].data(), lens[i]);
  }
  if (!f) throw std::runtime_error("short read on " + path);
  return out;
}

void check_roundtrip(cudf::column const& chars,
                     cudf::column const& offsets,
                     std::vector<std::string> const& expected,
                     rmm::cuda_stream_view stream)
{
  expect(offsets.type().id() == cudf::type_id::INT32, "offsets column is not INT32");
  auto const n_rows = offsets.size() - 1;
  expect(static_cast<std::size_t>(n_rows) == expected.size(), "row count mismatch");

  std::vector<std::int32_t> offsets_h(offsets.size());
  cudaMemcpyAsync(offsets_h.data(),
                  offsets.view().data<std::int32_t>(),
                  offsets_h.size() * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaStreamSynchronize(stream.value());

  std::size_t const total_bytes = static_cast<std::size_t>(offsets_h[n_rows]);
  expect(static_cast<std::size_t>(chars.size()) == total_bytes,
        "chars byte count disagrees with reconstructed offsets");
  std::vector<char> chars_h(total_bytes);
  if (total_bytes > 0) {
    cudaMemcpyAsync(chars_h.data(),
                    chars.view().data<char>(),
                    total_bytes,
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }

  for (cudf::size_type i = 0; i < n_rows; ++i) {
    std::int32_t const start = offsets_h[i];
    std::int32_t const end   = offsets_h[i + 1];
    std::string_view got(chars_h.data() + start, end - start);
    expect(got == expected[i], ("row mismatch at index " + std::to_string(i)).c_str());
  }
}

void bench_column(std::string const& dump_dir, std::string const& col, int iters)
{
  auto stream = rmm::cuda_stream_view{};
  auto mr     = rmm::mr::get_current_device_resource_ref();

  auto const dump_path = dump_dir + "/" + col + ".onpair.bin";
  auto rep = simpatico::onpair_compressed_representation::load_from_dump(dump_path, stream, mr);
  auto const row_offsets = simpatico::onpair_compressed_representation::load_row_offsets_from_dump(dump_path);
  auto expected           = read_strings_dump(dump_dir + "/" + col + ".strings.bin");

  std::size_t const uncompressed_bytes = rep->uncompressed_bytes;
  std::size_t const chars_compressed_bytes = rep->compressed_payload_bytes();
  double const chars_only_ratio = static_cast<double>(uncompressed_bytes) / chars_compressed_bytes;

  // Offsets: compressed via Sirius's real str_split-offsets bitpack cascade
  // (same plan FSST's own str_split.offsets uses), not reinvented here.
  auto offsets_compressed = compress_offsets_via_bitpack(row_offsets, stream, mr);
  auto offsets_decompressed = offsets_compressed.decompress(stream, mr);
  auto offsets_released      = offsets_decompressed->release();
  auto offsets_col            = std::move(offsets_released.front());

  // Correctness: decode chars (this codec's own job) + decode offsets (the
  // real bitpack cascade's job), zip, compare every row — same contract
  // FSST's str_split.chars/offsets split already has in production.
  auto chars_col = rep->decompress(stream, mr);
  check_roundtrip(*chars_col, *offsets_col, expected, stream);
  std::printf(
    "%s: round-trip OK (rows=%zu dict_size=%zu n_tokens=%zu onpair_chars_only_ratio=%.3fx)\n",
    col.c_str(),
    row_offsets.size() - 1,
    rep->dict_size,
    rep->n_tokens,
    chars_only_ratio);

  // Combined compressed size: this codec's chars payload + the real bitpack
  // cascade's persisted offsets buffers (summed via describe(), the same
  // introspection test_bitpack_layout_contract.cpp uses).
  auto const description = offsets_compressed.describe(stream);
  std::size_t offsets_compressed_bytes = 0;
  for (auto const& leaf : description[0]) {
    for (auto const& buf : leaf.buffers) { offsets_compressed_bytes += buf.size_bytes; }
  }
  std::size_t const combined_compressed_bytes = chars_compressed_bytes + offsets_compressed_bytes;
  double const combined_ratio = static_cast<double>(uncompressed_bytes) / combined_compressed_bytes;

  // Chars-only full pipeline: decode kernel + buffer alloc, via the public
  // decompress() entry point, warmup then timed iterations. With chunk_offsets
  // precomputed at dump time this differs from the isolated kernel only by
  // buffer allocation — see onpair_representation.hpp.
  for (int w = 0; w < 2; ++w) { rep->decompress(stream, mr); }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream.value());
  for (int i = 0; i < iters; ++i) { rep->decompress(stream, mr); }
  cudaEventRecord(stop, stream.value());
  cudaEventSynchronize(stop);
  float chars_ms = 0.0f;
  cudaEventElapsedTime(&chars_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Offsets decode: the real bitpack cascade's decompress(), timed the same
  // way — this is the literal cost FSST's own full-pipeline number already
  // includes for str_split.offsets.
  for (int w = 0; w < 2; ++w) { offsets_compressed.decompress(stream, mr); }
  cudaEvent_t ostart, ostop;
  cudaEventCreate(&ostart);
  cudaEventCreate(&ostop);
  cudaEventRecord(ostart, stream.value());
  for (int i = 0; i < iters; ++i) { offsets_compressed.decompress(stream, mr); }
  cudaEventRecord(ostop, stream.value());
  cudaEventSynchronize(ostop);
  float offsets_ms = 0.0f;
  cudaEventElapsedTime(&offsets_ms, ostart, ostop);
  cudaEventDestroy(ostart);
  cudaEventDestroy(ostop);

  double const chars_only_gbps =
    (static_cast<double>(uncompressed_bytes) * iters) / (chars_ms / 1000.0) / 1e9;
  double const combined_gbps = (static_cast<double>(uncompressed_bytes) * iters) /
                               ((chars_ms + offsets_ms) / 1000.0) / 1e9;

  // Isolated decode-kernel-only throughput (buffer alloc excluded).
  double const decode_ms = rep->time_decode_kernel_only_ms(stream, mr, iters);
  double const decode_gbps =
    (static_cast<double>(uncompressed_bytes) * iters) / (decode_ms / 1000.0) / 1e9;

  std::printf(
    "%s: Sirius-OnPair-GPU  chars_only_full_pipeline=%.2f GB/s  "
    "combined_full_pipeline(chars+bitpack_offsets)=%.2f GB/s  isolated_decode_kernel=%.2f GB/s  "
    "onpair_chars_only_ratio=%.3fx  combined_ratio=%.3fx  "
    "(%d iters, uncompressed=%zuB chars_compressed=%zuB offsets_compressed=%zuB)\n",
    col.c_str(),
    chars_only_gbps,
    combined_gbps,
    decode_gbps,
    chars_only_ratio,
    combined_ratio,
    iters,
    uncompressed_bytes,
    chars_compressed_bytes,
    offsets_compressed_bytes);
}

}  // namespace

int main(int argc, char** argv)
{
  std::string dump_dir = argc > 1 ? argv[1] : "/tmp/onpair_dump";
  int iters             = argc > 2 ? std::atoi(argv[2]) : 10;

  // Stream-ordered pool allocator (cudaMallocAsync-backed), matching what the
  // FSST-GPU CLI benchmark installs (simpatico_main.cpp's pool_mr_guard) — a
  // raw synchronous cudaMalloc/cudaFree per rmm::device_buffer is not how
  // Sirius's own operators are ever actually invoked in production.
  rmm::mr::cuda_async_memory_resource pool_mr{};
  rmm::device_async_resource_ref previous_mr{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::set_current_device_resource_ref(pool_mr);

  static const std::vector<std::string> kColumns = {
    "l_comment", "l_shipinstruct", "c_comment", "p_name", "p_comment", "s_comment"};

  int failures = 0;
  for (auto const& col : kColumns) {
    try {
      bench_column(dump_dir, col, iters);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s: FAIL: %s\n", col.c_str(), e.what());
      ++failures;
    }
  }
  rmm::mr::set_current_device_resource_ref(previous_mr);

  if (failures > 0) {
    std::fprintf(stderr, "bench_onpair_gpu: %d/%zu columns FAILED\n", failures, kColumns.size());
    return 1;
  }
  std::printf("bench_onpair_gpu: all columns PASS\n");
  return 0;
}
