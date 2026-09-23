// SPDX-License-Identifier: Apache-2.0
// OnPair-GPU decode — ported from Vortex's vortex-cuda kernels:
//   kernels/src/onpair_shmem_4tpt_split8read.cu  (the fast per-token decode kernel, u16 codes)
// adapted to Sirius's cudf/rmm buffer conventions, following fsst_compressor.cu's shape.
// The decode kernel body is a near-verbatim port (same algorithm, same launch geometry).
//
// v4: chars-only — row offsets are not this codec's concern at all (see
// onpair_representation.hpp). chunk_offsets (this codec's own per-128-token-
// batch placement metadata) is precomputed at dump time and loaded straight
// from disk, mirroring how fsst_compressor.cu's compressed frame header
// stores each chunk's uncompressed length directly (fsst_compressor.cu:227-
// 250) instead of having the GPU rediscover it — there is nothing left to
// derive from the token stream at decode time.

#include "onpair_representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace simpatico {
namespace {

inline void check_cuda(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("onpair: ") + what + ": " + cudaGetErrorString(e));
  }
}

constexpr std::uint32_t kMaxTokenSize   = 16;
constexpr std::uint32_t kTokensPerBatch = 128;

// -----------------------------------------------------------------------------
// Ported verbatim from vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu
// (u16-codes instantiation only — the dump tool always widens codes to u16).
// -----------------------------------------------------------------------------

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline std::uint32_t onpair_warp_inclusive_scan_u32(std::uint32_t x, std::uint32_t lane)
{
  constexpr unsigned mask = 0xffffffffu;
#pragma unroll
  for (std::uint8_t offset = 1; offset < 32; offset <<= 1) {
    std::uint32_t y = __shfl_up_sync(mask, x, offset);
    if (lane >= offset) { x += y; }
  }
  return x;
}

struct OnPairTokens {
  uint2 lo[4];
  std::uint32_t code[4];
  std::uint32_t len[4];
};

__device__ inline OnPairTokens onpair_load_tokens(const std::uint16_t* __restrict codes,
                                                  const std::uint8_t* __restrict dict_s8,
                                                  const std::uint8_t* __restrict lens,
                                                  std::uint64_t base_i,
                                                  std::uint64_t token_start,
                                                  std::uint64_t token_end)
{
  OnPairTokens t;
#pragma unroll
  for (std::uint8_t k = 0; k < 4; ++k) {
    const std::uint64_t i = base_i + (std::uint64_t)(k * 32);
    if (i >= token_start && i < token_end) {
      const std::uint32_t code = (std::uint32_t)codes[i];
      t.code[k]                = code;
      t.lo[k]                  = *reinterpret_cast<const uint2*>(dict_s8 + (size_t)code * 8u);
      t.len[k]                 = (std::uint32_t)lens[code];
    } else {
      t.code[k] = 0u;
      t.lo[k]   = make_uint2(0u, 0u);
      t.len[k]  = 0u;
    }
  }
  return t;
}

__device__ inline std::uint32_t onpair_scan_offsets(const std::uint32_t (&len)[4],
                                                     std::uint32_t lane,
                                                     std::uint32_t (&excl)[4])
{
  constexpr unsigned mask   = 0xffffffffu;
  std::uint32_t acc_base    = 0u;
#pragma unroll
  for (std::uint8_t k = 0; k < 4; ++k) {
    const std::uint32_t incl = onpair_warp_inclusive_scan_u32(len[k], lane);
    excl[k]                  = acc_base + (incl - len[k]);
    acc_base += __shfl_sync(mask, incl, 31);
  }
  return acc_base;
}

__device__ inline void onpair_stage_tokens(const OnPairTokens& t,
                                           const std::uint32_t (&excl)[4],
                                           const std::uint8_t* __restrict dict_padded,
                                           std::uint8_t* __restrict s_buf)
{
#pragma unroll
  for (std::uint8_t k = 0; k < 4; ++k) {
    const std::uint32_t len = t.len[k];
    if (len == 0u) { continue; }
    const std::uint32_t base      = excl[k];
    const std::uint8_t* lob       = reinterpret_cast<const std::uint8_t*>(&t.lo[k]);
    const std::uint32_t nlo       = len < 8u ? len : 8u;
#pragma unroll
    for (std::uint8_t j = 0; j < 8; ++j) {
      if (j < nlo) { s_buf[base + j] = lob[j]; }
    }
    if (len > 8u) {
      const uint2 hi = *reinterpret_cast<const uint2*>(dict_padded + (size_t)t.code[k] * kMaxTokenSize + 8u);
      const std::uint8_t* hib = reinterpret_cast<const std::uint8_t*>(&hi);
#pragma unroll
      for (std::uint8_t j = 0; j < 8; ++j) {
        if (8u + j < len) { s_buf[base + 8 + j] = hib[j]; }
      }
    }
  }
}

__device__ inline void onpair_drain(const std::uint8_t* __restrict s_buf,
                                    std::uint8_t* __restrict output_bytes,
                                    std::uint64_t out_start,
                                    std::uint32_t head_pre,
                                    std::uint32_t warp_total,
                                    std::uint32_t lane)
{
  const std::uint32_t head = head_pre < warp_total ? head_pre : warp_total;
  if (lane < head) { output_bytes[out_start + (std::uint64_t)lane] = s_buf[lane]; }
  if (head >= warp_total) { return; }

  const std::uint32_t body_chunks = (warp_total - head) >> 4;
  for (std::uint8_t k = (std::uint8_t)lane; k < body_chunks; k += 32u) {
    const std::uint32_t off = head + k * 16u;
    const uint4 v            = *reinterpret_cast<const uint4*>(s_buf + off);
    __stcs(reinterpret_cast<uint4*>(output_bytes + out_start + off), v);
  }

  const std::uint32_t tail_start = head + (body_chunks << 4);
  if (lane < warp_total - tail_start) {
    output_bytes[out_start + (std::uint64_t)tail_start + (std::uint64_t)lane] = s_buf[tail_start + lane];
  }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_u16(
  const std::uint16_t* __restrict codes,
  const std::uint64_t* __restrict chunk_offsets,
  const std::uint8_t* __restrict dict_s8,
  const std::uint8_t* __restrict dict_padded,
  const std::uint8_t* __restrict lens,
  std::uint8_t* __restrict output_bytes,
  std::uint64_t token_start,
  std::uint64_t token_end,
  std::uint64_t first_batch,
  std::uint64_t byte_start)
{
  const std::uint32_t lane    = threadIdx.x & 31;
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint64_t window_chunk =
    (std::uint64_t)blockIdx.x * (std::uint64_t)(blockDim.x >> 5) + (std::uint64_t)warp_id;
  const std::uint64_t chunk = first_batch + window_chunk;
  if (chunk * 128u >= token_end) { return; }

  __shared__ __align__(16) std::uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
  std::uint8_t* s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

  const OnPairTokens t =
    onpair_load_tokens(codes, dict_s8, lens, chunk * 128u + (std::uint64_t)lane, token_start, token_end);

  std::uint32_t excl[4];
  const std::uint32_t warp_total = onpair_scan_offsets(t.len, lane, excl);

  const std::uint64_t out_start = chunk == first_batch ? 0 : chunk_offsets[chunk] - byte_start;
  const std::uint32_t head_pre  = (16u - (std::uint32_t)(out_start & 15u)) & 15u;
  std::uint8_t* s_buf           = s_buf_base + ((16u - head_pre) & 15u);

  onpair_stage_tokens(t, excl, dict_padded, s_buf);
  __syncwarp();

  onpair_drain(s_buf, output_bytes, out_start, head_pre, warp_total, lane);
}

}  // namespace

// -----------------------------------------------------------------------------
// Host orchestration
// -----------------------------------------------------------------------------

std::unique_ptr<cudf::column> onpair_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  std::uint64_t const total_bytes = static_cast<std::uint64_t>(uncompressed_bytes);
  if (n_tokens == 0 || total_bytes == 0) {
    return std::make_unique<cudf::column>(
      cudf::data_type(cudf::type_id::UINT8), 0, rmm::device_buffer{0, stream, mr}, rmm::device_buffer{}, 0);
  }

  auto const* dict_padded   = static_cast<const std::uint8_t*>(dict_padded_buf.data());
  auto const* dict_s8       = static_cast<const std::uint8_t*>(dict_s8_buf.data());
  auto const* lens          = static_cast<const std::uint8_t*>(lens_buf.data());
  auto const* codes         = static_cast<const std::uint16_t*>(codes_buf.data());
  auto const* chunk_offsets = static_cast<const std::uint64_t*>(chunk_offsets_buf.data());

  // Chars-only decode: chunk_offsets is precomputed at dump time (this
  // codec's own placement metadata, not row offsets), so total_bytes is known
  // from the dump header and there is nothing to synchronize on before
  // sizing the output buffer.
  rmm::device_buffer output_bytes_buf(total_bytes, stream, mr);
  {
    constexpr int kThreads     = 256;  // matches ONPAIR_LAUNCH_BOUNDS(256, 4)
    int const warps_per_block  = kThreads / 32;
    int const blocks = static_cast<int>((static_cast<std::int64_t>(num_batches) + warps_per_block - 1) /
                                        warps_per_block);
    onpair_shmem_4tpt_split8read_u16<<<blocks, kThreads, 0, stream.value()>>>(
      codes,
      chunk_offsets,
      dict_s8,
      dict_padded,
      lens,
      static_cast<std::uint8_t*>(output_bytes_buf.data()),
      /*token_start=*/0,
      /*token_end=*/n_tokens,
      /*first_batch=*/0,
      /*byte_start=*/0);
    check_cuda(cudaGetLastError(), "decode launch");
  }
  stream.synchronize();

  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT8),
                                        static_cast<cudf::size_type>(total_bytes),
                                        std::move(output_bytes_buf),
                                        rmm::device_buffer{},
                                        0);
}

double onpair_compressed_representation::time_decode_kernel_only_ms(rmm::cuda_stream_view stream,
                                                                     rmm::device_async_resource_ref mr,
                                                                     int iters) const
{
  if (n_tokens == 0 || uncompressed_bytes == 0 || iters <= 0) { return 0.0; }

  auto const* dict_padded   = static_cast<const std::uint8_t*>(dict_padded_buf.data());
  auto const* dict_s8       = static_cast<const std::uint8_t*>(dict_s8_buf.data());
  auto const* lens          = static_cast<const std::uint8_t*>(lens_buf.data());
  auto const* codes         = static_cast<const std::uint16_t*>(codes_buf.data());
  auto const* chunk_offsets = static_cast<const std::uint64_t*>(chunk_offsets_buf.data());

  std::int64_t const num_batches_i64 = static_cast<std::int64_t>(num_batches);
  std::uint64_t const total_bytes    = static_cast<std::uint64_t>(uncompressed_bytes);

  rmm::device_buffer output_bytes_buf(total_bytes, stream, mr);
  constexpr int kThreads    = 256;
  int const warps_per_block = kThreads / 32;
  int const blocks   = static_cast<int>((num_batches_i64 + warps_per_block - 1) / warps_per_block);
  auto* out           = static_cast<std::uint8_t*>(output_bytes_buf.data());

  // 2 warmup launches (not timed), then `iters` timed launches bracketed by
  // one cudaEvent pair — same methodology as the CUDA-event/nsys measurements
  // used elsewhere in this comparison (kernel time only, no host overhead).
  for (int w = 0; w < 2; ++w) {
    onpair_shmem_4tpt_split8read_u16<<<blocks, kThreads, 0, stream.value()>>>(
      codes, chunk_offsets, dict_s8, dict_padded, lens, out, 0, n_tokens, 0, 0);
  }
  check_cuda(cudaGetLastError(), "warmup decode launch");
  stream.synchronize();

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "event create start");
  check_cuda(cudaEventCreate(&stop), "event create stop");
  check_cuda(cudaEventRecord(start, stream.value()), "event record start");
  for (int i = 0; i < iters; ++i) {
    onpair_shmem_4tpt_split8read_u16<<<blocks, kThreads, 0, stream.value()>>>(
      codes, chunk_offsets, dict_s8, dict_padded, lens, out, 0, n_tokens, 0, 0);
  }
  check_cuda(cudaGetLastError(), "timed decode launch");
  check_cuda(cudaEventRecord(stop, stream.value()), "event record stop");
  check_cuda(cudaEventSynchronize(stop), "event sync stop");
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms);
}

std::unique_ptr<onpair_compressed_representation> onpair_compressed_representation::load_from_dump(
  std::string const& path, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("onpair: cannot open dump file " + path); }

  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "ONP4", 4) != 0) {
    throw std::runtime_error(
      "onpair: bad magic in dump file " + path +
      " (expected \"ONP4\" — regenerate with the current dump_onpair.rs; older \"ONP2\"/\"ONP3\" "
      "formats that bundled row offsets into the compressed payload are no longer supported)");
  }
  auto read_u64 = [&]() {
    std::uint64_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };
  std::uint64_t const n_rows             = read_u64();
  std::uint64_t const dict_size          = read_u64();
  std::uint64_t const n_tokens           = read_u64();
  std::uint64_t const code_width         = read_u64();
  std::uint64_t const uncompressed_bytes = read_u64();
  std::uint64_t const num_batches        = read_u64();
  if (code_width != 2) {
    throw std::runtime_error("onpair: dump file has code_width=" + std::to_string(code_width) +
                             ", expected 2 (dump tool always widens to u16)");
  }

  std::vector<std::uint8_t> dict_padded_h(dict_size * kMaxTokenSize);
  std::vector<std::uint8_t> dict_s8_h(dict_size * 8);
  std::vector<std::uint8_t> lens_h(dict_size);
  std::vector<std::uint16_t> codes_h(n_tokens);
  f.read(reinterpret_cast<char*>(dict_padded_h.data()), dict_padded_h.size());
  f.read(reinterpret_cast<char*>(dict_s8_h.data()), dict_s8_h.size());
  f.read(reinterpret_cast<char*>(lens_h.data()), lens_h.size());
  f.read(reinterpret_cast<char*>(codes_h.data()), codes_h.size() * sizeof(std::uint16_t));

  std::vector<std::uint64_t> chunk_offsets_h(num_batches + 1);
  f.read(reinterpret_cast<char*>(chunk_offsets_h.data()), chunk_offsets_h.size() * sizeof(std::uint64_t));
  if (!f) { throw std::runtime_error("onpair: short read on dump file " + path); }
  // Out-of-band ground-truth row_offsets follow, deliberately not read here —
  // see load_row_offsets_from_dump().

  auto to_device = [&](void const* p, std::size_t bytes) {
    rmm::device_buffer b(bytes, stream, mr);
    check_cuda(cudaMemcpyAsync(b.data(), p, bytes, cudaMemcpyHostToDevice, stream.value()), "upload");
    return b;
  };
  auto dict_padded_d   = to_device(dict_padded_h.data(), dict_padded_h.size());
  auto dict_s8_d        = to_device(dict_s8_h.data(), dict_s8_h.size());
  auto lens_d           = to_device(lens_h.data(), lens_h.size());
  auto codes_d          = to_device(codes_h.data(), codes_h.size() * sizeof(std::uint16_t));
  auto chunk_offsets_d  = to_device(chunk_offsets_h.data(), chunk_offsets_h.size() * sizeof(std::uint64_t));
  stream.synchronize();

  return std::make_unique<onpair_compressed_representation>(static_cast<cudf::size_type>(n_rows),
                                                             static_cast<std::size_t>(dict_size),
                                                             static_cast<std::size_t>(n_tokens),
                                                             static_cast<std::size_t>(num_batches),
                                                             static_cast<std::size_t>(uncompressed_bytes),
                                                             std::move(dict_padded_d),
                                                             std::move(dict_s8_d),
                                                             std::move(lens_d),
                                                             std::move(codes_d),
                                                             std::move(chunk_offsets_d));
}

std::vector<std::int32_t> onpair_compressed_representation::load_row_offsets_from_dump(
  std::string const& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("onpair: cannot open dump file " + path); }

  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "ONP4", 4) != 0) {
    throw std::runtime_error("onpair: bad magic in dump file " + path + " (expected \"ONP4\")");
  }
  auto read_u64 = [&]() {
    std::uint64_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };
  std::uint64_t const n_rows             = read_u64();
  std::uint64_t const dict_size          = read_u64();
  std::uint64_t const n_tokens           = read_u64();
  std::uint64_t const code_width         = read_u64();
  read_u64();  // uncompressed_bytes — not needed here
  std::uint64_t const num_batches        = read_u64();
  (void)n_rows;
  (void)code_width;

  // Skip the chars-only payload this function doesn't need.
  f.seekg(static_cast<std::streamoff>(dict_size * kMaxTokenSize + dict_size * 8 + dict_size), std::ios::cur);
  f.seekg(static_cast<std::streamoff>(n_tokens * sizeof(std::uint16_t)), std::ios::cur);
  f.seekg(static_cast<std::streamoff>((num_batches + 1) * sizeof(std::uint64_t)), std::ios::cur);

  std::uint64_t const row_offsets_len = read_u64();
  std::vector<std::int32_t> row_offsets(row_offsets_len);
  f.read(reinterpret_cast<char*>(row_offsets.data()), row_offsets.size() * sizeof(std::int32_t));
  if (!f) { throw std::runtime_error("onpair: short read on row_offsets in " + path); }
  return row_offsets;
}

}  // namespace simpatico
