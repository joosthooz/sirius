// SPDX-License-Identifier: Apache-2.0
//
// dict_keys.cu — the dictionary build that runs before a fused Dict encode.
//
// A narrow key range (max - min < kDictMaxBitmapRange) is marked into a presence bitmap, whose
// exclusive prefix popcount is at once the sorted key list and the rank table the encoder maps
// values through. A wider range is sorted and deduplicated instead.

#include "codegen/bridge/dict_keys.hpp"

#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <array>
#include <stdexcept>
#include <string>

namespace simpatico {
namespace {

// A block keeps a private bitmap in shared memory when the whole bitmap fits; low-cardinality
// columns would otherwise funnel every row into a handful of global atomics.
constexpr std::int32_t kPrivateBitmapWords = 8192;  // 32 KB
constexpr int kMarkBlock                   = 256;

void check(cudaError_t e, char const* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("simpatico dict keys: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

template <class T>
using unsigned_t = typename ::cuda::std::make_unsigned<T>::type;

template <class T>
__global__ void mark_private(T const* __restrict__ data,
                             std::int64_t n,
                             T minv,
                             std::int32_t words,
                             std::uint32_t* __restrict__ bitmap)
{
  extern __shared__ std::uint32_t local[];
  for (std::int32_t w = threadIdx.x; w < words; w += blockDim.x)
    local[w] = 0u;
  __syncthreads();
  using U = unsigned_t<T>;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const o = static_cast<std::uint32_t>(static_cast<U>(data[i]) - static_cast<U>(minv));
    std::uint32_t const bit = 1u << (o & 31u);
    if ((local[o >> 5] & bit) == 0u) atomicOr(&local[o >> 5], bit);
  }
  __syncthreads();
  for (std::int32_t w = threadIdx.x; w < words; w += blockDim.x)
    if (local[w] != 0u) atomicOr(&bitmap[w], local[w]);
}

template <class T>
__global__ void mark_global(T const* __restrict__ data,
                            std::int64_t n,
                            T minv,
                            std::uint32_t* __restrict__ bitmap)
{
  using U = unsigned_t<T>;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const o = static_cast<std::uint32_t>(static_cast<U>(data[i]) - static_cast<U>(minv));
    std::uint32_t const bit = 1u << (o & 31u);
    if ((bitmap[o >> 5] & bit) == 0u) atomicOr(&bitmap[o >> 5], bit);
  }
}

template <class T>
__global__ void emit_bitmap_keys(std::uint32_t const* __restrict__ bitmap,
                                 std::int32_t const* __restrict__ rank,
                                 std::int32_t words,
                                 T minv,
                                 T* __restrict__ keys)
{
  using U              = unsigned_t<T>;
  std::int32_t const w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= words) return;
  std::uint32_t bits = bitmap[w];
  std::int32_t out   = rank[w];
  while (bits != 0u) {
    int const b = __ffs(bits) - 1;
    keys[out++] = static_cast<T>(static_cast<U>(minv) + static_cast<U>(w * 32 + b));
    bits &= bits - 1u;
  }
}

struct popcount_word {
  __device__ std::int32_t operator()(std::uint32_t w) const { return __popc(w); }
};

unsigned grid_for(std::int64_t n)
{
  int device = 0, sms = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
  auto const needed = (n + kMarkBlock - 1) / kMarkBlock;
  return static_cast<unsigned>(std::max<std::int64_t>(1, std::min<std::int64_t>(needed, sms * 4)));
}

void write_meta(rmm::device_buffer& meta,
                std::array<std::int64_t, 4> const& host,
                ::cuda::stream_ref stream)
{
  check(
    cudaMemcpyAsync(meta.data(), host.data(), sizeof(host), cudaMemcpyHostToDevice, stream.get()),
    "meta upload");
  // `host` is a stack array; the copy must land before it goes away.
  check(cudaStreamSynchronize(stream.get()), "meta upload sync");
}

template <class T>
std::optional<dict_keys> build(T const* data,
                               std::int64_t n,
                               std::int64_t max_keys,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  using U = unsigned_t<T>;
  dict_keys out;
  out.meta          = rmm::device_buffer(4 * sizeof(std::int64_t), stream, mr);
  auto const policy = rmm::exec_policy_nosync(stream, mr);

  if (n == 0) {
    out.bitmap = rmm::device_buffer(sizeof(std::uint32_t), stream, mr);
    out.rank   = rmm::device_buffer(sizeof(std::int32_t), stream, mr);
    write_meta(out.meta, {0, 0, kDictLookupSearch, 0}, stream);
    return out;
  }

  auto const [lo_it, hi_it] = thrust::minmax_element(policy, data, data + n);
  std::array<T, 2> bounds{};
  check(cudaMemcpyAsync(&bounds[0], lo_it, sizeof(T), cudaMemcpyDeviceToHost, stream.get()),
        "min download");
  check(cudaMemcpyAsync(&bounds[1], hi_it, sizeof(T), cudaMemcpyDeviceToHost, stream.get()),
        "max download");
  check(cudaStreamSynchronize(stream.get()), "minmax sync");
  T const minv = bounds[0];
  std::uint64_t const span =
    static_cast<std::uint64_t>(static_cast<U>(bounds[1]) - static_cast<U>(minv));

  if (span < kDictMaxBitmapRange) {
    auto const words = static_cast<std::int32_t>(span / 32 + 1);
    out.bitmap       = rmm::device_buffer(words * sizeof(std::uint32_t), stream, mr);
    out.rank         = rmm::device_buffer(words * sizeof(std::int32_t), stream, mr);
    auto* bitmap     = static_cast<std::uint32_t*>(out.bitmap.data());
    auto* rank       = static_cast<std::int32_t*>(out.rank.data());
    check(cudaMemsetAsync(bitmap, 0, out.bitmap.size(), stream.get()), "bitmap clear");
    auto const grid = grid_for(n);
    if (words <= kPrivateBitmapWords) {
      auto const smem = static_cast<int>(words * sizeof(std::uint32_t));
      mark_private<T><<<grid, kMarkBlock, smem, stream.get()>>>(data, n, minv, words, bitmap);
    } else {
      mark_global<T><<<grid, kMarkBlock, 0, stream.get()>>>(data, n, minv, bitmap);
    }
    check(cudaGetLastError(), "mark launch");
    auto const counts = thrust::make_transform_iterator(bitmap, popcount_word{});
    thrust::exclusive_scan(policy, counts, counts + words, rank);
    std::array<std::int32_t, 1> last_rank{};
    std::array<std::uint32_t, 1> last_word{};
    check(cudaMemcpyAsync(last_rank.data(),
                          rank + words - 1,
                          sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()),
          "rank download");
    check(cudaMemcpyAsync(last_word.data(),
                          bitmap + words - 1,
                          sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()),
          "word download");
    check(cudaStreamSynchronize(stream.get()), "key count sync");
    std::int64_t const k = last_rank[0] + __builtin_popcount(last_word[0]);
    if (k > max_keys) return std::nullopt;
    out.num_keys = static_cast<std::int32_t>(k);
    out.keys     = rmm::device_buffer(k * sizeof(T), stream, mr);
    emit_bitmap_keys<T><<<(words + 255) / 256, 256, 0, stream.get()>>>(
      bitmap, rank, words, minv, static_cast<T*>(out.keys.data()));
    check(cudaGetLastError(), "key emit launch");
    write_meta(out.meta, {static_cast<std::int64_t>(minv), k, kDictLookupBitmap, words}, stream);
    return out;
  }

  rmm::device_buffer sorted(n * sizeof(T), stream, mr);
  auto* s = static_cast<T*>(sorted.data());
  thrust::copy(policy, data, data + n, s);
  thrust::sort(policy, s, s + n);
  auto const k = static_cast<std::int64_t>(thrust::unique(policy, s, s + n) - s);
  if (k > max_keys) return std::nullopt;
  out.num_keys = static_cast<std::int32_t>(k);
  out.keys     = rmm::device_buffer(k * sizeof(T), stream, mr);
  check(cudaMemcpyAsync(out.keys.data(), s, k * sizeof(T), cudaMemcpyDeviceToDevice, stream.get()),
        "keys copy");
  out.bitmap = rmm::device_buffer(sizeof(std::uint32_t), stream, mr);
  out.rank   = rmm::device_buffer(sizeof(std::int32_t), stream, mr);
  write_meta(out.meta, {static_cast<std::int64_t>(minv), k, kDictLookupSearch, 0}, stream);
  return out;
}

}  // namespace

std::optional<dict_keys> build_dict_keys(void const* data,
                                         std::int64_t n,
                                         std::size_t elem_size,
                                         std::int64_t max_keys,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  switch (elem_size) {
    case 1: return build(static_cast<std::int8_t const*>(data), n, max_keys, stream, mr);
    case 2: return build(static_cast<std::int16_t const*>(data), n, max_keys, stream, mr);
    case 4: return build(static_cast<std::int32_t const*>(data), n, max_keys, stream, mr);
    case 8: return build(static_cast<std::int64_t const*>(data), n, max_keys, stream, mr);
    default:
      throw std::invalid_argument("simpatico dict keys: unsupported element size " +
                                  std::to_string(elem_size));
  }
}

}  // namespace simpatico
