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

// Offset-value-code experiments on lineitem/orders-shaped data, sorted on l_orderkey.
//
//   join: lineitem (probe, sorted) joined with orders (unique build, sorted, full or filtered).
//         cuDF hash, distinct-hash and sort-merge joins against custom kernels that search every
//         probe row, walk sorted chunks, or search only the first row of each key run (the rows
//         whose offset value code says the join key changed).
//   agg:  GROUP BY l_orderkey (q18 shape) and GROUP BY l_orderkey, o_orderdate, o_shippriority
//         (q3 shape). cuDF hash and sorted group-by against labels derived from the codes and a
//         segmented reduction over the runs the codes delimit.
//   ovc:  cost of computing codes for (l_orderkey, l_linenumber) and of carrying them through a
//         filter instead of recomputing them.
//
// OVC_KEYS sets the number of distinct order keys (default 16M); lineitem has 1-7 rows per key.

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using index_t = cudf::size_type;

#define CUDA_CHECK(expr)                                                                 \
  do {                                                                                   \
    cudaError_t const err_ = (expr);                                                     \
    if (err_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

template <typename T>
using dvec = rmm::device_uvector<T>;

// ---------------------------------------------------------------------------------------------
// Timing and CUB plumbing

template <typename Fn>
double mean_ms(
  cudaStream_t stream, Fn&& fn, char const* range = "timed", int iters = 5, int warmup = 2)
{
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  nvtxRangePushA(range);
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  nvtxRangePop();
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// Runs a two-phase CUB algorithm: `call(temp, bytes)` with temp == nullptr sizes the scratch.
template <typename Call>
void cub_run(cudaStream_t stream, Call&& call)
{
  std::size_t bytes = 0;
  CUDA_CHECK(call(nullptr, bytes));
  rmm::device_buffer temp(bytes, stream);
  CUDA_CHECK(call(temp.data(), bytes));
}

template <typename T>
T read_scalar(T const* device_ptr, cudaStream_t stream)
{
  T value{};
  CUDA_CHECK(cudaMemcpyAsync(&value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return value;
}

template <typename T>
int64_t device_sum(T const* data, std::size_t n, cudaStream_t stream)
{
  return thrust::reduce(thrust::cuda::par.on(stream), data, data + n, int64_t{0});
}

constexpr int block_size = 256;
inline int grid_for(std::size_t n) { return static_cast<int>((n + block_size - 1) / block_size); }

// ---------------------------------------------------------------------------------------------
// Data: orders keys are sparse like o_orderkey; lineitem has one to seven lines per order.

int64_t key_of(std::size_t k) { return static_cast<int64_t>(k) * 4 + 1; }
std::size_t lines_of(std::size_t k) { return 1 + ((k * 0x9E3779B97F4A7C15ull) >> 61) % 7; }
bool order_passes_filter(std::size_t k) { return ((k * 0xBF58476D1CE4E5B9ull) >> 63) == 0; }

struct dataset {
  std::size_t num_orders;
  std::size_t num_lines;
  dvec<int64_t> orders_key;           // all orders, sorted
  dvec<int64_t> orders_key_filtered;  // about half of the orders, sorted (q3-style date filter)
  dvec<int64_t> l_orderkey;           // sorted
  dvec<int32_t> l_linenumber;
  dvec<int32_t> o_orderdate;  // functionally dependent on l_orderkey
  dvec<int32_t> o_shippriority;
  dvec<double> l_quantity;
  dvec<double> revenue;
  dvec<uint8_t> line_keep;  // a 50% row filter, like a predicate on l_shipdate
};

template <typename T>
dvec<T> upload(std::vector<T> const& host, cudaStream_t stream)
{
  dvec<T> out(host.size(), stream);
  CUDA_CHECK(cudaMemcpyAsync(
    out.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  return out;
}

dataset make_dataset(std::size_t num_orders, cudaStream_t stream)
{
  std::vector<int64_t> orders(num_orders);
  std::vector<int64_t> filtered;
  std::vector<int64_t> orderkey;
  std::vector<int32_t> linenumber;
  std::vector<int32_t> orderdate;
  std::vector<int32_t> shippriority;
  std::vector<double> quantity;
  std::vector<double> revenue;
  std::vector<uint8_t> keep;
  orderkey.reserve(num_orders * 4);
  linenumber.reserve(num_orders * 4);
  orderdate.reserve(num_orders * 4);
  shippriority.reserve(num_orders * 4);
  quantity.reserve(num_orders * 4);
  revenue.reserve(num_orders * 4);
  keep.reserve(num_orders * 4);
  std::mt19937_64 rng{7};
  for (std::size_t k = 0; k < num_orders; ++k) {
    orders[k] = key_of(k);
    if (order_passes_filter(k)) { filtered.push_back(key_of(k)); }
    auto const date = static_cast<int32_t>(8000 + k % 2400);
    for (std::size_t line = 1; line <= lines_of(k); ++line) {
      orderkey.push_back(key_of(k));
      linenumber.push_back(static_cast<int32_t>(line));
      orderdate.push_back(date);
      shippriority.push_back(0);
      quantity.push_back(static_cast<double>(1 + rng() % 50));
      revenue.push_back(static_cast<double>(rng() % 100000) / 100.0);
      keep.push_back(static_cast<uint8_t>(rng() & 1));
    }
  }
  dataset data{num_orders,
               orderkey.size(),
               upload(orders, stream),
               upload(filtered, stream),
               upload(orderkey, stream),
               upload(linenumber, stream),
               upload(orderdate, stream),
               upload(shippriority, stream),
               upload(quantity, stream),
               upload(revenue, stream),
               upload(keep, stream)};
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return data;
}

template <typename T>
cudf::column_view col(dvec<T> const& v, cudf::type_id type)
{
  return cudf::column_view{
    cudf::data_type{type}, static_cast<index_t>(v.size()), v.data(), nullptr, 0};
}

cudf::column_view col(dvec<int64_t> const& v) { return col(v, cudf::type_id::INT64); }
cudf::column_view col(dvec<int32_t> const& v) { return col(v, cudf::type_id::INT32); }
cudf::column_view col(dvec<double> const& v) { return col(v, cudf::type_id::FLOAT64); }

// ---------------------------------------------------------------------------------------------
// Offset value codes. For a stream sorted on k key columns, a row's offset is the index of the
// first key column in which it differs from the previous row (k when equal; 0 for the first
// row). The value half of the code is the row's value in that column; the experiments only need
// the offset, which is what decides run boundaries and whether a lookup can be skipped.

__global__ void ovc_offsets_kernel(int64_t const* key0,
                                   int32_t const* key1,
                                   index_t n,
                                   uint8_t* offsets)
{
  auto const i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) { return; }
  if (i == 0 || key0[i] != key0[i - 1]) {
    offsets[i] = 0;
  } else {
    offsets[i] = key1[i] != key1[i - 1] ? 1 : 2;
  }
}

// Run heads on the first key column: rows whose offset is 0.
__global__ void run_heads_from_offsets_kernel(uint8_t const* offsets, index_t n, uint8_t* heads)
{
  auto const i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) { heads[i] = offsets[i] == 0; }
}

__global__ void run_heads_from_keys_kernel(int64_t const* keys, index_t n, uint8_t* heads)
{
  auto const i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) { heads[i] = i == 0 || keys[i] != keys[i - 1]; }
}

dvec<uint8_t> compute_offsets(dataset const& d, cudaStream_t stream)
{
  dvec<uint8_t> offsets(d.num_lines, stream);
  ovc_offsets_kernel<<<grid_for(d.num_lines), block_size, 0, stream>>>(
    d.l_orderkey.data(), d.l_linenumber.data(), static_cast<index_t>(d.num_lines), offsets.data());
  return offsets;
}

dvec<uint8_t> heads_from_offsets(dvec<uint8_t> const& offsets, cudaStream_t stream)
{
  dvec<uint8_t> heads(offsets.size(), stream);
  run_heads_from_offsets_kernel<<<grid_for(offsets.size()), block_size, 0, stream>>>(
    offsets.data(), static_cast<index_t>(offsets.size()), heads.data());
  return heads;
}

// Positions of the run heads, followed by a sentinel equal to n.
dvec<index_t> head_positions(dvec<uint8_t> const& heads, cudaStream_t stream)
{
  auto const n = static_cast<index_t>(heads.size());
  dvec<index_t> positions(heads.size() + 1, stream);
  dvec<index_t> count(1, stream);
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::Flagged(temp,
                                      bytes,
                                      cuda::counting_iterator<index_t>{0},
                                      heads.data(),
                                      positions.data(),
                                      count.data(),
                                      n,
                                      stream);
  });
  auto const num_runs = read_scalar(count.data(), stream);
  CUDA_CHECK(cudaMemcpyAsync(
    positions.data() + num_runs, &n, sizeof(index_t), cudaMemcpyHostToDevice, stream));
  positions.resize(num_runs + 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return positions;
}

// ---------------------------------------------------------------------------------------------
// Join kernels. Build keys are sorted and unique; probe keys are sorted.

__device__ inline index_t lower_bound(int64_t const* keys, index_t lo, index_t hi, int64_t key)
{
  while (lo < hi) {
    index_t const mid = lo + (hi - lo) / 2;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void search_each_kernel(
  int64_t const* probe, index_t num_probe, int64_t const* build, index_t num_build, index_t* match)
{
  auto const i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_probe) { return; }
  auto const key = probe[i];
  auto const pos = lower_bound(build, 0, num_build, key);
  match[i]       = (pos < num_build && build[pos] == key) ? pos : -1;
}

// Each thread merges a chunk of consecutive probe rows: one search for the first row, then a
// short forward walk, falling back to a search when the walk would be long.
template <int Chunk>
__global__ void merge_walk_kernel(
  int64_t const* probe, index_t num_probe, int64_t const* build, index_t num_build, index_t* match)
{
  auto const chunk    = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t const begin = chunk * Chunk;
  if (begin >= num_probe) { return; }
  index_t const end = min(begin + Chunk, num_probe);
  index_t pos       = lower_bound(build, 0, num_build, probe[begin]);
  for (index_t i = begin; i < end; ++i) {
    auto const key = probe[i];
    int steps      = 0;
    while (pos < num_build && build[pos] < key && steps < 8) {
      ++pos;
      ++steps;
    }
    if (steps == 8) { pos = lower_bound(build, pos, num_build, key); }
    match[i] = (pos < num_build && build[pos] == key) ? pos : -1;
  }
}

struct has_match {
  __device__ bool operator()(cuda::std::tuple<index_t, index_t> const& t) const
  {
    return cuda::std::get<1>(t) >= 0;
  }
};

struct join_output {
  dvec<index_t> probe_idx;
  dvec<index_t> build_idx;
};

// Compacts a per-probe-row match array into [probe index, build index] pairs, in probe order.
join_output compact_matches(dvec<index_t> const& match, cudaStream_t stream)
{
  auto const n = static_cast<index_t>(match.size());
  dvec<index_t> probe_idx(match.size(), stream);
  dvec<index_t> build_idx(match.size(), stream);
  dvec<index_t> count(1, stream);
  auto const in  = cuda::make_zip_iterator(cuda::counting_iterator<index_t>{0}, match.data());
  auto const out = cuda::make_zip_iterator(probe_idx.data(), build_idx.data());
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::If(temp, bytes, in, out, count.data(), n, has_match{}, stream);
  });
  auto const total = read_scalar(count.data(), stream);
  probe_idx.resize(total, stream);
  build_idx.resize(total, stream);
  return {std::move(probe_idx), std::move(build_idx)};
}

join_output join_search_each(dvec<int64_t> const& probe,
                             dvec<int64_t> const& build,
                             cudaStream_t stream)
{
  dvec<index_t> match(probe.size(), stream);
  search_each_kernel<<<grid_for(probe.size()), block_size, 0, stream>>>(
    probe.data(),
    static_cast<index_t>(probe.size()),
    build.data(),
    static_cast<index_t>(build.size()),
    match.data());
  return compact_matches(match, stream);
}

join_output join_merge_walk(dvec<int64_t> const& probe,
                            dvec<int64_t> const& build,
                            cudaStream_t stream)
{
  constexpr int chunk = 8;
  dvec<index_t> match(probe.size(), stream);
  merge_walk_kernel<chunk><<<grid_for((probe.size() + chunk - 1) / chunk), block_size, 0, stream>>>(
    probe.data(),
    static_cast<index_t>(probe.size()),
    build.data(),
    static_cast<index_t>(build.size()),
    match.data());
  return compact_matches(match, stream);
}

// One search per run of equal probe keys; the run's rows inherit the result.
__global__ void search_runs_kernel(int64_t const* probe,
                                   index_t const* run_starts,
                                   index_t num_runs,
                                   int64_t const* build,
                                   index_t num_build,
                                   index_t* run_match,
                                   index_t* run_out_count)
{
  auto const r = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r > num_runs) { return; }
  if (r == num_runs) {
    run_out_count[r] = 0;
    return;
  }
  auto const start = run_starts[r];
  auto const key   = probe[start];
  auto const pos   = lower_bound(build, 0, num_build, key);
  bool const found = pos < num_build && build[pos] == key;
  run_match[r]     = found ? pos : -1;
  run_out_count[r] = found ? run_starts[r + 1] - start : 0;
}

__global__ void expand_runs_kernel(index_t const* run_starts,
                                   index_t num_runs,
                                   index_t const* run_match,
                                   index_t const* run_out_offset,
                                   index_t* probe_idx,
                                   index_t* build_idx)
{
  auto const r = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= num_runs || run_match[r] < 0) { return; }
  auto const start = run_starts[r];
  auto const len   = run_starts[r + 1] - start;
  auto const out   = run_out_offset[r];
  for (index_t t = 0; t < len; ++t) {
    probe_idx[out + t] = start + t;
    build_idx[out + t] = run_match[r];
  }
}

join_output join_search_runs(dvec<int64_t> const& probe,
                             dvec<uint8_t> const& heads,
                             dvec<int64_t> const& build,
                             cudaStream_t stream)
{
  auto const run_starts = head_positions(heads, stream);
  auto const num_runs   = static_cast<index_t>(run_starts.size() - 1);
  dvec<index_t> run_match(num_runs, stream);
  dvec<index_t> run_count(num_runs + 1, stream);
  search_runs_kernel<<<grid_for(num_runs + 1), block_size, 0, stream>>>(
    probe.data(),
    run_starts.data(),
    num_runs,
    build.data(),
    static_cast<index_t>(build.size()),
    run_match.data(),
    run_count.data());
  dvec<index_t> run_offset(num_runs + 1, stream);
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::ExclusiveSum(
      temp, bytes, run_count.data(), run_offset.data(), num_runs + 1, stream);
  });
  auto const total = read_scalar(run_offset.data() + num_runs, stream);
  dvec<index_t> probe_idx(total, stream);
  dvec<index_t> build_idx(total, stream);
  expand_runs_kernel<<<grid_for(num_runs), block_size, 0, stream>>>(run_starts.data(),
                                                                    num_runs,
                                                                    run_match.data(),
                                                                    run_offset.data(),
                                                                    probe_idx.data(),
                                                                    build_idx.data());
  return {std::move(probe_idx), std::move(build_idx)};
}

struct join_check {
  std::size_t rows;
  int64_t probe_sum;
  int64_t build_sum;
  bool operator==(join_check const&) const = default;
};

join_check check_of(index_t const* probe_idx,
                    index_t const* build_idx,
                    std::size_t rows,
                    cudaStream_t stream)
{
  return {rows, device_sum(probe_idx, rows, stream), device_sum(build_idx, rows, stream)};
}

using cudf_pair = std::pair<std::unique_ptr<rmm::device_uvector<index_t>>,
                            std::unique_ptr<rmm::device_uvector<index_t>>>;

join_check check_of(cudf_pair const& p, cudaStream_t stream)
{
  return check_of(p.first->data(), p.second->data(), p.first->size(), stream);
}

join_check check_of(join_output const& o, cudaStream_t stream)
{
  return check_of(o.probe_idx.data(), o.build_idx.data(), o.probe_idx.size(), stream);
}

void report_join(std::string const& scenario,
                 std::string const& impl,
                 double ms,
                 join_check const& c,
                 join_check const& reference)
{
  std::printf("[join] %-22s %-38s %8.3f ms  rows=%zu  %s\n",
              scenario.c_str(),
              impl.c_str(),
              ms,
              c.rows,
              c == reference ? "ok" : "MISMATCH");
}

void bench_left_join(std::string const& scenario,
                     cudf::table_view const& build_view,
                     cudf::table_view const& probe_view,
                     cudaStream_t stream)
{
  auto const nulls = cudf::null_equality::EQUAL;
  auto hash        = [&] {
    cudf::hash_join join(build_view, nulls, stream);
    return join.left_join(probe_view, std::nullopt, stream);
  };
  auto const reference = check_of(hash(), stream);
  report_join(scenario,
              "cudf hash_join, left",
              mean_ms(stream, hash, "cudf hash_join, left"),
              reference,
              reference);

  auto smj = [&] {
    cudf::sort_merge_join join(build_view, cudf::sorted::YES, nulls, stream);
    return join.left_join(probe_view, stream);
  };
  report_join(scenario,
              "cudf sort_merge_join, left",
              mean_ms(stream, smj, "cudf sort_merge_join, left"),
              check_of(smj(), stream),
              reference);
}

// cuDF joins only: shows the probe-order dependence of the joins that accept any probe order.
void bench_cudf_join(std::string const& scenario,
                     dvec<int64_t> const& build,
                     dvec<int64_t> const& probe,
                     cudaStream_t stream)
{
  auto const build_view = cudf::table_view{{col(build)}};
  auto const probe_view = cudf::table_view{{col(probe)}};
  auto const nulls      = cudf::null_equality::EQUAL;

  auto distinct = [&] {
    cudf::distinct_hash_join join(build_view, nulls, 0.5, stream);
    return join.inner_join(probe_view, stream);
  };
  auto const reference = check_of(distinct(), stream);
  report_join(scenario,
              "cudf distinct_hash_join",
              mean_ms(stream, distinct, "cudf distinct_hash_join"),
              reference,
              reference);

  auto smj = [&] {
    cudf::sort_merge_join join(build_view, cudf::sorted::YES, nulls, stream);
    return join.inner_join(probe_view, stream);
  };
  report_join(scenario,
              "cudf sort_merge_join (build sorted)",
              mean_ms(stream, smj, "cudf sort_merge_join (build sorted)"),
              check_of(smj(), stream),
              reference);

  bench_left_join(scenario, build_view, probe_view, stream);
}

dvec<int64_t> shuffled(dvec<int64_t> const& in, cudaStream_t stream)
{
  dvec<int64_t> out(in.size(), stream);
  thrust::shuffle_copy(thrust::cuda::par.on(stream),
                       in.data(),
                       in.data() + in.size(),
                       out.data(),
                       thrust::default_random_engine{42});
  return out;
}

void bench_join(std::string const& scenario,
                dataset const& d,
                dvec<int64_t> const& build,
                dvec<uint8_t> const& offsets,
                cudaStream_t stream)
{
  auto const build_view = cudf::table_view{{col(build)}};
  auto const probe_view = cudf::table_view{{col(d.l_orderkey)}};
  auto const nulls      = cudf::null_equality::EQUAL;

  auto distinct = [&] {
    cudf::distinct_hash_join join(build_view, nulls, 0.5, stream);
    return join.inner_join(probe_view, stream);
  };
  auto const reference = check_of(distinct(), stream);
  report_join(scenario,
              "cudf distinct_hash_join",
              mean_ms(stream, distinct, "cudf distinct_hash_join"),
              reference,
              reference);

  auto hash = [&] {
    cudf::hash_join join(build_view, nulls, stream);
    return join.inner_join(probe_view, std::nullopt, stream);
  };
  report_join(scenario,
              "cudf hash_join",
              mean_ms(stream, hash, "cudf hash_join"),
              check_of(hash(), stream),
              reference);

  auto smj = [&] {
    cudf::sort_merge_join join(build_view, cudf::sorted::YES, nulls, stream);
    return join.inner_join(probe_view, stream);
  };
  report_join(scenario,
              "cudf sort_merge_join (build sorted)",
              mean_ms(stream, smj, "cudf sort_merge_join (build sorted)"),
              check_of(smj(), stream),
              reference);
  bench_left_join(scenario, build_view, probe_view, stream);

  auto each = [&] { return join_search_each(d.l_orderkey, build, stream); };
  report_join(scenario,
              "custom: search every probe row",
              mean_ms(stream, each, "custom: search every probe row"),
              check_of(each(), stream),
              reference);

  auto walk = [&] { return join_merge_walk(d.l_orderkey, build, stream); };
  report_join(scenario,
              "custom: merge walk, 8 rows/thread",
              mean_ms(stream, walk, "custom: merge walk, 8 rows/thread"),
              check_of(walk(), stream),
              reference);

  // The codes exist already (computed at pin time); only the run-head flags are derived here.
  auto runs_ovc = [&] {
    auto const heads = heads_from_offsets(offsets, stream);
    return join_search_runs(d.l_orderkey, heads, build, stream);
  };
  report_join(scenario,
              "custom: search run heads (from OVC)",
              mean_ms(stream, runs_ovc, "custom: search run heads (from OVC)"),
              check_of(runs_ovc(), stream),
              reference);

  auto runs_keys = [&] {
    dvec<uint8_t> heads(d.num_lines, stream);
    run_heads_from_keys_kernel<<<grid_for(d.num_lines), block_size, 0, stream>>>(
      d.l_orderkey.data(), static_cast<index_t>(d.num_lines), heads.data());
    return join_search_runs(d.l_orderkey, heads, build, stream);
  };
  report_join(scenario,
              "custom: search run heads (keys compared)",
              mean_ms(stream, runs_keys, "custom: search run heads (keys compared)"),
              check_of(runs_keys(), stream),
              reference);
}

// ---------------------------------------------------------------------------------------------
// Grouped aggregation.

struct agg_check {
  std::size_t groups;
  double total;
  int64_t key_total;
  bool close_to(agg_check const& o) const
  {
    return groups == o.groups && key_total == o.key_total &&
           std::abs(total - o.total) <= 1e-6 * std::abs(o.total) + 1e-3;
  }
};

double device_sum_double(cudf::column_view const& v, cudaStream_t stream)
{
  auto const* p = v.data<double>();
  return thrust::reduce(thrust::cuda::par.on(stream), p, p + v.size(), 0.0);
}

void report_agg(std::string const& scenario,
                std::string const& impl,
                double ms,
                agg_check const& c,
                agg_check const& reference)
{
  std::printf("[agg]  %-22s %-38s %8.3f ms  groups=%zu  %s\n",
              scenario.c_str(),
              impl.c_str(),
              ms,
              c.groups,
              c.close_to(reference) ? "ok" : "MISMATCH");
}

struct agg_result {
  std::unique_ptr<cudf::table> keys;
  std::unique_ptr<cudf::column> sums;
};

agg_result cudf_groupby(cudf::table_view const& keys,
                        cudf::column_view const& values,
                        cudf::sorted sorted,
                        cudaStream_t stream)
{
  std::vector<cudf::order> order(keys.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_order(keys.num_columns(), cudf::null_order::AFTER);
  cudf::groupby::groupby gb(keys, cudf::null_policy::INCLUDE, sorted, order, null_order);
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto [out_keys, results] = gb.aggregate(requests, stream);
  return {std::move(out_keys), std::move(results[0].results[0])};
}

agg_check check_of(agg_result const& r, cudaStream_t stream)
{
  auto const first_key = r.keys->get_column(0).view();
  auto const* k        = first_key.data<int64_t>();
  auto const key_total =
    thrust::reduce(thrust::cuda::par.on(stream), k, k + first_key.size(), int64_t{0});
  return {
    static_cast<std::size_t>(r.sums->size()), device_sum_double(r.sums->view(), stream), key_total};
}

struct to_int {
  __device__ index_t operator()(uint8_t v) const { return v; }
};

// Group labels from run heads: label = number of heads up to and including the row.
dvec<index_t> labels_from_heads(dvec<uint8_t> const& heads, cudaStream_t stream)
{
  dvec<index_t> labels(heads.size(), stream);
  auto const in = cuda::make_transform_iterator(heads.data(), to_int{});
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::InclusiveSum(
      temp, bytes, in, labels.data(), static_cast<index_t>(heads.size()), stream);
  });
  return labels;
}

// Group keys are the key columns at the run heads.
std::unique_ptr<cudf::table> keys_at(cudf::table_view const& keys,
                                     dvec<index_t> const& run_starts,
                                     cudaStream_t stream)
{
  cudf::column_view const map{cudf::data_type{cudf::type_id::INT32},
                              static_cast<index_t>(run_starts.size() - 1),
                              run_starts.data(),
                              nullptr,
                              0};
  return cudf::gather(keys, map, cudf::out_of_bounds_policy::DONT_CHECK, stream);
}

agg_result ovc_label_groupby(cudf::table_view const& keys,
                             cudf::column_view const& values,
                             dvec<uint8_t> const& offsets,
                             cudaStream_t stream)
{
  auto const heads      = heads_from_offsets(offsets, stream);
  auto const labels     = labels_from_heads(heads, stream);
  auto const run_starts = head_positions(heads, stream);
  auto result = cudf_groupby(cudf::table_view{{col(labels)}}, values, cudf::sorted::YES, stream);
  return {keys_at(keys, run_starts, stream), std::move(result.sums)};
}

// Runs average under four rows, so one thread per run is enough.
__global__ void sum_runs_kernel(double const* values,
                                index_t const* run_starts,
                                index_t num_runs,
                                double* sums)
{
  auto const r = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= num_runs) { return; }
  double sum = 0.0;
  for (index_t i = run_starts[r]; i < run_starts[r + 1]; ++i) {
    sum += values[i];
  }
  sums[r] = sum;
}

agg_result ovc_sum_runs(cudf::table_view const& keys,
                        cudf::column_view const& values,
                        dvec<uint8_t> const& offsets,
                        cudaStream_t stream)
{
  auto const heads      = heads_from_offsets(offsets, stream);
  auto const run_starts = head_positions(heads, stream);
  auto const num_runs   = static_cast<index_t>(run_starts.size() - 1);
  auto sums             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, num_runs, cudf::mask_state::UNALLOCATED, stream);
  sum_runs_kernel<<<grid_for(num_runs), block_size, 0, stream>>>(
    values.data<double>(), run_starts.data(), num_runs, sums->mutable_view().data<double>());
  return {keys_at(keys, run_starts, stream), std::move(sums)};
}

// Needs only sortedness on the first key column, which the other keys depend on.
agg_result cub_reduce_by_key(cudf::table_view const& keys,
                             cudf::column_view const& values,
                             cudaStream_t stream)
{
  auto const n      = keys.num_rows();
  auto const* first = keys.column(0).data<int64_t>();
  auto unique_keys  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream);
  auto sums = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream);
  dvec<index_t> count(1, stream);
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceReduce::ReduceByKey(temp,
                                          bytes,
                                          first,
                                          unique_keys->mutable_view().data<int64_t>(),
                                          values.data<double>(),
                                          sums->mutable_view().data<double>(),
                                          count.data(),
                                          cuda::std::plus<>{},
                                          n,
                                          stream);
  });
  auto const groups = read_scalar(count.data(), stream);
  // Only the first key column comes out of the reduction; the dependent keys would be gathered
  // at the run starts as in the OVC variants. For the comparison, report the reduction alone.
  auto trimmed =
    std::make_unique<cudf::column>(cudf::slice(sums->view(), {0, groups}, stream)[0], stream);
  std::vector<std::unique_ptr<cudf::column>> key_cols;
  key_cols.push_back(std::make_unique<cudf::column>(
    cudf::slice(unique_keys->view(), {0, groups}, stream)[0], stream));
  return {std::make_unique<cudf::table>(std::move(key_cols)), std::move(trimmed)};
}

void bench_agg(std::string const& scenario,
               cudf::table_view const& keys,
               cudf::column_view const& values,
               dvec<uint8_t> const& offsets,
               cudaStream_t stream)
{
  auto hash            = [&] { return cudf_groupby(keys, values, cudf::sorted::NO, stream); };
  auto const reference = check_of(hash(), stream);
  report_agg(scenario,
             "cudf groupby, hash",
             mean_ms(stream, hash, "cudf groupby, hash"),
             reference,
             reference);

  auto sorted_checked = [&] {
    std::vector<cudf::order> order(keys.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_order(keys.num_columns(), cudf::null_order::AFTER);
    bool const is_sorted = cudf::is_sorted(keys, order, null_order, stream);
    return cudf_groupby(keys, values, is_sorted ? cudf::sorted::YES : cudf::sorted::NO, stream);
  };
  report_agg(scenario,
             "cudf is_sorted + sorted groupby",
             mean_ms(stream, sorted_checked, "cudf is_sorted + sorted groupby"),
             check_of(sorted_checked(), stream),
             reference);

  auto sorted_trusted = [&] { return cudf_groupby(keys, values, cudf::sorted::YES, stream); };
  report_agg(scenario,
             "cudf sorted groupby, no check",
             mean_ms(stream, sorted_trusted, "cudf sorted groupby, no check"),
             check_of(sorted_trusted(), stream),
             reference);

  auto labels = [&] { return ovc_label_groupby(keys, values, offsets, stream); };
  report_agg(scenario,
             "OVC labels + cudf sorted groupby",
             mean_ms(stream, labels, "OVC labels + cudf sorted groupby"),
             check_of(labels(), stream),
             reference);

  auto per_run = [&] { return ovc_sum_runs(keys, values, offsets, stream); };
  report_agg(scenario,
             "OVC runs + thread-per-run sum",
             mean_ms(stream, per_run, "OVC runs + thread-per-run sum"),
             check_of(per_run(), stream),
             reference);

  auto by_key = [&] { return cub_reduce_by_key(keys, values, stream); };
  report_agg(scenario,
             "cub reduce_by_key, first key only",
             mean_ms(stream, by_key, "cub reduce_by_key, first key only"),
             check_of(by_key(), stream),
             reference);
}

// ---------------------------------------------------------------------------------------------
// Carrying codes through a filter.

struct offset_segment {
  uint8_t offset;
  bool starts;  // the row follows a surviving row, so the dropped-run minimum restarts here
};

struct min_within_segment {
  __device__ offset_segment operator()(offset_segment a, offset_segment b) const
  {
    return b.starts ? b : offset_segment{static_cast<uint8_t>(min(a.offset, b.offset)), a.starts};
  }
};

struct make_offset_segment {
  uint8_t const* offsets;
  uint8_t const* keep;
  __device__ offset_segment operator()(index_t i) const
  {
    return {offsets[i], i == 0 || keep[i - 1] != 0};
  }
};

struct segment_offset {
  __device__ uint8_t operator()(offset_segment s) const { return s.offset; }
};

// A surviving row's new offset is the minimum offset over itself and the rows dropped since the
// previous survivor.
dvec<uint8_t> propagate_offsets(dvec<uint8_t> const& offsets,
                                dvec<uint8_t> const& keep,
                                cudaStream_t stream)
{
  auto const n = static_cast<index_t>(offsets.size());
  dvec<offset_segment> scanned(offsets.size(), stream);
  auto const in = cuda::make_transform_iterator(cuda::counting_iterator<index_t>{0},
                                                make_offset_segment{offsets.data(), keep.data()});
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::InclusiveScan(
      temp, bytes, in, scanned.data(), min_within_segment{}, n, stream);
  });
  dvec<uint8_t> out(offsets.size(), stream);
  dvec<index_t> count(1, stream);
  auto const values = cuda::make_transform_iterator(scanned.data(), segment_offset{});
  cub_run(stream, [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::Flagged(
      temp, bytes, values, keep.data(), out.data(), count.data(), n, stream);
  });
  out.resize(read_scalar(count.data(), stream), stream);
  return out;
}

void bench_ovc(dataset const& d, dvec<uint8_t> const& offsets, cudaStream_t stream)
{
  auto compute = [&] { return compute_offsets(d, stream); };
  std::printf("[ovc]  compute codes for (l_orderkey, l_linenumber), %zu rows: %8.3f ms\n",
              d.num_lines,
              mean_ms(stream, compute));

  auto const n = static_cast<index_t>(d.num_lines);
  // Filtering compacts the key columns either way; recomputing then compares adjacent keys again.
  auto filter_then_recompute = [&] {
    dvec<int64_t> key0(d.num_lines, stream);
    dvec<int32_t> key1(d.num_lines, stream);
    dvec<index_t> count(1, stream);
    auto const in  = cuda::make_zip_iterator(d.l_orderkey.data(), d.l_linenumber.data());
    auto const out = cuda::make_zip_iterator(key0.data(), key1.data());
    cub_run(stream, [&](void* temp, std::size_t& bytes) {
      return cub::DeviceSelect::Flagged(
        temp, bytes, in, d.line_keep.data(), out, count.data(), n, stream);
    });
    auto const kept = read_scalar(count.data(), stream);
    dvec<uint8_t> codes(kept, stream);
    ovc_offsets_kernel<<<grid_for(kept), block_size, 0, stream>>>(
      key0.data(), key1.data(), kept, codes.data());
    return codes;
  };
  auto propagate = [&] { return propagate_offsets(offsets, d.line_keep, stream); };

  auto const recomputed = filter_then_recompute();
  auto const propagated = propagate();
  std::vector<uint8_t> a(recomputed.size());
  std::vector<uint8_t> b(propagated.size());
  CUDA_CHECK(
    cudaMemcpyAsync(a.data(), recomputed.data(), a.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
    cudaMemcpyAsync(b.data(), propagated.data(), b.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  bool const same = a == b;
  std::printf("[ovc]  50%% filter, compact keys + recompute codes:       %8.3f ms\n",
              mean_ms(stream, filter_then_recompute));
  std::printf("[ovc]  50%% filter, carry codes through (segmented min): %8.3f ms  %s\n",
              mean_ms(stream, propagate),
              same ? "ok" : "MISMATCH");
}

class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : previous_(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(previous_)); }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous_;
};

}  // namespace

int main()
{
  std::size_t num_orders = std::size_t{16} << 20;
  if (char const* env = std::getenv("OVC_KEYS"); env != nullptr && *env != '\0') {
    num_orders = static_cast<std::size_t>(std::atoll(env));
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    rmm::mr::cuda_async_memory_resource mr;
    current_resource_guard guard{mr};

    auto const d       = make_dataset(num_orders, stream);
    auto const offsets = compute_offsets(d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("orders=%zu filtered orders=%zu lineitem=%zu\n",
                d.num_orders,
                d.orders_key_filtered.size(),
                d.num_lines);

    bench_join("full orders build", d, d.orders_key, offsets, stream);
    bench_join("filtered orders build", d, d.orders_key_filtered, offsets, stream);
    bench_cudf_join("shuffled probe", d.orders_key, shuffled(d.l_orderkey, stream), stream);

    auto const q18_keys = cudf::table_view{{col(d.l_orderkey)}};
    bench_agg("q18: by l_orderkey", q18_keys, col(d.l_quantity), offsets, stream);
    auto const q3_keys =
      cudf::table_view{{col(d.l_orderkey), col(d.o_orderdate), col(d.o_shippriority)}};
    bench_agg("q3: by 3 keys", q3_keys, col(d.revenue), offsets, stream);

    bench_ovc(d, offsets, stream);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
