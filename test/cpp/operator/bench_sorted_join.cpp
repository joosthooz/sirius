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

//===----------------------------------------------------------------------===//
// Join-algorithm microbench for sorted inputs. Compares cudf::hash_join,
// cudf::distinct_hash_join and cudf::sort_merge_join on lineitem/orders-shaped
// INT64 keys, sorted and shuffled, and reports whether each keeps the probe
// order in its output. Tagged `[!benchmark]` so default unittest runs skip it.
// SIRIUS_BENCH_JOIN_BUILD_ROWS sets the number of distinct keys (default 16M).
//===----------------------------------------------------------------------===//

#include <cudf/column/column_view.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using index_vector = rmm::device_uvector<cudf::size_type>;
using index_pair   = std::pair<std::unique_ptr<index_vector>, std::unique_ptr<index_vector>>;

class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : previous_(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(previous_)); }
  current_resource_guard(current_resource_guard const&)            = delete;
  current_resource_guard& operator=(current_resource_guard const&) = delete;

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous_;
};

std::size_t build_rows_from_env()
{
  char const* rows = std::getenv("SIRIUS_BENCH_JOIN_BUILD_ROWS");
  if (rows == nullptr || *rows == '\0') { return std::size_t{16} << 20; }
  return static_cast<std::size_t>(std::atoll(rows));
}

// Sparse like o_orderkey, so no dense-domain shortcut applies.
int64_t key_of(std::size_t k) { return static_cast<int64_t>(k) * 4 + 1; }

// One to seven rows per key, about four on average, like lineitem per order.
std::size_t repeats_of(std::size_t k) { return 1 + ((k * 0x9E3779B97F4A7C15ull) >> 61) % 7; }

std::vector<int64_t> unique_keys(std::size_t num_keys)
{
  std::vector<int64_t> keys(num_keys);
  for (std::size_t k = 0; k < num_keys; ++k) {
    keys[k] = key_of(k);
  }
  return keys;
}

std::vector<int64_t> repeated_keys(std::size_t num_keys)
{
  std::vector<int64_t> keys;
  keys.reserve(num_keys * 4);
  for (std::size_t k = 0; k < num_keys; ++k) {
    keys.insert(keys.end(), repeats_of(k), key_of(k));
  }
  return keys;
}

std::vector<int64_t> shuffled(std::vector<int64_t> keys)
{
  std::mt19937_64 rng{42};
  std::shuffle(keys.begin(), keys.end(), rng);
  return keys;
}

struct device_keys {
  rmm::device_buffer data;
  cudf::size_type rows;
  bool sorted;

  [[nodiscard]] cudf::table_view view() const
  {
    return cudf::table_view{
      {cudf::column_view{cudf::data_type{cudf::type_id::INT64}, rows, data.data(), nullptr, 0}}};
  }
};

device_keys upload(std::vector<int64_t> const& keys, bool sorted, rmm::cuda_stream& stream)
{
  return {rmm::device_buffer{keys.data(), keys.size() * sizeof(int64_t), stream.view()},
          static_cast<cudf::size_type>(keys.size()),
          sorted};
}

template <typename Fn>
double mean_ms(rmm::cuda_stream& stream, Fn&& fn, int iters = 5, int warmup = 2)
{
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  stream.synchronize();
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream.value());
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  cudaEventRecord(stop, stream.value());
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

bool non_decreasing(index_vector const& indices, rmm::cuda_stream& stream)
{
  cudf::column_view col{cudf::data_type{cudf::type_id::INT32},
                        static_cast<cudf::size_type>(indices.size()),
                        indices.data(),
                        nullptr,
                        0};
  return cudf::is_sorted(cudf::table_view{{col}}, {cudf::order::ASCENDING}, {}, stream.view());
}

struct join_result {
  double build_ms;
  double probe_ms;
  std::size_t out_rows;
  bool probe_order_kept;
};

// `make_join` builds the join object; `probe` runs one inner-join probe and returns the
// [probe indices, build indices] pair.
template <typename MakeJoin, typename Probe>
join_result run_join(rmm::cuda_stream& stream, MakeJoin&& make_join, Probe&& probe)
{
  double const build_ms    = mean_ms(stream, [&] { (void)make_join(); });
  auto join                = make_join();
  double const probe_ms    = mean_ms(stream, [&] { (void)probe(*join); });
  auto const probe_indices = probe(*join).first;
  return {build_ms, probe_ms, probe_indices->size(), non_decreasing(*probe_indices, stream)};
}

void report(std::string const& scenario, std::string const& impl, join_result const& r)
{
  std::printf(
    "[bench] %-34s %-24s build=%8.3f ms  probe=%8.3f ms  total=%8.3f ms  rows=%zu  "
    "probe_order=%s\n",
    scenario.c_str(),
    impl.c_str(),
    r.build_ms,
    r.probe_ms,
    r.build_ms + r.probe_ms,
    r.out_rows,
    r.probe_order_kept ? "kept" : "lost");
}

cudf::sorted sorted_hint(device_keys const& keys)
{
  return keys.sorted ? cudf::sorted::YES : cudf::sorted::NO;
}

void bench_scenario(std::string const& scenario,
                    device_keys const& build,
                    device_keys const& probe,
                    bool build_unique,
                    rmm::cuda_stream& stream)
{
  auto const build_view = build.view();
  auto const probe_view = probe.view();
  auto const nulls      = cudf::null_equality::EQUAL;
  std::vector<std::size_t> row_counts;

  auto const hash = run_join(
    stream,
    [&] { return std::make_unique<cudf::hash_join>(build_view, nulls, stream.view()); },
    [&](cudf::hash_join const& join) {
      return join.inner_join(probe_view, std::nullopt, stream.view());
    });
  report(scenario, "hash_join", hash);
  row_counts.push_back(hash.out_rows);

  if (build_unique) {
    auto const distinct = run_join(
      stream,
      [&] {
        return std::make_unique<cudf::distinct_hash_join>(build_view, nulls, 0.5, stream.view());
      },
      [&](cudf::distinct_hash_join const& join) {
        return join.inner_join(probe_view, stream.view());
      });
    report(scenario, "distinct_hash_join", distinct);
    row_counts.push_back(distinct.out_rows);
  }

  auto const run_smj = [&](cudf::sorted build_hint, cudf::sorted probe_hint) {
    return run_join(
      stream,
      [&] {
        return std::make_unique<cudf::sort_merge_join>(
          build_view, build_hint, nulls, stream.view());
      },
      [&](cudf::sort_merge_join const& join) -> index_pair {
        return join.inner_join(probe_view, probe_hint, stream.view());
      });
  };
  auto const smj_unhinted = run_smj(cudf::sorted::NO, cudf::sorted::NO);
  report(scenario, "sort_merge_join(no hint)", smj_unhinted);
  row_counts.push_back(smj_unhinted.out_rows);
  if (build.sorted || probe.sorted) {
    auto const smj_hinted = run_smj(sorted_hint(build), sorted_hint(probe));
    report(scenario, "sort_merge_join(hinted)", smj_hinted);
    row_counts.push_back(smj_hinted.out_rows);
  }

  for (auto const rows : row_counts) {
    REQUIRE(rows == row_counts.front());
  }
}

}  // namespace

TEST_CASE("bench sorted-input joins on lineitem/orders-shaped INT64 keys",
          "[!benchmark][operator][join]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  current_resource_guard resource_guard{mr};

  auto const num_keys = build_rows_from_env();
  auto const orders   = unique_keys(num_keys);
  auto const lineitem = repeated_keys(num_keys);
  std::printf("[bench] distinct keys=%zu  repeated rows=%zu\n", orders.size(), lineitem.size());

  auto const orders_sorted     = upload(orders, true, stream);
  auto const orders_shuffled   = upload(shuffled(orders), false, stream);
  auto const lineitem_sorted   = upload(lineitem, true, stream);
  auto const lineitem_shuffled = upload(shuffled(lineitem), false, stream);
  stream.synchronize();

  // Build on the unique side, probe with the repeated side: lineitem probing orders.
  bench_scenario("unique build sorted, probe sorted", orders_sorted, lineitem_sorted, true, stream);
  bench_scenario(
    "unique build sorted, probe shuffled", orders_sorted, lineitem_shuffled, true, stream);
  bench_scenario(
    "unique build shuffled, probe sorted", orders_shuffled, lineitem_sorted, true, stream);
  bench_scenario(
    "unique build shuffled, probe shuffled", orders_shuffled, lineitem_shuffled, true, stream);

  // Build on the repeated side: each probe row matches about four build rows.
  bench_scenario("multi build sorted, probe sorted", lineitem_sorted, orders_sorted, false, stream);
  bench_scenario(
    "multi build shuffled, probe shuffled", lineitem_shuffled, orders_shuffled, false, stream);
}
