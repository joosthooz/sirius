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
// Ad hoc microbench: simpatico FSST-GPU vs. Vortex FSST, on real SF1000 TPC-H
// string columns. Tagged [!benchmark] so default unittest runs skip it.
// Not part of the permanent test suite — thrown together to answer one
// question and safe to delete afterwards.
//===----------------------------------------------------------------------===//

#include "operator/mgpu_test_utils.hpp"

#include <catch.hpp>
#include <cuda_runtime.h>
#include <duckdb.hpp>
#include <utils/pinned_entry_census.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  return count >= 1;
}

bool no_gpu()
{
  if (has_gpu()) { return false; }
  WARN("Bench requires a GPU — skipping");
  return true;
}

void require_ok(const duckdb::unique_ptr<duckdb::MaterializedQueryResult>& res,
                const std::string& what)
{
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO(what << " error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
}

duckdb::unique_ptr<duckdb::MaterializedQueryResult> run_ok(duckdb::Connection& con,
                                                           const std::string& sql,
                                                           const std::string& what)
{
  auto res = con.Query(sql);
  require_ok(res, what);
  return res;
}

void write_plan_file(const fs::path& plan_dir, const std::string& table_name, const std::string& dsl)
{
  fs::create_directories(plan_dir);
  std::ofstream f(plan_dir / (table_name + ".txt"));
  f << dsl;
}

// "fsst" = plain str_split -> fsst(chars) + bitpack(offsets): the isolated-FSST plan.
// "default" plans below are copy-pasted verbatim from
// src/compression/simpatico_codegen/plans/tpch_sf1000/{lineitem,part}.txt (the explorer's own
// Pareto-picked choice for that column). Columns where the explorer already picks plain fsst
// (l_shipinstruct, c_comment, p_name, s_comment) get only one "fsst" row — default == fsst there.
constexpr const char* kFsstPlan = "input -> str_split -> offsets, chars\n"
                                  "str_split.chars -> fsst\n"
                                  "str_split.offsets -> bitpack -> chunk_min, chunk_count, "
                                  "chunk_bits, packed\n";

constexpr const char* kDataDir =
  "/tmp/claude-1000/-home-nvidia-joost-sirius/b03bc846-c2e5-4831-b672-2f4b3614917a/scratchpad/"
  "fsst_bench_data/";

double now_s()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

TEST_CASE("bench: simpatico FSST vs default-plan compression on SF1000 string columns",
          "[!benchmark][compression][fsst_vs_vortex]")
{
  if (no_gpu()) { return; }

  auto tmp = fs::temp_directory_path() / ("sirius-fsst-bench-" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto yaml_path = tmp / "comp.yaml";

  sirius::test::mgpu::mgpu_env_params params;
  params.num_gpus              = 1;
  params.cache                 = "none";
  params.scan_task_batch_size  = 2'000'000'000ull;
  params.usage_limit_fraction  = 0.6;
  sirius::test::mgpu::write_mgpu_yaml(yaml_path, params);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  struct row {
    std::string col;
    std::string variant;  // "fsst" or "default"
    std::string plan;
  };

  std::vector<row> rows = {
    {"l_comment", "fsst", kFsstPlan},
    {"l_comment",
     "default",
     "input -> str_split -> offsets, chars\n"
     "str_split.offsets -> delta -> differences\n"
     "str_split.chars -> snappy\n"
     "str_split.offsets.differences -> ans\n"},
    {"l_shipinstruct", "fsst", kFsstPlan},
    {"c_comment", "fsst", kFsstPlan},
    {"p_name", "fsst", kFsstPlan},
    {"p_comment", "fsst", kFsstPlan},
    {"p_comment",
     "default",
     "input -> str_split -> offsets, chars\n"
     "str_split.offsets -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
     "str_split.offsets.chunk_bits -> rle -> runs, values\n"
     "str_split.offsets.chunk_bits.values -> rle -> runs, values\n"
     "str_split.chars -> snappy\n"},
    {"s_comment", "fsst", kFsstPlan},
  };

  std::printf(
    "\n%-16s %-9s %12s %12s %8s %10s %10s\n", "column", "variant", "rows", "uncompr_MB", "ratio",
    "comp_GB/s", "decomp_GB/s");

  for (auto const& r : rows) {
    std::string tname      = r.col + "_" + r.variant;
    std::string path       = std::string(kDataDir) + r.col + ".parquet";
    std::string select_sql = "SELECT COUNT(*), MAX(LENGTH(s)), SUM(LENGTH(s)) FROM read_parquet('" +
                              path + "')";

    write_plan_file(plan_dir, tname, r.plan);

    double t0 = now_s();
    auto pin  = con.Query("CALL pin_table('" + path + "', tier='host', name='" + tname + "');");
    double t1 = now_s();
    require_ok(pin, "pin:" + tname);
    double compress_s = t1 - t0;

    auto census             = sirius::test::census_entry(con, tname);
    double compressed_bytes = double(census.stored_bytes);

    // First (uncached-JIT) run warms up NVRTC compile of this plan's kernel; exclude it
    // from the timed decompress measurement.
    auto warm = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
    require_ok(warm, "warmup:" + tname);
    long long rows_n      = warm->GetValue(0, 0).GetValue<int64_t>();
    long long total_chars = warm->GetValue(2, 0).GetValue<int64_t>();

    constexpr int kIters = 5;
    double best_s        = 1e18;
    for (int i = 0; i < kIters; ++i) {
      double s0 = now_s();
      auto res  = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
      double s1 = now_s();
      require_ok(res, "decompress:" + tname);
      best_s = std::min(best_s, s1 - s0);
    }

    // Uncompressed logical size: 4-byte offset per row + raw char bytes (matches what
    // str_split feeds to fsst/bitpack).
    double uncompressed_bytes = double(total_chars) + 4.0 * double(rows_n);

    double ratio       = uncompressed_bytes / compressed_bytes;
    double comp_gbps   = (uncompressed_bytes / 1e9) / compress_s;
    double decomp_gbps = (uncompressed_bytes / 1e9) / best_s;

    std::printf(
      "%-16s %-9s %12lld %12.1f %8.3f %10.2f %10.2f\n", r.col.c_str(), r.variant.c_str(), rows_n,
      uncompressed_bytes / 1e6, ratio, comp_gbps, decomp_gbps);

    run_ok(con, "CALL unpin_table('" + tname + "');", "unpin:" + tname);
  }

  fs::remove_all(tmp);
}
