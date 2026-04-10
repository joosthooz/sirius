// =============================================================================
// Copyright 2025, Sirius Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.
// =============================================================================
// Micro-benchmark: libcudf hash_join in build/probe pattern with row batches.
//
// Build: cmake --build build/release --target join_batch_benchmark
// Binary: build/release/extension/sirius/test/cpp/join_batch_benchmark
//
// Usage:
//   join_batch_benchmark --join-name q21 --build /path/to/x_build.parquet \
//       --probe /path/to/x_probe.parquet --key-cols 1 \
//       --batch-bytes 16M --batch-bytes 64M --iterations 3
//
// First key_cols columns of build and probe tables must be equi-join keys with
// matching types (see extract_join_data.py metadata).

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

std::size_t device_memory_used_bytes()
{
  std::size_t free  = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) { return 0; }
  return total - free;
}

std::size_t update_peak(std::size_t& peak)
{
  std::size_t const u = device_memory_used_bytes();
  if (u > peak) { peak = u; }
  return u;
}

std::uint64_t parse_byte_size(std::string const& s_in)
{
  if (s_in.empty()) { throw std::runtime_error("empty size"); }
  std::string s     = s_in;
  std::uint64_t mul = 1;
  char last         = static_cast<char>(std::tolower(static_cast<unsigned char>(s.back())));
  if (last == 'k') {
    mul = 1024;
    s.pop_back();
  } else if (last == 'm') {
    mul = 1024ULL * 1024ULL;
    s.pop_back();
  } else if (last == 'g') {
    mul = 1024ULL * 1024ULL * 1024ULL;
    s.pop_back();
  }
  char* end            = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (end == s.c_str() || v == 0) { throw std::runtime_error("invalid byte size: " + s_in); }
  return static_cast<std::uint64_t>(v) * mul;
}

std::size_t estimate_row_bytes(cudf::table_view const& tbl, rmm::cuda_stream_view stream)
{
  std::size_t sum = 0;
  for (cudf::size_type i = 0; i < tbl.num_columns(); ++i) {
    cudf::column_view const& col = tbl.column(i);
    cudf::type_id const id       = col.type().id();
    if (id == cudf::type_id::STRING) {
      cudf::strings_column_view const sv(col);
      cudf::size_type const n = std::max(col.size(), static_cast<cudf::size_type>(1));
      std::size_t const chars =
        static_cast<std::size_t>(sv.chars_size(stream)) / static_cast<std::size_t>(n);
      sum += chars + sizeof(std::int32_t);
    } else {
      sum += cudf::size_of(col.type());
    }
  }
  return std::max<std::size_t>(sum, 1);
}

cudf::table_view key_subset(cudf::table_view const& full, int key_cols)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(static_cast<std::size_t>(key_cols));
  for (int i = 0; i < key_cols; ++i) {
    cols.push_back(full.column(i));
  }
  return cudf::table_view(cols);
}

std::vector<std::pair<cudf::size_type, cudf::size_type>> make_row_ranges(
  cudf::size_type num_rows, cudf::size_type rows_per_batch)
{
  std::vector<std::pair<cudf::size_type, cudf::size_type>> ranges;
  if (num_rows <= 0 || rows_per_batch <= 0) { return ranges; }
  for (cudf::size_type start = 0; start < num_rows; start += rows_per_batch) {
    cudf::size_type const end = std::min(start + rows_per_batch, num_rows);
    ranges.emplace_back(start, end);
  }
  return ranges;
}

struct Args {
  std::string join_name;
  std::string build_path;
  std::string probe_path;
  std::string csv_path;
  int key_cols   = 1;
  int iterations = 3;
  int warmup     = 1;
  std::vector<std::uint64_t> batch_bytes;
};

void print_usage(char const* argv0)
{
  std::cerr << "Usage: " << argv0
            << " --join-name NAME --build BUILD.parquet --probe PROBE.parquet\n"
               "  [--key-cols N] [--batch-bytes SIZE]... [--iterations N] [--warmup N]\n"
               "  [--csv OUT.csv]\n"
               "SIZE suffix: K, M, G (binary). Example: --batch-bytes 16M --batch-bytes 1G\n";
}

Args parse_args(int argc, char** argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    auto need             = [&](char const* name) -> char* {
      if (i + 1 >= argc) { throw std::runtime_error(std::string("missing value for ") + name); }
      return argv[++i];
    };
    if (arg == "--join-name") {
      a.join_name = need("--join-name");
    } else if (arg == "--build") {
      a.build_path = need("--build");
    } else if (arg == "--probe") {
      a.probe_path = need("--probe");
    } else if (arg == "--key-cols") {
      a.key_cols = std::atoi(need("--key-cols"));
    } else if (arg == "--iterations") {
      a.iterations = std::atoi(need("--iterations"));
    } else if (arg == "--warmup") {
      a.warmup = std::atoi(need("--warmup"));
    } else if (arg == "--batch-bytes") {
      a.batch_bytes.push_back(parse_byte_size(need("--batch-bytes")));
    } else if (arg == "--csv") {
      a.csv_path = need("--csv");
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (a.join_name.empty() || a.build_path.empty() || a.probe_path.empty()) {
    print_usage(argv[0]);
    throw std::runtime_error("required: --join-name, --build, --probe");
  }
  if (a.key_cols < 1) { throw std::runtime_error("--key-cols must be >= 1"); }
  if (a.batch_bytes.empty()) {
    static constexpr std::uint64_t sizes[] = {
      1ULL * 1024 * 1024,
      4ULL * 1024 * 1024,
      16ULL * 1024 * 1024,
      64ULL * 1024 * 1024,
      256ULL * 1024 * 1024,
    };
    a.batch_bytes.assign(std::begin(sizes), std::end(sizes));
  }
  return a;
}

std::unique_ptr<cudf::table> read_parquet_table(std::string const& path,
                                                rmm::cuda_stream_view stream)
{
  cudf::io::parquet_reader_options opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{path}).build();
  cudf::io::table_with_metadata result = cudf::io::read_parquet(opts, stream);
  stream.synchronize();
  if (!result.tbl) { throw std::runtime_error("read_parquet returned null: " + path); }
  return std::move(result.tbl);
}

struct BenchResult {
  double build_ms            = 0;
  double probe_ms            = 0;
  double total_ms            = 0;
  std::uint64_t output_rows  = 0;
  std::size_t peak_mem       = 0;
  cudf::size_type build_rows = 0;
  cudf::size_type probe_rows = 0;
};

BenchResult run_benchmark(cudf::table_view const& build_tbl,
                          cudf::table_view const& probe_tbl,
                          int key_cols,
                          std::uint64_t target_batch_bytes,
                          int iterations,
                          int warmup,
                          rmm::cuda_stream_view stream)
{
  std::size_t const build_row_b = estimate_row_bytes(build_tbl, stream);
  std::size_t const probe_row_b = estimate_row_bytes(probe_tbl, stream);
  cudf::size_type const build_rows_per =
    std::max<cudf::size_type>(1, static_cast<cudf::size_type>(target_batch_bytes / build_row_b));
  cudf::size_type const probe_rows_per =
    std::max<cudf::size_type>(1, static_cast<cudf::size_type>(target_batch_bytes / probe_row_b));

  auto const build_ranges = make_row_ranges(build_tbl.num_rows(), build_rows_per);
  auto const probe_ranges = make_row_ranges(probe_tbl.num_rows(), probe_rows_per);

  BenchResult acc;
  acc.build_rows = build_tbl.num_rows();
  acc.probe_rows = probe_tbl.num_rows();

  for (int it = 0; it < warmup + iterations; ++it) {
    bool const count_it    = (it >= warmup);
    double build_ms        = 0;
    double probe_ms        = 0;
    std::uint64_t out_rows = 0;
    std::size_t peak       = device_memory_used_bytes();

    for (auto const& br : build_ranges) {
      std::vector<cudf::size_type> const slice_ix  = {br.first, br.second};
      std::vector<cudf::table_view> const b_slices = cudf::slice(build_tbl, slice_ix, stream);
      if (b_slices.empty()) { continue; }
      cudf::table_view const b_view     = b_slices.front();
      cudf::table_view const build_keys = key_subset(b_view, key_cols);

      auto const t0 = clock_type::now();
      cudf::hash_join hj(build_keys, cudf::null_equality::UNEQUAL, stream);
      stream.synchronize();
      auto const t1 = clock_type::now();
      build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      update_peak(peak);

      for (auto const& pr : probe_ranges) {
        std::vector<cudf::size_type> const pix       = {pr.first, pr.second};
        std::vector<cudf::table_view> const p_slices = cudf::slice(probe_tbl, pix, stream);
        if (p_slices.empty()) { continue; }
        cudf::table_view const p_view     = p_slices.front();
        cudf::table_view const probe_keys = key_subset(p_view, key_cols);

        auto const tp0 = clock_type::now();
        std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                  std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
          jr = hj.inner_join(probe_keys, std::nullopt, stream);
        stream.synchronize();
        auto const tp1 = clock_type::now();
        probe_ms += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
        update_peak(peak);
        if (jr.first) { out_rows += static_cast<std::uint64_t>(jr.first->size()); }
      }
    }

    acc.peak_mem = std::max(acc.peak_mem, peak);
    if (count_it) {
      acc.build_ms += build_ms;
      acc.probe_ms += probe_ms;
      acc.output_rows += out_rows;
    }
  }

  int const n = std::max(iterations, 1);
  acc.build_ms /= static_cast<double>(n);
  acc.probe_ms /= static_cast<double>(n);
  acc.output_rows /= static_cast<std::uint64_t>(n);
  acc.total_ms = acc.build_ms + acc.probe_ms;
  return acc;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    Args const args = parse_args(argc, argv);
    rmm::cuda_stream stream;

    auto build_table                  = read_parquet_table(args.build_path, stream);
    auto probe_table                  = read_parquet_table(args.probe_path, stream);
    cudf::table_view const build_view = build_table->view();
    cudf::table_view const probe_view = probe_table->view();

    if (build_view.num_columns() < args.key_cols || probe_view.num_columns() < args.key_cols) {
      throw std::runtime_error("table has fewer columns than --key-cols");
    }

    std::ofstream csv_out;
    bool const write_csv = !args.csv_path.empty();
    if (write_csv) {
      csv_out.open(args.csv_path, std::ios::out | std::ios::app);
      if (!csv_out) { throw std::runtime_error("failed to open csv: " + args.csv_path); }
      if (csv_out.tellp() == 0) {
        csv_out << "join_name,batch_size_bytes,build_rows,probe_rows,build_time_ms,"
                   "probe_time_ms,total_time_ms,peak_memory_bytes,"
                   "throughput_probe_rows_per_sec,avg_output_rows\n";
      }
    }

    for (std::uint64_t const bb : args.batch_bytes) {
      BenchResult const r = run_benchmark(
        build_view, probe_view, args.key_cols, bb, args.iterations, args.warmup, stream);

      double const thr =
        (r.probe_ms > 1e-9) ? (static_cast<double>(r.probe_rows) / (r.probe_ms / 1000.0)) : 0.0;

      std::cout << std::fixed << std::setprecision(3) << "join=" << args.join_name
                << " batch_bytes=" << bb << " build_ms=" << r.build_ms << " probe_ms=" << r.probe_ms
                << " total_ms=" << r.total_ms << " peak_mem=" << r.peak_mem
                << " probe_throughput_rows_s=" << thr << " avg_output_rows=" << r.output_rows
                << '\n';

      if (write_csv) {
        csv_out << args.join_name << ',' << bb << ',' << r.build_rows << ',' << r.probe_rows << ','
                << r.build_ms << ',' << r.probe_ms << ',' << r.total_ms << ',' << r.peak_mem << ','
                << thr << ',' << r.output_rows << '\n';
      }
    }

    return 0;
  } catch (std::exception const& e) {
    std::cerr << "join_batch_benchmark: " << e.what() << '\n';
    return 1;
  }
}
