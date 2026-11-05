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

/**
 * @file test_filtered_mark_joins.cpp
 * @brief A MARK join whose marker only feeds the FILTER above it becomes a SEMI join.
 *
 * DuckDB keeps such a MARK join for `x IN (subquery)` inside the CTE of a correlated subquery, the
 * shape of TPC-H q20.
 */

#include "transparent/filtered_mark_joins.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <string>

namespace {

constexpr char const* tables[] = {
  // p_name is NULL for every seventh part, so `pk IN (...)` has NULL markers.
  "CREATE TABLE p AS SELECT i AS pk, CASE WHEN i % 7 = 0 THEN NULL ELSE i END AS pn "
  "FROM range(100) t(i)",
  "CREATE TABLE ps AS SELECT i % 100 AS pk, i % 10 AS sk, i AS qty FROM range(1000) t(i)",
  "CREATE TABLE l AS SELECT i % 100 AS pk, i % 10 AS sk, i % 5 AS q FROM range(5000) t(i)",
};

constexpr char const* q20_shape =
  "SELECT sk, qty FROM ps WHERE pk IN (SELECT pn FROM p WHERE pk < 50) AND qty > "
  "(SELECT 0.5 * sum(q) FROM l WHERE l.pk = ps.pk AND l.sk = ps.sk)";

std::size_t count_joins(duckdb::LogicalOperator const& op, duckdb::JoinType type)
{
  std::size_t count = 0;
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
      op.Cast<duckdb::LogicalComparisonJoin>().join_type == type) {
    ++count;
  }
  for (auto const& child : op.children) {
    count += count_joins(*child, type);
  }
  return count;
}

struct plan_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  plan_fixture()
  {
    for (auto const* sql : tables) {
      REQUIRE_FALSE(con.Query(sql)->HasError());
    }
  }
};

}  // namespace

TEST_CASE_METHOD(plan_fixture,
                 "A MARK join under a FILTER on its marker becomes a SEMI join",
                 "[transparent][filtered_mark_joins]")
{
  auto plan = con.ExtractPlan(q20_shape);
  REQUIRE(count_joins(*plan, duckdb::JoinType::MARK) == 1);
  auto const semi_before = count_joins(*plan, duckdb::JoinType::SEMI);

  REQUIRE(sirius::transparent::convert_filtered_mark_joins(plan) == 1);
  CHECK(count_joins(*plan, duckdb::JoinType::MARK) == 0);
  CHECK(count_joins(*plan, duckdb::JoinType::SEMI) == semi_before + 1);
}

TEST_CASE_METHOD(plan_fixture,
                 "A MARK join whose marker is read elsewhere stays a MARK join",
                 "[transparent][filtered_mark_joins]")
{
  auto plan = con.ExtractPlan("SELECT sk, pk IN (SELECT pn FROM p WHERE pk < 50) AS found FROM ps");
  auto const marks = count_joins(*plan, duckdb::JoinType::MARK);
  REQUIRE(marks >= 1);
  CHECK(sirius::transparent::convert_filtered_mark_joins(plan) == 0);
  CHECK(count_joins(*plan, duckdb::JoinType::MARK) == marks);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "The GPU answers the q20 shape like the CPU",
                 "[integration][gpu_execution][transparent][filtered_mark_joins]")
{
  for (auto const* sql : tables) {
    run_ok(sql);
  }
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(q20_shape);
  compare_gpu_vs_cpu(
    "SELECT sk, qty FROM ps WHERE pk NOT IN (SELECT pn FROM p WHERE pk < 50) AND qty > "
    "(SELECT 0.5 * sum(q) FROM l WHERE l.pk = ps.pk AND l.sk = ps.sk)");
}
