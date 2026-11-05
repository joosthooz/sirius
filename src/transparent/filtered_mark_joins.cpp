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

#include "transparent/filtered_mark_joins.hpp"

#include <duckdb/common/table_index.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>

#include <algorithm>

namespace sirius::transparent {

namespace {

/// The marker column, by position in a resolved plan and by binding in an unresolved one.
struct marker_column {
  duckdb::idx_t position;
  duckdb::TableIndex table_index;

  [[nodiscard]] bool is(duckdb::Expression const& expression) const
  {
    switch (expression.GetExpressionClass()) {
      case duckdb::ExpressionClass::BOUND_REF:
        return expression.Cast<duckdb::BoundReferenceExpression>().Index() == position;
      case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
        auto const& ref = expression.Cast<duckdb::BoundColumnRefExpression>();
        return ref.Depth() == 0 && ref.Binding().table_index == table_index;
      }
      default: return false;
    }
  }

  [[nodiscard]] bool read_by(duckdb::Expression const& expression) const
  {
    if (is(expression)) { return true; }
    bool found = false;
    duckdb::ExpressionIterator::EnumerateChildren(
      expression, [&](duckdb::Expression const& child) { found = found || read_by(child); });
    return found;
  }
};

/// Converts the MARK join below @p node when @p node is a FILTER that keeps the rows whose marker
/// is TRUE and its projection map hides the marker from every operator above.
bool convert(duckdb::unique_ptr<duckdb::LogicalOperator>& node)
{
  if (node->type != duckdb::LogicalOperatorType::LOGICAL_FILTER || node->children.size() != 1) {
    return false;
  }
  auto& filter = node->Cast<duckdb::LogicalFilter>();
  auto& child  = *filter.children[0];
  if (child.type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return false; }
  auto& join = child.Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::MARK || join.conditions.empty()) { return false; }
  for (auto const& condition : join.conditions) {
    if (!condition.IsComparison() ||
        condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
      return false;
    }
  }
  // The marker is the join's last output column. A copied plan carries no operator types until
  // they are resolved again.
  join.ResolveOperatorTypes();
  if (join.types.empty()) { return false; }
  marker_column const marker{join.types.size() - 1, join.mark_index};
  if (!filter.HasProjectionMap() || std::ranges::any_of(filter.projection_map, [&](auto index) {
        return index == marker.position;
      })) {
    return false;
  }
  auto& expressions = filter.expressions;
  auto const found  = std::ranges::find_if(
    expressions, [&](auto const& expression) { return marker.is(*expression); });
  if (found == expressions.end()) { return false; }
  if (std::any_of(expressions.begin(), expressions.end(), [&](auto const& expression) {
        return expression != *found && marker.read_by(*expression);
      })) {
    return false;
  }

  join.join_type = duckdb::JoinType::SEMI;
  expressions.erase(found);
  if (!expressions.empty()) {
    join.ResolveOperatorTypes();
    filter.ResolveOperatorTypes();
    return true;
  }
  // Nothing left to filter: the join selects the FILTER's columns itself.
  duckdb::vector<duckdb::ProjectionIndex> left;
  left.reserve(filter.projection_map.size());
  for (auto const index : filter.projection_map) {
    left.push_back(join.left_projection_map.empty() ? index : join.left_projection_map[index]);
  }
  join.left_projection_map = std::move(left);
  join.ResolveOperatorTypes();
  auto const cardinality          = filter.estimated_cardinality;
  auto const has_cardinality      = filter.has_estimated_cardinality;
  node                            = std::move(filter.children[0]);
  node->estimated_cardinality     = cardinality;
  node->has_estimated_cardinality = has_cardinality;
  return true;
}

std::size_t visit(duckdb::unique_ptr<duckdb::LogicalOperator>& op)
{
  std::size_t converted = convert(op) ? 1 : 0;
  for (auto& child : op->children) {
    converted += visit(child);
  }
  return converted;
}

}  // namespace

std::size_t convert_filtered_mark_joins(duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  return plan ? visit(plan) : 0;
}

}  // namespace sirius::transparent
