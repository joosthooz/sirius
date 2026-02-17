/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/sirius_physical_delim_join.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/utilities/type_dispatcher.hpp>

namespace sirius {
namespace op {

class sirius_left_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
  // duckdb::shared_ptr<GPUIntermediateRelation> lhs_data;
  duckdb::ColumnDataAppendState append_state;
};

class sirius_right_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> join_state;
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
};

sirius_physical_delim_join::sirius_physical_delim_join(
  SiriusPhysicalOperatorType type,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_operator(type, std::move(types), estimated_cardinality),
    join(std::move(original_join)),
    delim_scans(std::move(delim_scans))
{
  D_ASSERT(type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
           type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
}

sirius_physical_right_delim_join::sirius_physical_right_delim_join(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_delim_join(SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[1]));

  join->children[1] =
    duckdb::make_uniq<sirius_physical_dummy_scan>(children[0]->get_types(), estimated_cardinality);
}

sirius_physical_left_delim_join::sirius_physical_left_delim_join(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_delim_join(SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[0]));

  auto cached_chunk_scan = duckdb::make_uniq<sirius_physical_column_data_scan>(
    children[0]->get_types(),
    SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
    estimated_cardinality,
    nullptr);
  if (delim_idx.IsValid()) { cached_chunk_scan->cte_index = delim_idx.GetIndex(); }
  join->children[0] = std::move(cached_chunk_scan);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_left_delim_join::build_pipelines(pipeline::sirius_pipeline& current,
                                                      pipeline::sirius_meta_pipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the RHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::sirius_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }
  join->build_pipelines(current, meta_pipeline);
}

void sirius_physical_right_delim_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the LHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::sirius_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }

  sirius_physical_hash_join::build_join_pipelines(current, meta_pipeline, *join, false);
}

std::unique_ptr<operator_data> sirius_physical_right_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  const auto& batches = input_data.get_data_batches();
  if (!batches.empty() && batches[0] && batches[0]->get_data()) {
    auto pipeline_id            = get_pipeline() ? get_pipeline()->get_pipeline_id() : 0u;
    cudf::table_view input_view = sirius::get_cudf_table_view(*batches[0]);
    for (cudf::size_type c = 0; c < input_view.num_columns(); ++c) {
      SIRIUS_LOG_DEBUG("Pipeline {}: Right delim join input  col[{}]: {}",
                       pipeline_id,
                       c,
                       cudf::type_to_name(input_view.column(c).type()));
    }
    for (duckdb::idx_t c = 0; c < types.size(); ++c) {
      SIRIUS_LOG_DEBUG("Pipeline {}: Right delim join expected output col[{}]: {}",
                       pipeline_id,
                       c,
                       types[c].ToString());
    }
    SIRIUS_LOG_DEBUG("Pipeline {}: Right delim join output (pass-through, same as input)",
                     pipeline_id);
  }
  return std::make_unique<operator_data>(input_data);
}

void sirius_physical_right_delim_join::sink(const operator_data& input_data,
                                            rmm::cuda_stream_view stream)
{
  auto pipeline_id      = get_pipeline() ? get_pipeline()->get_pipeline_id() : 0u;
  auto log_column_types = [pipeline_id](const operator_data& data, const char* label) {
    const auto& batches = data.get_data_batches();
    if (!batches.empty() && batches[0] && batches[0]->get_data()) {
      cudf::table_view tv = sirius::get_cudf_table_view(*batches[0]);
      for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
        SIRIUS_LOG_DEBUG("Pipeline {}: sink {} col[{}]: {}",
                         pipeline_id,
                         label,
                         c,
                         cudf::type_to_name(tv.column(c).type()));
      }
    }
  };

  log_column_types(input_data, "before (input_data)");
  // call partition join execute
  auto partition_join_output = partition_join->execute(input_data, stream);
  log_column_types(*partition_join_output, "after partition_join->execute");
  // call distinct execute
  auto distinct_output = distinct->execute(input_data, stream);
  log_column_types(*distinct_output, "after distinct->execute");
  // call partition distinct execute
  auto partition_distinct_output = partition_distinct->execute(*distinct_output, stream);
  log_column_types(*partition_distinct_output, "after partition_distinct->execute");
  // call partition join sink
  partition_join->sink(*partition_join_output, stream);
  // call partition distinct sink
  partition_distinct->sink(*partition_distinct_output, stream);
  SIRIUS_LOG_DEBUG("Pipeline {}: sink right delim join done", pipeline_id);
}

std::unique_ptr<operator_data> sirius_physical_left_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  const auto& batches = input_data.get_data_batches();
  if (!batches.empty() && batches[0] && batches[0]->get_data()) {
    auto pipeline_id            = get_pipeline() ? get_pipeline()->get_pipeline_id() : 0u;
    cudf::table_view input_view = sirius::get_cudf_table_view(*batches[0]);
    for (cudf::size_type c = 0; c < input_view.num_columns(); ++c) {
      SIRIUS_LOG_DEBUG("Pipeline {}: Left delim join input  col[{}]: {}",
                       pipeline_id,
                       c,
                       cudf::type_to_name(input_view.column(c).type()));
    }
    for (duckdb::idx_t c = 0; c < types.size(); ++c) {
      SIRIUS_LOG_DEBUG("Pipeline {}: Left delim join expected output col[{}]: {}",
                       pipeline_id,
                       c,
                       types[c].ToString());
    }
    SIRIUS_LOG_DEBUG("Pipeline {}: Left delim join output (pass-through, same as input)",
                     pipeline_id);
  }
  return std::make_unique<operator_data>(input_data);
}

void sirius_physical_left_delim_join::sink(const operator_data& input_data,
                                           rmm::cuda_stream_view stream)
{
  auto pipeline_id      = get_pipeline() ? get_pipeline()->get_pipeline_id() : 0u;
  auto log_column_types = [pipeline_id](const operator_data& data, const char* label) {
    const auto& batches = data.get_data_batches();
    if (!batches.empty() && batches[0] && batches[0]->get_data()) {
      cudf::table_view tv = sirius::get_cudf_table_view(*batches[0]);
      for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
        SIRIUS_LOG_DEBUG("Pipeline {}: sink {} col[{}]: {}",
                         pipeline_id,
                         label,
                         c,
                         cudf::type_to_name(tv.column(c).type()));
      }
    }
  };

  log_column_types(input_data, "before (input_data)");
  // call distinct execute
  auto distinct_output = distinct->execute(input_data, stream);
  log_column_types(*distinct_output, "after distinct->execute");
  // call column data scan execute
  auto column_data_scan_output = column_data_scan->execute(*distinct_output, stream);
  log_column_types(*column_data_scan_output, "after column_data_scan->execute");
  // call partition distinct execute
  auto partition_distinct_output = partition_distinct->execute(*distinct_output, stream);
  log_column_types(*partition_distinct_output, "after partition_distinct->execute");
  // call partition join sink
  column_data_scan->sink(*column_data_scan_output, stream);
  // call partition distinct sink
  partition_distinct->sink(*partition_distinct_output, stream);
  SIRIUS_LOG_DEBUG("Pipeline {}: sink left delim join done", pipeline_id);
}

}  // namespace op
}  // namespace sirius
