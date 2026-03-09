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

#pragma once

#include <cucascade/data/data_repository.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief A lightweight scan task used in preload (cache) mode.
 *
 * Unlike parquet_scan_task, this task performs no file I/O — no footer reads,
 * no datasource creation, no data reads. It carries only the metadata needed
 * for the scan executor to serve cached data: pipeline identity, data
 * repository target, and output consumer references.
 *
 * Its compute_task() should never be called; the scan executor's preload path
 * returns cached data directly.
 */
class preload_scan_task : public pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;

 public:
  preload_scan_task(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                    shared_data_repository* data_repo,
                    std::vector<sirius_physical_operator*> output_consumers)
    : pipeline::sirius_pipeline_itask(
        nullptr, std::make_shared<pipeline::sirius_pipeline_task_global_state>(pipeline)),
      _pipeline(std::move(pipeline)),
      _data_repo(data_repo),
      _output_consumers(std::move(output_consumers))
  {
  }

  ~preload_scan_task() override
  {
    if (_pipeline) { _pipeline->mark_task_completed(); }
  }

  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view) override
  {
    throw std::runtime_error("preload_scan_task::compute_task should never be called");
  }

  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view) override
  {
    for (auto& batch : output_data.get_data_batches()) {
      _data_repo->add_data_batch(std::move(batch));
    }
  }

  [[nodiscard]] size_t get_estimated_reservation_size() const override { return 0; }

  std::vector<sirius_physical_operator*> get_output_consumers() override
  {
    return _output_consumers;
  }

 private:
  duckdb::shared_ptr<pipeline::sirius_pipeline> _pipeline;
  shared_data_repository* _data_repo;
  std::vector<sirius_physical_operator*> _output_consumers;
};

}  // namespace sirius::op::scan
