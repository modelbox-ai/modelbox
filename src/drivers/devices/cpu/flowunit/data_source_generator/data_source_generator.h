/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_GENERATOR_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_GENERATOR_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "data_source_generator";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: The operator can generator test data source config for "
    "data_source_parser. \n"
    "\t@Port parameter:  The output port buffer data indicate data source "
    "config. \n"
    "\t@Constraint: This flowunit is usually followed by 'data_source_parser'.";

class DataSourceGeneratorFlowUnit : public modelbox::FlowUnit {
 public:
  DataSourceGeneratorFlowUnit();
  ~DataSourceGeneratorFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  /* run when processing data */
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_GENERATOR_CPU_H_
