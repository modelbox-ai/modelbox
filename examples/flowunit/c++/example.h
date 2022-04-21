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

#ifndef MODELBOX_FLOWUNIT_EXAMPLE_CPU_H_
#define MODELBOX_FLOWUNIT_EXAMPLE_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_NAME = "example";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_VERSION = "1.0.0";
constexpr const char *FLOWUNIT_DESC = "\n\t@Brief: A example flowunit on cpu";

class ExampleFlowUnit : public modelbox::FlowUnit {
 public:
  ExampleFlowUnit();
  virtual ~ExampleFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();
  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> ct);
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ct);
  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> ct);
  modelbox::Status DataGroupPre(std::shared_ptr<modelbox::DataContext> ct);
  modelbox::Status DataGroupPost(std::shared_ptr<modelbox::DataContext> ct);
};

#endif  // MODELBOX_FLOWUNIT_EXAMPLE_CPU_H_
