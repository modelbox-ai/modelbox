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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_CPU_H_
#define MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_CPU_H_

#include <modelbox/flowunit.h>

#include "mindspore_inference.h"

constexpr const char *FLOWUNIT_TYPE = "cpu";

class MindSporeInferenceCPUFlowUnit : public modelbox::FlowUnit {
 public:
  MindSporeInferenceCPUFlowUnit();
  ~MindSporeInferenceCPUFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::shared_ptr<MindSporeInference> infer_;
};

class MindSporeInferenceCPUFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  MindSporeInferenceCPUFlowUnitFactory() = default;
  ~MindSporeInferenceCPUFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; };
  std::string GetVirtualType() override { return INFERENCE_TYPE; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_CPU_H_
