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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_ASEND_H_
#define MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_ASEND_H_

#include <modelbox/flowunit.h>

#include "mindspore_inference.h"
#include "modelbox/device/ascend/device_ascend.h"

constexpr const char *FLOWUNIT_TYPE = "ascend";

class MindSporeInferenceAsendFlowUnit : public modelbox::AscendFlowUnit {
 public:
  MindSporeInferenceAsendFlowUnit();
  virtual ~MindSporeInferenceAsendFlowUnit();

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status AscendProcess(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      aclrtStream stream) override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status Close() override;

 private:
  std::shared_ptr<MindSporeInference> infer_;
};

class MindSporeInferenceAsendFlowUnitFactory
    : public modelbox::FlowUnitFactory {
 public:
  MindSporeInferenceAsendFlowUnitFactory() = default;
  virtual ~MindSporeInferenceAsendFlowUnitFactory() = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type);

  std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };
  std::string GetVirtualType() { return INFERENCE_TYPE; };
  std::string GetFlowUnitInputDeviceType() override { return "cpu"; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_ASEND_H_
