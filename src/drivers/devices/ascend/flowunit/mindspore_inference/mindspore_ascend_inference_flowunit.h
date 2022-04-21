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

#include "mindspore_inference_flowunit.h"

constexpr const char *FLOWUNIT_TYPE = "ascend";

class MindSporeInferenceAsendFlowUnit : public MindSporeInferenceFlowUnit {
 public:
  MindSporeInferenceAsendFlowUnit();
  virtual ~MindSporeInferenceAsendFlowUnit();

 protected:
  virtual std::shared_ptr<mindspore::DeviceInfoContext> GetDeviceInfoContext(
      std::shared_ptr<modelbox::Configuration> &config);
};

class MindSporeInferenceAsendFlowUnitFactory
    : public modelbox::FlowUnitFactory {
 public:
  MindSporeInferenceAsendFlowUnitFactory() = default;
  virtual ~MindSporeInferenceAsendFlowUnitFactory() = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type);

  const std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };
  const std::string GetVirtualType() { return INFERENCE_TYPE; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_ASEND_H_
