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


#ifndef MODELBOX_FLOWUNIT_MINDSPRORE_ASCEND_INFERENCE_H_
#define MODELBOX_FLOWUNIT_MINDSPRORE_ASCEND_INFERENCE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/ascend/device_ascend.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include "mindspore_inference.h"

constexpr const char *FLOWUNIT_TYPE = "ascend";
constexpr const char *INFERENCE_TYPE = "mindspore";

class MindSporeInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  MindSporeInferenceFlowUnit();
  virtual ~MindSporeInferenceFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status GetFlowUnitIO(std::vector<std::string> &input_name_list,
                               std::vector<std::string> &output_name_list,
                               std::vector<std::string> &input_type_list,
                               std::vector<std::string> &output_type_list);

  std::shared_ptr<MindSporeInference> infer_;
};

class MindSporeInferenceFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  MindSporeInferenceFlowUnitDesc() = default;
  virtual ~MindSporeInferenceFlowUnitDesc() = default;

  void SetModelEntry(const std::string model_entry);
  const std::string GetModelEntry();

 private:
  std::string model_entry_;
};

class MindSporeInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  MindSporeInferenceFlowUnitFactory() = default;
  virtual ~MindSporeInferenceFlowUnitFactory() = default;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe() {
    return {};
  }

  const std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };
  const std::string GetVirtualType() { return INFERENCE_TYPE; };
  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;
};

#endif  // MODELBOX_FLOWUNIT_ASCEND_INFERENCE_H_
