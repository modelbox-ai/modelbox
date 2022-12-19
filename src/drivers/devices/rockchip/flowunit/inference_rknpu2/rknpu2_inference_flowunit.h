/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_FLOWUNIT_RKNPU2_INFERENCE_H_
#define MODELBOX_FLOWUNIT_RKNPU2_INFERENCE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include "rknpu2_inference.h"

constexpr const char *RKNPU2_FLOWUNIT_TYPE = "rockchip";
constexpr const char *RKNPU2_INFERENCE_TYPE = "rknpu2";

class RKNPU2InferenceFlowUnit : public modelbox::FlowUnit {
 public:
  RKNPU2InferenceFlowUnit();
  ~RKNPU2InferenceFlowUnit() override;
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  modelbox::Status GetFlowUnitIO(
      std::shared_ptr<modelbox::InferenceRKNPUParams> &params);
  std::shared_ptr<modelbox::RKNPU2Inference> infer_;
};

class RKNPU2InferenceFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  RKNPU2InferenceFlowUnitDesc();
  ~RKNPU2InferenceFlowUnitDesc() override;
  void SetModelEntry(const std::string &model_entry);
  std::string GetModelEntry();

 private:
  std::string model_entry_;
};

class RKNPU2InferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  RKNPU2InferenceFlowUnitFactory();
  ~RKNPU2InferenceFlowUnitFactory() override;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return {};
  }
  std::string GetFlowUnitFactoryType() { return RKNPU2_FLOWUNIT_TYPE; };
  std::string GetVirtualType() { return RKNPU2_INFERENCE_TYPE; };
  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;
};

#endif  // MODELBOX_FLOWUNIT_RKNPU2_INFERENCE_H_
