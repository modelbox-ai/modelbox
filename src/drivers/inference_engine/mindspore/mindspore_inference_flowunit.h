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

#ifndef MODELBOX_FLOWUNIT_MINDSPRORE_INFERENCE_H_
#define MODELBOX_FLOWUNIT_MINDSPRORE_INFERENCE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include "mindspore_inference.h"

constexpr const char *INFERENCE_TYPE = "mindspore";

class MODELBOX_DLL_PUBLIC MindSporeInferenceFlowUnit
    : public modelbox::FlowUnit {
 public:
  MindSporeInferenceFlowUnit();
  ~MindSporeInferenceFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 protected:
  virtual std::shared_ptr<mindspore::DeviceInfoContext> GetDeviceInfoContext(
      std::shared_ptr<modelbox::Configuration> &config) = 0;

 private:
  modelbox::Status GetFlowUnitIO(struct MindSporeIOList &io_list);
  std::shared_ptr<mindspore::Context> InitMindSporeContext(
      std::shared_ptr<modelbox::Configuration> &config);

  std::shared_ptr<MindSporeInference> infer_;
};

#endif  // MODELBOX_FLOWUNIT_MINDSPRORE_INFERENCE_H_
