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

#include "mindspore_ascend_inference_flowunit.h"

MindSporeInferenceAsendFlowUnit::MindSporeInferenceAsendFlowUnit() = default;

MindSporeInferenceAsendFlowUnit::~MindSporeInferenceAsendFlowUnit() = default;

modelbox::Status MindSporeInferenceAsendFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto context = std::make_shared<mindspore::Context>();
  auto &device_list = context->MutableDeviceInfo();
  auto ascend_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  ascend_device_info->SetDeviceID(dev_id_);
  device_list.push_back(ascend_device_info);

  auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  device_list.push_back(cpu_device_info);

  infer_ = std::make_shared<MindSporeInference>();
  return infer_->Open(opts, this->GetFlowUnitDesc(),
                      GetBindDevice()->GetDeviceManager()->GetDrivers(),
                      context);
}

modelbox::Status MindSporeInferenceAsendFlowUnit::AscendProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream) {
  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInferenceAsendFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return infer_->Infer(data_ctx);
}

modelbox::Status MindSporeInferenceAsendFlowUnit::Close() {
  infer_ = nullptr;
  return modelbox::STATUS_OK;
}

std::shared_ptr<modelbox::FlowUnit>
MindSporeInferenceAsendFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  auto inference_flowunit = std::make_shared<MindSporeInferenceAsendFlowUnit>();
  return inference_flowunit;
};
