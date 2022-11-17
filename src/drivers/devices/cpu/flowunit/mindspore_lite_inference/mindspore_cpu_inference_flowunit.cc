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

#include "mindspore_cpu_inference_flowunit.h"

MindSporeInferenceCPUFlowUnit::MindSporeInferenceCPUFlowUnit() = default;

MindSporeInferenceCPUFlowUnit::~MindSporeInferenceCPUFlowUnit() = default;

modelbox::Status MindSporeInferenceCPUFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto context = std::make_shared<mindspore::Context>();
  auto &device_list = context->MutableDeviceInfo();
  auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  device_list.push_back(cpu_device_info);

  infer_ = std::make_shared<MindSporeInference>(GetBindDevice(), context);
  return infer_->Open(opts, this->GetFlowUnitDesc());
}

modelbox::Status MindSporeInferenceCPUFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return infer_->Infer(data_ctx);
}

modelbox::Status MindSporeInferenceCPUFlowUnit::Close() {
  infer_ = nullptr;
  return modelbox::STATUS_OK;
}

std::shared_ptr<modelbox::FlowUnit>
MindSporeInferenceCPUFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  auto inference_flowunit = std::make_shared<MindSporeInferenceCPUFlowUnit>();
  return inference_flowunit;
};
