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

#include "mindspore_inference_flowunit.h"

#include <fstream>

#include "virtualdriver_inference.h"

MindSporeInferenceFlowUnit::MindSporeInferenceFlowUnit() = default;
MindSporeInferenceFlowUnit::~MindSporeInferenceFlowUnit() = default;

std::shared_ptr<mindspore::Context>
MindSporeInferenceFlowUnit::InitMindSporeContext(
    std::shared_ptr<modelbox::Configuration> &config) {
  modelbox::StatusError = modelbox::STATUS_BADCONF;
  auto context = std::make_shared<mindspore::Context>();
  auto &device_list = context->MutableDeviceInfo();
  auto info = GetDeviceInfoContext(config);
  if (info == nullptr) {
    modelbox::StatusError = {modelbox::StatusError,
                             "Init device info context failed."};
    return nullptr;
  }

  device_list.push_back(info);
  return context;
}

modelbox::Status MindSporeInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto config = unit_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  merge_config->Add(*config);
  merge_config->Add(*opts);

  struct MindSporeIOList iolist;

  auto ret = GetFlowUnitIO(iolist);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  auto context = InitMindSporeContext(merge_config);
  if (context == nullptr) {
    return {modelbox::StatusError, "init mindspore context failed."};
  }

  merge_config->SetProperty("deviceid", dev_id_);
  infer_ = std::make_shared<MindSporeInference>();
  ret = infer_->Init(context, unit_desc->GetModelEntry(), merge_config, iolist,
                     GetBindDevice()->GetDeviceManager()->GetDrivers());
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init inference failed, " << ret;
    return ret;
  }

  config->SetProperty<uint32_t>("batch_size", infer_->GetBatchSize());

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInferenceFlowUnit::GetFlowUnitIO(
    struct MindSporeIOList &io_list) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto input_desc = unit_desc->GetFlowUnitInput();
  auto output_desc = unit_desc->GetFlowUnitOutput();
  for (auto &input : input_desc) {
    io_list.input_name_list.push_back(input.GetPortName());
    io_list.input_type_list.push_back(input.GetPortType());
  }

  for (auto &output : output_desc) {
    io_list.output_name_list.push_back(output.GetPortName());
    io_list.output_type_list.push_back(output.GetPortType());
  }

  if (io_list.input_name_list.empty() || io_list.output_name_list.empty()) {
    MBLOG_ERROR << "Wrong input name [" << io_list.input_name_list.size()
                << "] or output name [" << io_list.output_name_list.size()
                << "] number";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto ret = infer_->Infer(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Execute infer failed, detail:" << ret.Errormsg();
    return {modelbox::STATUS_FAULT, ret.Errormsg()};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInferenceFlowUnit::Close() {
  infer_ = nullptr;
  return modelbox::STATUS_OK;
}
