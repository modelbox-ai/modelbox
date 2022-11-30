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

#include "rknpu2_inference_flowunit.h"

#include "virtualdriver_inference.h"

RKNPU2InferenceFlowUnit::RKNPU2InferenceFlowUnit() = default;
RKNPU2InferenceFlowUnit::~RKNPU2InferenceFlowUnit() = default;

modelbox::Status RKNPU2InferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  unit_desc->GetModelEntry();
  auto config = unit_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  merge_config->Add(*config);
  merge_config->Add(*opts);

  auto params = std::make_shared<modelbox::InferenceRKNPUParams>();
  auto ret = GetFlowUnitIO(params);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }
  params->device_id_ = dev_id_;

  infer_ = std::make_shared<modelbox::RKNPU2Inference>();
  return infer_->Init(unit_desc->GetModelEntry(),
                      this->GetBindDevice()->GetDeviceManager()->GetDrivers(),
                      merge_config, params);
}

modelbox::Status RKNPU2InferenceFlowUnit::GetFlowUnitIO(
    std::shared_ptr<modelbox::InferenceRKNPUParams> &params) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto input_desc = unit_desc->GetFlowUnitInput();
  auto output_desc = unit_desc->GetFlowUnitOutput();
  for (auto &input : input_desc) {
    params->input_name_list_.push_back(input.GetPortName());
    params->input_type_list_.push_back(input.GetPortType());
  }

  for (auto &output : output_desc) {
    params->output_name_list_.push_back(output.GetPortName());
    params->output_type_list_.push_back(output.GetPortType());
  }

  if (params->input_name_list_.empty() || params->output_name_list_.empty()) {
    MBLOG_ERROR << "rknpu2 Wrong input[" << params->input_name_list_.size()
                << "] or output[" << params->output_name_list_.size()
                << "] number";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status RKNPU2InferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto ret = infer_->Infer(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Execute infer failed, detail:" << ret.Errormsg();
    return {modelbox::STATUS_FAULT, ret.Errormsg()};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status RKNPU2InferenceFlowUnit::Close() {
  MBLOG_INFO << "rknn2 inference close";
  auto ret = infer_->Deinit();
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Deinit inference failed";
    return ret;
  }

  return modelbox::STATUS_SUCCESS;
}

RKNPU2InferenceFlowUnitDesc::RKNPU2InferenceFlowUnitDesc() = default;
RKNPU2InferenceFlowUnitDesc::~RKNPU2InferenceFlowUnitDesc() = default;

void RKNPU2InferenceFlowUnitDesc::SetModelEntry(
    const std::string &model_entry) {
  model_entry_ = model_entry;
}

std::string RKNPU2InferenceFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}

RKNPU2InferenceFlowUnitFactory::RKNPU2InferenceFlowUnitFactory() = default;
RKNPU2InferenceFlowUnitFactory::~RKNPU2InferenceFlowUnitFactory() = default;

std::shared_ptr<modelbox::FlowUnit>
RKNPU2InferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<RKNPU2InferenceFlowUnit>();
};