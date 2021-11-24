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

#include "modelbox/device/ascend/device_ascend.h"
#include "virtualdriver_inference.h"

MindSporeInferenceFlowUnit::MindSporeInferenceFlowUnit(){};
MindSporeInferenceFlowUnit::~MindSporeInferenceFlowUnit(){};

modelbox::Status MindSporeInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto config = unit_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  merge_config->Add(*config);
  merge_config->Add(*opts);

  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  std::vector<std::string> input_type_list;
  std::vector<std::string> output_type_list;
  auto ret = GetFlowUnitIO(input_name_list, output_name_list, input_type_list,
                           output_type_list);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  merge_config->SetProperty("deviceid", dev_id_);
  infer_ = std::make_shared<MindSporeInference>();
  ret = infer_->Init(unit_desc->GetModelEntry(), merge_config, input_name_list,
                     output_name_list, input_type_list, output_type_list,
                     GetBindDevice()->GetDeviceManager()->GetDrivers());
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init inference failed, " << ret;
    return ret;
  }

  config->SetProperty<uint32_t>("batch_size", infer_->GetBatchSize());

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInferenceFlowUnit::GetFlowUnitIO(
    std::vector<std::string> &input_name_list,
    std::vector<std::string> &output_name_list,
    std::vector<std::string> &input_type_list,
    std::vector<std::string> &output_type_list) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto input_desc = unit_desc->GetFlowUnitInput();
  auto output_desc = unit_desc->GetFlowUnitOutput();
  for (auto &input : input_desc) {
    input_name_list.push_back(input.GetPortName());
    input_type_list.push_back(input.GetPortType());
  }

  for (auto &output : output_desc) {
    output_name_list.push_back(output.GetPortName());
    output_type_list.push_back(output.GetPortType());
  }

  if (input_name_list.empty() || output_name_list.empty()) {
    MBLOG_ERROR << "Wrong input name [" << input_name_list.size()
                << "] or output name [" << output_name_list.size()
                << "] number";
    return modelbox::STATUS_BADCONF;
  }

  if (input_type_list.empty() || output_type_list.empty()) {
    MBLOG_ERROR << "Wrong input type [" << input_type_list.size()
                << "] or output type [" << output_type_list.size()
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

modelbox::Status MindSporeInferenceFlowUnit::Close() { return modelbox::STATUS_OK; }

void MindSporeInferenceFlowUnitDesc::SetModelEntry(
    const std::string model_entry) {
  model_entry_ = model_entry;
}

const std::string MindSporeInferenceFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}

std::shared_ptr<modelbox::FlowUnit>
MindSporeInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<MindSporeInferenceFlowUnit>();
};
