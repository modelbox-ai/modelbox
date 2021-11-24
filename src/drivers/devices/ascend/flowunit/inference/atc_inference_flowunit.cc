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


#include "atc_inference_flowunit.h"

#include <fstream>

#include "modelbox/device/ascend/device_ascend.h"
#include "virtualdriver_inference.h"

AtcInferenceFlowUnit::AtcInferenceFlowUnit(){};
AtcInferenceFlowUnit::~AtcInferenceFlowUnit(){};

modelbox::Status AtcInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  unit_desc->GetModelEntry();
  auto config = unit_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  merge_config->Add(*config);
  merge_config->Add(*opts);

  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  auto ret = GetFlowUnitIO(input_name_list, output_name_list);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  merge_config->SetProperty("deviceid", dev_id_);
  infer_ = std::make_shared<AtcInference>();
  ret = infer_->Init(unit_desc->GetModelEntry(), merge_config, input_name_list,
                     output_name_list, GetBindDevice()->GetDeviceManager()->GetDrivers());
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init inference failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status AtcInferenceFlowUnit::GetFlowUnitIO(
    std::vector<std::string> &input_name_list,
    std::vector<std::string> &output_name_list) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto input_desc = unit_desc->GetFlowUnitInput();
  auto output_desc = unit_desc->GetFlowUnitOutput();
  for (auto &input : input_desc) {
    input_name_list.push_back(input.GetPortName());
  }

  for (auto &output : output_desc) {
    output_name_list.push_back(output.GetPortName());
  }

  if (input_name_list.empty() || output_name_list.empty()) {
    MBLOG_ERROR << "Wrong input[" << input_name_list.size() << "] or output["
                << output_name_list.size() << "] number";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status AtcInferenceFlowUnit::AscendProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream) {
  auto ret = infer_->Infer(data_ctx, stream);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Execute infer failed, detail:" << ret.Errormsg();
    return {modelbox::STATUS_FAULT, ret.Errormsg()};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status AtcInferenceFlowUnit::Close() {
  auto ret = infer_->Deinit();
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Deinit inference failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

void AtcInferenceFlowUnitDesc::SetModelEntry(const std::string model_entry) {
  model_entry_ = model_entry;
}

const std::string AtcInferenceFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}

std::shared_ptr<modelbox::FlowUnit>
AtcInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<AtcInferenceFlowUnit>();
};