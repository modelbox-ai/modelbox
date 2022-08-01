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

#include "modelbox/flowunit.h"

#include "modelbox/tensor_list.h"

namespace modelbox {

static const std::regex REGROUPTYPE("^[A-Z][\\w/]*");

IFlowUnit::IFlowUnit() = default;
IFlowUnit::~IFlowUnit() = default;

/* class when unit is close */
Status IFlowUnit::Close() { return STATUS_OK; }

Status IFlowUnit::DataPre(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

Status IFlowUnit::DataPost(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

Status IFlowUnit::DataGroupPre(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

Status IFlowUnit::DataGroupPost(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

void FlowUnit::SetFlowUnitDesc(std::shared_ptr<FlowUnitDesc> desc) {
  flowunit_desc_ = desc;
}

void FlowUnit::SetBindDevice(std::shared_ptr<Device> device) {
  device_ = device;
  if (device == nullptr) {
    return;
  }

  auto dev_id_str = device->GetDeviceID();
  try {
    dev_id_ = std::stoi(dev_id_str);
  } catch (const std::exception &e) {
    MBLOG_WARN << "Convert device id to int failed, id " << dev_id_str
               << ", err " << e.what() << "; use device 0 as default";
  }
}

std::shared_ptr<Device> FlowUnit::GetBindDevice() { return device_; }

std::shared_ptr<FlowUnitDesc> FlowUnit::GetFlowUnitDesc() {
  return flowunit_desc_;
}

void FlowUnitDesc::SetFlowUnitName(const std::string &flowunit_name) {
  flowunit_name_ = flowunit_name;
}

Status FlowUnitDesc::AddFlowUnitInput(const FlowUnitInput &flowunit_input) {
  if (CheckInputDuplication(flowunit_input) != STATUS_OK) {
    MBLOG_WARN << "The flowunit input has already added.";
    return STATUS_EXIST;
  }

  flowunit_input_list_.push_back(flowunit_input);
  return STATUS_OK;
}

Status FlowUnitDesc::AddFlowUnitOutput(const FlowUnitOutput &flowunit_output) {
  if (CheckOutputDuplication(flowunit_output) != STATUS_OK) {
    MBLOG_WARN << "The flowunit input has already added.";
    return STATUS_EXIST;
  }

  flowunit_output_list_.push_back(flowunit_output);
  return STATUS_OK;
}

Status FlowUnitDesc::AddFlowUnitOption(const FlowUnitOption &flowunit_option) {
  if (CheckOptionDuplication(flowunit_option) != STATUS_OK) {
    MBLOG_WARN << "The flowunit input has already added.";
    return STATUS_EXIST;
  }

  flowunit_option_list_.push_back(flowunit_option);
  return STATUS_OK;
}

Status FlowUnitDesc::CheckGroupType(const std::string &group_type) {
  if (!std::regex_match(group_type, modelbox::REGROUPTYPE)) {
    auto err_msg = group_type +
                   " is not match, you can use a-z, A-Z, 1-9, _ and uppercase "
                   "the first character.";
    MBLOG_WARN << err_msg;
    return {STATUS_INVALID, err_msg};
  }

  if (group_type.find("/") == std::string::npos) {
    return modelbox::STATUS_SUCCESS;
  }

  if (group_type.find_first_of("/") != group_type.find_last_of("/")) {
    auto err_msg = "there are more than one / in " + group_type;
    MBLOG_WARN << err_msg;
    return {STATUS_INVALID, err_msg};
  }

  return modelbox::STATUS_SUCCESS;
}

Status FlowUnitDesc::CheckInputDuplication(
    const FlowUnitInput &flowunit_input) {
  for (const auto &input : flowunit_input_list_) {
    if (input.GetPortName() != flowunit_input.GetPortName()) {
      continue;
    }

    if (input.GetDeviceType() != flowunit_input.GetDeviceType()) {
      continue;
    }

    return STATUS_EXIST;
  }

  return STATUS_OK;
}

Status FlowUnitDesc::CheckOutputDuplication(
    const FlowUnitOutput &flowunit_output) {
  for (const auto &output : flowunit_output_list_) {
    if (output.GetPortName() != flowunit_output.GetPortName()) {
      continue;
    }

    if (output.GetDeviceType() != flowunit_output.GetDeviceType()) {
      continue;
    }

    return STATUS_EXIST;
  }

  return STATUS_OK;
}

Status FlowUnitDesc::CheckOptionDuplication(
    const FlowUnitOption &flowunit_option) {
  for (const auto &option : flowunit_option_list_) {
    if (option.GetOptionName() != flowunit_option.GetOptionName()) {
      continue;
    }

    if (option.GetOptionType() != flowunit_option.GetOptionType()) {
      continue;
    }

    return STATUS_EXIST;
  }

  return STATUS_OK;
}

}  // namespace modelbox
