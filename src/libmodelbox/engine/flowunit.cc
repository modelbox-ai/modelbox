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

#include <utility>

#include "modelbox/tensor_list.h"

namespace modelbox {

static const std::regex REGROUPTYPE("^[A-Z][\\w/]*");

IFlowUnit::IFlowUnit() = default;
IFlowUnit::~IFlowUnit() = default;

/* class when unit is close */
Status IFlowUnit::Close() { return STATUS_OK; }

// NOLINTNEXTLINE
Status IFlowUnit::DataPre(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

// NOLINTNEXTLINE
Status IFlowUnit::DataPost(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

// NOLINTNEXTLINE
Status IFlowUnit::DataGroupPre(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

// NOLINTNEXTLINE
Status IFlowUnit::DataGroupPost(std::shared_ptr<DataContext> data_ctx) {
  return STATUS_OK;
}

void FlowUnit::SetFlowUnitDesc(std::shared_ptr<FlowUnitDesc> desc) {
  flowunit_desc_ = std::move(desc);
}

FlowUnit::FlowUnit() = default;
FlowUnit::~FlowUnit() = default;

Status FlowUnit::Open(const std::shared_ptr<Configuration> &config) {
  return STATUS_OK;
}

Status FlowUnit::Close() { return STATUS_OK; }

void FlowUnit::SetBindDevice(const std::shared_ptr<Device> &device) {
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

void FlowUnit::SetExternalData(
    const CreateExternalDataFunc &create_external_data) {
  create_ext_data_func_ = create_external_data;
}

std::shared_ptr<ExternalData> FlowUnit::CreateExternalData() const {
  if (!create_ext_data_func_) {
    return nullptr;
  }

  return create_ext_data_func_(device_);
}

CreateExternalDataFunc FlowUnit::GetCreateExternalDataFunc() {
  return create_ext_data_func_;
}

FlowUnitPort::FlowUnitPort(std::string name) : port_name_(std::move(name)){};

FlowUnitPort::FlowUnitPort(std::string name, std::string device_type)
    : port_name_(std::move(name)), device_type_(std::move(device_type)){};

FlowUnitPort::FlowUnitPort(std::string name, uint32_t device_mem_flags)
    : port_name_(std::move(name)), device_mem_flags_(device_mem_flags){};

FlowUnitPort::FlowUnitPort(std::string name, std::string device_type,
                           uint32_t device_mem_flags)
    : port_name_(std::move(name)),
      device_type_(std::move(device_type)),
      device_mem_flags_(device_mem_flags){};

FlowUnitPort::FlowUnitPort(std::string name, std::string device_type,
                           std::string type)
    : port_name_(std::move(name)),
      device_type_(std::move(device_type)),
      port_type_(std::move(type)){};

FlowUnitPort::FlowUnitPort(std::string name, std::string device_type,
                           std::string type,
                           std::map<std::string, std::string> ext)
    : port_name_(std::move(name)),
      device_type_(std::move(device_type)),
      port_type_(std::move(type)),
      ext_(std::move(ext)){};

FlowUnitPort::~FlowUnitPort() = default;

void FlowUnitPort::SetDeviceType(const std::string &device_type) {
  device_type_ = device_type;
};

void FlowUnitPort::SetPortName(const std::string &port_name) {
  port_name_ = port_name;
};

void FlowUnitPort::SetPortType(const std::string &port_type) {
  port_type_ = port_type;
};

void FlowUnitPort::SetDevice(std::shared_ptr<Device> device) {
  device_ = std::move(device);
}

void FlowUnitPort::SetProperity(const std::string &key,
                                const std::string &value) {
  ext_[key] = value;
}

std::string FlowUnitPort::GetDeviceType() const { return device_type_; };

std::string FlowUnitPort::GetPortName() const { return port_name_; };

std::string FlowUnitPort::GetPortType() const { return port_type_; };

std::shared_ptr<Device> FlowUnitPort::GetDevice() const { return device_; }

uint32_t FlowUnitPort::GetDeviceMemFlags() const { return device_mem_flags_; }

std::string FlowUnitPort::GetProperity(const std::string &key) {
  if (ext_.find(key) == ext_.end()) {
    return "";
  }

  return ext_[key];
}

FlowUnitOption::FlowUnitOption(std::string name, std::string type)
    : option_name_(std::move(name)), option_type_(std::move(type)){};

FlowUnitOption::FlowUnitOption(std::string name, std::string type, bool require)
    : option_name_(std::move(name)),
      option_type_(std::move(type)),
      option_require_{require} {};

FlowUnitOption::FlowUnitOption(std::string name, std::string type, bool require,
                               std::string default_value, std::string desc,
                               std::map<std::string, std::string> values)
    : option_name_(std::move(name)),
      option_type_(std::move(type)),
      option_require_(require),
      option_default_(std::move(default_value)),
      option_desc_(std::move(desc)),
      option_values_(std::move(values)){};

FlowUnitOption::FlowUnitOption(std::string name, std::string type, bool require,
                               std::string default_value, std::string desc)
    : option_name_(std::move(name)),
      option_type_(std::move(type)),
      option_require_(require),
      option_default_(std::move(default_value)),
      option_desc_(std::move(desc)){};

FlowUnitOption::~FlowUnitOption() { option_values_.clear(); }

void FlowUnitOption::SetOptionName(const std::string &option_name) {
  option_name_ = option_name;
}

void FlowUnitOption::SetOptionType(const std::string &option_type) {
  option_type_ = option_type;
}

void FlowUnitOption::SetOptionRequire(bool option_require) {
  option_require_ = option_require;
}

void FlowUnitOption::SetOptionDesc(const std::string &option_desc) {
  option_desc_ = option_desc;
}

void FlowUnitOption::AddOptionValue(const std::string &key,
                                    const std::string &value) {
  option_values_.emplace(key, value);
}

std::string FlowUnitOption::GetOptionName() const { return option_name_; }

std::string FlowUnitOption::GetOptionType() const { return option_type_; }

bool FlowUnitOption::IsRequire() const { return option_require_; }

std::string FlowUnitOption::GetOptionDefault() const { return option_default_; }

std::string FlowUnitOption::GetOptionDesc() const { return option_desc_; }

std::map<std::string, std::string> FlowUnitOption::GetOptionValues() {
  return option_values_;
}

std::string FlowUnitOption::GetOptionValue(const std::string &key) {
  auto iter = option_values_.find(key);
  if (iter == option_values_.end()) {
    return "";
  }

  return option_values_[key];
}

std::shared_ptr<Device> FlowUnit::GetBindDevice() { return device_; }

std::shared_ptr<FlowUnitDesc> FlowUnit::GetFlowUnitDesc() {
  return flowunit_desc_;
}

std::string FlowUnitDesc::GetFlowUnitName() { return flowunit_name_; };

std::string FlowUnitDesc::GetFlowUnitAliasName() { return alias_name_; };

std::string FlowUnitDesc::GetFlowUnitArgument() { return argument_; };

bool FlowUnitDesc::IsCollapseAll() {
  if (loop_type_ != LOOP) {
    if (output_type_ != COLLAPSE) {
      return false;
    }
    return is_collapse_all_;
  }

  return true;
};

bool FlowUnitDesc::IsStreamSameCount() {
  if (flow_type_ == NORMAL) {
    return true;
  }
  return is_stream_same_count_;
};

bool FlowUnitDesc::IsInputContiguous() const { return is_input_contiguous_; }

bool FlowUnitDesc::IsResourceNice() const { return is_resource_nice_; }

bool FlowUnitDesc::IsExceptionVisible() { return is_exception_visible_; };

ConditionType FlowUnitDesc::GetConditionType() { return condition_type_; };

FlowOutputType FlowUnitDesc::GetOutputType() { return output_type_; };

bool FlowUnitDesc::IsUserSetFlowType() { return is_user_set_flow_type_; }

FlowType FlowUnitDesc::GetFlowType() { return flow_type_; };

LoopType FlowUnitDesc::GetLoopType() { return loop_type_; };

std::string FlowUnitDesc::GetGroupType() { return group_type_; };

uint32_t FlowUnitDesc::GetMaxBatchSize() {
  if (max_batch_size_ != 0) {
    return max_batch_size_;
  }

  // return default value
  if (flow_type_ == STREAM) {
    return STREAM_MAX_BATCH_SIZE;
  }
  return NORMAL_MAX_BATCH_SIZE;
};

uint32_t FlowUnitDesc::GetDefaultBatchSize() {
  if (default_batch_size_ != 0) {
    return default_batch_size_;
  }

  // return default value
  if (flow_type_ == STREAM) {
    return STREAM_DEFAULT_BATCH_SIZE;
  }
  return NORMAL_DEFAULT_BATCH_SIZE;
};

std::vector<FlowUnitInput> &FlowUnitDesc::GetFlowUnitInput() {
  return flowunit_input_list_;
};
const std::vector<FlowUnitOutput> &FlowUnitDesc::GetFlowUnitOutput() {
  return flowunit_output_list_;
};

std::vector<FlowUnitOption> &FlowUnitDesc::GetFlowUnitOption() {
  return flowunit_option_list_;
}

std::shared_ptr<DriverDesc> FlowUnitDesc::GetDriverDesc() {
  return driver_desc_;
}

std::string FlowUnitDesc::GetDescription() { return flowunit_description_; }

std::string FlowUnitDesc::GetVirtualType() { return virtual_type_; }

void FlowUnitDesc::SetFlowUnitName(const std::string &flowunit_name) {
  flowunit_name_ = flowunit_name;
}

void FlowUnitDesc::SetFlowUnitGroupType(const std::string &group_type) {
  if (CheckGroupType(group_type) != STATUS_SUCCESS) {
    MBLOG_WARN << "check group type failed , your group_type is " << group_type
               << ", the right group_type is a or a/b , for instance input "
                  "or input/http.";
    return;
  }

  group_type_ = group_type;
};

void FlowUnitDesc::SetDriverDesc(std::shared_ptr<DriverDesc> driver_desc) {
  driver_desc_ = std::move(driver_desc);
}

void FlowUnitDesc::SetFlowUnitAliasName(const std::string &alias_name) {
  alias_name_ = alias_name;
};

void FlowUnitDesc::SetFlowUnitArgument(const std::string &argument) {
  argument_ = argument;
};

void FlowUnitDesc::SetConditionType(ConditionType condition_type) {
  condition_type_ = condition_type;
}

void FlowUnitDesc::SetLoopType(LoopType loop_type) { loop_type_ = loop_type; }

void FlowUnitDesc::SetOutputType(FlowOutputType output_type) {
  output_type_ = output_type;
}

void FlowUnitDesc::SetFlowType(FlowType flow_type) {
  is_user_set_flow_type_ = true;
  flow_type_ = flow_type;
}

void FlowUnitDesc::SetStreamSameCount(bool is_stream_same_count) {
  if (flow_type_ == STREAM) {
    is_stream_same_count_ = is_stream_same_count;
  }
};

void FlowUnitDesc::SetInputContiguous(bool is_input_contiguous) {
  is_input_contiguous_ = is_input_contiguous;
}

void FlowUnitDesc::SetResourceNice(bool is_resource_nice) {
  is_resource_nice_ = is_resource_nice;
}

void FlowUnitDesc::SetCollapseAll(bool is_collapse_all) {
  if (output_type_ == COLLAPSE) {
    is_collapse_all_ = is_collapse_all;
  }
};

void FlowUnitDesc::SetExceptionVisible(bool is_exception_visible) {
  is_exception_visible_ = is_exception_visible;
};

void FlowUnitDesc::SetVirtualType(const std::string &virtual_type) {
  virtual_type_ = virtual_type;
}

void FlowUnitDesc::SetDescription(const std::string &description) {
  flowunit_description_ = description;
}

void FlowUnitDesc::SetMaxBatchSize(const uint32_t &max_batch_size) {
  if (max_batch_size == 0) {
    MBLOG_ERROR << "max_batch_size must be greater than zero.";
    return;
  }
  max_batch_size_ = max_batch_size;
}

void FlowUnitDesc::SetDefaultBatchSize(const uint32_t &default_batch_size) {
  if (default_batch_size == 0) {
    MBLOG_ERROR << "default_batch_size must be greater than zero.";
    return;
  }
  default_batch_size_ = default_batch_size;
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
  if (!std::regex_match(group_type, REGROUPTYPE)) {
    auto err_msg = group_type +
                   " is not match, you can use a-z, A-Z, 1-9, _ and uppercase "
                   "the first character.";
    MBLOG_WARN << err_msg;
    return {STATUS_INVALID, err_msg};
  }

  if (group_type.find('/') == std::string::npos) {
    return STATUS_SUCCESS;
  }

  if (group_type.find_first_of('/') != group_type.find_last_of('/')) {
    auto err_msg = "there are more than one / in " + group_type;
    MBLOG_WARN << err_msg;
    return {STATUS_INVALID, err_msg};
  }

  return STATUS_SUCCESS;
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

FlowUnitFactory::FlowUnitFactory() = default;
FlowUnitFactory::~FlowUnitFactory() = default;

std::map<std::string, std::shared_ptr<FlowUnitDesc>>
FlowUnitFactory::FlowUnitProbe() {
  return std::map<std::string, std::shared_ptr<FlowUnitDesc>>();
}

void FlowUnitFactory::SetDriver(const std::shared_ptr<Driver> &driver) {
  driver_ = driver;
}

std::shared_ptr<Driver> FlowUnitFactory::GetDriver() { return driver_; }

std::string FlowUnitFactory::GetFlowUnitFactoryType() { return ""; };

std::string FlowUnitFactory::GetFlowUnitFactoryName() { return ""; };

std::vector<std::string> FlowUnitFactory::GetFlowUnitNames() {
  return std::vector<std::string>();
};

std::string FlowUnitFactory::GetVirtualType() { return ""; };

void FlowUnitFactory::SetVirtualType(const std::string &virtual_type){};

std::shared_ptr<FlowUnit> FlowUnitFactory::CreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type) {
  if (GetVirtualType().empty()) {
    StatusError = {STATUS_FAULT, "invalid Flow Unit"};
    return nullptr;
  }

  return VirtualCreateFlowUnit(unit_name, unit_type, GetVirtualType());
};

std::shared_ptr<FlowUnit> FlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  StatusError = {STATUS_FAULT, "Invalid virtual flowunit"};
  return nullptr;
}

void FlowUnitFactory::SetFlowUnitFactory(
    const std::vector<std::shared_ptr<DriverFactory>>
        &bind_flowunit_factory_list){};

}  // namespace modelbox
