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


#include <modelbox/modelbox.h>
#include <modelbox/base/config.h>

#include <nlohmann/json.hpp>

#include "modelbox/common/flowunit_info.h"

namespace modelbox {
FlowUnitInfo::FlowUnitInfo() {}

FlowUnitInfo::~FlowUnitInfo() {}

Status FlowUnitInfo::Init(std::shared_ptr<Configuration> config) {
  ConfigurationBuilder config_builder;
  config_ = config_builder.Build();
  if (config) {
    config_->Add(*config);
  }

  drivers_ = std::make_shared<Drivers>();
  auto status = drivers_->Initialize(config_->GetSubConfig("driver"));
  if (!status) {
    MBLOG_ERROR << "initialize drivers failed, "
                << status.WrapErrormsgs().c_str();
    return {status, "init driver failed."};
  }

  status = drivers_->Scan();
  if (!status) {
    MBLOG_ERROR << "scan failed, " << status.WrapErrormsgs().c_str();
    return {status, "scan failed."};
  }

  device_ = std::make_shared<DeviceManager>();
  status = device_->Initialize(drivers_, config_);
  if (!status) {
    MBLOG_ERROR << "init device manager failed, "
                << status.WrapErrormsgs().c_str();
    return {status, "init device manager failed."};
  }

  flowunit_ = std::make_shared<FlowUnitManager>();
  status = flowunit_->Initialize(drivers_, device_, config_);
  if (!status) {
    MBLOG_ERROR << "init flowunit manager failed, "
                << status.WrapErrormsgs().c_str();
    return {status, "init flowunit manager failed."};
  }

  return STATUS_OK;
}

std::shared_ptr<DeviceManager> FlowUnitInfo::GetDeviceManager() {
  return device_;
}

std::shared_ptr<FlowUnitManager> FlowUnitInfo::GetFlowUnitManager() {
  return flowunit_;
}

std::shared_ptr<Drivers> FlowUnitInfo::GetDriverManager() {
  return drivers_;
}


Status FlowUnitInfo::GetInfoInJson(std::string *result) {
  nlohmann::json result_json;
  nlohmann::json flowunits;
  nlohmann::json devices;

  if (result == nullptr) {
    return STATUS_INVALID;
  }

  try {
    auto device_desc_list = device_->GetDeviceDescList();
    for (const auto &itr_list : device_desc_list) {
      for (const auto &itr_device : itr_list.second) {
        nlohmann::json json;
        auto desc = itr_device.second;
        json["name"] = itr_device.first;
        json["type"] = desc->GetDeviceType();
        json["version"] = desc->GetDeviceVersion();
        json["descryption"] = desc->GetDeviceDesc();
        devices.push_back(json);
      }
    }
    
    auto flow_list = flowunit_->GetAllFlowUnitDesc();
    for (const auto &flow : flow_list) {
      nlohmann::json json;
      nlohmann::json json_inputs = nlohmann::json::array();
      nlohmann::json json_outputs = nlohmann::json::array();
      nlohmann::json json_options = nlohmann::json::array();

      auto driverdesc = flow->GetDriverDesc();
      json["name"] = flow->GetFlowUnitName();
      json["type"] = driverdesc->GetType();
      json["version"] = driverdesc->GetVersion();
      json["descryption"] = flow->GetDescription();
      json["group"] = [&]() -> std::string {
        auto type = flow->GetGroupType();
        if (type.empty()) {
          return "Generic";
        }

        return type;
      }();

      json["virtual"] = false;

      for (const auto &input : flow->GetFlowUnitInput()) {
        nlohmann::json json_input;
        json_input["name"] = input.GetPortName();
        json_input["device_type"] = input.GetDeviceType();
        json_input["port_type"] = input.GetPortType();
        json_inputs.push_back(json_input);
      }
      json["inputports"] = json_inputs;

      for (const auto &output : flow->GetFlowUnitOutput()) {
        nlohmann::json json_output;
        json_output["name"] = output.GetPortName();
        json_output["device_type"] = output.GetDeviceType();
        json_output["port_type"] = output.GetPortType();
        json_outputs.push_back(json_output);
      }
      json["outputports"] = json_outputs;

      for (auto &option : flow->GetFlowUnitOption()) {
        nlohmann::json json_option;
        json_option["name"] = option.GetOptionName();
        json_option["type"] = option.GetOptionType();
        json_option["default"] = option.GetOptionDefault();
        json_option["desc"] = option.GetOptionDesc();
        json_option["required"] = option.IsRequire();
        auto values = option.GetOptionValues();
        if (values.size() > 0) {
          nlohmann::json json_values;
          for (const auto &value : values) {
            json_values[value.first] = value.second;
          }

          json_option["values"] = json_values;
        }
        json_options.push_back(json_option);
      }
      json["options"] = json_options;

      flowunits.push_back(json);
    }

    result_json["flowunits"] = flowunits;
    result_json["devices"] = devices;
  } catch (const std::exception &e) {
    MBLOG_INFO << e.what();
    return {STATUS_INTERNAL, e.what()};
  }

  *result = result_json.dump();

  return STATUS_OK;
}

}  // namespace modelbox