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

#include <modelbox/base/config.h>
#include <modelbox/modelbox.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "modelbox/common/flowunit_info.h"

namespace modelbox {
FlowUnitInfo::FlowUnitInfo() = default;

FlowUnitInfo::~FlowUnitInfo() = default;

Status FlowUnitInfo::Init(const std::shared_ptr<Configuration> &config) {
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

  if (config_->GetSubConfig("driver") != nullptr) {
    auto paths = config_->GetSubConfig("driver")->GetStrings(DRIVER_DIR);
    for (const auto &search_path : paths) {
      modelbox::ListSubDirectoryFiles(search_path, "*.toml",
                                      &flowunits_from_files_);
    }
  }

  return STATUS_OK;
}

std::shared_ptr<DeviceManager> FlowUnitInfo::GetDeviceManager() {
  return device_;
}

std::shared_ptr<FlowUnitManager> FlowUnitInfo::GetFlowUnitManager() {
  return flowunit_;
}

std::shared_ptr<Drivers> FlowUnitInfo::GetDriverManager() { return drivers_; }

Status GetInfoFromTomlFile(const std::string &file, nlohmann::json &json) {
  MBLOG_DEBUG << "flowunit from file: " << file;
  std::string json_data;
  std::ifstream infile(file);
  if (infile.fail()) {
    return {modelbox::STATUS_NOTFOUND,
            "Get file failed" + modelbox::StrError(errno)};
  }
  Defer { infile.close(); };

  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());
  if (data.length() <= 0) {
    return {modelbox::STATUS_BADCONF, "toml file is invalid."};
  }

  auto ret = modelbox::TomlToJson(data, &json_data);
  if (!ret) {
    MBLOG_WARN << "Get flowunit info failed. " << ret.WrapErrormsgs();
    return {STATUS_BADCONF, "Get flowunit info failed."};
  }

  try {
    auto json_flowunit = nlohmann::json::parse(json_data);
    // only add c++ virtual flowunit
    if (json_flowunit.contains("base") == false) {
      return {STATUS_BADCONF, "not a flowunit toml file"};
    }

    if (json_flowunit["base"].contains("type") == false) {
      return {STATUS_BADCONF, "not a flowunit toml file"};
    }

    if (json_flowunit["base"]["type"] != "c++") {
      return {STATUS_BADCONF, "not a flowunit toml file"};
    }

    nlohmann::json json_inputs = nlohmann::json::array();
    nlohmann::json json_outputs = nlohmann::json::array();
    nlohmann::json json_options = nlohmann::json::array();

    json = json_flowunit["base"];
    json["type"] = json_flowunit["base"]["device"];
    json.erase("device");
    json["group"] = json_flowunit["base"]["group_type"];
    json.erase("group_type");
    if (json_flowunit.contains("input")) {
      for (auto &input : json_flowunit["input"]) {
        nlohmann::json json_input;
        json_input["name"] = input["name"];
        json_input["port_type"] = input["type"];
        json_input["device_type"] = input["device"];
        json_inputs.push_back(json_input);
      }
      json["inputports"] = json_inputs;
    }

    if (json_flowunit.contains("output")) {
      for (auto &output : json_flowunit["output"]) {
        nlohmann::json json_output;
        json_output["name"] = output["name"];
        json_output["port_type"] = output["type"];
        json_output["device_type"] = output["device"];
        json_outputs.push_back(json_output);
      }
      json["outputports"] = json_outputs;
    }

    if (json_flowunit.contains("options")) {
      for (auto &output : json_flowunit["options"]) {
        json_outputs.push_back(output);
      }
      json["options"] = json_options;
    }
  } catch (const std::exception &e) {
    std::string errmsg = "Get flowunit info failed. ";
    errmsg += e.what();
    MBLOG_WARN << errmsg;
    return {STATUS_BADCONF, errmsg};
  }

  return STATUS_OK;
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
        json["description"] = desc->GetDeviceDesc();
        devices.push_back(json);
      }
    }

    auto flow_list = flowunit_->GetAllFlowUnitDesc();
    std::map<std::string, bool> flowunit_map;
    for (const auto &flow : flow_list) {
      nlohmann::json json;
      nlohmann::json json_inputs = nlohmann::json::array();
      nlohmann::json json_outputs = nlohmann::json::array();
      nlohmann::json json_options = nlohmann::json::array();

      auto driverdesc = flow->GetDriverDesc();
      json["name"] = flow->GetFlowUnitName();
      json["type"] = driverdesc->GetType();
      json["version"] = driverdesc->GetVersion();
      json["description"] = flow->GetDescription();
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

      std::string key = json["name"];
      key += ":";
      key += json["type"];
      key += ":";
      key += json["version"];
      flowunit_map[key] = true;

      flowunits.push_back(json);
    }

    for (const auto &f : flowunits_from_files_) {
      nlohmann::json json_flowunit;
      auto ret = GetInfoFromTomlFile(f, json_flowunit);
      if (!ret) {
        if (ret == STATUS_BADCONF) {
          continue;
        }

        MBLOG_WARN << "Get flowunit info failed. " << ret.WrapErrormsgs();
        continue;
      }

      std::string key = json_flowunit["name"];
      key += ":";
      key += json_flowunit["type"];
      key += ":";
      key += json_flowunit["version"];
      if (flowunit_map.find(key) != flowunit_map.end()) {
        continue;
      }

      flowunits.push_back(json_flowunit);
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