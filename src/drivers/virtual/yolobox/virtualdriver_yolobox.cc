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

#include "virtualdriver_yolobox.h"

using namespace modelbox;

constexpr const char *VIRTUAL_FLOWUNIT_TYPE = "yolo_postprocess";

std::shared_ptr<modelbox::DriverFactory> YoloBoxVirtualDriver::CreateFactory() {
  auto factory = std::make_shared<YoloBoxVirtualFlowUnitFactory>();
  auto real_driver_list = GetBindDriver();
  factory->SetDriver(shared_from_this());
  auto real_factory_list =
      std::vector<std::shared_ptr<modelbox::DriverFactory>>();
  for (auto &real_driver : real_driver_list) {
    auto real_factory = real_driver->CreateFactory();
    if (real_factory == nullptr) {
      auto driver_desc = real_driver->GetDriverDesc();
      MBLOG_ERROR << "real driver binded by virtual yolo driver create "
                     "factory failed, real drivers is "
                  << driver_desc->GetName() << ", " << driver_desc->GetType()
                  << ", " << driver_desc->GetFilePath();
      continue;
    }
    real_factory_list.push_back(real_factory);
  }
  factory->SetFlowUnitFactory(real_factory_list);
  return factory;
}

std::vector<std::shared_ptr<modelbox::Driver>>
YoloBoxVirtualDriver::GetBindDriver() {
  return flowunit_driver_list_;
}

void YoloBoxVirtualDriver::SetBindDriver(
    std::vector<std::shared_ptr<modelbox::Driver>> driver_list) {
  flowunit_driver_list_ = driver_list;
}

modelbox::Status YoloBoxVirtualDriverManager::Init(modelbox::Drivers &driver) {
  auto ret = GetTargetDriverList(driver);
  return ret;
}

modelbox::Status YoloBoxVirtualDriverManager::Scan(const std::string &path) {
  std::vector<std::string> config_file_list;
  std::string filter = "*.toml";
  auto status =
      modelbox::ListSubDirectoryFiles(path, filter, &config_file_list);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "list directory:  " + path + "/" + filter + " failed.";
    return {status, err_msg};
  }

  for (auto &config_file : config_file_list) {
    auto result = Add(config_file);
    if (result) {
      MBLOG_INFO << "Add virtual driver " << config_file << " success";
    }

    if (result == STATUS_NOTSUPPORT) {
      MBLOG_DEBUG << "add file: " << config_file << " failed, "
                  << result.WrapErrormsgs();
    } else if (!result) {
      MBLOG_ERROR << "add file: " << config_file << " failed, "
                  << result.WrapErrormsgs();
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status YoloBoxVirtualDriverManager::Add(const std::string &file) {
  std::string name, type, version, description, entry, flowunit_type;
  auto builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config = builder->Build(file);
  if (config == nullptr) {
    MBLOG_ERROR << StatusError.Errormsg();
    return STATUS_BADCONF;
  }

  flowunit_type = config->GetString("base.type");
  if (flowunit_type.empty()) {
    auto err_msg = "config " + file + " the config does not have 'type'.";
    MBLOG_ERROR << err_msg;
    return {STATUS_BADCONF, err_msg};
  }

  if (flowunit_type != VIRTUAL_FLOWUNIT_TYPE) {
    auto err_msg = "config " + file + " type is " + flowunit_type +
                   ", will not load as " + std::string(VIRTUAL_FLOWUNIT_TYPE);
    return {STATUS_NOTSUPPORT, err_msg};
  }

  name = config->GetString("base.name");
  if (name.empty()) {
    auto err_msg = "config " + file + " does not have 'name'.";
    MBLOG_ERROR << err_msg;
    return {STATUS_BADCONF, err_msg};
  }

  type = config->GetString("base.device");
  if (type.empty()) {
    auto err_msg = "config " + file + " does not have 'device'.";
    MBLOG_ERROR << err_msg;
    return {STATUS_BADCONF, err_msg};
  }

  version = config->GetString("base.version");
  if (version.empty()) {
    auto err_msg = "config " + file + " does not have 'version'.";
    MBLOG_ERROR << err_msg;
    return {STATUS_BADCONF, err_msg};
  }

  description = config->GetString("base.description");
  if (description.empty()) {
    auto err_msg = "config " + file + " does not have 'description'.";
    MBLOG_ERROR << err_msg;
    return {STATUS_BADCONF, err_msg};
  }

  std::shared_ptr<YoloBoxVirtualDriver> driver =
      std::make_shared<YoloBoxVirtualDriver>();
  std::shared_ptr<DriverDesc> driver_desc = std::make_shared<DriverDesc>();
  driver_desc->SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  driver_desc->SetFilePath(file);
  driver_desc->SetName(name);
  driver_desc->SetType(type);
  auto status = driver_desc->SetVersion(version);
  if (status != STATUS_SUCCESS) {
    auto err_msg = "SetVersion failed, version: " + version;
    return {STATUS_FAULT, err_msg};
  }

  driver_desc->SetDescription(description);
  driver->SetDriverDesc(driver_desc);
  driver->SetVirtual(true);
  driver->SetBindDriver(flowunit_driver_list_);
  drivers_list_.push_back(driver);
  return STATUS_OK;
}

modelbox::Status YoloBoxVirtualDriverManager::GetTargetDriverList(
    modelbox::Drivers &drivers) {
  for (auto &bind_type : BIND_FLOWUNIT_TYPE) {
    auto tmp_driver =
        drivers.GetDriver(modelbox::DRIVER_CLASS_FLOWUNIT, bind_type,
                          BIND_FLOWUNIT_NAME, BIND_FLOWUNIT_VERSION);
    if (tmp_driver == nullptr) {
      continue;
    }

    flowunit_driver_list_.push_back(tmp_driver);
  }

  if (flowunit_driver_list_.empty()) {
    return {modelbox::STATUS_NOTFOUND, "can not find yolo flowunit"};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status YoloBoxVirtualFlowUnitFactory::FillInput(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<YoloBoxVirtualFlowUnitDesc> &flowunit_desc) {
  auto unit_input = config->GetSubKeys("input");
  if (unit_input.empty()) {
    MBLOG_ERROR << "the key 'input' is not found in config file.";
    return modelbox::STATUS_BADCONF;
  }

  for (size_t i = 1; i <= unit_input.size(); ++i) {
    std::string input_device, input_name, input_type;
    auto key = "input.input" + std::to_string(i);
    auto input_item_table = config->GetSubKeys(key);
    if (input_item_table.empty()) {
      MBLOG_ERROR << "the key " << key << " is not found in config file.";
      return modelbox::STATUS_BADCONF;
    }

    auto name_index = key + ".name";
    input_name = config->GetString(name_index);
    if (input_name.empty()) {
      MBLOG_ERROR << "the key " << key << " should have key name.";
      return modelbox::STATUS_BADCONF;
    }

    flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput(input_name));
  }

  return modelbox::STATUS_OK;
}

modelbox::Status YoloBoxVirtualFlowUnitFactory::FillOutput(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<YoloBoxVirtualFlowUnitDesc> &flowunit_desc) {
  auto unit_output = config->GetSubKeys("output");
  if (unit_output.empty()) {
    MBLOG_ERROR << "the key 'output' is not found in config file.";
    return modelbox::STATUS_BADCONF;
  }

  for (size_t i = 1; i <= unit_output.size(); ++i) {
    std::string output_device, output_name, output_type;
    auto key = "output.output" + std::to_string(i);
    auto output_item_table = config->GetSubKeys(key);
    if (output_item_table.empty()) {
      MBLOG_ERROR << "the key " << key << " is not found in config file.";
      return modelbox::STATUS_BADCONF;
    }

    auto name_index = key + ".name";
    output_name = config->GetString(name_index);
    if (output_name.empty()) {
      MBLOG_ERROR << "the key " << key << " should have key name.";
      return modelbox::STATUS_BADCONF;
    }

    flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput(output_name));
  }

  return modelbox::STATUS_OK;
}

std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
YoloBoxVirtualFlowUnitFactory::FlowUnitProbe() {
  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> return_map;
  auto driver_desc = GetDriver()->GetDriverDesc();
  auto toml_file = driver_desc->GetFilePath();

  auto flowunit_desc = std::make_shared<YoloBoxVirtualFlowUnitDesc>();
  flowunit_desc->SetVirtualType(VIRTUAL_FLOWUNIT_TYPE);
  flowunit_desc->SetInputContiguous(false);

  std::shared_ptr<ConfigurationBuilder> builder =
      std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config = builder->Build(toml_file);

  auto ret = FillInput(config, flowunit_desc);
  if (!ret) {
    return return_map;
  }

  ret = FillOutput(config, flowunit_desc);
  if (!ret) {
    return return_map;
  }

  auto virtual_type = config->GetString("base.virtual_type");
  if (virtual_type.empty()) {
    MBLOG_ERROR << "the key 'virtual_type' is not found under base.";
    return return_map;
  }

  flowunit_desc->SetFlowUnitName(driver_desc->GetName());
  flowunit_desc->SetVirtualType(virtual_type);
  flowunit_desc->SetConfiguration(config->GetSubConfig("config"));
  flowunit_desc->SetFlowType(modelbox::FlowType::NORMAL);
  flowunit_desc->SetInputContiguous(false);
  flowunit_desc->SetFlowUnitGroupType("Image");
  flowunit_desc->SetDescription(driver_desc->GetDescription());
  return_map.insert(std::make_pair(driver_desc->GetName(), flowunit_desc));
  return return_map;
}

void YoloBoxVirtualFlowUnitFactory::SetFlowUnitFactory(
    std::vector<std::shared_ptr<modelbox::DriverFactory>>
        bind_flowunit_factory_list) {
  for (auto &bind_flowunit_factory : bind_flowunit_factory_list) {
    bind_flowunit_factory_list_.push_back(
        std::dynamic_pointer_cast<FlowUnitFactory>(bind_flowunit_factory));
  }
}

std::shared_ptr<modelbox::FlowUnit>
YoloBoxVirtualFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  for (auto &flowunit_factory : bind_flowunit_factory_list_) {
    if (std::dynamic_pointer_cast<FlowUnitFactory>(flowunit_factory)
            ->GetFlowUnitFactoryType() != unit_type) {
      continue;
    }

    if (std::dynamic_pointer_cast<FlowUnitFactory>(flowunit_factory)
            ->GetVirtualType() != virtual_type) {
      continue;
    }

    return std::dynamic_pointer_cast<FlowUnitFactory>(flowunit_factory)
        ->CreateFlowUnit(unit_name, unit_type);
  }

  return nullptr;
};

void YoloBoxVirtualFlowUnitFactory::SetVirtualType(
    const std::string &virtual_type) {
  virtual_type_ = virtual_type;
}

const std::string YoloBoxVirtualFlowUnitFactory::GetVirtualType() {
  return virtual_type_;
}

void YoloBoxVirtualFlowUnitDesc::SetConfiguration(
    const std::shared_ptr<modelbox::Configuration> config) {
  config_ = config;
}

std::shared_ptr<modelbox::Configuration>
YoloBoxVirtualFlowUnitDesc::GetConfiguration() {
  return config_;
}