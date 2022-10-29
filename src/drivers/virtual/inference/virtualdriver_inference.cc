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

#include "virtualdriver_inference.h"

#include <utility>

#include "modelbox/base/driver.h"

constexpr const char *VIRTUAL_FLOWUNIT_TYPE = "inference";

void VirtualInferenceFlowUnitDesc::SetModelEntry(std::string model_entry) {
  model_entry_ = std::move(model_entry);
}

std::shared_ptr<modelbox::DriverFactory>
InferenceVirtualDriver::CreateFactory() {
  auto factory = std::make_shared<VirtualInferenceFlowUnitFactory>();
  auto real_driver_list = GetBindDriver();
  factory->SetDriver(shared_from_this());
  auto real_factory_list =
      std::vector<std::shared_ptr<modelbox::DriverFactory>>();
  for (auto &real_driver : real_driver_list) {
    auto real_factory = real_driver->CreateFactory();
    if (real_factory == nullptr) {
      auto driver_desc = real_driver->GetDriverDesc();
      MBLOG_ERROR << "real driver binded by virtual inference driver create "
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
InferenceVirtualDriver::GetBindDriver() {
  return inference_flowunit_driver_list_;
}

void InferenceVirtualDriver::SetBindDriver(
    const std::vector<std::shared_ptr<modelbox::Driver>> &driver_list) {
  inference_flowunit_driver_list_ = driver_list;
}

modelbox::Status InferenceVirtualDriverManager::Init(
    modelbox::Drivers &driver) {
  auto ret = BindBaseDriver(driver);
  return ret;
}

modelbox::Status InferenceVirtualDriverManager::Scan(const std::string &path) {
  std::vector<std::string> drivers_list;
  std::string filter = "*.toml";
  auto status = modelbox::ListSubDirectoryFiles(path, filter, &drivers_list);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "list directory:  " + path + "/" + filter + " failed.";
    return {status, err_msg};
  }

  for (auto &driver_file : drivers_list) {
    auto result = Add(driver_file);
    if (result) {
      MBLOG_INFO << "Add virtual driver " << driver_file << " success";
    }

    if (result == modelbox::STATUS_NOTSUPPORT) {
      MBLOG_DEBUG << "add file: " << driver_file << " failed, "
                  << result.WrapErrormsgs();
    } else if (!result) {
      MBLOG_ERROR << "add file: " << driver_file << " failed, "
                  << result.WrapErrormsgs();
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceVirtualDriverManager::Add(const std::string &file) {
  std::string name;
  std::string type;
  std::string version;
  std::string description;
  std::string entry;
  std::string flowunit_type;
  std::shared_ptr<modelbox::ConfigurationBuilder> builder =
      std::make_shared<modelbox::ConfigurationBuilder>();
  std::shared_ptr<modelbox::Configuration> config = builder->Build(file);
  if (config == nullptr) {
    const auto &err_msg = modelbox::StatusError.Errormsg();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_BADCONF, err_msg};
  }

  flowunit_type = config->GetString("base.type");
  if (flowunit_type.empty()) {
    MBLOG_ERROR << "the config does not have 'type'.";
    return {modelbox::STATUS_BADCONF, "the config does not have 'type'."};
  }

  if (flowunit_type != VIRTUAL_FLOWUNIT_TYPE) {
    auto err_msg = "the config type is " + flowunit_type +
                   ", but the so type is " + std::string(VIRTUAL_FLOWUNIT_TYPE);
    return {modelbox::STATUS_NOTSUPPORT, err_msg};
  }

  name = config->GetString("base.name");
  if (name.empty()) {
    MBLOG_ERROR << "the config does not have 'name'.";
    return {modelbox::STATUS_BADCONF, "the config does not have 'name'."};
  }

  type = config->GetString("base.device");
  if (type.empty()) {
    MBLOG_ERROR << "the config does not have 'device'.";
    return {modelbox::STATUS_BADCONF, "the config does not have 'device'."};
  }

  version = config->GetString("base.version");
  if (version.empty()) {
    MBLOG_ERROR << "the config does not have 'version'.";
    return {modelbox::STATUS_BADCONF, "the config does not have 'version'."};
  }

  description = config->GetString("base.description");
  if (description.empty()) {
    MBLOG_ERROR << "the config does not have 'description'.";
    return {modelbox::STATUS_BADCONF,
            "the config does not have 'description'."};
  }

  std::shared_ptr<InferenceVirtualDriver> driver =
      std::make_shared<InferenceVirtualDriver>();
  std::shared_ptr<modelbox::DriverDesc> driver_desc =
      std::make_shared<modelbox::DriverDesc>();
  driver_desc->SetClass("DRIVER-FLOWUNIT");
  driver_desc->SetFilePath(file);
  driver_desc->SetName(name);
  driver_desc->SetType(type);
  auto status = driver_desc->SetVersion(version);
  if (status != modelbox::STATUS_SUCCESS) {
    auto err_msg = "SetVersion failed, version: " + version;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  driver_desc->SetDescription(description);
  driver->SetDriverDesc(driver_desc);
  driver->SetVirtual(true);
  driver->SetBindDriver(inference_flowunit_driver_list_);
  // TODO: 判断是否重复存在
  drivers_list_.push_back(driver);
  return modelbox::STATUS_OK;
}

modelbox::Status InferenceVirtualDriverManager::BindBaseDriver(
    modelbox::Drivers &driver) {
  auto inference_drivers =
      driver.GetDriverListByClass(modelbox::DRIVER_CLASS_INFERENCE);
  for (const auto &infer_driver : inference_drivers) {
    inference_flowunit_driver_list_.push_back(infer_driver);
  }

  if (inference_flowunit_driver_list_.empty()) {
    return {modelbox::STATUS_NOTFOUND, "can not find inference flowunit"};
  }

  return modelbox::STATUS_OK;
}

std::string
VirtualInferenceFlowUnitFactory::GetInferenceFlowUintInputDeviceType(
    const std::string &unit_type, const std::string &virtual_type) {
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
        ->GetFlowUnitInputDeviceType();
  }

  return "";
}

modelbox::Status VirtualInferenceFlowUnitFactory::FillItem(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc,
    const std::string &device, const std::string &type) {
  auto item = config->GetSubKeys(type);
  if (item.empty()) {
    MBLOG_ERROR << "the key " << type << " is not found in config file.";
    return modelbox::STATUS_BADCONF;
  }

  for (unsigned int i = 1; i <= item.size(); ++i) {
    std::string item_device = device;
    std::string item_name;
    std::string item_type;
    auto key = type;
    key += "." + type;
    key += std::to_string(i);
    auto item_table = config->GetSubKeys(key);
    if (item_table.empty()) {
      MBLOG_ERROR << "the key " << key << " is not found in config file.";
      return modelbox::STATUS_BADCONF;
    }

    std::map<std::string, std::string> ext_map;
    for (const auto &inner_item : item_table) {
      auto item_index = key;
      item_index += "." + inner_item;
      if (inner_item == "name") {
        item_name = config->GetString(item_index);
        if (item_name.empty()) {
          MBLOG_ERROR << "the key " << key << " should have key name.";
          return modelbox::STATUS_BADCONF;
        }
        continue;
      }

      if (inner_item == "type") {
        auto config_type = config->GetString(item_index);
        if (!config_type.empty()) {
          item_type = config_type;
        }
        continue;
      }

      if (inner_item == "device") {
        auto config_device = config->GetString(item_index);
        if (!config_device.empty()) {
          item_device = config_device;
        }
        continue;
      }

      ext_map[inner_item] = config->GetString(item_index);
    }

    if (type == "input") {
      auto device_type = GetInferenceFlowUintInputDeviceType(
          GetDriver()->GetDriverDesc()->GetType(),
          flowunit_desc->GetVirtualType());
      item_device = device_type.empty() ? item_device : device_type;
      flowunit_desc->AddFlowUnitInput(
          modelbox::FlowUnitInput(item_name, item_device, item_type, ext_map));
    } else {
      flowunit_desc->AddFlowUnitOutput(
          modelbox::FlowUnitOutput(item_name, item_device, item_type, ext_map));
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VirtualInferenceFlowUnitFactory::FillBaseInfo(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc,
    const std::string &toml_file, std::string *device) {
  auto model_entry = config->GetString("base.entry");
  if (model_entry.empty()) {
    MBLOG_ERROR << "the key 'entry' is not found under base.";
    return modelbox::STATUS_BADCONF;
  }

  if (!modelbox::IsAbsolutePath(model_entry)) {
    auto relpath = modelbox::GetDirName(toml_file);
    model_entry = relpath + "/" + model_entry;
  }
  MBLOG_DEBUG << "module entry path: " << model_entry;
  flowunit_desc->SetModelEntry(model_entry);

  auto virtual_type = config->GetString("base.virtual_type");
  if (virtual_type.empty()) {
    MBLOG_ERROR << "the key 'virtual_type' is not found under base.";
    return modelbox::STATUS_BADCONF;
  }

  flowunit_desc->SetVirtualType(virtual_type);
  *device = config->GetString("base.device");
  if (device->empty()) {
    MBLOG_ERROR << "the key 'device' is not found under base.";
    return modelbox::STATUS_BADCONF;
  }

  auto group_type = config->GetString("base.group_type");
  if (!group_type.empty()) {
    flowunit_desc->SetFlowUnitGroupType(group_type);
  }

  bool is_input_contiguous = true;
  auto contiguous_str = config->GetString("base.is_input_contiguous");
  if (contiguous_str.empty()) {
    // if set it as bool
    is_input_contiguous = config->GetBool("base.is_input_contiguous", true);
  } else {
    // key word is "false", so I need check it is false
    is_input_contiguous = !(contiguous_str == "false");
  }
  if (!is_input_contiguous) {
    // generally, it true, but some hw, for ex. sdc, must be false
    flowunit_desc->SetInputContiguous(false);
  }

  return modelbox::STATUS_OK;
}

void VirtualInferenceFlowUnitFactory::FillFlowUnitType(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc) {
  flowunit_desc->SetFlowType(modelbox::NORMAL);
  flowunit_desc->SetOutputType(modelbox::ORIGIN);
}

std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
VirtualInferenceFlowUnitFactory::FlowUnitProbe() {
  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> return_map;
  auto driver_desc = GetDriver()->GetDriverDesc();
  auto toml_file = driver_desc->GetFilePath();

  std::shared_ptr<VirtualInferenceFlowUnitDesc> flowunit_desc =
      std::make_shared<VirtualInferenceFlowUnitDesc>();
  modelbox::Status status;

  std::shared_ptr<modelbox::ConfigurationBuilder> builder =
      std::make_shared<modelbox::ConfigurationBuilder>();
  std::shared_ptr<modelbox::Configuration> config = builder->Build(toml_file);

  std::string device;
  auto ret = FillBaseInfo(config, flowunit_desc, toml_file, &device);
  if (ret != modelbox::STATUS_OK) {
    return return_map;
  }

  ret = FillItem(config, flowunit_desc, device, "input");
  if (ret != modelbox::STATUS_OK) {
    return return_map;
  }

  ret = FillItem(config, flowunit_desc, device, "output");
  if (ret != modelbox::STATUS_OK) {
    return return_map;
  }

  FillFlowUnitType(config, flowunit_desc);
  flowunit_desc->SetFlowUnitName(driver_desc->GetName());
  flowunit_desc->SetConfiguration(config);
  flowunit_desc->SetDescription(driver_desc->GetDescription());
  return_map.insert(std::make_pair(driver_desc->GetName(), flowunit_desc));
  return return_map;
}

void VirtualInferenceFlowUnitFactory::SetFlowUnitFactory(
    const std::vector<std::shared_ptr<modelbox::DriverFactory>>
        &bind_flowunit_factory_list) {
  for (const auto &bind_flowunit_factory : bind_flowunit_factory_list) {
    bind_flowunit_factory_list_.push_back(
        std::dynamic_pointer_cast<FlowUnitFactory>(bind_flowunit_factory));
  }
}

std::shared_ptr<modelbox::FlowUnit>
VirtualInferenceFlowUnitFactory::VirtualCreateFlowUnit(
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
  modelbox::StatusError = {
      modelbox::STATUS_NOTFOUND,
      "current environment does not support the inference type: '" +
          virtual_type + ":" + unit_type + "'"};

  return nullptr;
};

std::string VirtualInferenceFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}

void VirtualInferenceFlowUnitDesc::SetConfiguration(
    const std::shared_ptr<modelbox::Configuration> &config) {
  config_ = config;
}

std::shared_ptr<modelbox::Configuration>
VirtualInferenceFlowUnitDesc::GetConfiguration() {
  return config_;
}

std::string VirtualInferenceFlowUnitFactory::GetVirtualType() {
  return virtual_type_;
};

void VirtualInferenceFlowUnitFactory::SetVirtualType(
    const std::string &virtual_type) {
  virtual_type_ = virtual_type;
};
