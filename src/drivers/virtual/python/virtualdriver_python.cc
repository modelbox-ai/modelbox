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

#include "virtualdriver_python.h"

#include <libgen.h>

#include <utility>

constexpr const char *VIRTUAL_FLOWUNIT_TYPE = "python";

PythonVirtualDriverDesc::PythonVirtualDriverDesc() = default;

PythonVirtualDriverDesc::~PythonVirtualDriverDesc() = default;

std::shared_ptr<modelbox::DriverFactory> PythonVirtualDriver::CreateFactory() {
  auto factory = std::make_shared<VirtualPythonFlowUnitFactory>();
  auto real_driver_list = GetBindDriver();
  factory->SetDriver(shared_from_this());
  auto real_factory_list =
      std::vector<std::shared_ptr<modelbox::DriverFactory>>();
  for (auto &real_driver : real_driver_list) {
    auto real_factory = real_driver->CreateFactory();
    if (real_factory == nullptr) {
      auto driver_desc = real_driver->GetDriverDesc();
      MBLOG_ERROR << "real driver binded by virtual python driver create "
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

PythonVirtualDriver::PythonVirtualDriver() = default;

PythonVirtualDriver::~PythonVirtualDriver() = default;

std::vector<std::shared_ptr<modelbox::Driver>>
PythonVirtualDriver::GetBindDriver() {
  return python_flowunit_driver_;
}

void PythonVirtualDriver::SetBindDriver(
    const std::vector<std::shared_ptr<modelbox::Driver>> &driver_list) {
  python_flowunit_driver_ = driver_list;
}

PythonVirtualDriverManager::PythonVirtualDriverManager() = default;

PythonVirtualDriverManager::~PythonVirtualDriverManager() = default;

modelbox::Status PythonVirtualDriverManager::Init(modelbox::Drivers &driver) {
  auto ret = BindBaseDriver(driver);
  return ret;
}

modelbox::Status PythonVirtualDriverManager::Scan(const std::string &path) {
  std::vector<std::string> drivers_list;
  std::string filter = "*.toml";
  auto status = modelbox::ListSubDirectoryFiles(path, filter, &drivers_list);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "list directory:  " + path + "/" + filter + " failed.";
    return status;
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

modelbox::Status PythonVirtualDriverManager::Add(const std::string &file) {
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

  std::shared_ptr<PythonVirtualDriver> driver =
      std::make_shared<PythonVirtualDriver>();
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
  driver->SetBindDriver(python_flowunit_driver_list_);
  // TODO: 判断是否重复存在
  drivers_list_.push_back(driver);
  return modelbox::STATUS_OK;
}

modelbox::Status PythonVirtualDriverManager::BindBaseDriver(
    modelbox::Drivers &driver) {
  for (const auto &bind_type : BIND_PYTHON_FLOWUNIT_TYPE) {
    auto tmp_driver = driver.GetDriver(modelbox::DRIVER_CLASS_FLOWUNIT,
                                       bind_type, BIND_PYTHON_FLOWUNIT_NAME,
                                       BIND_PYTHON_FLOWUNIT_VERSION);
    if (tmp_driver == nullptr) {
      continue;
    }

    python_flowunit_driver_list_.push_back(tmp_driver);
  }

  if (python_flowunit_driver_list_.empty()) {
    return {modelbox::STATUS_NOTFOUND, "can not find python flowunit"};
  }

  return modelbox::STATUS_OK;
}

VirtualPythonFlowUnitFactory::VirtualPythonFlowUnitFactory() = default;
VirtualPythonFlowUnitFactory::~VirtualPythonFlowUnitFactory() = default;

std::shared_ptr<modelbox::Driver> VirtualPythonFlowUnitFactory::GetDriver() {
  return driver_;
}

void VirtualPythonFlowUnitFactory::SetDriver(
    const std::shared_ptr<modelbox::Driver> &driver) {
  driver_ = driver;
}

modelbox::Status VirtualPythonFlowUnitFactory::FillInput(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
    const std::string &device) {
  auto input = config->GetSubKeys("input");
  if (input.empty()) {
    return modelbox::STATUS_NOTFOUND;
  }

  for (unsigned int i = 1; i <= input.size(); ++i) {
    std::string input_device;
    std::string input_name;
    std::string input_type;
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

    auto device_index = key + ".device";
    input_device = config->GetString(device_index);
    if (input_device.empty()) {
      input_device = device;
    }

    flowunit_desc->AddFlowUnitInput(
        modelbox::FlowUnitInput(input_name, input_device, input_type));
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VirtualPythonFlowUnitFactory::FillOutput(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
    const std::string &device) {
  auto output = config->GetSubKeys("output");
  if (output.empty()) {
    return modelbox::STATUS_NOTFOUND;
  }

  for (unsigned int i = 1; i <= output.size(); ++i) {
    std::string output_device;
    std::string output_name;
    std::string output_type;
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

    auto device_index = key + ".device";
    output_device = config->GetString(device_index);
    if (output_device.empty()) {
      output_device = device;
    }

    flowunit_desc->AddFlowUnitOutput(
        modelbox::FlowUnitOutput(output_name, output_device, output_type));
  }
  return modelbox::STATUS_OK;
}

modelbox::Status VirtualPythonFlowUnitFactory::FillBaseInfo(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
    const std::string &toml_file, std::string *device) {
  auto python_entry = config->GetString("base.entry");
  if (python_entry.empty()) {
    MBLOG_ERROR << "the key 'entry' is not found under base.";
    return modelbox::STATUS_BADCONF;
  }

  flowunit_desc->SetPythonEntry(python_entry);

  *device = config->GetString("base.device");
  if (device->empty()) {
    MBLOG_ERROR << "the key 'device' is not found under base.";
    return modelbox::STATUS_BADCONF;
  }

  auto group_type = config->GetString("base.group_type");
  if (group_type.empty()) {
    MBLOG_WARN << "the key group type is empty, so classify it into Undefined.";
  }
  flowunit_desc->SetFlowUnitGroupType(group_type);

  return modelbox::STATUS_OK;
}

void VirtualPythonFlowUnitFactory::FillFlowUnitType(
    std::shared_ptr<modelbox::Configuration> &config,
    std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc) {
  auto config_op = config->GetSubKeys("config");
  if (!config_op.empty()) {
    flowunit_desc->SetConfiguration(config->GetSubConfig("config"));
  }

  auto is_stream = config->GetBool("base.stream", true);
  if (is_stream) {
    flowunit_desc->SetFlowType(modelbox::STREAM);
  } else {
    flowunit_desc->SetFlowType(modelbox::NORMAL);
  }

  auto is_condition = config->GetBool("base.condition", false);
  if (is_condition) {
    flowunit_desc->SetConditionType(modelbox::IF_ELSE);
  } else {
    flowunit_desc->SetConditionType(modelbox::NONE);
  }

  flowunit_desc->SetOutputType(modelbox::ORIGIN);

  auto is_collapse = config->GetBool("base.collapse", false);
  if (is_collapse) {
    flowunit_desc->SetOutputType(modelbox::COLLAPSE);
    auto is_collapse_all = config->GetBool("base.collapse_all", true);
    flowunit_desc->SetCollapseAll(is_collapse_all);
  }

  auto is_expand = config->GetBool("base.expand", false);
  if (is_expand) {
    flowunit_desc->SetOutputType(modelbox::EXPAND);
  }

  auto is_same_count = config->GetBool("base.stream_same_count", false);
  if (is_same_count) {
    flowunit_desc->SetStreamSameCount(true);
  } else {
    flowunit_desc->SetStreamSameCount(false);
  }

  auto exception_visible = config->GetBool("base.exception_visible", false);
  if (exception_visible) {
    flowunit_desc->SetExceptionVisible(true);
  } else {
    flowunit_desc->SetExceptionVisible(false);
  }

  if (config->Contain("base.max_batch_size")) {
    auto max_batch_size = config->GetInt32("base.max_batch_size", 1);
    flowunit_desc->SetMaxBatchSize(max_batch_size);
  }

  if (config->Contain("base.default_batch_size")) {
    auto default_batch_size = config->GetInt32("base.default_batch_size", 1);
    flowunit_desc->SetDefaultBatchSize(default_batch_size);
  }
}

std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
VirtualPythonFlowUnitFactory::FlowUnitProbe() {
  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> return_map;
  auto driver_desc = GetDriver()->GetDriverDesc();
  auto toml_file = driver_desc->GetFilePath();
  std::shared_ptr<VirtualPythonFlowUnitDesc> flowunit_desc =
      std::make_shared<VirtualPythonFlowUnitDesc>();
  modelbox::Status status;

  std::shared_ptr<modelbox::ConfigurationBuilder> builder =
      std::make_shared<modelbox::ConfigurationBuilder>();
  std::shared_ptr<modelbox::Configuration> config = builder->Build(toml_file);

  std::string device;
  auto ret = FillBaseInfo(config, flowunit_desc, toml_file, &device);
  if (ret != modelbox::STATUS_OK) {
    return return_map;
  }

  auto input_ret = FillInput(config, flowunit_desc, device);
  if (input_ret == modelbox::STATUS_BADCONF) {
    return return_map;
  }

  auto output_ret = FillOutput(config, flowunit_desc, device);
  if (output_ret == modelbox::STATUS_BADCONF) {
    return return_map;
  }

  if (output_ret == modelbox::STATUS_NOTFOUND &&
      input_ret == modelbox::STATUS_NOTFOUND) {
    MBLOG_ERROR
        << "neither the key 'input' nor 'output' is not found in config file.";
    return return_map;
  }

  FillFlowUnitType(config, flowunit_desc);

  const auto &tom_file_path = driver_desc->GetFilePath();
  auto dir_name = modelbox::GetDirName(tom_file_path);
  flowunit_desc->SetPythonFilePath(std::string(dir_name));

  flowunit_desc->SetFlowUnitName(driver_desc->GetName());
  return_map.insert(std::make_pair(driver_desc->GetName(), flowunit_desc));
  return return_map;
}

void VirtualPythonFlowUnitFactory::SetFlowUnitFactory(
    const std::vector<std::shared_ptr<modelbox::DriverFactory>>
        &bind_flowunit_factory_list) {
  for (const auto &bind_flowunit_factory : bind_flowunit_factory_list) {
    bind_flowunit_factory_list_.push_back(
        std::dynamic_pointer_cast<FlowUnitFactory>(bind_flowunit_factory));
  }
}

std::shared_ptr<modelbox::FlowUnit>
VirtualPythonFlowUnitFactory::CreateFlowUnit(const std::string &unit_name,
                                             const std::string &unit_type) {
  for (auto &flowunit_factory : bind_flowunit_factory_list_) {
    if (std::dynamic_pointer_cast<FlowUnitFactory>(flowunit_factory)
            ->GetFlowUnitFactoryType() != unit_type) {
      continue;
    }
    return std::dynamic_pointer_cast<FlowUnitFactory>(flowunit_factory)
        ->CreateFlowUnit(unit_name, unit_type);
  }
  modelbox::StatusError = {modelbox::STATUS_NOTFOUND,
                           "not found flowunit " + std::string(unit_name)};
  return nullptr;
}

VirtualPythonFlowUnitDesc::VirtualPythonFlowUnitDesc() = default;

VirtualPythonFlowUnitDesc::~VirtualPythonFlowUnitDesc() = default;

void SetConfiguration(const std::shared_ptr<modelbox::Configuration> &config);
std::shared_ptr<modelbox::Configuration> GetConfiguration();

void VirtualPythonFlowUnitDesc::SetPythonFilePath(const std::string &path) {
  python_file_path_ = path;
}
const std::string &VirtualPythonFlowUnitDesc::GetPythonFilePath() const {
  return python_file_path_;
}

void VirtualPythonFlowUnitDesc::SetPythonEntry(std::string python_entry) {
  python_entry_ = std::move(python_entry);
}

std::string VirtualPythonFlowUnitDesc::GetPythonEntry() {
  return python_entry_;
}

void VirtualPythonFlowUnitDesc::SetConfiguration(
    const std::shared_ptr<modelbox::Configuration> &config) {
  config_ = config;
}

std::shared_ptr<modelbox::Configuration>
VirtualPythonFlowUnitDesc::GetConfiguration() {
  return config_;
}