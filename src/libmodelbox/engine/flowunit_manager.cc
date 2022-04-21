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

#include <algorithm>

#include "modelbox/base/log.h"
#include "modelbox/flowunit.h"

namespace modelbox {

FlowUnitManager::FlowUnitManager(){};
FlowUnitManager::~FlowUnitManager(){};

std::shared_ptr<FlowUnitManager> FlowUnitManager::GetInstance() {
  static std::shared_ptr<FlowUnitManager> flowunit_mgr =
      std::make_shared<FlowUnitManager>();
  return flowunit_mgr;
}

std::shared_ptr<FlowUnitDesc> FlowUnitManager::GetFlowUnitDesc(
    const std::string &flowunit_type, const std::string &flowunit_name) {
  auto iter_device_type = flowunit_desc_list_.find(flowunit_type);
  if (iter_device_type == flowunit_desc_list_.end()) {
    MBLOG_ERROR << "do not find device_type " << flowunit_type
                << " in the flowunit desc map, please check the device type.";
    return nullptr;
  }

  auto iter_flowunit_name =
      flowunit_desc_list_[flowunit_type].find(flowunit_name);
  if (iter_flowunit_name == flowunit_desc_list_[flowunit_type].end()) {
    MBLOG_ERROR << "do not find flowunit name " << flowunit_name
                << " in device type " << flowunit_type
                << " in the flowunit desc map, please check the device name.";
    return nullptr;
  }

  return flowunit_desc_list_[flowunit_type][flowunit_name];
}

Status FlowUnitManager::Initialize(std::shared_ptr<Drivers> driver,
                                   std::shared_ptr<DeviceManager> device_mgr,
                                   std::shared_ptr<Configuration> config) {
  SetDeviceManager(device_mgr);
  Status status;
  status = InitFlowUnitFactory(driver);
  if (status != STATUS_SUCCESS) {
    return status;
  }

  status = FlowUnitProbe();
  if (status != STATUS_SUCCESS) {
    return status;
  }

  status = SetUpFlowUnitDesc();
  if (status != STATUS_SUCCESS) {
    return status;
  }

  return status;
}

Status FlowUnitManager::InitFlowUnitFactory(std::shared_ptr<Drivers> driver) {
  std::vector<std::shared_ptr<Driver>> driver_list =
      driver->GetDriverListByClass("DRIVER-FLOWUNIT");
  std::vector<std::shared_ptr<Driver>> inference_driver_list =
      driver->GetDriverListByClass("DRIVER-INFERENCE");
  for (auto &infer_driver : inference_driver_list) {
    driver_list.emplace_back(infer_driver);
  }

  std::shared_ptr<DriverDesc> desc;
  for (auto &flowunit_driver : driver_list) {
    auto temp_factory = flowunit_driver->CreateFactory();
    if (nullptr == temp_factory) {
      continue;
    }
    desc = flowunit_driver->GetDriverDesc();
    std::shared_ptr<FlowUnitFactory> flowunit_factory =
        std::dynamic_pointer_cast<FlowUnitFactory>(temp_factory);

    flowunit_factory->SetDriver(flowunit_driver);

    auto names = flowunit_factory->GetFlowUnitNames();
    if (names.empty()) {
      flowunit_factory_.insert(std::make_pair(
          std::make_pair(desc->GetType(), desc->GetName()), flowunit_factory));
    } else {
      for (const auto &name : names) {
        flowunit_factory_.insert(std::make_pair(
            std::make_pair(desc->GetType(), name), flowunit_factory));
      }
    }
  }

  return STATUS_OK;
}

Status FlowUnitManager::FlowUnitProbe() {
  for (auto &iter : flowunit_factory_) {
    auto tmp = iter.second->FlowUnitProbe();
    if (!tmp.size()) {
      continue;
    }

    auto value = flowunit_desc_list_.find(iter.first.first);
    if (value == flowunit_desc_list_.end()) {
      flowunit_desc_list_.insert(std::make_pair(iter.first.first, tmp));
    } else {
      for (const auto &item : tmp)
        value->second.insert(std::make_pair(item.first, item.second));
    }

    for (const auto &iter_flow : tmp) {
      auto flowunit_desc = iter_flow.second;
      MBLOG_DEBUG << "add flowunit:";
      MBLOG_DEBUG << "  name: " << flowunit_desc->GetFlowUnitName();
      MBLOG_DEBUG << "  type: " << iter.first.first;
    }
  }

  return STATUS_OK;
}

Status FlowUnitManager::Register(std::shared_ptr<FlowUnitFactory> factory) {
  std::string factory_type = factory->GetFlowUnitFactoryType();
  std::string factory_unit_name = factory->GetFlowUnitFactoryName();
  if (flowunit_factory_.count(
          std::make_pair(factory_type, factory_unit_name))) {
    MBLOG_WARN << "The type " << factory_type << " has already existed.";
    return Status(STATUS_EXIST);
  }

  flowunit_factory_.insert(
      std::make_pair(std::make_pair(factory_type, factory_unit_name), factory));
  return STATUS_OK;
}

std::vector<std::string> FlowUnitManager::GetFlowUnitTypes() {
  std::vector<std::string> flowunit_type;
  std::set<std::string> tmp_set;
  for (auto &iter : flowunit_factory_) {
    tmp_set.insert(iter.first.first);
  }
  std::copy(tmp_set.begin(), tmp_set.end(), std::back_inserter(flowunit_type));

  return flowunit_type;
}

std::vector<std::string> FlowUnitManager::GetFlowUnitTypes(
    const std::string &unit_name) {
  std::vector<std::string> unit_types;
  for (auto &iter : flowunit_desc_list_) {
    auto &dev_type = iter.first;
    auto &units = iter.second;
    auto unit_item = units.find(unit_name);
    if (unit_item == units.end()) {
      continue;
    }

    unit_types.push_back(dev_type);
  }

  return unit_types;
}

std::vector<std::string> FlowUnitManager::GetFlowUnitList(
    const std::string &unit_type) {
  std::vector<std::string> flowunit_name;
  auto iter = flowunit_desc_list_.find(unit_type);
  if (iter == flowunit_desc_list_.end()) {
    return std::vector<std::string>();
  }

  for (auto &name : flowunit_desc_list_[unit_type]) {
    flowunit_name.push_back(name.first);
  }

  return flowunit_name;
}

modelbox::Status FlowUnitManager::CheckParams(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &unit_device_id) {
  if (unit_name.empty()) {
    MBLOG_WARN << "FlowUnit name should not be empty.";
    return modelbox::STATUS_BADCONF;
  }

  if (unit_type.empty() && !unit_device_id.empty()) {
    MBLOG_WARN << "Empty flowUnit type and not empty flowunit device id are "
                  "not allowed.";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status FlowUnitManager::ParseUnitDeviceConf(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &unit_device_id, FlowUnitDeviceConfig &dev_cfg) {
  auto ret = ParseUserDeviceConf(unit_type, unit_device_id, dev_cfg);
  if (!ret) {
    return ret;
  }

  ret = AutoFillDeviceConf(unit_name, dev_cfg);
  if (!ret) {
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status FlowUnitManager::ParseUserDeviceConf(
    const std::string &unit_type, const std::string &unit_device_id,
    FlowUnitDeviceConfig &dev_cfg) {
  /**
   * user format: unit_type = "cuda:0,1;cpu"
   * what we get from configuration will be: unit_type = "cuda:0~1;cpu"
   **/
  auto unit_type_formatted = unit_type;
  std::replace(unit_type_formatted.begin(), unit_type_formatted.end(), '~',
               ',');
  auto device_list = modelbox::StringSplit(unit_type_formatted, ';');
  for (auto &device_info : device_list) {
    auto data = modelbox::StringSplit(device_info, ':');
    if (data.empty() || data.size() > 2) {
      return {modelbox::STATUS_BADCONF,
              "device info " + unit_type + " format error"};
    }

    auto &device_type = data[0];
    auto &single_dev_cfg = dev_cfg[device_type];
    if (data.size() == 1) {
      continue;
    }

    auto &ids = data[1];
    auto id_list = modelbox::StringSplit(ids, ',');
    for (size_t id_index = 0; id_index < id_list.size(); ++id_index) {
      single_dev_cfg.push_back(id_list[id_index]);
    }
  }

  if (unit_device_id.empty()) {
    return modelbox::STATUS_OK;
  }

  // For compatibility, check old config: device="cpu", deviceid="0"
  if (dev_cfg.size() != 1) {
    return {modelbox::STATUS_BADCONF,
            "should not set deviceid param when use multi device"};
  }

  for (auto &cfg_item : dev_cfg) {
    auto &ids = cfg_item.second;
    if (!ids.empty()) {
      return {modelbox::STATUS_BADCONF,
              "should not set deviceid param when use [device:id] format"};
    }

    ids.push_back(unit_device_id);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status FlowUnitManager::AutoFillDeviceConf(
    const std::string &unit_name, FlowUnitDeviceConfig &dev_cfg) {
  if (dev_cfg.empty()) {
    // will auto fill all device type if no device selected
    auto unit_types = GetFlowUnitTypes(unit_name);
    for (auto type : unit_types) {
      dev_cfg[type];  // create empty list
    }
  }

  for (auto &cfg_item : dev_cfg) {
    auto &dev_type = cfg_item.first;
    auto &ids = cfg_item.second;
    if (ids.empty()) {
      // will auto fill all id if no id selected
      auto real_ids = device_mgr_->GetDevicesIdList(dev_type);
      ids.assign(real_ids.begin(), real_ids.end());
    }
  }

  return modelbox::STATUS_OK;
}

std::vector<std::shared_ptr<FlowUnit>> FlowUnitManager::CreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &unit_device_id) {
  std::vector<std::shared_ptr<FlowUnit>> flowunit_list;

  StatusError = {STATUS_NOTFOUND};

  auto ret = CheckParams(unit_name, unit_type, unit_device_id);
  if (ret != modelbox::STATUS_OK) {
    return flowunit_list;
  }

  FlowUnitDeviceConfig unit_dev_cfg;
  ret = ParseUnitDeviceConf(unit_name, unit_type, unit_device_id, unit_dev_cfg);
  if (!ret) {
    MBLOG_ERROR << "Parse unit device config failed, err " << ret;
    return flowunit_list;
  }

  for (auto &cfg_item : unit_dev_cfg) {
    auto &dev_type = cfg_item.first;
    auto &ids = cfg_item.second;
    if (ids.empty()) {
      MBLOG_WARN << "CreateFlowUnit: " << unit_name << "," << dev_type
                 << " failed, No available device for type " << dev_type;
      continue;
    }

    for (auto &id : ids) {
      auto flowunit = CreateSingleFlowUnit(unit_name, dev_type, id);
      if (flowunit == nullptr) {
        MBLOG_WARN << "CreateFlowUnit: " << unit_name << " failed, "
                   << StatusError;
        continue;
      }

      flowunit_list.push_back(flowunit);
    }
  }

  return flowunit_list;
}

std::shared_ptr<FlowUnit> FlowUnitManager::CreateSingleFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &unit_device_id) {
  if (unit_device_id.empty()) {
    StatusError = {STATUS_INVALID, "FlowUnit device id is none."};
    MBLOG_WARN << StatusError.Errormsg();
    return nullptr;
  }

  if (unit_type.empty()) {
    StatusError = {STATUS_INVALID, "FlowUnit device type is none."};
    MBLOG_WARN << StatusError.Errormsg();
    return nullptr;
  }

  std::shared_ptr<FlowUnit> flowunit;
  std::shared_ptr<Device> device;
  std::shared_ptr<modelbox::DeviceManager> device_mgr = GetDeviceManager();

  auto iter = flowunit_factory_.find(std::make_pair(unit_type, unit_name));
  if (iter == flowunit_factory_.end()) {
    StatusError = {STATUS_NOTFOUND, "can not find flowunit[type: " + unit_type +
                                        ", name:" + unit_name +
                                        "], may not loaded."};
    return nullptr;
  }

  auto &factory = iter->second;
  const auto &flowunit_desc_map = factory->FlowUnitProbe();
  auto item = flowunit_desc_map.find(unit_name);
  if (item == flowunit_desc_map.end()) {
    StatusError = {STATUS_FAULT,
                   "flowunit probe for unit " + unit_name + " failed."};
    return nullptr;
  }

  auto flowunit_desc = item->second;
  auto driver_desc = factory->GetDriver()->GetDriverDesc();
  flowunit_desc->SetDriverDesc(driver_desc);
  factory->SetVirtualType(flowunit_desc->GetVirtualType());
  flowunit = factory->CreateFlowUnit(unit_name, unit_type);
  if (flowunit == nullptr) {
    return nullptr;
  }

  device = device_mgr->CreateDevice(unit_type, unit_device_id);
  if (device == nullptr) {
    return nullptr;
  }

  flowunit->SetBindDevice(device);
  std::vector<FlowUnitInput> &in_list = flowunit_desc->GetFlowUnitInput();
  for (auto &in_item : in_list) {
    const auto &device_type = in_item.GetDeviceType();
    if (device_type.empty() || device_type == device->GetType()) {
      in_item.SetDevice(device);
      continue;
    }

    const auto &dev_id_list = device_mgr->GetDevicesIdList(device_type);
    // TODO if this device type has not device, what should we do?
    if (dev_id_list.empty()) {
      in_item.SetDevice(device);
      continue;
    }
    // TODO what device id select?
    auto in_device = device_mgr->CreateDevice(device_type, dev_id_list[0]);
    if (in_device == nullptr) {
      in_device = device;
    }

    in_item.SetDevice(in_device);
  }
  flowunit->SetFlowUnitDesc(flowunit_desc);

  return flowunit;
}

void FlowUnitManager::Clear() {
  flowunit_desc_list_.clear();
  flowunit_factory_.clear();
}

std::map<std::pair<std::string, std::string>, std::shared_ptr<FlowUnitFactory>>
FlowUnitManager::GetFlowUnitFactoryList() {
  return flowunit_factory_;
}

std::map<std::string, std::map<std::string, std::shared_ptr<FlowUnitDesc>>>
FlowUnitManager::GetFlowUnitDescList() {
  return flowunit_desc_list_;
}

void FlowUnitManager::InsertFlowUnitFactory(
    const std::string &name, const std::string &type,
    const std::shared_ptr<FlowUnitFactory> &flowunit_factory) {
  flowunit_factory_.insert(
      std::make_pair(std::make_pair(type, name), flowunit_factory));
}

Status FlowUnitManager::SetUpFlowUnitDesc() {
  for (auto &iter_device : flowunit_desc_list_) {
    for (auto &iter_name : flowunit_desc_list_[iter_device.first]) {
      auto pair = std::make_pair(iter_device.first, iter_name.first);
      auto iter_driver_desc = flowunit_factory_.find(pair);
      if (iter_driver_desc == flowunit_factory_.end()) {
        auto err_msg = "flowunit_factory find " + iter_device.first + ", " +
                       iter_name.first + " failed";
        MBLOG_ERROR << err_msg;
        return {STATUS_FAULT, err_msg};
      }

      auto driver = flowunit_factory_[pair]->GetDriver();
      auto driver_desc = driver->GetDriverDesc();
      flowunit_desc_list_[iter_device.first][iter_name.first]->SetDriverDesc(
          driver_desc);
    }
  }
  return STATUS_SUCCESS;
}

std::vector<std::shared_ptr<FlowUnitDesc>>
FlowUnitManager::GetAllFlowUnitDesc() {
  std::vector<std::shared_ptr<FlowUnitDesc>> desc_vec;
  for (auto &iter_device : flowunit_desc_list_) {
    for (auto &iter_name : flowunit_desc_list_[iter_device.first]) {
      desc_vec.push_back(
          flowunit_desc_list_[iter_device.first][iter_name.first]);
    }
  }

  return desc_vec;
}

void FlowUnitManager::SetDeviceManager(
    std::shared_ptr<DeviceManager> device_mgr) {
  device_mgr_ = device_mgr;
}

std::shared_ptr<DeviceManager> FlowUnitManager::GetDeviceManager() {
  return device_mgr_;
}

}  // namespace modelbox