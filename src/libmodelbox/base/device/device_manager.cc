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


#include <stdio.h>

#include <string>
#include <vector>

#include "modelbox/base/device.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"

namespace modelbox {
DeviceManager::DeviceManager() = default;
DeviceManager::~DeviceManager() = default;

Status DeviceManager::Register(std::shared_ptr<DeviceFactory> factory) {
  std::string factory_type = factory->GetDeviceFactoryType();
  if (device_factory_.count(factory_type)) {
    MBLOG_WARN << "The type " << factory_type << " has already existed.";
    return STATUS_EXIST;
  }

  device_factory_.insert(std::make_pair(factory_type, factory));
  // TODO: register device id
  return STATUS_OK;
}

std::shared_ptr<DeviceManager> DeviceManager::GetInstance() {
  static std::shared_ptr<DeviceManager> device_mgr =
      std::make_shared<DeviceManager>();
  return device_mgr;
}

void DeviceManager::Clear() {
  device_list_.clear();
  device_desc_list_.clear();
  device_factory_.clear();
}

Status DeviceManager::Initialize(std::shared_ptr<Drivers> driver,
                                 std::shared_ptr<Configuration> config) {
  if (driver == nullptr) {
    return STATUS_FAULT;
  }

  SetDrivers(driver);

  InitDeviceFactory(driver);
  Status status = DeviceProbe();

  return status;
}

Status DeviceManager::CheckDeviceManagerInit() {
  if (device_factory_.empty() || device_desc_list_.empty()) {
    MBLOG_ERROR << "Please init devicemanager first.";
    return STATUS_FAULT;
  }

  return STATUS_OK;
}

Status DeviceManager::InitDeviceFactory(std::shared_ptr<Drivers> driver) {
  std::vector<std::shared_ptr<Driver>> driver_list =
      driver->GetDriverListByClass("DRIVER-DEVICE");
  std::shared_ptr<DriverDesc> desc;
  for (auto &device_driver : driver_list) {
    auto temp_factory = device_driver->CreateFactory();
    if (nullptr == temp_factory) {
      continue;
    }
    desc = device_driver->GetDriverDesc();
    std::shared_ptr<DeviceFactory> device_factory =
        std::dynamic_pointer_cast<DeviceFactory>(temp_factory);

    device_factory_.insert(std::make_pair(desc->GetType(), device_factory));
  }
  return STATUS_OK;
}

std::vector<std::string> DeviceManager::GetDevicesTypes() {
  std::vector<std::string> device_type;

  for (auto &iter : device_factory_) {
    device_type.push_back(iter.first);
  }

  return device_type;
}

std::vector<std::string> DeviceManager::GetDevicesIdList(
    const std::string &device_type) {
  std::vector<std::string> device_id;
  auto iter = device_desc_list_.find(device_type);
  if (iter == device_desc_list_.end()) {
    return device_id;
  }

  for (auto &id : device_desc_list_[device_type]) {
    device_id.push_back(id.first);
  }

  return device_id;
}

std::shared_ptr<Device> DeviceManager::CreateDevice(
    const std::string &device_type, const std::string &device_id) {
  if (CheckDeviceManagerInit() != STATUS_OK) {
    StatusError = {STATUS_FAULT, "check device failed."};
    return nullptr;
  }

  if (device_type.empty() || device_id.empty()) {
    StatusError = {STATUS_INVALID, "device type or id is invalid."};
    MBLOG_ERROR << StatusError.Errormsg();
    return nullptr;
  }

  std::shared_ptr<Device> device;
  device = GetDevice(device_type, device_id);
  if (device != nullptr) {
    return device;
  }

  auto type_desc = device_desc_list_.find(device_type);
  if (type_desc == device_desc_list_.end()) {
    StatusError = {STATUS_NOTFOUND, "Can't support type:" + device_type};
    MBLOG_ERROR << StatusError.Errormsg();
    return nullptr;
  }

  auto id_desc = device_desc_list_[device_type].find(device_id);
  if (id_desc == device_desc_list_[device_type].end()) {
    StatusError = {STATUS_NOTFOUND,
                   "Can't find device, type " + device_type + " id: " + device_id};
    MBLOG_ERROR << StatusError.Errormsg();
    return nullptr;
  }
  
  auto iter = device_factory_.find(device_type);
  if (iter == device_factory_.end()) {
    StatusError = {STATUS_NOTFOUND, "device type not found: " + device_type};
    return nullptr;
  }

  device = device_factory_[device_type]->CreateDevice(device_id);
  if (device == nullptr) {
    return device;
  }

  device->SetDeviceManager(shared_from_this());
  auto ret = device->Init();
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Device init failed";
    return nullptr;
  }


  device->SetDeviceDesc(device_desc_list_[device_type][device_id]);
  auto &id_map = device_list_[device_type];
  id_map[device_id] = device;
  return device;
}

std::shared_ptr<Device> DeviceManager::GetDevice(const std::string &device_type,
                                                 const std::string &device_id) {
  auto iter = device_list_.find(device_type);
  if (iter == device_list_.end()) {
    StatusError = {STATUS_NOTFOUND, "cannot found device: " + device_type +
                                        ", id: " + device_id};
    return nullptr;
  }

  auto id = device_list_[device_type].find(device_id);
  if (id == device_list_[device_type].end()) {
    StatusError = {STATUS_NOTFOUND, "cannot found device: " + device_type +
                                        ", id: " + device_id};
    return nullptr;
  }

  return device_list_[device_type][device_id];
}

Status DeviceManager::DeviceProbe() {
  for (auto &iter : device_factory_) {
    auto tmp = iter.second->DeviceProbe();
    device_desc_list_.insert(std::make_pair(iter.first, tmp));
    for (auto const &itdev : tmp) {
      auto dev_desc = itdev.second;
      MBLOG_DEBUG << "add device:";
      MBLOG_DEBUG << "  type: " << dev_desc->GetDeviceType();
      MBLOG_DEBUG << "  id: " << dev_desc->GetDeviceId();
      MBLOG_DEBUG << "  memory: " << dev_desc->GetDeviceMemory();
      MBLOG_DEBUG << "  version: " << dev_desc->GetDeviceVersion();
      MBLOG_DEBUG << "  description: " << dev_desc->GetDeviceDesc();
    }
  }

  return STATUS_OK;
}

const std::map<std::string, std::shared_ptr<DeviceFactory>>
    &DeviceManager::GetDeviceFactoryList() {
  return device_factory_;
}

const std::map<std::string, std::map<std::string, std::shared_ptr<DeviceDesc>>>
    &DeviceManager::GetDeviceDescList() {
  return device_desc_list_;
}

const std::map<std::string, std::map<std::string, std::shared_ptr<Device>>>
    &DeviceManager::GetDeviceList() {
  return device_list_;
}

std::shared_ptr<Drivers> DeviceManager::GetDrivers() { return drivers_; }

void DeviceManager::SetDrivers(std::shared_ptr<Drivers> drivers) {
  drivers_ = drivers;
}

}  // namespace modelbox
