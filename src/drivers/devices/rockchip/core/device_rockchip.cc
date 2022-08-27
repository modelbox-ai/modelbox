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

#include "modelbox/device/rockchip/device_rockchip.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/kernel.h>
#include <linux/unistd.h>
#include <stdio.h>
#include <sys/sysinfo.h>

#include <fstream>
#include <iostream>

#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "rknn_api.h"

const std::string LIB_RKNN_API_PATH = "librknn_api.so";

namespace modelbox {

RockChip::RockChip(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

std::string RockChip::GetType() const { return DEVICE_TYPE; }

Status RockChip::DeviceExecute(const DevExecuteCallBack &rkfun,
                               int32_t priority, size_t rkcount) {
  if (0 == rkcount) {
    return STATUS_OK;
  }

  for (size_t i = 0; i < rkcount; ++i) {
    auto status = rkfun(i);
    if (!status) {
      MBLOG_WARN << "executor rkfunc failed: " << status
                 << " stack trace:" << GetStackTrace();
      return status;
    }
  }

  return STATUS_OK;
};

bool RockChip::NeedResourceNice() { return true; }

std::map<std::string, std::shared_ptr<DeviceDesc>>
RockChipFactory::ProbeRKNNDevice() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> device_desc_map;

  void *handler = dlopen(LIB_RKNN_API_PATH.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handler == nullptr) {
    MBLOG_ERROR << "dlopen " << LIB_RKNN_API_PATH << " failed.";
    return device_desc_map;
  }

  Defer { dlclose(handler); };

  std::shared_ptr<rknn_devices_id> dev_ids =
      std::make_shared<rknn_devices_id>();
  if (dev_ids == nullptr) {
    MBLOG_ERROR << "make dev ids fail";
    return device_desc_map;
  }

  typedef int (*find_device_func)(rknn_devices_id *);
  auto find_device =
      reinterpret_cast<find_device_func>(dlsym(handler, "rknn_find_devices"));

  if (find_device(dev_ids.get()) != RKNN_SUCC || dev_ids->n_devices == 0) {
    MBLOG_ERROR << "find none rknn device";
    return device_desc_map;
  }

  std::vector<std::string> rknn_devs;
  for (size_t i = 0; i < dev_ids->n_devices; i++) {
    rknn_devs.emplace_back(std::string(dev_ids->ids[i]));
  }

  struct sysinfo s_info;
  auto ret = sysinfo(&s_info);
  if (ret != 0) {
    MBLOG_ERROR << "failed to sysinfo ret = " << ret;
  }

  for (size_t i = 0; i < rknn_devs.size(); i++) {
    auto device_desc = std::make_shared<RockChipDesc>();
    device_desc->SetDeviceDesc("This is a rockchip device description.");
    // inference module will bind all rockchip device to one
    auto id_str = std::to_string(i);
    device_desc->SetDeviceId(id_str);
    device_desc->SetDeviceMemory(GetBytesReadable(s_info.totalram));
    device_desc->SetDeviceType(DEVICE_TYPE);
    device_desc_map.insert(std::make_pair(id_str, device_desc));
  }

  RKNNDevs::Instance().SetNames(rknn_devs);

  return device_desc_map;
}

std::map<std::string, std::shared_ptr<DeviceDesc>>
RockChipFactory::DeviceProbe() {
  RKNNDevs::Instance().UpdateDeviceType();

  std::map<std::string, std::shared_ptr<DeviceDesc>> device_desc_map =
      ProbeRKNNDevice();
  auto deviceType = RKNNDevs::Instance().GetDeviceType();
  if (deviceType == RKNNDevs::RKNN_DEVICE_TYPE_RK356X ||
      deviceType == RKNNDevs::RKNN_DEVICE_TYPE_RK358X ||
      deviceType == RKNNDevs::RKNN_DEVICE_TYPE_RV110X) {
    MBLOG_INFO << "find rknpu2 type inference.";
    struct sysinfo s_info;
    auto ret = sysinfo(&s_info);
    if (ret != 0) {
      MBLOG_ERROR << "failed to sysinfo ret = " << ret;
    }

    auto device_desc = std::make_shared<RockChipDesc>();
    device_desc->SetDeviceDesc("This is a rknpu2 device description.");
    auto id_str = std::to_string(device_desc_map.size());
    device_desc->SetDeviceId(id_str);
    device_desc->SetDeviceMemory(GetBytesReadable(s_info.totalram));
    device_desc->SetDeviceType(DEVICE_TYPE);
    device_desc_map.insert(std::make_pair(id_str, device_desc));
  }

  return device_desc_map;
}

std::string RockChipFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::shared_ptr<Device> RockChipFactory::CreateDevice(
    const std::string &device_id) {
  auto mem_mgr = std::make_shared<RockChipMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }

  return std::make_shared<RockChip>(mem_mgr);
}

void RKNNDevs::SetNames(std::vector<std::string> &dev_names) {
  dev_names_.swap(dev_names);
}

const std::vector<std::string> &RKNNDevs::GetNames() { return dev_names_; }

void RKNNDevs::UpdateDeviceType() {
  type_ = RKNN_DEVICE_TYPE_OTHERS;
  std::ifstream dev_file("/proc/device-tree/compatible", std::ios::in);
  if (!dev_file.is_open()) {
    MBLOG_ERROR << "failed to open device file";
    return;
  }

  std::string strLine;

  std::unordered_map<std::string, RKNNDevs::RKNN_DEVICE_TYPE> deviceDictionary =
      {{"rk3399pro", RKNN_DEVICE_TYPE_RK3399PRO},
       {"rk356", RKNN_DEVICE_TYPE_RK356X},
       {"rk358", RKNN_DEVICE_TYPE_RK358X},
       {"rv110", RKNN_DEVICE_TYPE_RV110X}};

  Defer { dev_file.close(); };

  while (getline(dev_file, strLine)) {
    for (const auto &item : deviceDictionary) {
      if (strLine.find(item.first) != std::string::npos) {
        type_ = item.second;
        return;
      }
    }

    MBLOG_ERROR << strLine << " type not support";
    break;
  }
}

RKNNDevs::RKNN_DEVICE_TYPE RKNNDevs::GetDeviceType() { return type_; }

}  // namespace modelbox
