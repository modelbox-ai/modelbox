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


#ifndef MODELBOX_DEVICE_CPU_H_
#define MODELBOX_DEVICE_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>
#include "modelbox/device/cpu/cpu_memory.h"

namespace modelbox {

constexpr const char *DEVICE_TYPE = "cpu";
constexpr const char *DEVICE_DRIVER_NAME = "device-cpu";
constexpr const char *DEVICE_DRIVER_DESCRIPTION = "A cpu device driver";

class CPU : public Device {
 public:
  CPU(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  ~CPU() override;
  std::string GetType() const override;

  Status DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                       size_t count) override;
};

class CPUFactory : public DeviceFactory {
 public:
  CPUFactory();
  ~CPUFactory() override;

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe() override;
  std::string GetDeviceFactoryType() override;
  std::vector<std::string> GetDeviceList() override;
  std::shared_ptr<Device> CreateDevice(const std::string &device_id) override;
};

class CPUDesc : public DeviceDesc {
 public:
  CPUDesc();
  ~CPUDesc() override;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_CPU_H_
