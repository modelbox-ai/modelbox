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


#include "modelbox/device/cpu/device_cpu.h"

#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/base/utils.h"

namespace modelbox {
CPU::CPU(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

CPU::~CPU() = default;

std::string CPU::GetType() const { return DEVICE_TYPE; }

Status CPU::DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                          size_t count) {
  if (0 == count) {
    return STATUS_OK;
  }

  std::vector<std::future<Status>> future_list(count - 1);
  for (size_t i = 0; i < count - 1; ++i) {
    auto future_status = executor_->Run(fun, priority, i);
    future_list[i] = std::move(future_status);
  }

  auto status = fun(count - 1);
  std::vector<Status> future_status(count, STATUS_OK);
  future_status[count - 1] = status;
  for (size_t i = 0; i < future_list.size(); ++i) {
    future_list[i].wait();
    future_status[i] = future_list[i].get();
  }

  auto ret = STATUS_OK;
  for (const auto &status : future_status) {
    if (!status) {
      return status;
    }
  }

  return STATUS_OK;
};

CPUFactory::CPUFactory() = default;
CPUFactory::~CPUFactory() = default;

std::map<std::string, std::shared_ptr<DeviceDesc>> CPUFactory::DeviceProbe() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> return_map;
  size_t free;
  size_t total;
  std::shared_ptr<CPUDesc> device_desc = std::make_shared<CPUDesc>();
  device_desc->SetDeviceDesc("Host cpu device.");
  device_desc->SetDeviceId("0");
  os->GetMemoryUsage(&free, &total);
  device_desc->SetDeviceMemory(GetBytesReadable(total));
  device_desc->SetDeviceType("CPU");
  return_map.insert(std::make_pair("0", device_desc));
  return return_map;
}

std::string CPUFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::vector<std::string> CPUFactory::GetDeviceList() {
  std::vector<std::string> cpuIds;
  cpuIds.emplace_back("0");
  return cpuIds;
}

std::shared_ptr<Device> CPUFactory::CreateDevice(const std::string &device_id) {
  auto mem_mgr = std::make_shared<CpuMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }
  return std::make_shared<CPU>(mem_mgr);
}

CPUDesc::CPUDesc() = default;

CPUDesc::~CPUDesc() = default;

}  // namespace modelbox
