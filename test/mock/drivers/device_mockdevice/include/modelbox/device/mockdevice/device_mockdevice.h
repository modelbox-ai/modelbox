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


#ifndef MODELBOX_DEVICE_MOCKDEVICE_H_
#define MODELBOX_DEVICE_MOCKDEVICE_H_

#include <modelbox/base/device.h>
#include <modelbox/device/cpu/device_cpu.h>
#include <modelbox/flow.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

namespace modelbox {

constexpr const char *MOCK_DEVICE_TYPE = "MOCKDEVICE";
constexpr const char *MOCK_DEVICE_DRIVER_NAME = "device-mockdevice";
constexpr const char *MOCK_DEVICE_DRIVER_DESCRIPTION =
    "A mockdevice device driver";

class FakeDeviceMemoryManager : public DeviceMemoryManager {
 public:
  FakeDeviceMemoryManager() : DeviceMemoryManager("0") {}

  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, void *mem_ptr, size_t size) {
    return nullptr;
  };

  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) {
    return nullptr;
  };

  void *Malloc(size_t size, uint32_t mem_flags = 0) { return nullptr; };

  void Free(void *mem_ptr, uint32_t mem_flags = 0){};

  Status Write(const void *host_data, size_t host_size, void *device_buffer,
               size_t device_size) {
    return STATUS_SUCCESS;
  };

  Status Read(const void *device_data, size_t device_size, void *host_buffer,
              size_t host_size) {
    return STATUS_SUCCESS;
  };

  Status DeviceMemoryCopy(
      const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
      const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
      size_t src_size,
      DeviceMemoryCopyKind copy_kind = DeviceMemoryCopyKind::FromHost) {
    return STATUS_SUCCESS;
  };

  Status GetDeviceMemUsage(size_t *free, size_t *total) const {
    return STATUS_SUCCESS;
  };
};

class MockDevice : public Device {
 public:
  MockDevice() : Device(std::make_shared<FakeDeviceMemoryManager>()) {
    EXPECT_CALL(*this, Malloc)
        .WillRepeatedly([](size_t size, const std::string &user_id) {
          return nullptr;
        });
  };

  virtual ~MockDevice() = default;

  std::vector<std::shared_ptr<DeviceMemory>> GetDeviceMemories() {
    return std::vector<std::shared_ptr<DeviceMemory>>();
  }

  using DeviceMem = std::shared_ptr<DeviceMemory>;
  MOCK_METHOD(DeviceMem, Malloc, (size_t, const std::string &));

 private:
  std::shared_ptr<Device> device_;
};

class MockDeviceFactory : public DeviceFactory {
 public:
  MockDeviceFactory() {
    EXPECT_CALL(*this, DeviceProbe).WillRepeatedly([this]() {
      return bind_factory_->DeviceProbe();
    });

    EXPECT_CALL(*this, CreateDevice)
        .WillRepeatedly([this](const std::string &device_id) {
          return bind_factory_->CreateDevice(device_id);
        });
  };

  virtual ~MockDeviceFactory(){};

  using DescMap = std::map<std::string, std::shared_ptr<DeviceDesc>>;
  MOCK_METHOD(DescMap, DeviceProbe, ());
  using DevicePtr = std::shared_ptr<Device>;
  MOCK_METHOD(DevicePtr, CreateDevice, (const std::string &));

 private:
  std::shared_ptr<DeviceFactory> bind_factory_ = std::make_shared<CPUFactory>();
};

class MockDriverDevice : public modelbox::MockDriver {
 public:
  MockDriverDevice(){};
  virtual ~MockDriverDevice(){};

  static MockDriverDevice *Instance() { return &desc_; };

 private:
  static MockDriverDevice desc_;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_MockDevice_H_
