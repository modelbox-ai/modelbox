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


#include "modelbox/base/device.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "modelbox/base/driver.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

using ::testing::_;

namespace modelbox {

class DeviceManagerTest : public testing::Test {
 public:
  DeviceManagerTest() = default;

 protected:
  void SetUp() override{

  };

  void TearDown() override {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    device_mgr->Clear();
    drivers->Clear();
  };
};

class DeviceMemoryTest : public testing::Test {
 public:
  DeviceMemoryTest() = default;

 protected:
  void SetUp() override {
    auto drivers = Drivers::GetInstance();
    ConfigurationBuilder config_builder;

    auto device_cpu_src_path = std::string(DEVICE_CPU_SO_PATH);
    auto device_cpu_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
    CopyFile(device_cpu_src_path, device_cpu_dest_path, 0, true);

    drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    device_mgr->Initialize(drivers, config_builder.Build());
    device_ = device_mgr->CreateDevice("cpu", "0");

    auto device_cuda_src_path = std::string(DEVICE_CUDA_SO_PATH);
    device_cuda_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-device-cuda.so";
    CopyFile(device_cuda_src_path, device_cuda_dest_path, 0, true);

    auto device_ascend_src_path = std::string(DEVICE_ASCEND_SO_PATH);
    device_ascend_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-device-ascend.so";
    CopyFile(device_ascend_src_path, device_ascend_dest_path, 0, true);
  };

  void TearDown() override {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    device_ = nullptr;
    device_mgr->Clear();
    drivers->Clear();

    remove(device_cuda_dest_path.c_str());
  };

  std::shared_ptr<Device> device_;
  std::string device_cuda_dest_path;
  std::string device_ascend_dest_path;
};

TEST_F(DeviceManagerTest, CheckInit) {
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  auto device = device_mgr->CreateDevice("cpu", "0");
  EXPECT_EQ(device, nullptr);
}

TEST_F(DeviceManagerTest, InitDeviceFactory) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;

  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");

  EXPECT_TRUE(result);
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  Status result1 = device_mgr->InitDeviceFactory(drivers);
  auto factory_list = device_mgr->GetDeviceFactoryList();
  for (auto iter = factory_list.begin(); iter != factory_list.end(); iter++) {
    EXPECT_EQ(iter->first, "cpu");
    EXPECT_NE(iter->second, nullptr);
  }
}

TEST_F(DeviceManagerTest, Probe) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;

  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");

  EXPECT_TRUE(result);
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  Status status1 = device_mgr->InitDeviceFactory(drivers);
  auto cpu_factory = device_mgr->GetDeviceFactoryList().begin();
  auto mock_factory =
      std::dynamic_pointer_cast<MockDeviceFactory>(cpu_factory->second);
  EXPECT_CALL(*mock_factory, DeviceProbe())
      .WillRepeatedly(testing::Invoke([&]() {
        std::map<std::string, std::shared_ptr<DeviceDesc>> tmp_map;
        std::shared_ptr<DeviceDesc> device_desc =
            std::make_shared<DeviceDesc>();
        device_desc->SetDeviceId("0");
        device_desc->SetDeviceDesc("test desc");
        device_desc->SetDeviceMemory("8Gi");
        device_desc->SetDeviceVersion("xxxx");
        device_desc->SetDeviceType("CPU");
        tmp_map.insert(std::make_pair("0", device_desc));
        return tmp_map;
      }));

  Status status2 = device_mgr->DeviceProbe();
  EXPECT_EQ(status1, STATUS_OK);
  EXPECT_EQ(status2, STATUS_OK);

  auto desc_list = device_mgr->GetDeviceDescList();
  auto iter1 = desc_list.find("cpu");
  EXPECT_EQ(iter1->first, "cpu");

  auto iter2 = iter1->second.find("0");
  EXPECT_EQ(iter2->first, "0");
  auto device_desc = iter2->second;
  EXPECT_EQ(device_desc->GetDeviceDesc(), "test desc");
  EXPECT_EQ(device_desc->GetDeviceId(), "0");
  EXPECT_EQ(device_desc->GetDeviceMemory(), "8Gi");
  EXPECT_EQ(device_desc->GetDeviceType(), "CPU");
}

TEST_F(DeviceManagerTest, CreateDevice) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;

  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");
  EXPECT_TRUE(result);
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  Status status1 = device_mgr->InitDeviceFactory(drivers);

  auto cpu_factory = device_mgr->GetDeviceFactoryList().begin();
  auto mock_factory =
      std::dynamic_pointer_cast<MockDeviceFactory>(cpu_factory->second);
  EXPECT_CALL(*mock_factory, DeviceProbe())
      .WillRepeatedly(testing::Invoke([&]() {
        std::map<std::string, std::shared_ptr<DeviceDesc>> tmp_map;
        std::shared_ptr<DeviceDesc> device_desc =
            std::make_shared<DeviceDesc>();
        device_desc->SetDeviceId("0");
        device_desc->SetDeviceDesc("test desc");
        device_desc->SetDeviceMemory("8Gi");
        device_desc->SetDeviceVersion("xxxx");
        device_desc->SetDeviceType("CPU");
        tmp_map.insert(std::make_pair("0", device_desc));
        return tmp_map;
      }));

  Status status2 = device_mgr->DeviceProbe();
  auto ss = device_mgr->GetDevicesTypes();
  EXPECT_EQ(ss[0], "cpu");

  auto device_null = device_mgr->GetDevice("cpu", "0");
  EXPECT_EQ(device_null, nullptr);

  EXPECT_CALL(*mock_factory, CreateDevice(_))
      .WillRepeatedly(testing::Invoke([&](const std::string &device_id) {
        std::shared_ptr<MockDevice> temp_mockdevice =
            std::make_shared<MockDevice>();
        return temp_mockdevice;
      }));

  auto device = device_mgr->CreateDevice("cpu", "0");
  EXPECT_NE(device->GetDeviceManager(), nullptr);
  auto device_desc = device->GetDeviceDesc();
  EXPECT_EQ(device_desc->GetDeviceDesc(), "test desc");
  EXPECT_EQ(device_desc->GetDeviceId(), "0");
  EXPECT_EQ(device_desc->GetDeviceMemory(), "8Gi");
  EXPECT_EQ(device_desc->GetDeviceVersion(), "xxxx");
  EXPECT_EQ(device_desc->GetDeviceType(), "CPU");

  auto device_sec = device_mgr->CreateDevice("cpu", "0");
  EXPECT_EQ(device, device_sec);

  auto device_get = device_mgr->GetDevice("cpu", "0");
  EXPECT_EQ(device, device_get);

  auto test = device_mgr->GetDevicesIdList("cpu");
  EXPECT_EQ(test[0], "0");
}

TEST_F(DeviceManagerTest, CreateDeviceMemory) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;

  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");

  EXPECT_TRUE(result);
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  device_mgr->Initialize(drivers, configbuilder.Build());
  EXPECT_EQ(device_mgr->GetDrivers(), drivers);
  auto device = device_mgr->CreateDevice("cpu", "0");
  device->SetMemQuota(1024);
  auto device_memory = device->MemAlloc(100);

  EXPECT_EQ(*((uint64_t *)(device_memory->GetConstPtr<uint8_t>().get() + 100)),
            DeviceMemory::MEM_MAGIC_CODE);
}

TEST_F(DeviceMemoryTest, MemAlloc) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(1024);
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->GetSize(), 1024);
  EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
  EXPECT_NE(mem1->GetPtr<void>().get(), nullptr);
  EXPECT_NE(mem1->GetConstPtr<void>().get(), nullptr);

  EXPECT_EQ(device_->GetAllocatedMemSize(), 1024);

  auto mem2 = device_->MemAlloc(0);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->GetSize(), 0);
  EXPECT_EQ(mem2->Verify(), STATUS_SUCCESS);
  EXPECT_EQ(mem2->GetPtr<void>().get(), nullptr);
  EXPECT_EQ(mem2->GetConstPtr<void>().get(), nullptr);

  auto mem3 = device_->MemAlloc(1);
  EXPECT_EQ(mem3.get(), nullptr);

  EXPECT_EQ(device_->GetAllocatedMemSize(), 1024);
}

TEST_F(DeviceMemoryTest, MemWrite) {
  device_->SetMemQuota(1024);

  std::shared_ptr<uint32_t> data(new uint32_t[100],
                                 [](const uint32_t *ptr) { delete[] ptr; });
  data.get()[13] = 111333;
  data.get()[33] = 333333;
  auto mem1 = device_->MemWrite(data.get(), 100 * sizeof(uint32_t));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->GetSize(), 100 * sizeof(uint32_t));
  EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetPtr<uint32_t>().get()[13], data.get()[13]);
  EXPECT_EQ(mem1->GetConstPtr<uint32_t>().get()[33], data.get()[33]);

  std::shared_ptr<uint32_t> data2(new uint32_t[300],
                                  [](const uint32_t *ptr) { delete[] ptr; });
  auto mem2 = device_->MemWrite(data.get(), 300 * sizeof(uint32_t));
  EXPECT_EQ(mem2.get(), nullptr);
}

TEST_F(DeviceMemoryTest, MemClone) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  auto ptr1 = mem1->GetPtr<uint32_t>();
  EXPECT_NE(mem1.get(), nullptr);
  ptr1.get()[13] = 111333;
  ptr1.get()[17] = 111777;

  auto mem2 = device_->MemClone(mem1);
  EXPECT_NE(mem2.get(), nullptr);
  auto ptr2 = mem2->GetPtr<uint32_t>();
  EXPECT_EQ(ptr2.get()[13], 111333);
  EXPECT_EQ(ptr2.get()[17], 111777);

  mem1->SetContentMutable(false);
  auto mem3 = device_->MemClone(mem1);
  EXPECT_EQ(mem3.get(), mem1.get());
}

// Read & Write
TEST_F(DeviceMemoryTest, DeviceMemoryReadWrite) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1.get(), nullptr);
  auto ptr = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr.get(), nullptr);
  ptr.get()[0] = 1;
  ptr.get()[1] = 2;
  auto ptr2 = mem1->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr2.get(), nullptr);
  EXPECT_EQ(ptr2.get()[0], 1);
  EXPECT_EQ(ptr2.get()[1], 2);
}

// Copy
TEST_F(DeviceMemoryTest, DeviceMemoryCopy) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1.get(), nullptr);

  auto mem2 = device_->MemAlloc(100);
  EXPECT_NE(mem2.get(), nullptr);

  auto ptr1 = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr1.get(), nullptr);
  ptr1.get()[3] = 3;
  ptr1.get()[15] = 15;
  ptr1.get()[27] = 27;

  auto ret = mem2->ReadFrom(mem1, 2, 97, 3);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  auto ptr2 = mem2->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr2.get(), nullptr);
  EXPECT_EQ(ptr2.get()[4], 3);
  EXPECT_EQ(ptr2.get()[16], 15);
  EXPECT_EQ(ptr2.get()[28], 27);

  ret = mem1->WriteTo(mem2, 3, 30, 1);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  auto ptr3 = mem2->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr3.get(), nullptr);
  EXPECT_EQ(ptr3.get()[1], 3);
  EXPECT_EQ(ptr3.get()[13], 15);
  EXPECT_EQ(ptr3.get()[25], 27);
}

// Mutale
TEST_F(DeviceMemoryTest, DeviceMemoryMutable) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1.get(), nullptr);

  auto ptr1 = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr1.get(), nullptr);

  ptr1.get()[13] = 3;
  ptr1.get()[25] = 5;
  ptr1.get()[37] = 7;

  mem1->SetContentMutable(false);
  EXPECT_EQ(mem1->IsContentMutable(), false);

  auto ptr2 = mem1->GetPtr<uint8_t>();
  EXPECT_EQ(ptr2.get(), nullptr);

  auto ptr3 = mem1->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr3.get(), nullptr);

  EXPECT_EQ(ptr3.get()[13], 3);
  EXPECT_EQ(ptr3.get()[25], 5);
  EXPECT_EQ(ptr3.get()[37], 7);
}

// Resize
TEST_F(DeviceMemoryTest, DeviceMemoryResize) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1.get(), nullptr);
  EXPECT_EQ(mem1->GetSize(), 100);

  auto ptr1 = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr1.get(), nullptr);
  ptr1.get()[25] = 3;
  ptr1.get()[13] = 5;
  ptr1.get()[1] = 7;

  auto ret = mem1->Resize(50);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetSize(), 50);

  ret = mem1->Resize(150);
  EXPECT_NE(ret, STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetSize(), 50);

  auto ptr2 = mem1->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr2.get(), nullptr);
  EXPECT_EQ(ptr2.get()[25], 3);
  EXPECT_EQ(ptr2.get()[13], 5);
  EXPECT_EQ(ptr2.get()[1], 7);
}

// Realloc
TEST_F(DeviceMemoryTest, DeviceMemoryRealloc) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1.get(), nullptr);
  EXPECT_EQ(mem1->GetSize(), 100);
  EXPECT_EQ(mem1->GetCapacity(), 100);

  auto ptr1 = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr1.get(), nullptr);
  ptr1.get()[25] = 3;
  ptr1.get()[13] = 5;
  ptr1.get()[1] = 7;

  auto ret = mem1->Realloc(50);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetSize(), 100);
  EXPECT_EQ(mem1->GetCapacity(), 100);

  ret = mem1->Realloc(150);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetSize(), 100);
  EXPECT_EQ(mem1->GetCapacity(), 150);

  auto ptr2 = mem1->GetConstPtr<uint8_t>();
  EXPECT_NE(ptr2.get(), nullptr);
  EXPECT_EQ(ptr2.get()[25], 3);
  EXPECT_EQ(ptr2.get()[13], 5);
  EXPECT_EQ(ptr2.get()[1], 7);

  ret = mem1->Realloc(1024);
  EXPECT_NE(ret, STATUS_SUCCESS);
  EXPECT_EQ(mem1->GetSize(), 100);
  EXPECT_EQ(mem1->GetCapacity(), 150);
}

// Memory size
TEST_F(DeviceMemoryTest, DeviceMemorySize) {
  size_t free;
  size_t total;
  auto ret = device_->GetMemInfo(&free, &total);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  EXPECT_NE(free, 0);
  EXPECT_NE(total, 0);

  EXPECT_EQ(device_->GetMemQuota(), total);
}

TEST_F(DeviceMemoryTest, CudaMemoryTest) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cuda.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto cuda_device = dev_mgr->CreateDevice("cuda", "0");
  if (cuda_device == nullptr) {
    GTEST_SKIP();
  }

  cuda_device->SetMemQuota(1024);
  device_->SetMemQuota(1024);
  auto cpu_mem = device_->MemAlloc(1024);
  {
    // Malloc
    auto mem1 = cuda_device->MemAlloc(1024);
    EXPECT_NE(mem1, nullptr);
    EXPECT_EQ(mem1->GetSize(), 1024);
    EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
    EXPECT_NE(mem1->GetPtr<void>().get(), nullptr);
    EXPECT_NE(mem1->GetConstPtr<void>().get(), nullptr);
    EXPECT_EQ(cuda_device->GetAllocatedMemSize(), 1024);
  }
  {
    // MemWrite
    std::shared_ptr<uint32_t> data(new uint32_t[100],
                                   [](const uint32_t *ptr) { delete[] ptr; });
    data.get()[35] = 333555;
    data.get()[53] = 555333;
    auto mem1 = cuda_device->MemWrite(data.get(), 100 * sizeof(uint32_t));
    EXPECT_NE(mem1, nullptr);
    EXPECT_EQ(mem1->GetSize(), 100 * sizeof(uint32_t));
    EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
    EXPECT_NE(mem1->GetPtr<uint8_t>(), nullptr);
    EXPECT_NE(mem1->GetConstPtr<uint8_t>(), nullptr);

    mem1->WriteTo(cpu_mem, 0, 100 * sizeof(uint32_t), 0);
    auto cpu_data = cpu_mem->GetConstPtr<uint32_t>();
    EXPECT_NE(cpu_data, nullptr);
    EXPECT_EQ(cpu_data.get()[35], 333555);
    EXPECT_EQ(cpu_data.get()[53], 555333);
  }
  {
    // MemClone & Copy
    auto ptr1 = cpu_mem->GetPtr<uint32_t>();
    ptr1.get()[17] = 111777;
    ptr1.get()[13] = 333111;
    auto mem1 = cuda_device->MemAlloc(100);
    auto ret = mem1->ReadFrom(cpu_mem, 0, 100, 0);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto mem2 = cuda_device->MemClone(mem1);
    ret = mem2->WriteTo(cpu_mem, 0, 100, 100);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto ptr2 = cpu_mem->GetPtr<uint32_t>();
    EXPECT_EQ(ptr2.get()[25 + 17], 111777);
    EXPECT_EQ(ptr2.get()[25 + 13], 333111);
  }
  {
    // Mutable
    auto mem1 = cuda_device->MemAlloc(100);
    mem1->SetContentMutable(false);
    auto ptr = mem1->GetPtr<uint8_t>();
    EXPECT_EQ(ptr, nullptr);
    auto ptr2 = mem1->GetConstPtr<uint8_t>();
    EXPECT_NE(ptr2, nullptr);
  }
  {
    // Realloc
    auto mem1 = cuda_device->MemAlloc(100);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 100);

    mem1->Realloc(50);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 100);

    mem1->Realloc(200);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 200);
    mem1->WriteTo(cpu_mem, 0, 100);  // sync stream
  }
  {
    // Memory size
    size_t free;
    size_t total;
    auto ret = cuda_device->GetMemInfo(&free, &total);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    EXPECT_NE(free, 0);
    EXPECT_NE(total, 0);
  }

  cuda_device = nullptr;
}

TEST_F(DeviceMemoryTest, CudaStreamTest) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cuda.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto cuda_device = dev_mgr->CreateDevice("cuda", "0");
  if (cuda_device == nullptr) {
    GTEST_SKIP();
  }

  cuda_device->SetMemQuota(1024);
  device_->SetMemQuota(1024);
  {
    auto unit3_output = cuda_device->MemAlloc(100);
    EXPECT_NE(unit3_output, nullptr);
    {
      auto unit2_output = cuda_device->MemAlloc(100);
      EXPECT_NE(unit2_output, nullptr);
      {
        auto unit1_output = cuda_device->MemAlloc(100);
        EXPECT_NE(unit1_output, nullptr);
        {
          // unit1
          uint32_t host_data[5]{1, 2, 3, 4, 5};
          auto size = 5 * sizeof(uint32_t);
          auto mem1 = cuda_device->MemWrite(host_data, size);
          EXPECT_NE(mem1, nullptr);
          auto ret = unit1_output->ReadFrom(mem1, 0, size);
          EXPECT_EQ(ret, STATUS_SUCCESS);
        }
        // unit2
        auto ret = unit2_output->ReadFrom(unit1_output, 0, 100);
        EXPECT_EQ(ret, STATUS_SUCCESS);
      }
      // unit3
      auto ret = unit3_output->ReadFrom(unit2_output, 0, 100);
      EXPECT_EQ(ret, STATUS_SUCCESS);
    }
    // check output
    auto host_mem = device_->MemAlloc(100);
    EXPECT_NE(host_mem, nullptr);
    auto ret = host_mem->ReadFrom(unit3_output, 0, 100);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto ptr = host_mem->GetConstPtr<uint32_t>();
    EXPECT_EQ(ptr.get()[0], 1);
    EXPECT_EQ(ptr.get()[1], 2);
    EXPECT_EQ(ptr.get()[2], 3);
    EXPECT_EQ(ptr.get()[3], 4);
    EXPECT_EQ(ptr.get()[4], 5);
  }

  cuda_device = nullptr;
}

TEST_F(DeviceMemoryTest, DeviceMemoryAppend) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(50, (size_t)100, 0);
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->GetSize(), 50);
  EXPECT_EQ(mem1->GetCapacity(), 100);
  auto ptr = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr, nullptr);
  ptr.get()[13] = 13;
  ptr.get()[23] = 23;
  ptr.get()[33] = 33;
  auto mem2 = device_->MemAlloc(50);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->GetSize(), 50);
  EXPECT_EQ(mem2->GetCapacity(), 50);
  auto ptr2 = mem2->GetPtr<uint8_t>();
  EXPECT_NE(ptr2, nullptr);
  ptr2.get()[13] = 33;
  ptr2.get()[23] = 32;
  ptr2.get()[33] = 31;

  auto mem3 = mem1->Append(mem2);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->GetSize(), 100);
  EXPECT_EQ(mem3->GetCapacity(), 100);
  EXPECT_EQ(mem3->GetPtr<uint8_t>(), mem1->GetPtr<uint8_t>());
  auto ptr3 = mem3->GetPtr<uint8_t>();
  EXPECT_NE(ptr3, nullptr);
  EXPECT_EQ(ptr3.get()[13], 13);
  EXPECT_EQ(ptr3.get()[23], 23);
  EXPECT_EQ(ptr3.get()[33], 33);
  EXPECT_EQ(ptr3.get()[63], 33);
  EXPECT_EQ(ptr3.get()[73], 32);
  EXPECT_EQ(ptr3.get()[83], 31);

  auto mem4 = mem3->Append(mem2);
  EXPECT_NE(mem4, nullptr);
  EXPECT_EQ(mem4->GetSize(), 150);
  EXPECT_EQ(mem4->GetCapacity(), 150);
  EXPECT_NE(mem4->GetPtr<uint8_t>(), mem3->GetPtr<uint8_t>());
  auto ptr4 = mem4->GetPtr<uint8_t>();
  EXPECT_NE(ptr4, nullptr);
  EXPECT_EQ(ptr4.get()[13], 13);
  EXPECT_EQ(ptr4.get()[23], 23);
  EXPECT_EQ(ptr4.get()[33], 33);
  EXPECT_EQ(ptr4.get()[63], 33);
  EXPECT_EQ(ptr4.get()[73], 32);
  EXPECT_EQ(ptr4.get()[83], 31);
  EXPECT_EQ(ptr4.get()[113], 33);
  EXPECT_EQ(ptr4.get()[123], 32);
  EXPECT_EQ(ptr4.get()[133], 31);
}

TEST_F(DeviceMemoryTest, DeviceMemoryAppend2) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(50, (size_t)100, 0);
  mem1->GetPtr<uint8_t>().get()[4] = 14;
  auto mem2 = device_->MemAlloc(50);
  mem2->GetPtr<uint8_t>().get()[4] = 24;
  auto mem3 = device_->MemAlloc(50);
  mem3->GetPtr<uint8_t>().get()[4] = 34;

  auto mem4 = mem1->Append({mem2, mem3});
  EXPECT_NE(mem4, nullptr);
  EXPECT_EQ(mem4->GetSize(), 150);
  EXPECT_EQ(mem4->GetCapacity(), 150);
  auto ptr4 = mem4->GetPtr<uint8_t>();
  EXPECT_EQ(ptr4.get()[4], 14);
  EXPECT_EQ(ptr4.get()[54], 24);
  EXPECT_EQ(ptr4.get()[104], 34);

  auto mem5 = DeviceMemory::Combine({mem1, mem2, mem3});
  EXPECT_NE(mem5, nullptr);
  EXPECT_EQ(mem5->GetSize(), 150);
  EXPECT_EQ(mem5->GetCapacity(), 150);
  auto ptr5 = mem5->GetPtr<uint8_t>();
  EXPECT_EQ(ptr5.get()[4], 14);
  EXPECT_EQ(ptr5.get()[54], 24);
  EXPECT_EQ(ptr5.get()[104], 34);
}

TEST_F(DeviceMemoryTest, DeviceMemoryAppend3) {
  device_->SetMemQuota(1024);
  auto mem1 = device_->MemAlloc(100, (size_t)100, 0);
  auto sub1 = mem1->Cut(0, 10);
  auto sub2 = mem1->Cut(10, 10);
  auto sub3 = mem1->Cut(20, 10);
  auto sub4 = mem1->Cut(29, 10);  // overlap
  // continuous
  auto mem2 = DeviceMemory::Combine({sub1, sub3, sub2});
  EXPECT_NE(mem2, nullptr);
  EXPECT_NE(mem2->GetConstPtr<void>(), sub1->GetConstPtr<void>());
  EXPECT_EQ(mem2->GetSize(), 30);
  EXPECT_EQ(mem2->GetCapacity(), 30);
  auto mem3 = DeviceMemory::Combine({sub3, sub2});
  EXPECT_NE(mem3, nullptr);
  EXPECT_NE(mem3->GetConstPtr<void>(), sub2->GetConstPtr<void>());
  EXPECT_EQ(mem3->GetSize(), 20);
  EXPECT_EQ(mem3->GetCapacity(), 20);
  // fragment
  auto mem4 = DeviceMemory::Combine({sub1, sub2, sub3, sub4});
  EXPECT_NE(mem4, nullptr);
  EXPECT_NE(mem4->GetConstPtr<void>(), sub1->GetConstPtr<void>());
  EXPECT_EQ(mem4->GetSize(), 40);
  EXPECT_EQ(mem4->GetCapacity(), 40);
  auto mem5 = DeviceMemory::Combine({sub1, sub3});
  EXPECT_NE(mem5, nullptr);
  EXPECT_NE(mem5->GetConstPtr<void>(), sub1->GetConstPtr<void>());
  EXPECT_EQ(mem5->GetSize(), 20);
  EXPECT_EQ(mem5->GetCapacity(), 20);
}

TEST_F(DeviceMemoryTest, CudaMemoryAppend) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cuda.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto cuda_device = dev_mgr->CreateDevice("cuda", "0");
  if (cuda_device == nullptr) {
    GTEST_SKIP();
  }
  // cuda meta
  device_->SetMemQuota(1024);
  auto mem1 = device_->MemAlloc(100, (size_t)100, 0);
  auto *ptr = mem1->GetPtr<uint8_t>().get();
  ptr[1] = 13;
  ptr[5] = 53;
  ptr[9] = 93;
  cuda_device->SetMemQuota(1024);
  auto cuda_mem1 = cuda_device->MemAlloc(100, (size_t)100, 0);
  auto cuda_sub1 = cuda_mem1->Cut(0, 10);
  auto cuda_sub2 = cuda_mem1->Cut(10, 10);
  // Create different stream
  auto ret = cuda_sub1->ReadFrom(mem1, 0, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  ret = cuda_sub2->ReadFrom(mem1, 0, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  auto cuda_mem2 = DeviceMemory::Combine({cuda_sub1, cuda_sub2});
  EXPECT_NE(cuda_mem2, nullptr);
  EXPECT_EQ(cuda_mem2->GetSize(), 20);
  EXPECT_EQ(cuda_mem2->GetCapacity(), 100);
  auto mem2 = device_->MemAlloc(100, (size_t)100, 0);
  EXPECT_NE(mem2, nullptr);
  ret = mem2->ReadFrom(cuda_mem2, 0, 20);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  const auto *ptr2 = mem2->GetConstPtr<uint8_t>().get();
  EXPECT_EQ(ptr2[1], 13);
  EXPECT_EQ(ptr2[5], 53);
  EXPECT_EQ(ptr2[9], 93);
  EXPECT_EQ(ptr2[11], 13);
  EXPECT_EQ(ptr2[15], 53);
  EXPECT_EQ(ptr2[19], 93);
  cuda_device = nullptr;
}

TEST_F(DeviceMemoryTest, DeviceMemoryCut) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->GetSize(), 100);
  EXPECT_EQ(mem1->GetCapacity(), 100);
  auto ptr = mem1->GetPtr<uint8_t>();
  EXPECT_NE(ptr, nullptr);
  ptr.get()[13] = 13;
  ptr.get()[23] = 23;
  ptr.get()[33] = 33;
  ptr.get()[43] = 43;
  ptr.get()[53] = 53;
  auto mem_part1 = mem1->Cut(10, 10);
  EXPECT_NE(mem_part1, nullptr);
  EXPECT_EQ(mem_part1->GetSize(), 10);
  EXPECT_EQ(mem_part1->GetCapacity(), 90);
  auto mem_part2 = mem1->Cut(20, 10);
  EXPECT_NE(mem_part2, nullptr);
  EXPECT_EQ(mem_part2->GetSize(), 10);
  EXPECT_EQ(mem_part2->GetCapacity(), 80);
  EXPECT_EQ(mem_part2->GetPtr<uint8_t>().get()[3], 23);
  auto mem_part3 = mem1->Cut(30, 10);
  EXPECT_NE(mem_part3, nullptr);
  auto mem_part4 = mem1->Cut(40, 10);
  EXPECT_NE(mem_part4, nullptr);
  auto mem_part5 = mem1->Cut(50, 50);
  EXPECT_NE(mem_part5, nullptr);
  EXPECT_EQ(mem_part5->GetSize(), 50);
  EXPECT_EQ(mem_part5->GetCapacity(), 50);
  EXPECT_EQ(mem_part5->GetPtr<uint8_t>().get()[3], 53);
}

TEST_F(DeviceMemoryTest, DeviceMemoryDelete) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1, nullptr);
  auto ptr = mem1->GetPtr<uint8_t>();
  ptr.get()[3] = 3;
  ptr.get()[13] = 13;
  ptr.get()[23] = 23;
  auto mem2 = mem1->Delete(0, 10, 100);
  EXPECT_EQ(mem2->GetSize(), 90);
  EXPECT_NE(mem2, nullptr);
  auto ptr2 = mem2->GetPtr<uint8_t>();
  EXPECT_EQ(ptr2.get()[3], 13);
  EXPECT_EQ(ptr2.get()[13], 23);
  auto mem3 = mem1->Delete(10, 10);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->GetSize(), 90);
  EXPECT_EQ(mem3->GetCapacity(), 90);
  auto ptr3 = mem3->GetPtr<uint8_t>();
  EXPECT_EQ(ptr3.get()[3], 3);
  EXPECT_EQ(ptr3.get()[13], 23);
  auto mem4 = mem1->Delete(90, 10, 100);
  EXPECT_NE(mem4, nullptr);
  EXPECT_EQ(mem4->GetSize(), 90);
  EXPECT_EQ(mem4->GetCapacity(), 100);
  auto mem5 = mem1->Delete(90, 20, 100);
  EXPECT_EQ(mem5, nullptr);
}

TEST_F(DeviceMemoryTest, DeviceMemoryCopy2) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1, nullptr);
  auto ptr = mem1->GetPtr<uint8_t>();
  ptr.get()[3] = 3;
  ptr.get()[13] = 13;
  ptr.get()[23] = 23;
  auto mem2 = mem1->Copy(0, 10, 100);
  EXPECT_NE(mem2, nullptr);
  auto ptr2 = mem2->GetPtr<uint8_t>();
  EXPECT_EQ(ptr2.get()[3], 3);
  auto mem3 = mem1->Copy(10, 20);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->GetSize(), 20);
  EXPECT_EQ(mem3->GetCapacity(), 20);
  auto ptr3 = mem3->GetPtr<uint8_t>();
  EXPECT_EQ(ptr3.get()[3], 13);
  EXPECT_EQ(ptr3.get()[13], 23);
  auto mem4 = mem1->Copy(50, 60, 100);
  EXPECT_EQ(mem4, nullptr);
}

TEST_F(DeviceMemoryTest, DeviceMemoryClone2) {
  device_->SetMemQuota(1024);

  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1, nullptr);
  auto ptr = mem1->GetPtr<uint8_t>();
  ptr.get()[3] = 3;
  ptr.get()[13] = 13;
  ptr.get()[23] = 23;

  auto mem2 = mem1->Clone();
  EXPECT_EQ(mem2->GetPtr<uint8_t>(), mem1->GetPtr<uint8_t>());
  auto mem3 = mem1->Clone(true);
  EXPECT_NE(mem3->GetPtr<uint8_t>(), mem1->GetPtr<uint8_t>());
  auto ptr2 = mem3->GetPtr<uint8_t>();
  EXPECT_EQ(ptr2.get()[3], 3);
  EXPECT_EQ(ptr2.get()[13], 13);
  EXPECT_EQ(ptr2.get()[23], 23);
}

TEST_F(DeviceMemoryTest, DeviceMemoryContiguous) {
  auto mem1 = device_->MemAlloc(100);
  EXPECT_NE(mem1, nullptr);
  auto mem2 = mem1->Cut(0, 10);
  auto mem3 = mem1->Cut(10, 20);
  auto mem4 = mem1->Cut(30, 50);
  auto mem5 = mem4->Cut(10, 30);
  EXPECT_EQ(device_->GetAllocatedMemSize(), 100);
  std::vector<std::shared_ptr<DeviceMemory>> mem_list = {mem2, mem3, mem4};
  std::vector<std::shared_ptr<DeviceMemory>> mem_list2 = {mem3, mem2, mem4};
  std::vector<std::shared_ptr<DeviceMemory>> mem_list3 = {mem3, mem2, mem4,
                                                          mem5};
  // Basic test
  EXPECT_TRUE(DeviceMemory::IsContiguous(mem_list));
  EXPECT_TRUE(DeviceMemory::IsContiguous(mem_list, false));
  // Test order
  EXPECT_FALSE(DeviceMemory::IsContiguous(mem_list2));
  EXPECT_TRUE(DeviceMemory::IsContiguous(mem_list2, false));
  // Test mem offset
  EXPECT_FALSE(DeviceMemory::IsContiguous(mem_list3));
  EXPECT_FALSE(DeviceMemory::IsContiguous(mem_list3, false));
  // Test mem block
  auto mem6 = device_->MemAlloc(100);
  mem_list.push_back(mem6);
  EXPECT_FALSE(DeviceMemory::IsContiguous(mem_list));
  EXPECT_FALSE(DeviceMemory::IsContiguous(mem_list, false));
}

TEST_F(DeviceMemoryTest, DeviceMemoryAcquire) {
  device_->SetMemQuota(1024);

  auto *data = new uint8_t[100];
  data[33] = 33;
  data[44] = 44;
  data[55] = 55;
  data[66] = 66;

  auto dev_mem = device_->MemAcquire(
      (void *)data, 100, [](void *ptr) { delete[](uint8_t *) ptr; });
  EXPECT_NE(dev_mem, nullptr);
  const auto *ptr = dev_mem->GetConstPtr<uint8_t>().get();
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(ptr[33], 33);
  EXPECT_EQ(ptr[44], 44);
  EXPECT_EQ(ptr[55], 55);
  EXPECT_EQ(ptr[66], 66);

  std::shared_ptr<uint8_t> data2(new uint8_t[100],
                                 [](const uint8_t *ptr) { delete[] ptr; });
  data2.get()[33] = 33;
  data2.get()[44] = 44;
  data2.get()[55] = 55;
  data2.get()[66] = 66;

  auto dev_mem2 = device_->MemAcquire(data2, 100);
  EXPECT_NE(dev_mem2, nullptr);
  const auto *ptr2 = dev_mem2->GetConstPtr<uint8_t>().get();
  EXPECT_NE(ptr2, nullptr);
  EXPECT_EQ(ptr2[33], 33);
  EXPECT_EQ(ptr2[44], 44);
  EXPECT_EQ(ptr2[55], 55);
  EXPECT_EQ(ptr2[66], 66);
}

TEST_F(DeviceMemoryTest, AscendMemoryTest) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-ascend.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto ascend_device = dev_mgr->CreateDevice("ascend", "0");
  if (ascend_device == nullptr) {
    GTEST_SKIP();
  }

  ascend_device->SetMemQuota(1024);
  device_->SetMemQuota(1024);
  auto cpu_mem = device_->MemAlloc(1024);
  {
    // Malloc
    auto mem1 = ascend_device->MemAlloc(1024);
    EXPECT_NE(mem1, nullptr);
    EXPECT_EQ(mem1->GetSize(), 1024);
    EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
    EXPECT_NE(mem1->GetPtr<void>().get(), nullptr);
    EXPECT_NE(mem1->GetConstPtr<void>().get(), nullptr);
    EXPECT_EQ(ascend_device->GetAllocatedMemSize(), 1024);
  }
  {
    // MemWrite
    std::shared_ptr<uint32_t> data(new uint32_t[100],
                                   [](const uint32_t *ptr) { delete[] ptr; });
    data.get()[35] = 333555;
    data.get()[53] = 555333;
    auto mem1 = ascend_device->MemWrite(data.get(), 100 * sizeof(uint32_t));
    EXPECT_NE(mem1, nullptr);
    EXPECT_EQ(mem1->GetSize(), 100 * sizeof(uint32_t));
    EXPECT_EQ(mem1->Verify(), STATUS_SUCCESS);
    EXPECT_NE(mem1->GetPtr<uint8_t>(), nullptr);
    EXPECT_NE(mem1->GetConstPtr<uint8_t>(), nullptr);

    mem1->WriteTo(cpu_mem, 0, 100 * sizeof(uint32_t), 0);
    auto cpu_data = cpu_mem->GetConstPtr<uint32_t>();
    EXPECT_NE(cpu_data, nullptr);
    EXPECT_EQ(cpu_data.get()[35], 333555);
    EXPECT_EQ(cpu_data.get()[53], 555333);
  }
  {
    // MemClone & Copy
    auto ptr1 = cpu_mem->GetPtr<uint32_t>();
    ptr1.get()[17] = 111777;
    ptr1.get()[13] = 333111;
    auto mem1 = ascend_device->MemAlloc(100);
    auto ret = mem1->ReadFrom(cpu_mem, 0, 100, 0);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto mem2 = ascend_device->MemClone(mem1);
    ret = mem2->WriteTo(cpu_mem, 0, 100, 100);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto ptr2 = cpu_mem->GetPtr<uint32_t>();
    EXPECT_EQ(ptr2.get()[25 + 17], 111777);
    EXPECT_EQ(ptr2.get()[25 + 13], 333111);
  }
  {
    // Mutable
    auto mem1 = ascend_device->MemAlloc(100);
    mem1->SetContentMutable(false);
    auto ptr = mem1->GetPtr<uint8_t>();
    EXPECT_EQ(ptr, nullptr);
    auto ptr2 = mem1->GetConstPtr<uint8_t>();
    EXPECT_NE(ptr2, nullptr);
  }
  {
    // Realloc
    auto mem1 = ascend_device->MemAlloc(100);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 100);

    mem1->Realloc(50);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 100);

    mem1->Realloc(200);
    EXPECT_EQ(mem1->GetSize(), 100);
    EXPECT_EQ(mem1->GetCapacity(), 200);

    // Sync
    mem1->WriteTo(cpu_mem, 0, 100);
  }
  {
    // Memory size
    size_t free;
    size_t total;
    auto ret = ascend_device->GetMemInfo(&free, &total);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    EXPECT_NE(free, 0);
    EXPECT_NE(total, 0);
  }
}

TEST_F(DeviceMemoryTest, AscendStreamTest) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-ascend.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto ascend_device = dev_mgr->CreateDevice("ascend", "0");
  if (ascend_device == nullptr) {
    GTEST_SKIP();
  }

  ascend_device->SetMemQuota(1024);
  device_->SetMemQuota(1024);
  {
    auto unit3_output = ascend_device->MemAlloc(100);
    EXPECT_NE(unit3_output, nullptr);
    {
      auto unit2_output = ascend_device->MemAlloc(100);
      EXPECT_NE(unit2_output, nullptr);
      {
        auto unit1_output = ascend_device->MemAlloc(100);
        EXPECT_NE(unit1_output, nullptr);
        {
          // unit1
          auto *host_data = new uint32_t[5]{1, 2, 3, 4, 5};
          auto size = 5 * sizeof(uint32_t);
          auto mem1 = ascend_device->MemWrite(host_data, size);
          delete []host_data;
          EXPECT_NE(mem1, nullptr);
          auto ret = unit1_output->ReadFrom(mem1, 0, size);
          EXPECT_EQ(ret, STATUS_SUCCESS);
        }
        // unit2
        auto ret = unit2_output->ReadFrom(unit1_output, 0, 100);
        EXPECT_EQ(ret, STATUS_SUCCESS);
      }
      // unit3
      auto ret = unit3_output->ReadFrom(unit2_output, 0, 100);
      EXPECT_EQ(ret, STATUS_SUCCESS);
    }
    // check output
    auto host_mem = device_->MemAlloc(100);
    EXPECT_NE(host_mem, nullptr);
    auto ret = host_mem->ReadFrom(unit3_output, 0, 100);
    EXPECT_EQ(ret, STATUS_SUCCESS);
    auto ptr = host_mem->GetConstPtr<uint32_t>();
    EXPECT_EQ(ptr.get()[0], 1);
    EXPECT_EQ(ptr.get()[1], 2);
    EXPECT_EQ(ptr.get()[2], 3);
    EXPECT_EQ(ptr.get()[3], 4);
    EXPECT_EQ(ptr.get()[4], 5);
  }
}

TEST_F(DeviceMemoryTest, AscendMemoryAppend) {
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-ascend.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  dev_mgr->Initialize(drivers, configbuilder.Build());
  auto ascend_device = dev_mgr->CreateDevice("ascend", "0");
  if (ascend_device == nullptr) {
    GTEST_SKIP();
  }

  // ascend meta
  device_->SetMemQuota(1024);
  auto mem1 = device_->MemAlloc(100, (size_t)100, 0);
  auto *ptr = mem1->GetPtr<uint8_t>().get();
  ptr[1] = 13;
  ptr[5] = 53;
  ptr[9] = 93;
  ascend_device->SetMemQuota(1024);
  auto ascend_mem1 = ascend_device->MemAlloc(100, (size_t)100, 0);
  auto ascend_sub1 = ascend_mem1->Cut(0, 10);
  auto ascend_sub2 = ascend_mem1->Cut(10, 10);
  // Create different stream
  auto ret = ascend_sub1->ReadFrom(mem1, 0, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  ret = ascend_sub2->ReadFrom(mem1, 0, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  auto ascend_mem2 = DeviceMemory::Combine({ascend_sub1, ascend_sub2});
  EXPECT_NE(ascend_mem2, nullptr);
  EXPECT_EQ(ascend_mem2->GetSize(), 20);
  EXPECT_EQ(ascend_mem2->GetCapacity(), 100);
  auto mem2 = device_->MemAlloc(100, (size_t)100, 0);
  EXPECT_NE(mem2, nullptr);
  ret = mem2->ReadFrom(ascend_mem2, 0, 20);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  const auto *ptr2 = mem2->GetConstPtr<uint8_t>().get();
  EXPECT_EQ(ptr2[1], 13);
  EXPECT_EQ(ptr2[5], 53);
  EXPECT_EQ(ptr2[9], 93);
  EXPECT_EQ(ptr2[11], 13);
  EXPECT_EQ(ptr2[15], 53);
  EXPECT_EQ(ptr2[19], 93);
}

}  // namespace modelbox