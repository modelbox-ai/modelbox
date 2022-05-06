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

#include "modelbox/buffer.h"

#include <functional>
#include <future>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"

namespace modelbox {
class BufferTest : public testing::Test {
 public:
  BufferTest() {}

 protected:
  std::shared_ptr<Device> device_;
  std::shared_ptr<MockDriverCtl> ctl_;

  virtual void SetUp() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    ctl_ = std::make_shared<MockDriverCtl>();
    modelbox::DriverDesc desc;

    desc.SetClass("DRIVER-DEVICE");
    desc.SetType("cpu");
    desc.SetName("device-driver-cpu");
    desc.SetDescription("the cpu device");
    desc.SetVersion("8.9.2");
    std::string file_path_device =
        std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
    desc.SetFilePath(file_path_device);
    ctl_->AddMockDriverDevice("cpu", desc);

    bool result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cpu.so");

    EXPECT_TRUE(result);
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    device_mgr->Initialize(drivers, config);
    device_ = device_mgr->CreateDevice("cpu", "0");
    device_->SetMemQuota(10240);
  };

  virtual void TearDown() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    device_mgr->Clear();
    drivers->Clear();
    device_ = nullptr;
  };
};

TEST_F(BufferTest, MutableData) {
  std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  Buffer buffer(device_);
  buffer.Build(data.size() * sizeof(int));
  auto dev_data = static_cast<int *>(buffer.MutableData());
  for (size_t i = 0; i < data.size(); ++i) {
    dev_data[i] = data[i];
  }

  const auto buffer_data = (const int *)buffer.ConstData();
  EXPECT_NE(nullptr, buffer_data);
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(buffer_data[i], data[i]);
  }

  buffer.SetError("BufferTest.ProcessError", "exception test");

  auto buffer_data2 = (int *)buffer.MutableData();
  EXPECT_EQ(nullptr, buffer_data2);
  EXPECT_TRUE(buffer.HasError());
}

TEST_F(BufferTest, SetException) {
  Buffer buffer;
  EXPECT_FALSE(buffer.HasError());

  buffer.SetError("BufferTest.ProcessError", "exception test");
  EXPECT_TRUE(buffer.HasError());
}

TEST_F(BufferTest, Size) {
  std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto dev_mem = device_->MemAlloc(data.size() * sizeof(int));
  Buffer buffer(device_);
  buffer.Build(data.size() * sizeof(int));
  EXPECT_EQ(buffer.GetBytes(), data.size() * sizeof(int));
}

TEST_F(BufferTest, Get) {
  Buffer buffer(device_);
  buffer.Set("Height", 720);
  buffer.Set("Width", 1280);

  buffer.Set("PTS", 10);
  buffer.Set("FPS", 30.1f);

  int i_value = 0;
  float f_valud = 0.0;
  EXPECT_TRUE(buffer.Get("Height", i_value));
  EXPECT_EQ(i_value, 720);

  EXPECT_TRUE(buffer.Get("Width", i_value));
  EXPECT_EQ(i_value, 1280);

  EXPECT_TRUE(buffer.Get("PTS", i_value));
  EXPECT_EQ(i_value, 10);

  EXPECT_TRUE(buffer.Get("FPS", f_valud));
  EXPECT_EQ(f_valud, 30.1f);

  std::shared_ptr<Buffer> buf_ptr(&buffer, [](void *p) {});

  Buffer buffer2(device_);
  buffer2.CopyMeta(buf_ptr);

  buffer2.Set("Height", 360);

  EXPECT_TRUE(buffer.Get("Height", i_value));
  EXPECT_EQ(i_value, 720);

  EXPECT_TRUE(buffer2.Get("Height", i_value));
  EXPECT_EQ(i_value, 360);

  buffer2.Get("Not_Found", i_value, 1000);
  EXPECT_EQ(i_value, 1000);

  buffer2.Get("Not_Found", f_valud, 100.f);
  EXPECT_EQ(f_valud, 100.f);
}

TEST_F(BufferTest, GetCast) {
  Buffer buffer(device_);
  int32_t weight = 720;
  buffer.Set("weight", weight);
  int64_t weight64;
  bool res = buffer.Get("weight", weight64);
  EXPECT_TRUE(res);
  EXPECT_EQ(weight64, 720);
}

TEST_F(BufferTest, Buffer1) {
  Buffer buffer(device_);
  Buffer buffer2 = buffer;

  EXPECT_EQ(buffer.MutableData(), buffer2.MutableData());
}

TEST_F(BufferTest, Copy) {
  Buffer buffer(device_);
  auto buffer2 = buffer.Copy();

  EXPECT_EQ(buffer.MutableData(), buffer2->MutableData());
}

TEST_F(BufferTest, DeepCopy) {
  Buffer buffer(device_);

  constexpr int DATA_SIZE = 10;
  std::vector<int> data(DATA_SIZE, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  buffer.Build(data.data(), data.size() * sizeof(int), [](void *ptr) {});
  buffer.Set("Height", 720);
  buffer.Set("Width", 1280);

  auto buffer2 = buffer.DeepCopy();

  int buffer_value = 0;
  int buffer2_value = -1;
  buffer.Get("Height", buffer_value);
  buffer2->Get("Height", buffer2_value);
  EXPECT_EQ(buffer_value, buffer2_value);

  buffer_value = 0;
  buffer2_value = -1;
  buffer.Get("Width", buffer_value);
  buffer2->Get("Width", buffer2_value);
  EXPECT_EQ(buffer_value, buffer2_value);

  auto buf_data = (int *)buffer.MutableData();
  auto buf_data2 = (int *)buffer2->MutableData();
  EXPECT_NE(buf_data, buf_data2);

  EXPECT_EQ(buffer.GetBytes(), buffer2->GetBytes());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(buf_data[i], data[i]);
    EXPECT_EQ(buf_data2[i], data[i]);
  }
}

class MockBuffer : public Buffer {
 public:
  MockBuffer(const std::shared_ptr<Device> &device) : Buffer(device){};
  ~MockBuffer() = default;
  void SetDelayedCopyDestinationDevice(std::shared_ptr<Device> dest_device) {
    Buffer::SetDelayedCopyDestinationDevice(dest_device);
  }
};

TEST_F(BufferTest, MoveToTargetDevice) {
  auto device_cuda_src_path = std::string(DEVICE_CUDA_SO_PATH);
  auto device_cuda_dest_path =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cuda.so";
  CopyFile(device_cuda_src_path, device_cuda_dest_path, 0, true);
  auto drivers = Drivers::GetInstance();
  drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-cuda.so");
  auto dev_mgr = DeviceManager::GetInstance();
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  dev_mgr->Clear();
  dev_mgr->Initialize(drivers, config);
  auto device_cuda = dev_mgr->CreateDevice("cuda", "0");
  if (device_cuda == nullptr) {
    GTEST_SKIP();
  }

  MockBuffer buffer(device_cuda);
  buffer.Build({3 * sizeof(int)});

  auto device_cpu = dev_mgr->CreateDevice("cpu", "0");
  buffer.SetDelayedCopyDestinationDevice(device_cpu);
  EXPECT_EQ("cuda", buffer.GetDevice()->GetType());
  auto data = buffer.ConstData();
  EXPECT_NE(data, nullptr);
  EXPECT_EQ("cpu", buffer.GetDevice()->GetType());
  EXPECT_EQ({3 * sizeof(int)}, buffer.GetBytes());
}

}  // namespace modelbox