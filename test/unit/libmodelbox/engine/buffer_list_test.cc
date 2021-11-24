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


#include <functional>
#include <future>
#include <thread>

#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "gmock/gmock.h"
#include "mock_driver_ctl.h"

#include "modelbox/base/log.h"

#include "modelbox/buffer_list.h"
#include "gtest/gtest.h"

namespace modelbox {
class BufferListTest : public testing::Test {
 public:
  BufferListTest() {}

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

TEST_F(BufferListTest, BufferList) {
  {
    BufferList buffer_list(device_);
    EXPECT_EQ(buffer_list.GetBytes(), 0);
  }
}

TEST_F(BufferListTest, Build) {
  BufferList buffer_list(device_);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});

  std::vector<size_t> lengths(shapes.size());
  std::transform(shapes.begin(), shapes.end(), lengths.begin(),
                 [](const std::vector<size_t> &shape) -> size_t {
                   return Volume(shape) * sizeof(int);
                 });
  auto status = buffer_list.Build(lengths);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(buffer_list.Size(), shapes.size());

  size_t size = BATCH_NUM * Volume(shapes[0]);
  auto data = (int *)buffer_list.MutableData();
  for (size_t i = 0; i < size; ++i) {
    data[i] = i;
  }

  for (size_t i = 0; i < buffer_list.Size(); ++i) {
    auto buffer = buffer_list[i];
    auto tensor_data = (int *)(buffer->ConstData());
    auto tensor_size = buffer->GetBytes() / sizeof(int);
    for (size_t j = 0; j < tensor_size; ++j) {
      EXPECT_EQ(tensor_data[j], i * tensor_size + j);
    }
  }

  BufferList buffer_list_2(device_);
  std::vector<size_t> lengths_2(BATCH_NUM, 0);
  status = buffer_list_2.Build(lengths_2);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(buffer_list_2.Size(), lengths_2.size());
  std::vector<int *> data_list;
  for (size_t i = 0; i < buffer_list_2.Size(); ++i) {
    EXPECT_EQ(nullptr, buffer_list_2.ConstBufferData(i));

    auto data = new int[6];
    buffer_list_2[i]->Build(data, 6 * sizeof(int),
                            [](void *ptr) { delete[](int *) ptr; });
    data_list.push_back(data);
  }

  for (size_t i = 0; i < buffer_list_2.Size(); ++i) {
    EXPECT_EQ(data_list[i], buffer_list_2.ConstBufferData(i));
  }
}

TEST_F(BufferListTest, Get) {
  BufferList buffer_list(device_);
  buffer_list.Build({10, 100});
  buffer_list.Set("Height", 720);
  buffer_list.Set("Width", 1280);

  buffer_list.Set("PTS", 10);
  buffer_list.Set("FPS", 30.1f);

  int i_value = 0;
  float f_valud = 0.0;
  for (size_t i = 0; i < buffer_list.Size(); ++i) {
    EXPECT_TRUE(buffer_list[i]->Get("Height", i_value));
    EXPECT_EQ(i_value, 720);

    EXPECT_TRUE(buffer_list[i]->Get("Width", i_value));
    EXPECT_EQ(i_value, 1280);

    EXPECT_TRUE(buffer_list[i]->Get("PTS", i_value));
    EXPECT_EQ(i_value, 10);

    EXPECT_TRUE(buffer_list[i]->Get("FPS", f_valud));
    EXPECT_EQ(f_valud, 30.1f);
  }

  std::shared_ptr<BufferList> bl_ptr(&buffer_list, [](void *p) {});

  BufferList buffer_list2(device_);
  buffer_list2.Build({10, 100});
  buffer_list2.CopyMeta(bl_ptr);

  buffer_list2.Set("Height", 360);

  for (size_t i = 0; i < buffer_list2.Size(); ++i) {
    EXPECT_TRUE(buffer_list[i]->Get("Height", i_value));
    EXPECT_EQ(i_value, 720);

    EXPECT_TRUE(buffer_list2[i]->Get("Height", i_value));
    EXPECT_EQ(i_value, 360);
  }
}

}  // namespace modelbox
