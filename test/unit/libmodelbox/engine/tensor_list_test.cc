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


#include "modelbox/tensor_list.h"

#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"

namespace modelbox {
class TensorListTest : public testing::Test {
 public:
  TensorListTest() = default;

 protected:
  std::shared_ptr<MockFlow> flow_;
  void SetUp() override {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };

  void TearDown() override { flow_->Destroy(); };
};

TEST_F(TensorListTest, TensorList) {
  auto device = flow_->GetDevice();

  {
    auto bl = std::make_shared<BufferList>(device);
    TensorList tensor_list(bl);
    EXPECT_EQ(tensor_list.GetBytes(), 0);
  }
}

TEST_F(TensorListTest, TensorListBuild) {
  auto device = flow_->GetDevice();
  auto bl = std::make_shared<BufferList>(device);
  TensorList tensor_list(bl);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});
  auto status = tensor_list.Build<int>(shapes);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(tensor_list.Size(), shapes.size());

  size_t size = BATCH_NUM * Volume(shapes[0]);
  auto *data = tensor_list.MutableData<int>();
  for (size_t i = 0; i < size; ++i) {
    data[i] = i;
  }

  for (size_t i = 0; i < tensor_list.Size(); ++i) {
    auto tensor_buffer = tensor_list[i];
    EXPECT_EQ(tensor_list[i]->Shape(), shapes[i]);
    const auto *tensor_data = tensor_list[i]->ConstData<int>();
    auto tensor_size = tensor_list[i]->GetBytes() / sizeof(int);
    for (size_t j = 0; j < tensor_size; ++j) {
      EXPECT_EQ(tensor_data[j], i * tensor_size + j);
    }
  }
}

TEST_F(TensorListTest, TensorListBuildFromHost) {
  auto device = flow_->GetDevice();
  auto bl = std::make_shared<BufferList>(device);
  TensorList tensor_list(bl);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});
  auto size = Volume(shapes);
  auto *data = (int *)malloc(size * sizeof(int));
  Defer {
    if (data) {
      free(data);
    }
  };
  EXPECT_NE(data, nullptr);
  for (size_t i = 0; i < size; ++i) {
    data[i] = i;
  }

  auto status =
      tensor_list.BuildFromHost<int>(shapes, data, size * sizeof(int));
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(tensor_list.Size(), shapes.size());

  for (size_t i = 0; i < tensor_list.Size(); ++i) {
    EXPECT_EQ(tensor_list[i]->Shape(), shapes[i]);
    const auto *tensor_data = tensor_list[i]->ConstData<int>();
    auto tensor_size = tensor_list[i]->GetBytes() / sizeof(int);
    for (size_t j = 0; j < tensor_size; ++j) {
      EXPECT_EQ(tensor_data[j], i * tensor_size + j);
    }
  }
}

TEST_F(TensorListTest, SetShape) {
  auto device = flow_->GetDevice();
  auto bl = std::make_shared<BufferList>(device);
  TensorList tensor_list(bl);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});
  auto status = tensor_list.Build<int>(shapes);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(tensor_list.Size(), shapes.size());

  status = tensor_list.SetShape<int>({BATCH_NUM, {3, 2, 1}});
  EXPECT_EQ(status, STATUS_OK);

  status = tensor_list.SetShape<float>({BATCH_NUM, {3, 2, 1}});
  EXPECT_NE(status, STATUS_OK);

  status = tensor_list.SetShape<int>({BATCH_NUM, {3, 2, 2}});
  EXPECT_NE(status, STATUS_OK);

  status = tensor_list.SetShape<int>({BATCH_NUM - 1, {3, 2, 1}});
  EXPECT_NE(status, STATUS_OK);
}

TEST_F(TensorListTest, Shape) {
  auto device = flow_->GetDevice();
  auto bl = std::make_shared<BufferList>(device);
  TensorList tensor_list(bl);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});
  auto status = tensor_list.Build<int>(shapes);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(tensor_list.Size(), shapes.size());

  for (size_t i = 0; i < tensor_list.Size(); ++i) {
    EXPECT_EQ(tensor_list[i]->Shape(), shapes[i]);
  }
}

TEST_F(TensorListTest, SetType) {
  auto device = flow_->GetDevice();
  auto bl = std::make_shared<BufferList>(device);
  TensorList tensor_list(bl);

  const int BATCH_NUM = 10;
  std::vector<std::vector<size_t>> shapes(BATCH_NUM, {1, 2, 3});
  auto status = tensor_list.Build<int>(shapes);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(tensor_list.Size(), shapes.size());

  for (size_t i = 0; i < tensor_list.Size(); ++i) {
    EXPECT_EQ(tensor_list[i]->Shape(), shapes[i]);
  }
}

class TensorBufferTest : public testing::Test {
 public:
  TensorBufferTest() = default;

 protected:
  std::shared_ptr<MockFlow> flow_;
  void SetUp() override {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };

  void TearDown() override { flow_->Destroy(); };
};

TEST_F(TensorBufferTest, TensorBuffer) {
  auto device = flow_->GetDevice();
  TensorBuffer tensor(device);

  constexpr int DATA_SIZE = 10;
  std::vector<int> data(DATA_SIZE, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  tensor.Build(data.data(), data.size() * sizeof(int), [](void *ptr) {});
  tensor.SetShape<int>({2, 5});
  tensor.Set("Height", 720);
  tensor.Set("Width", 1280);

  TensorBuffer tensor2(tensor);
  EXPECT_EQ(tensor.MutableData<int>(), tensor2.MutableData<int>());
  EXPECT_EQ(tensor.Shape(), tensor2.Shape());
  EXPECT_EQ(tensor.GetType(), tensor2.GetType());

  int tensor_value = 0;
  int tensor2_value = -1;
  tensor.Get("Height", tensor_value);
  tensor2.Get("Height", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);

  tensor_value = 0;
  tensor2_value = -1;
  tensor.Get("Width", tensor_value);
  tensor2.Get("Width", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);
}

TEST_F(TensorBufferTest, Copy) {
  auto device = flow_->GetDevice();
  TensorBuffer tensor(device);

  constexpr int DATA_SIZE = 10;
  std::vector<int> data(DATA_SIZE, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  tensor.Build(data.data(), data.size() * sizeof(int), [](void *ptr) {});
  tensor.SetShape<int>({2, 5});
  tensor.Set("Height", 720);
  tensor.Set("Width", 1280);

  auto buffer = tensor.Copy();
  auto tensor2 = std::dynamic_pointer_cast<TensorBuffer>(buffer);
  EXPECT_NE(nullptr, tensor2);

  EXPECT_EQ(tensor.MutableData<int>(), tensor2->MutableData<int>());
  EXPECT_EQ(tensor.Shape(), tensor2->Shape());
  EXPECT_EQ(tensor.GetType(), tensor2->GetType());

  int tensor_value = 0;
  int tensor2_value = -1;
  tensor.Get("Height", tensor_value);
  tensor2->Get("Height", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);

  tensor_value = 0;
  tensor2_value = -1;
  tensor.Get("Width", tensor_value);
  tensor2->Get("Width", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);
}

TEST_F(TensorBufferTest, DeepCopy) {
  auto device = flow_->GetDevice();
  TensorBuffer tensor(device);

  constexpr int DATA_SIZE = 10;
  std::vector<int> data(DATA_SIZE, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  tensor.Build(data.data(), data.size() * sizeof(int), [](void *ptr) {});
  tensor.SetShape<int>({2, 5});
  tensor.Set("Height", 720);
  tensor.Set("Width", 1280);

  auto buffer = tensor.DeepCopy();
  auto tensor2 = std::dynamic_pointer_cast<TensorBuffer>(buffer);
  EXPECT_NE(nullptr, tensor2);

  EXPECT_EQ(tensor.Shape(), tensor2->Shape());
  EXPECT_EQ(tensor.GetType(), tensor2->GetType());

  int tensor_value = 0;
  int tensor2_value = -1;
  tensor.Get("Height", tensor_value);
  tensor2->Get("Height", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);

  tensor_value = 0;
  tensor2_value = -1;
  tensor.Get("Width", tensor_value);
  tensor2->Get("Width", tensor2_value);
  EXPECT_EQ(tensor_value, tensor2_value);

  auto *buf_data = tensor.MutableData<int>();
  auto *buf_data2 = tensor2->MutableData<int>();
  EXPECT_NE(buf_data, buf_data2);

  EXPECT_EQ(tensor.GetBytes(), tensor2->GetBytes());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(buf_data[i], data[i]);
    EXPECT_EQ(buf_data2[i], data[i]);
  }
}

}  // namespace modelbox
