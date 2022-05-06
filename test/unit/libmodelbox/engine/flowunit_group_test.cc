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

#include "modelbox/flowunit_group.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/log.h"
#include "modelbox/data_context.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "modelbox/flowunit.h"

using ::testing::_;

namespace modelbox {

std::shared_ptr<FlowUnitDataContext> BuildFlowUnitDataContext(
    size_t size, int& begin_data, Node* node, std::shared_ptr<Device> device) {
  auto stream_data_map = std::make_shared<PortDataMap>();
  const auto& input_ports = node->GetInputPorts();
  for (const auto& in_port : input_ports) {
    auto& data_list = (*stream_data_map)[in_port->GetName()];
    for (size_t i = 0; i < size; ++i) {
      auto buffer = std::make_shared<Buffer>(device);
      buffer->Build(sizeof(int32_t));
      auto ptr = (int32_t*)buffer->MutableData();
      *ptr = begin_data + i;
      data_list.push_back(buffer);
    }
  }
  begin_data += size;
  auto data_ctx =
      std::make_shared<NormalFlowUnitDataContext>(node, nullptr, nullptr);
  data_ctx->WriteInputData(stream_data_map);
  return data_ctx;
}

void PrintDataContext(const std::shared_ptr<FlowUnitDataContext>& ctx) {
  const auto& input = ctx->GetInputs();
  for (const auto& in : input) {
    MBLOG_DEBUG << in.first;
    for (auto& data : in.second) {
      MBLOG_DEBUG << *((int*)data->ConstData());
    }
  }
}

template <typename FuncImpl>
void CheckDataContext(const std::shared_ptr<FlowUnitDataContext>& ctx,
                      Node* node, FuncImpl func) {
  const auto& outputs = ctx->GetOutputs();
  for (const auto& out : outputs) {
    auto buffer_list = out.second;
    for (auto& data : *buffer_list) {
      EXPECT_TRUE(func(*(int*)data->ConstData()));
    }
  }
}

class FlowUnitGroupTest : public testing::Test {
 public:
  FlowUnitGroupTest() {}

 protected:
  std::shared_ptr<MockFlow> flow_;
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };

  virtual void TearDown() { flow_->Destroy(); };
};

TEST_F(FlowUnitGroupTest, Run2_In_1) {
  auto device_ = flow_->GetDevice();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "3");
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>();
  node_->SetFlowUnitInfo("iflow_add_1", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(node_->Init({"In_1"}, {"Out_1"}, config), STATUS_OK);

  size_t fug_size = 10;
  int data = 0;
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;
  for (size_t i = 0; i < fug_size; i++) {
    data_ctx_list.push_back(
        BuildFlowUnitDataContext(i + 1, data, node_.get(), device_));
  }

  FlowUnitGroup fug("iflow_add_1", "cpu", "0", config, nullptr);
  fug.Init({"In_1"}, {"Out_1"}, flowunit_mgr_);
  fug.SetNode(node_);
  fug.Open([](std::shared_ptr<Device>) -> std::shared_ptr<ExternalData> {
    return nullptr;
  });
  fug.Run(data_ctx_list);

  int check_data = 0;
  for (const auto& ctx : data_ctx_list) {
    CheckDataContext(ctx, node_.get(), [&](int data) -> bool {
      return data == (1 + check_data++);
    });
  }
}

TEST_F(FlowUnitGroupTest, Run2_In_2) {
  auto device_ = flow_->GetDevice();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "3");
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>();
  node_->SetFlowUnitInfo("add", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(node_->Init({"In_1", "In_2"}, {"Out_1"}, config), STATUS_OK);

  size_t fug_size = 10;
  int data = 0;
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;

  for (size_t i = 0; i < fug_size; i++) {
    data_ctx_list.push_back(
        BuildFlowUnitDataContext(i + 1, data, node_.get(), device_));
  }

  FlowUnitGroup fug("add", "cpu", "0", config, nullptr);
  fug.Init({"In_1", "In_2"}, {"Out_1"}, flowunit_mgr_);
  fug.SetNode(node_);
  fug.Open([](std::shared_ptr<Device>) -> std::shared_ptr<ExternalData> {
    return nullptr;
  });
  fug.Run(data_ctx_list);

  int check_data = 0;
  for (const auto& ctx : data_ctx_list) {
    CheckDataContext(ctx, node_.get(), [&](int data) -> bool {
      MBLOG_DEBUG << data << " = " << check_data << " + " << check_data;
      return data == (2 * check_data++);
    });
  }
}

TEST_F(FlowUnitGroupTest, Run2_Status_Error) {
  auto device_ = flow_->GetDevice();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "3");
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>();
  node_->SetFlowUnitInfo("add_1_and_error", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(node_->Init({"In_1"}, {"Out_1"}, config), STATUS_OK);

  size_t fug_size = 10;
  int data = 0;
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;

  for (size_t i = 0; i < fug_size; i++) {
    data_ctx_list.push_back(
        BuildFlowUnitDataContext(i + 1, data, node_.get(), device_));
  }

  FlowUnitGroup fug("add_1_and_error", "cpu", "0", config, nullptr);
  fug.Init({"In_1"}, {"Out_1"}, flowunit_mgr_);
  fug.SetNode(node_);
  fug.Open([](std::shared_ptr<Device>) -> std::shared_ptr<ExternalData> {
    return nullptr;
  });
  fug.Run(data_ctx_list);

  int check_data = 0;
  int idx = 0;

  auto func = [&](int data) -> bool {
    MBLOG_DEBUG << data << " = " << 1 << " + " << check_data;
    return data == (1 + check_data++);
  };

  for (const auto& ctx : data_ctx_list) {
    const auto& outputs = ctx->GetOutputs();

    for (const auto& out : outputs) {
      auto buffer_list = out.second;

      for (auto& data : *buffer_list) {
        if (data->HasError()) {
          EXPECT_EQ(data->ConstData(), nullptr);
          ++check_data;
          continue;
        }

        EXPECT_TRUE(func(*(int*)data->ConstData()));
      }
    }

    idx++;
  }
}

TEST_F(FlowUnitGroupTest, Run2_Condition) {
  auto device_ = flow_->GetDevice();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "1");
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>();
  node_->SetFlowUnitInfo("test_condition", "cpu", "0", flowunit_mgr_);

  EXPECT_EQ(node_->Init({"In_1"}, {"Out_1", "Out_2"}, config), STATUS_OK);

  size_t fug_size = 10;
  int data = 0;
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;

  for (size_t i = 0; i < fug_size; i++) {
    data_ctx_list.push_back(
        BuildFlowUnitDataContext(i + 1, data, node_.get(), device_));
  }

  FlowUnitGroup fug("test_condition", "cpu", "0", config, nullptr);
  fug.Init({"In_1"}, {"Out_1", "Out_2"}, flowunit_mgr_);
  fug.SetNode(node_);
  fug.Open([](std::shared_ptr<Device>) -> std::shared_ptr<ExternalData> {
    return nullptr;
  });
  fug.Run(data_ctx_list);

  int check_data = 0;
  int idx = 0;

  for (const auto& ctx : data_ctx_list) {
    EXPECT_FALSE(ctx->HasError());
    const auto& outputs = ctx->GetOutputs();
    const auto& output_1 = outputs.at("Out_1");
    const auto& output_2 = outputs.at("Out_2");

    EXPECT_EQ(output_1->Size(), output_2->Size());
    for (size_t i = 0; i < output_1->Size(); ++i) {
      if ((idx * (idx + 1) / 2 + i) % 2 == 0 &&
          (idx * (idx + 1) / 2 + i) != 10) {
        if (output_1->At(i)->HasError()) {
          EXPECT_EQ(output_1->ConstBufferData(i), nullptr);
          ++check_data;
          continue;
        }
        EXPECT_NE(output_1->At(i), nullptr);
        auto data = (int*)output_1->ConstBufferData(i);
        EXPECT_EQ(*data, check_data++);
      } else {
        if (output_2->At(i) == nullptr) {
          ++check_data;
          continue;
        }
        EXPECT_NE(output_2->At(i), nullptr);
        if (!output_2->At(i)->HasError()) {
          auto data = (int*)output_2->ConstBufferData(i);
          EXPECT_EQ(*data, check_data);
        }
        ++check_data;
      }
    }

    idx++;
  }
}

TEST_F(FlowUnitGroupTest, Init) {
  ConfigurationBuilder configbuilder;
  auto flowunit_mgr = FlowUnitManager::GetInstance();

  auto config = configbuilder.Build();
  auto valid_flg =
      std::make_shared<FlowUnitGroup>("test_0_2", "cpu", "0", config, nullptr);
  EXPECT_EQ(valid_flg->Init({}, {"Out_1"}, flowunit_mgr), STATUS_BADCONF);
  EXPECT_EQ(valid_flg->Init({}, {"Out_1", "Out_2"}, flowunit_mgr),
            STATUS_SUCCESS);
  auto invalid_flg = std::make_shared<FlowUnitGroup>("invalid_test", "cpu", "0",
                                                     config, nullptr);
  EXPECT_EQ(invalid_flg->Init({}, {"Out_1", "Out_2"}, flowunit_mgr),
            STATUS_NOTFOUND);
}

TEST_F(FlowUnitGroupTest, Open_Close) {
  ConfigurationBuilder configbuilder;
  auto flowunit_mgr = FlowUnitManager::GetInstance();

  bool flag = false;

  auto func_2 =
      [&](std::shared_ptr<Device> event) -> std::shared_ptr<ExternalData> {
    flag = true;
    return nullptr;
  };

  {
    auto config = configbuilder.Build();
    FlowUnitGroup fug("listen", "cpu", "0", config, nullptr);
    EXPECT_TRUE(fug.Init({}, {"Out_1", "Out_2"}, flowunit_mgr));
    auto flowunit = fug.GetExecutorUnit();
    EXPECT_EQ(fug.Open(func_2), STATUS_OK);
    EXPECT_EQ(fug.Close(), STATUS_OK);
  }

  {
    auto config = configbuilder.Build();
    FlowUnitGroup fug("listen", "cpu", "0", config, nullptr);
    EXPECT_TRUE(fug.Init({}, {"Out_1", "Out_2"}, flowunit_mgr));
    auto flowunit = fug.GetExecutorUnit();
    EXPECT_EQ(fug.Open(nullptr), STATUS_OK);
    EXPECT_EQ(fug.Close(), STATUS_OK);
  }
}

}  // namespace modelbox
