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

#include <modelbox/flowunit_balancer.h>
#include <modelbox/node.h>
#include <modelbox/session.h>

#include <chrono>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {

class BalancerMockFlowUnit : public FlowUnit {
 public:
  Status Open(const std::shared_ptr<Configuration> &config) override {
    return STATUS_OK;
  }

  Status Close() override { return STATUS_OK; }

  Status Process(std::shared_ptr<DataContext> data_ctx) override {
    return STATUS_OK;
  }
};

class BalancerMockDevice : public Device {
 public:
  MOCK_CONST_METHOD0(GetDeviceID, std::string());
};

class BalancerMockMemory : public DeviceMemory {
 public:
  BalancerMockMemory(const std::shared_ptr<Device> &device,
                     const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                     const std::shared_ptr<void> &device_mem_ptr, size_t size)
      : DeviceMemory(device, mem_mgr, device_mem_ptr, size) {}
};

class FlowUnitBalancerTest : public testing::Test {
 public:
  std::shared_ptr<FlowUnitDataContext> BuildFlowUnitDataContext(
      Node *node, const std::shared_ptr<DeviceMemory> &mem) {
    auto data_ctx =
        std::make_shared<NormalFlowUnitDataContext>(node, nullptr, nullptr);
    auto stream_data_map = std::make_shared<PortDataMap>();
    auto &buffer_list = (*stream_data_map)["test_port"];
    buffer_list.push_back(std::make_shared<Buffer>(mem));
    data_ctx->WriteInputData(stream_data_map);
    return data_ctx;
  }

  std::vector<std::shared_ptr<Device>> CreateDevices(size_t count) {
    std::vector<std::shared_ptr<Device>> devices;
    devices.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto device = std::make_shared<BalancerMockDevice>();
      EXPECT_CALL(*device, GetDeviceID())
          .WillOnce(testing::Return(std::to_string(i)));
      devices.push_back(device);
    }

    return devices;
  }

  std::vector<std::shared_ptr<FlowUnit>> CreateFlowUnits(
      size_t count, const std::vector<std::shared_ptr<Device>> &devices) {
    std::vector<std::shared_ptr<FlowUnit>> flowunits;
    flowunits.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto fu = std::make_shared<BalancerMockFlowUnit>();
      if (i >= devices.size()) {
        fu->SetBindDevice(devices.back());
      } else {
        fu->SetBindDevice(devices[i]);
      }

      flowunits.push_back(fu);
    }

    return flowunits;
  }

  std::vector<std::shared_ptr<DeviceMemory>> CreateMems(
      size_t count, const std::vector<std::shared_ptr<Device>> &devices) {
    std::vector<std::shared_ptr<DeviceMemory>> mems;
    mems.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      std::shared_ptr<Device> device;
      if (i >= devices.size()) {
        device = devices.back();
      } else {
        device = devices[i];
      }

      mems.push_back(
          std::make_shared<BalancerMockMemory>(device, nullptr, nullptr, 0));
    }

    return mems;
  }

 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class MockBalancer : public FlowUnitBalancer {
 public:
  MOCK_METHOD0(GetType, FlowUnitBalanceStrategy());
  MOCK_METHOD0(OnInit, Status());
  MOCK_METHOD1(BindFlowUnit, std::shared_ptr<FlowUnit>(
                                 const std::shared_ptr<FlowUnitDataContext> &));
};

TEST_F(FlowUnitBalancerTest, BalancerFactoryTest) {
  auto mock_balancer = std::make_shared<MockBalancer>();
  EXPECT_CALL(*mock_balancer, GetType())
      .WillOnce(testing::Return(FlowUnitBalanceStrategy::FU_NULL));
  EXPECT_CALL(*mock_balancer, OnInit()).WillOnce(testing::Return(STATUS_OK));
  EXPECT_CALL(*mock_balancer, BindFlowUnit(testing::_))
      .WillOnce(testing::Return(nullptr));

  auto create_func = [mock_balancer]() -> std::shared_ptr<FlowUnitBalancer> {
    return mock_balancer;
  };

  auto factory = FlowUnitBalancerFactory::GetInstance();
  factory.RegistBalancer(create_func);
  auto balancer = factory.CreateBalancer(FlowUnitBalanceStrategy::FU_NULL);
  EXPECT_EQ(mock_balancer, balancer);
  std::vector<std::shared_ptr<FlowUnit>> flowunits;
  EXPECT_EQ(balancer->Init(flowunits), STATUS_FAULT);
  flowunits.push_back(nullptr);
  EXPECT_EQ(balancer->Init(flowunits), STATUS_OK);
  EXPECT_EQ(balancer->GetFlowUnit(nullptr), nullptr);
}

TEST_F(FlowUnitBalancerTest, RoundRobinTest) {
  auto balancer = FlowUnitBalancerFactory::GetInstance().CreateBalancer(
      FlowUnitBalanceStrategy::FU_ROUND_ROBIN);
  ASSERT_NE(balancer, nullptr);
  EXPECT_EQ(balancer->GetType(), FlowUnitBalanceStrategy::FU_ROUND_ROBIN);
  auto devices = CreateDevices(3);
  EXPECT_EQ(devices[2]->GetDeviceID(), "2");
  auto flowunits = CreateFlowUnits(2, devices);
  balancer->Init(flowunits);
  auto mems = CreateMems(3, devices);
  auto node = std::make_shared<Node>();
  {
    auto ctx1 = BuildFlowUnitDataContext(node.get(), mems[1]);
    auto ctx2 = BuildFlowUnitDataContext(node.get(), mems[0]);
    auto ctx3 = BuildFlowUnitDataContext(node.get(), mems[2]);
    auto ctx4 = BuildFlowUnitDataContext(node.get(), mems[2]);
    // first round
    auto get_fu = balancer->GetFlowUnit(ctx1);
    EXPECT_EQ(get_fu, flowunits[1]);
    get_fu = balancer->GetFlowUnit(ctx2);
    EXPECT_EQ(get_fu, flowunits[0]);
    get_fu = balancer->GetFlowUnit(ctx3);
    EXPECT_EQ(get_fu, flowunits[0]);
    get_fu = balancer->GetFlowUnit(ctx4);
    EXPECT_EQ(get_fu, flowunits[1]);
    // second round
    get_fu = balancer->GetFlowUnit(ctx4);
    EXPECT_EQ(get_fu, flowunits[1]);
    get_fu = balancer->GetFlowUnit(ctx3);
    EXPECT_EQ(get_fu, flowunits[0]);
    get_fu = balancer->GetFlowUnit(ctx2);
    EXPECT_EQ(get_fu, flowunits[0]);
    get_fu = balancer->GetFlowUnit(ctx1);
    EXPECT_EQ(get_fu, flowunits[1]);
  }
  {
    auto ctx1 = BuildFlowUnitDataContext(node.get(), mems[0]);
    auto ctx2 = BuildFlowUnitDataContext(node.get(), mems[1]);
    auto ctx3 = BuildFlowUnitDataContext(node.get(), mems[2]);
    auto ctx4 = BuildFlowUnitDataContext(node.get(), mems[2]);
    // first round after ctx clear
    auto get_fu = balancer->GetFlowUnit(ctx4);
    EXPECT_EQ(get_fu, flowunits[0]);
    get_fu = balancer->GetFlowUnit(ctx3);
    EXPECT_EQ(get_fu, flowunits[1]);
    get_fu = balancer->GetFlowUnit(ctx2);
    EXPECT_EQ(get_fu, flowunits[1]);
    get_fu = balancer->GetFlowUnit(ctx1);
    EXPECT_EQ(get_fu, flowunits[0]);
  }
}

TEST_F(FlowUnitBalancerTest, RoundRobinPerfTest) {
  auto balancer = FlowUnitBalancerFactory::GetInstance().CreateBalancer(
      FlowUnitBalanceStrategy::FU_ROUND_ROBIN);
  ASSERT_NE(balancer, nullptr);
  EXPECT_EQ(balancer->GetType(), FlowUnitBalanceStrategy::FU_ROUND_ROBIN);
  auto devices = CreateDevices(101);
  EXPECT_EQ(devices.back()->GetDeviceID(), "100");
  auto flowunits = CreateFlowUnits(100, devices);
  auto mems = CreateMems(200, devices);
  std::vector<std::shared_ptr<FlowUnitDataContext>> ctx_list;
  auto node = std::make_shared<Node>();
  for (size_t i = 0; i < 200; ++i) {
    ctx_list.push_back(BuildFlowUnitDataContext(node.get(), mems[i]));
  }

  balancer->Init(flowunits);
  const size_t test_loop_count = 1000;
  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < test_loop_count; ++i) {
    for (auto &ctx : ctx_list) {
      balancer->GetFlowUnit(ctx);
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
  MBLOG_INFO << "[RoundRobin] flowunits: " << flowunits.size()
             << ", ctx:" << ctx_list.size()
             << ", avg cost:" << cost / test_loop_count << " microsec";
}

}  // namespace modelbox