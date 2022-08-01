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

#include "engine/scheduler/flow_scheduler.h"

#include <functional>
#include <future>
#include <thread>

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockflow.h"
#include "modelbox/base/log.h"

namespace modelbox {

class FlowSchedulerTest : public testing::Test {
 public:
  FlowSchedulerTest() = default;

 protected:
  std::shared_ptr<MockFlow> flow_;

  void SetUp() override {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };

  void TearDown() override { flow_->Destroy(); };
};

class MockNode : public Node {
 public:
  MockNode() = default;
  MOCK_METHOD1(Run, Status(RunType type));
};

static SessionManager g_test_session_manager;

TEST_F(FlowSchedulerTest, ShowScheduleStatus) {
  auto device_ = flow_->GetDevice();

  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<Node> node_a = nullptr;
  std::shared_ptr<Node> node_b = nullptr;
  std::shared_ptr<Node> node_c = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("queue_size", "1");
    config->SetProperty("interval_time", 1000);
    config->SetProperty("queue_size_event", 1);

    node_a = std::make_shared<Node>();
    node_a->SetFlowUnitInfo("listen", "cpu", "0", flowunit_mgr);
    node_a->SetName("gendata");
    node_a->Init({}, {"Out_1", "Out_2"}, config);
    node_a->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_a));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("queue_size", "1");

    node_b = std::make_shared<Node>();
    node_b->SetFlowUnitInfo("tensorlist_test_1", "cpu", "0", flowunit_mgr);
    node_b->SetName("tensorlist_test_1");
    node_b->Init({"IN1"}, {"OUT1"}, config);
    node_b->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_b));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("queue_size", "1");

    config->SetProperty("max_count", 1);
    config->SetProperty("batch_size", 1);

    node_c = std::make_shared<Node>();
    node_c->SetFlowUnitInfo("slow", "cpu", "0", flowunit_mgr);
    node_c->SetName("slow");
    node_c->Init({"IN1", "IN2"}, {}, config);
    node_c->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_c));
  }

  graph->AddLink(node_a->GetName(), "Out_1", node_b->GetName(), "IN1");
  graph->AddLink(node_a->GetName(), "Out_2", node_c->GetName(), "IN1");
  graph->AddLink(node_b->GetName(), "OUT1", node_c->GetName(), "IN2");

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  auto scheduler = std::make_shared<FlowScheduler>();
  auto status = scheduler->Init(config);
  EXPECT_EQ(status, STATUS_OK);
  status = scheduler->Build(*graph);
  EXPECT_EQ(status, STATUS_OK);

  EXPECT_EQ(scheduler->GetCheckCount(), 0);

  scheduler->SetMaxCheckTimeoutCount(1);

  scheduler->RunAsync();

  auto scheduler_status = scheduler->Wait(3000, &status);
  EXPECT_GT(scheduler->GetCheckCount(), 0);
  MBLOG_INFO << "count: " << scheduler->GetCheckCount();
  EXPECT_EQ(scheduler_status, STATUS_TIMEDOUT);
  scheduler->Shutdown();
}

}  // namespace modelbox