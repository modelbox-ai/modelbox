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

#include "engine/common/data_hub.h"

#include <functional>
#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "modelbox/base/log.h"

namespace modelbox {

class DefaultDataHubTest : public testing::Test {
 public:
  DefaultDataHubTest() = default;

 protected:
  std::shared_ptr<Node> node_;
  void SetUp() override { node_ = std::make_shared<Node>(); };
  void TearDown() override{};
};

TEST_F(DefaultDataHubTest, AddPort) {
  DefaultDataHub data_hub;
  int priority = 0;
  auto port = std::make_shared<InPort>("input_1", node_);
  port->SetPriority(priority);
  auto priority_port = std::make_shared<PriorityPort>(port);

  EXPECT_EQ(data_hub.GetPortNum(), 0);

  auto status = data_hub.AddPort(priority_port);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_EQ(data_hub.GetPortNum(), 1);

  status = data_hub.AddPort(priority_port);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_EQ(data_hub.GetPortNum(), 1);
}

TEST_F(DefaultDataHubTest, SelectActivePort) {
  DefaultDataHub data_hub;
  int priority_0 = 2;
  auto port_0 = std::make_shared<InPort>("input_0", node_);
  port_0->SetPriority(priority_0);
  auto priority_port_0 = std::make_shared<PriorityPort>(port_0);

  int priority_1 = 1;
  auto port_1 = std::make_shared<InPort>("input_1", node_);
  port_1->SetPriority(priority_1);
  auto priority_port_1 = std::make_shared<PriorityPort>(port_1);

  int priority_3 = 1;
  auto port_3 = std::make_shared<InPort>("input_2", node_);
  port_3->SetPriority(priority_3);
  auto priority_port_3 = std::make_shared<PriorityPort>(port_3);
  EXPECT_EQ(data_hub.GetPortNum(), 0);

  auto status = data_hub.AddPort(priority_port_0);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_EQ(data_hub.GetPortNum(), 1);

  status = data_hub.AddPort(priority_port_1);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_EQ(data_hub.GetPortNum(), 2);

  status = data_hub.AddPort(priority_port_3);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_EQ(data_hub.GetPortNum(), 3);

  auto buffer_1_0 = std::make_shared<Buffer>();
  BufferManageView::SetPriority(buffer_1_0, priority_1);
  auto in_port_1 =
      std::dynamic_pointer_cast<InPort>(priority_port_1->GetPort());
  in_port_1->GetQueue()->Push(buffer_1_0);
  in_port_1->NotifyPushEvent();
  auto in_port_0 =
      std::dynamic_pointer_cast<InPort>(priority_port_0->GetPort());
  auto buffer_0_0 = std::make_shared<Buffer>();
  BufferManageView::SetPriority(buffer_0_0, priority_0);
  in_port_0->GetQueue()->Push(buffer_0_0);
  in_port_0->NotifyPushEvent();
  auto in_port_3 =
      std::dynamic_pointer_cast<InPort>(priority_port_3->GetPort());
  auto buffer_3_0 = std::make_shared<Buffer>();
  BufferManageView::SetPriority(buffer_3_0, priority_3);
  in_port_3->GetQueue()->Push(buffer_3_0);
  in_port_3->NotifyPushEvent();
  EXPECT_EQ(data_hub.GetActivePortNum(), 3);

  std::shared_ptr<PriorityPort> active_port = nullptr;
  status = data_hub.SelectActivePort(&active_port);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(active_port, priority_port_0);
  EXPECT_EQ(data_hub.GetActivePortNum(), 2);

  status = data_hub.SelectActivePort(&active_port);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(active_port, priority_port_1);
  EXPECT_EQ(data_hub.GetActivePortNum(), 1);

  status = data_hub.SelectActivePort(&active_port);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(active_port, priority_port_3);
  EXPECT_EQ(data_hub.GetActivePortNum(), 0);

  status = data_hub.SelectActivePort(&active_port, 1000);
  EXPECT_EQ(status, STATUS_TIMEDOUT);

  status = data_hub.SelectActivePort(&active_port, -1);
  EXPECT_EQ(status, STATUS_NODATA);

  data_hub.AddToActivePort(priority_port_0);
  priority_port_0->GetPort()->NotifyPushEvent();
  priority_port_0->GetPort()->NotifyPopEvent();
  EXPECT_EQ(data_hub.GetActivePortNum(), 1);
}

TEST_F(DefaultDataHubTest, InactivePort) {
  DefaultDataHub data_hub;
  int priority_0 = 2;
  auto port_0 = std::make_shared<InPort>("input_0", node_);
  port_0->SetPriority(priority_0);
  auto priority_port_0 = std::make_shared<PriorityPort>(port_0);

  int priority_1 = 1;
  auto port_1 = std::make_shared<InPort>("input_1", node_);
  port_1->SetPriority(priority_1);
  auto priority_port_1 = std::make_shared<PriorityPort>(port_1);

  int priority_2 = 1;
  auto port_2 = std::make_shared<InPort>("input_2", node_);
  port_2->SetPriority(priority_2);
  auto priority_port_2 = std::make_shared<PriorityPort>(port_2);
  EXPECT_EQ(data_hub.GetPortNum(), 0);

  data_hub.AddPort(priority_port_0);
  data_hub.AddPort(priority_port_1);
  data_hub.AddPort(priority_port_2);

  EXPECT_EQ(data_hub.GetPortNum(), 3);

  auto buffer = std::make_shared<Buffer>();
  BufferManageView::SetPriority(buffer, priority_1);
  auto in_port_0 =
      std::dynamic_pointer_cast<InPort>(priority_port_0->GetPort());
  in_port_0->GetQueue()->Push(buffer);
  in_port_0->SetActiveState(false);
  in_port_0->NotifyPushEvent();

  std::shared_ptr<PriorityPort> active_port = nullptr;
  auto status = data_hub.SelectActivePort(&active_port, 1000);
  EXPECT_EQ(data_hub.GetActivePortNum(), 0);

  in_port_0->SetActiveState(true);
  in_port_0->NotifyPushEvent();

  status = data_hub.SelectActivePort(&active_port);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(active_port, priority_port_0);
  EXPECT_EQ(data_hub.GetActivePortNum(), 0);

  auto in_port_1 =
      std::dynamic_pointer_cast<InPort>(priority_port_1->GetPort());
  in_port_1->GetQueue()->Push(buffer);
  in_port_1->SetActiveState(false);
  in_port_1->NotifyPushEvent();

  auto in_port_2 =
      std::dynamic_pointer_cast<InPort>(priority_port_2->GetPort());
  in_port_2->GetQueue()->Push(buffer);
  in_port_2->NotifyPushEvent();

  status = data_hub.SelectActivePort(&active_port);
  EXPECT_EQ(status, STATUS_OK);
  EXPECT_EQ(active_port, priority_port_2);
  EXPECT_EQ(data_hub.GetActivePortNum(), 0);
}

}  // namespace modelbox