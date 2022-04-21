
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

#include "modelbox/port.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockflow.h"

namespace modelbox {

class PortTest : public testing::Test {
 public:
  PortTest() {}

 protected:
  std::shared_ptr<Node> node_;
  virtual void SetUp() {
    node_ = std::make_shared<Node>();
    node_->SetFlowUnitInfo("test_2_inputs_2_outputs", "cpu", "0", nullptr);
  };
  virtual void TearDown(){};
};

class InPortTest : public testing::Test {
 public:
  InPortTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(PortTest, Construct) {
  auto port = Port("In_1", node_);
  auto name = port.GetName();
  auto get_node = port.GetNode();
  EXPECT_EQ("In_1", name);
  EXPECT_EQ(node_, get_node);
}

TEST_F(InPortTest, GetDataCount) {
  auto port = std::make_shared<InPort>("In_1", nullptr);
  EXPECT_EQ(port->GetDataCount(), 0);

  {
    auto buffer = std::make_shared<Buffer>();
    port->Send(buffer);
    EXPECT_EQ(port->GetDataCount(), 1);
  }

  {
    auto buffer = std::make_shared<Buffer>();
    port->Send(buffer);
    EXPECT_EQ(port->GetDataCount(), 2);
  }

  auto notify_port =
      std::dynamic_pointer_cast<NotifyPort<Buffer, CustomCompare>>(port);
  notify_port->Recv();
  EXPECT_EQ(port->GetDataCount(), 1);
}

class EventPortTest : public testing::Test {
 public:
  EventPortTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(EventPortTest, Send_Recv) {
  EventPort event_port("test_event_port", nullptr);

  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  const int MAX_SEND_COUNT = 10;
  for (int i = 0; i < MAX_SEND_COUNT; i++) {
    EXPECT_EQ(event_port.Send(event), STATUS_OK);
  }

  FlowunitEventList events = nullptr;
  EXPECT_EQ(event_port.Recv(events), STATUS_OK);

  EXPECT_NE(events, nullptr);
  EXPECT_EQ(events->size(), MAX_SEND_COUNT);
  for (size_t i = 0; i < MAX_SEND_COUNT; i++) {
    EXPECT_EQ(events->at(i), event);
  }
}

TEST_F(EventPortTest, Empty) {
  EventPort event_port("test_event_port", nullptr);

  EXPECT_TRUE(event_port.Empty());

  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  const int MAX_SEND_COUNT = 10;
  for (int i = 0; i < MAX_SEND_COUNT; i++) {
    EXPECT_EQ(event_port.Send(event), STATUS_OK);
  }

  EXPECT_FALSE(event_port.Empty());

  FlowunitEventList events = nullptr;
  EXPECT_EQ(event_port.Recv(events), STATUS_OK);

  EXPECT_TRUE(event_port.Empty());
}

TEST_F(EventPortTest, GetPriority_SetPriority) {
  EventPort event_port("test_event_port", nullptr);

  EXPECT_EQ(event_port.GetPriority(), 0);
  event_port.SetPriority(10);
  EXPECT_EQ(event_port.GetPriority(), 10);
}

TEST_F(EventPortTest, NotifyPushEvent) {
  EventPort event_port("test_event_port", nullptr);

  bool flag = false;
  auto func = [&](bool no_need_flag) { flag = true; };

  EXPECT_TRUE(!flag);

  event_port.SetPushEventCallBack(func);
  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  EXPECT_EQ(event_port.Send(event), STATUS_OK);
  event_port.NotifyPushEvent();

  EXPECT_TRUE(flag);
}

TEST_F(EventPortTest, NotifyPopEvent) {
  EventPort event_port("test_event_port", nullptr);

  bool flag = false;
  auto func = [&]() { flag = true; };

  EXPECT_TRUE(!flag);

  event_port.SetPopEventCallBack(func);
  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  EXPECT_EQ(event_port.Send(event), STATUS_OK);
  event_port.NotifyPushEvent();

  EXPECT_TRUE(!flag);
  FlowunitEventList events = nullptr;
  EXPECT_EQ(event_port.Recv(events), STATUS_OK);
  event_port.NotifyPopEvent();

  EXPECT_TRUE(flag);
}

}  // namespace modelbox