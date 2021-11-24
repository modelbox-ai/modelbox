
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
  std::shared_ptr<MockFlow> flow_;
  std::shared_ptr<Node> node_;
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    node_ = std::make_shared<Node>("test_2_inputs_2_outputs", "cpu", "0",
                                   nullptr, nullptr);
  };
  virtual void TearDown() {
    node_ = nullptr;
    flow_->Destroy();
  };
};

class InPortTest : public testing::Test {
 public:
  InPortTest() {}

 protected:
  std::shared_ptr<MockFlow> flow_;

  std::shared_ptr<Node> node_;
  std::vector<std::shared_ptr<IndexBuffer>> root_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> root_vector_2_;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_2_;
  std::vector<std::shared_ptr<IndexBuffer>> other_buffer_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> other_buffer_vector_2_;

  std::shared_ptr<IndexBuffer> root_buffer_1_;
  std::shared_ptr<IndexBuffer> root_buffer_2_;
  std::shared_ptr<IndexBuffer> buffer_01_;
  std::shared_ptr<IndexBuffer> buffer_02_;
  std::shared_ptr<IndexBuffer> buffer_03_;
  std::shared_ptr<IndexBuffer> buffer_11_;
  std::shared_ptr<IndexBuffer> buffer_12_;
  std::shared_ptr<IndexBuffer> other_buffer_01_;
  std::shared_ptr<IndexBuffer> other_buffer_02_;
  std::shared_ptr<IndexBuffer> other_buffer_03_;

  std::shared_ptr<SingleMatch> single_match_;
  std::shared_ptr<StreamMatch> group_match_;
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
    auto flowunit_mgr = FlowUnitManager::GetInstance();
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    node_ = std::make_shared<Node>("test_2_inputs_2_outputs", "cpu", "0",
                                   flowunit_mgr, nullptr);

    std::set<std::string> input_port_names = {"In_1", "In_2"};
    std::set<std::string> output_port_names = {"Out_1", "Out_2"};
    auto status = node_->Init(input_port_names, output_port_names, config);

    single_match_ = node_->GetSingleMatchCache()->GetReceiveBuffer();
    group_match_ = node_->GetStreamMatchCache()->GetStreamReceiveBuffer();

    root_buffer_1_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindToRoot();

    root_buffer_2_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->CopyMetaTo(root_buffer_2_);

    buffer_01_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindDownLevelTo(buffer_01_, true, false);

    buffer_02_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindDownLevelTo(buffer_02_, false, false);

    buffer_03_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindDownLevelTo(buffer_03_, false, true);

    buffer_11_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindDownLevelTo(buffer_11_, true, false);

    buffer_12_ = std::make_shared<IndexBuffer>();
    root_buffer_1_->BindDownLevelTo(buffer_12_, false, true);

    other_buffer_01_ = std::make_shared<IndexBuffer>();
    root_buffer_2_->BindDownLevelTo(other_buffer_01_, true, false);

    other_buffer_02_ = std::make_shared<IndexBuffer>();
    root_buffer_2_->BindDownLevelTo(other_buffer_02_, false, false);

    other_buffer_03_ = std::make_shared<IndexBuffer>();
    root_buffer_2_->BindDownLevelTo(other_buffer_03_, false, true);

    root_vector_1_.push_back(root_buffer_1_);
    root_vector_2_.push_back(root_buffer_2_);

    buffer_vector_1_.push_back(buffer_03_);
    buffer_vector_1_.push_back(buffer_01_);
    buffer_vector_1_.push_back(buffer_11_);
    buffer_vector_1_.push_back(buffer_12_);

    buffer_vector_2_.push_back(buffer_11_);
    buffer_vector_2_.push_back(buffer_12_);
    buffer_vector_2_.push_back(buffer_01_);

    other_buffer_vector_1_.push_back(other_buffer_01_);
    other_buffer_vector_1_.push_back(other_buffer_02_);
    other_buffer_vector_1_.push_back(buffer_01_);
    other_buffer_vector_1_.push_back(buffer_03_);
    other_buffer_vector_1_.push_back(buffer_11_);
    other_buffer_vector_1_.push_back(buffer_12_);

    other_buffer_vector_2_.push_back(other_buffer_01_);
    other_buffer_vector_2_.push_back(other_buffer_03_);
    other_buffer_vector_2_.push_back(buffer_01_);
    other_buffer_vector_2_.push_back(buffer_03_);
    other_buffer_vector_2_.push_back(buffer_11_);
    other_buffer_vector_2_.push_back(buffer_12_);
  };
  virtual void TearDown() {
    node_ = nullptr;
    flow_->Destroy();
  };
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
    auto buffer = std::make_shared<IndexBuffer>();
    port->Send(buffer);
    EXPECT_EQ(port->GetDataCount(), 1);
  }

  {
    auto buffer = std::make_shared<IndexBuffer>();
    port->Send(buffer);
    EXPECT_EQ(port->GetDataCount(), 2);
  }

  auto notify_port =
      std::dynamic_pointer_cast<NotifyPort<IndexBuffer, CustomCompare>>(port);
  notify_port->Recv();
  EXPECT_EQ(port->GetDataCount(), 1);
}

TEST_F(InPortTest, RecvEmpty) {
  auto port_1 = node_->GetInputPort("In_1");
  auto port_2 = node_->GetInputPort("In_2");

  EXPECT_EQ(node_->ReceiveBuffer(port_1), STATUS_SUCCESS);
  EXPECT_EQ(0, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
  EXPECT_EQ(node_->ReceiveBuffer(port_2), STATUS_SUCCESS);
  EXPECT_EQ(0, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
}

TEST_F(InPortTest, RecvRoot) {
  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&root_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&root_vector_2_);

  EXPECT_EQ(node_->ReceiveBuffer(port_1), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(1, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
  EXPECT_EQ(node_->ReceiveBuffer(port_2), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(0, single_match_->size());
  EXPECT_EQ(1, group_match_->size());
}

TEST_F(InPortTest, RecvStream) {
  auto key_group_1 = std::make_tuple(buffer_01_->GetStreamLevelGroup(), 1);
  auto key_group_2 = std::make_tuple(buffer_11_->GetStreamLevelGroup(), 2);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&buffer_vector_2_);

  EXPECT_EQ(node_->ReceiveBuffer(port_1), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(4, single_match_->size());
  EXPECT_EQ(0, group_match_->size());

  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_01_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_03_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_11_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_12_->GetSameLevelGroup()));

  EXPECT_EQ(node_->ReceiveBuffer(port_2), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(1, single_match_->size());
  EXPECT_EQ(2, group_match_->size());

  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_03_->GetSameLevelGroup()));

  EXPECT_NE(group_match_->end(), group_match_->find(key_group_1));
  EXPECT_NE(group_match_->end(), group_match_->find(key_group_2));
  EXPECT_EQ(1, group_match_->find(key_group_1)->second.size());
  EXPECT_EQ(2, group_match_->find(key_group_2)->second.size());
}

TEST_F(InPortTest, RecvDiffrenceStream) {
  auto key_group_1 = std::make_tuple(buffer_01_->GetStreamLevelGroup(), 1);
  auto key_group_2 = std::make_tuple(buffer_11_->GetStreamLevelGroup(), 2);
  auto key_other_group_1 =
      std::make_tuple(other_buffer_01_->GetStreamLevelGroup(), 1);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&other_buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&other_buffer_vector_2_);

  EXPECT_EQ(node_->ReceiveBuffer(port_1), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(6, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
  EXPECT_NE(single_match_->end(),
            single_match_->find(other_buffer_01_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(other_buffer_02_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_01_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_03_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_11_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(buffer_12_->GetSameLevelGroup()));

  EXPECT_EQ(node_->ReceiveBuffer(port_2), STATUS_SUCCESS);
  node_->ReceiveGroupBuffer();
  EXPECT_EQ(2, single_match_->size());
  EXPECT_EQ(3, group_match_->size());
  EXPECT_NE(single_match_->end(),
            single_match_->find(other_buffer_02_->GetSameLevelGroup()));
  EXPECT_NE(single_match_->end(),
            single_match_->find(other_buffer_03_->GetSameLevelGroup()));

  EXPECT_NE(group_match_->end(), group_match_->find(key_group_1));
  EXPECT_NE(group_match_->end(), group_match_->find(key_group_2));
  EXPECT_NE(group_match_->end(), group_match_->find(key_other_group_1));

  EXPECT_EQ(2, group_match_->find(key_group_1)->second.size());
  EXPECT_EQ(2, group_match_->find(key_group_2)->second.size());
  EXPECT_EQ(1, group_match_->find(key_other_group_1)->second.size());
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