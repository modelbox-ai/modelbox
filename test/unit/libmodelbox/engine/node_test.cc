
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

#include "modelbox/node.h"

#include <string>

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockflow.h"
#include "modelbox/data_context.h"
#include "modelbox/stream.h"

namespace modelbox {

using ::testing::Sequence;

void BuildDataEventStart(
    std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>&
        input_map,
    std::shared_ptr<Device> device) {}

void BuildDataEventStop(
    std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>&
        input_map) {}

void BuildDataQueue(
    Node* match_at,
    std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>&
        input_map,
    std::shared_ptr<Device> device) {
  auto session = std::make_shared<Session>(nullptr);
  auto init_stream = std::make_shared<Stream>(session);
  auto init_buffer_index_info = std::make_shared<BufferIndexInfo>();
  init_buffer_index_info->SetStream(init_stream);
  init_buffer_index_info->SetIndex(0);
  init_stream->IncreaseBufferCount();

  auto inherit_info = std::make_shared<BufferInheritInfo>();
  inherit_info->SetInheritFrom(init_buffer_index_info);
  inherit_info->SetType(BufferProcessType::EXPAND);

  std::vector<std::shared_ptr<Buffer>> p1_bl(1);
  std::vector<std::shared_ptr<Buffer>> p2_bl(1);

  std::vector<int> data_1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> data_2 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  auto buf_1 = std::make_shared<Buffer>(device);
  buf_1->Build(10 * sizeof(int));
  auto dev_data_1 = (int*)buf_1->MutableData();
  for (size_t i = 0; i < data_1.size(); ++i) {
    dev_data_1[i] = data_1[i];
  }
  auto b1_index = BufferManageView::GetIndexInfo(buf_1);
  auto b1_s1 = std::make_shared<Stream>(session);
  b1_index->SetStream(b1_s1);
  b1_index->SetIndex(0);
  b1_index->SetInheritInfo(inherit_info);
  p1_bl[0] = buf_1;

  auto buf_2 = std::make_shared<Buffer>(device);
  buf_2->Build(10 * sizeof(int));
  auto dev_data_2 = (int*)buf_2->MutableData();
  for (size_t i = 0; i < data_2.size(); ++i) {
    dev_data_2[i] = data_2[i];
  }
  auto b2_index = BufferManageView::GetIndexInfo(buf_2);
  auto b2_s1 = std::make_shared<Stream>(session);
  b2_index->SetStream(b2_s1);
  b2_index->SetIndex(0);
  b2_index->SetInheritInfo(inherit_info);
  p2_bl[0] = buf_2;

  input_map.emplace("In_1", p1_bl);
  input_map.emplace("In_2", p2_bl);
}

void CheckQueueHasDataError(std::shared_ptr<BufferQueue> queue,
                            uint32_t queue_size) {
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error_buffer_vector.size(), queue_size);
  EXPECT_NE(error, nullptr);
  queue->PushBatch(&error_buffer_vector);
}

void CheckQueueNotHasDataError(std::shared_ptr<BufferQueue> queue) {
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error, nullptr);
  queue->PushBatch(&error_buffer_vector);
}

std::shared_ptr<Buffer> CreateBuffer(size_t idx = 0,
                                     std::shared_ptr<Stream> stream = nullptr) {
  auto buffer = std::make_shared<Buffer>();
  auto input_index = BufferManageView::GetIndexInfo(buffer);
  if (stream == nullptr) {
    auto session = std::make_shared<Session>(nullptr);
    stream = std::make_shared<Stream>(session);
  }
  input_index->SetIndex(idx);
  input_index->SetStream(stream);
  stream->IncreaseBufferCount();
  return buffer;
}

class TestNode : public Node {
 public:
  Status Recv(
      RunType type,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) override {
    return Node::Recv(type, data_ctx_list);
  }

  Status GenInputMatchStreamData(RunType type,
                                 std::list<std::shared_ptr<MatchStreamData>>&
                                     match_stream_data_list) override {
    return Node::GenInputMatchStreamData(type, match_stream_data_list);
  }

  Status InitNodeProperties() override { return Node::InitNodeProperties(); }

  void SetInputInOrder(bool input_in_order) {
    input_match_stream_mgr_->SetInputBufferInOrder(input_in_order);
  }

  void SetInputGatherAll(bool input_gather_all) {
    input_match_stream_mgr_->SetInputStreamGatherAll(input_gather_all);
  }
};

class NodeTest : public testing::Test {
 public:
  NodeTest() {}

 protected:
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };
  virtual void TearDown() { flow_->Destroy(); };

 private:
  std::shared_ptr<MockFlow> flow_;
};

class NodeRecvTest : public testing::Test {
 public:
  NodeRecvTest() {}

 protected:
  std::shared_ptr<TestNode> node_;
  std::vector<std::shared_ptr<Buffer>> node1_input_;
  std::vector<std::shared_ptr<Buffer>> node2_input1_;
  std::vector<std::shared_ptr<Buffer>> node2_input1_end_;
  std::vector<std::shared_ptr<Buffer>> node2_input1_mismatch_;
  std::vector<std::shared_ptr<Buffer>> node2_input2_;
  std::vector<std::shared_ptr<Buffer>> node2_input2_end_;
  std::vector<std::shared_ptr<Buffer>> node2_input2_mismatch_;

  std::shared_ptr<Node> node1_;
  std::shared_ptr<Node> node2_;

  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
    auto flowunit_mgr = FlowUnitManager::GetInstance();

    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    node_ = std::make_shared<TestNode>();
    node_->SetFlowUnitInfo("test_2_inputs_2_outputs", "cpu", "0", flowunit_mgr);
    node_->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config);

    /**
     *                      -> node2_input1
     * node1_input -> node1                 -> node2
     *                      -> node2_input2
     *
     **/

    node1_ = std::make_shared<Node>();
    node2_ = std::make_shared<Node>();

    auto session = std::make_shared<Session>(nullptr);
    auto init_stream = std::make_shared<Stream>(session);
    auto root1 = std::make_shared<BufferIndexInfo>();
    root1->SetIndex(0);
    root1->SetStream(init_stream);
    auto root2 = std::make_shared<BufferIndexInfo>();
    root2->SetIndex(0);
    root2->SetStream(init_stream);

    // node1 p1 s1
    auto node1_input_s1 = std::make_shared<Stream>(session);
    auto node1_input_s1_buffer = CreateBuffer(0, node1_input_s1);
    auto node1_input_s1_end_flag = CreateBuffer(1, node1_input_s1);
    BufferManageView::GetIndexInfo(node1_input_s1_end_flag)->MarkAsEndFlag();
    node1_input_.push_back(node1_input_s1_buffer);
    node1_input_.push_back(node1_input_s1_end_flag);
    // node1 p1 s2
    auto node1_input_s2_buffer = CreateBuffer();
    node1_input_.push_back(node1_input_s2_buffer);

    // node2 p1 s1
    auto node2_input1_s1 = std::make_shared<Stream>(session);
    auto node2_input1_s1_buffer = CreateBuffer(0, node2_input1_s1);
    auto node2_input1_s1_end_flag = CreateBuffer(1, node2_input1_s1);
    BufferManageView::GetIndexInfo(node2_input1_s1_end_flag)->MarkAsEndFlag();
    node2_input1_.push_back(node2_input1_s1_buffer);
    node2_input1_end_.push_back(node2_input1_s1_end_flag);
    // node2 p1 s2
    auto node2_input1_s2_buffer2 = CreateBuffer(1);
    node2_input1_.push_back(node2_input1_s2_buffer2);
    // node2 p1 mismatch
    node2_input1_mismatch_.push_back(node2_input1_s1_buffer);
    node2_input1_mismatch_.push_back(node2_input1_s1_end_flag);
    // node2 p2 s1
    auto node2_input2_s1 = std::make_shared<Stream>(session);
    auto node2_input2_s1_buffer = CreateBuffer(0, node2_input2_s1);
    auto node2_input2_s1_end_flag = CreateBuffer(1, node2_input2_s1);
    BufferManageView::GetIndexInfo(node2_input2_s1_end_flag)->MarkAsEndFlag();
    node2_input2_.push_back(node2_input2_s1_buffer);
    node2_input2_end_.push_back(node2_input2_s1_end_flag);
    // node2 p2 s2
    auto node2_input2_s2_buffer2 = CreateBuffer(1);
    node2_input2_.push_back(node2_input2_s2_buffer2);
    // node2 p2 mismatch
    auto node2_input2_mis_end_flag = CreateBuffer(0, node2_input2_s1);
    BufferManageView::GetIndexInfo(node2_input2_mis_end_flag)->MarkAsEndFlag();
    node2_input2_mismatch_.push_back(node2_input2_mis_end_flag);

    // inherit init
    auto inherit1 = std::make_shared<BufferInheritInfo>();
    inherit1->SetType(BufferProcessType::EXPAND);
    inherit1->SetInheritFrom(root1);
    auto inherit2 = std::make_shared<BufferInheritInfo>();
    inherit2->SetType(BufferProcessType::EXPAND);
    inherit2->SetInheritFrom(root2);
    // node1 input index init
    BufferManageView::GetIndexInfo(node1_input_s1_buffer)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node1_input_s1_end_flag)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node1_input_s2_buffer)
        ->SetInheritInfo(inherit2);
    // node2 input index init
    BufferManageView::GetIndexInfo(node2_input1_s1_buffer)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node2_input2_s1_buffer)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node2_input1_s1_end_flag)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node2_input2_s1_end_flag)
        ->SetInheritInfo(inherit1);
    BufferManageView::GetIndexInfo(node2_input2_mis_end_flag)
        ->SetInheritInfo(inherit1);

    BufferManageView::GetIndexInfo(node2_input1_s2_buffer2)
        ->SetInheritInfo(inherit2);
    BufferManageView::GetIndexInfo(node2_input2_s2_buffer2)
        ->SetInheritInfo(inherit2);
  };

  virtual void TearDown() {
    node_ = nullptr;
    flow_->Destroy();
  };

 private:
  std::shared_ptr<MockFlow> flow_;
};

class NodeRunTest : public testing::Test {
 public:
  NodeRunTest() {}
  void TestAdd(std::string add_flowunit_name);
  void TestWrongAdd(std::string add_flowunit_name, Status run_status);

 protected:
  std::shared_ptr<MockFlow> flow_;
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };
  virtual void TearDown() { flow_->Destroy(); };
};

static SessionManager node_test_session_manager;

std::shared_ptr<Node> Add_Node(
    std::string name, std::set<std::string> inputs,
    std::set<std::string> outputs,
    std::shared_ptr<Configuration> config = nullptr) {
  if (config == nullptr) {
    ConfigurationBuilder configbuilder;
    config = configbuilder.Build();
  }
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto node = std::make_shared<Node>();
  node->SetFlowUnitInfo(name, "cpu", "0", flowunit_mgr);
  node->SetSessionManager(&node_test_session_manager);
  EXPECT_EQ(node->Init(inputs, outputs, config), STATUS_SUCCESS);
  EXPECT_EQ(node->Open(), STATUS_SUCCESS);
  return node;
}

std::shared_ptr<Node> Add_Test_2_0_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_2_0", {"In_1", "In_2"}, {}, config);
}

std::shared_ptr<Node> Add_Test_1_0_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_1_0", {"In_1"}, {}, config);
}

std::shared_ptr<Node> Add_Test_1_0_Batch_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_1_0_batch", {"In_1"}, {}, config);
}

std::shared_ptr<Node> Add_Test_0_2_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_0_2", {}, {"Out_1", "Out_2"}, config);
}

std::shared_ptr<Node> Add_Test_0_1_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_0_1", {}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Test_0_1_Batch_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_0_1_batch", {}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Add_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("add", {"In_1", "In_2"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Stream_Add_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("stream_add", {"In_1", "In_2"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Test_Orgin_0_2_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("test_orgin_0_2", {}, {"Out_1", "Out_2"}, config);
}

std::shared_ptr<Node> Add_Half_Condition_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("half-condition", {"In_1"}, {"Out_1", "Out_2"}, config);
}

std::shared_ptr<Node> Add_Conditionn_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("condition", {"In_1"}, {"Out_1", "Out_2"}, config);
}

std::shared_ptr<Node> Add_Switch_Case_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("switch_case", {"In_1"}, {"Out_1", "Out_2", "Out_3"}, config);
}

std::shared_ptr<Node> Add_Loop_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("loop", {"In_1"}, {"Out_1", "Out_2"}, config);
}

std::shared_ptr<Node> Add_Loop_End_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("loop_end", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Expand_Normal_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("expand_normal", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Collapse_Normal_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("collapse_normal", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Expand_Stream_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("expand_stream", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Collapse_Stream_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("collapse_stream", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Garther_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("garther", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Scatter_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("scatter", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Simple_Pass_Node(
    int times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("simple_pass", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto pass_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*pass_fu, Process(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Process_Node(
    std::vector<uint32_t> times,
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto process_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_EQ(times.size(), 3);
  EXPECT_CALL(*process_fu, DataPre(testing::_)).Times(times[0]).InSequence(s1);
  EXPECT_CALL(*process_fu, Process(testing::_)).Times(times[1]).InSequence(s1);
  EXPECT_CALL(*process_fu, DataPost(testing::_)).Times(times[2]).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Info_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_info", {}, {"Out_1"}, config);
  Sequence s1;
  auto stream_info_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*stream_info_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_info_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_info_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Start_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_start", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto stream_start_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*stream_start_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_start_fu, Process(testing::_))
      .Times(times)
      .InSequence(s1);
  EXPECT_CALL(*stream_start_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Mid_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_mid", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto stream_mid_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*stream_mid_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_mid_fu, Process(testing::_)).Times(15).InSequence(s1);
  EXPECT_CALL(*stream_mid_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_End_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_end", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto stream_end_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*stream_end_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Garther_Gen_More_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  return Add_Node("garther_gen_more", {"In_1"}, {"Out_1"}, config);
}

std::shared_ptr<Node> Add_Stream_Normal_Info_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_normal_info", {}, {"Out_1"}, config);
  Sequence s1;
  auto stream_info_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*stream_info_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_info_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_info_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Simple_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("simple_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto simple_error_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*simple_error_fu, Process(testing::_))
      .Times(times)
      .InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Datapre_Error_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_datapre_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Collapse_Recieve_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("collapse_recieve_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  for (uint32_t i = 0; i < times; i++) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  }
  EXPECT_CALL(*node_fu, DataGroupPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Process_Error_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Error_Start_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("error_start", {}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Error_Start_Normal_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("error_start_normal", {}, {"Out_1"}, config);
  return node;
}

std::shared_ptr<Node> Add_Error_End_Normal_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("error_end_normal", {"In_1"}, {}, config);
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times);
  return node;
}

std::shared_ptr<Node> Add_Normal_Start_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("normal_start", {}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Expand_Process_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("normal_expand_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Collapse_Recieve_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("normal_collapse_recieve_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Expand_Process_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("normal_expand_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_In_Process_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_in_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Expand_Start_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("normal_expand_start", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Expand_Datapre_Error_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("expand_datapre_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Expand_Process_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("expand_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Expand_Process_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("expand_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  if (times == 0) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(0).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  } else {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  }
  return node;
}

std::shared_ptr<Node> Add_Collapse_Datagrouppre_Error_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("collapse_datagrouppre_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataGroupPost(testing::_)).Times(0).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Collapse_DataPre_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("collapse_datapre_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataGroupPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Collapse_Process_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("collapse_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  for (uint32_t i = 0; i < times; i++) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  }
  EXPECT_CALL(*node_fu, DataGroupPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Collapse_Process_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("collapse_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  for (uint32_t i = 0; i < times; i++) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  }
  return node;
}

std::shared_ptr<Node> Add_Normal_Collapse_Datapre_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("normal_collapse_datapre_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(0).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(0).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Collapse_Process_Error_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("normal_collapse_process_error", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(times).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Collapse_Process_Node2(
    uint32_t stream_times, uint32_t process_times,
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("normal_collapse_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(stream_times).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_))
      .Times(process_times)
      .InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_))
      .Times(stream_times)
      .InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Collapse_Process_Node(
    uint32_t times, bool repeat,
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("normal_collapse_process", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  if (!repeat) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(times).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(times).InSequence(s1);
  } else {
    for (uint32_t i = 0; i < times; i++) {
      EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
      EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
      EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
    }
  }
  return node;
}

std::shared_ptr<Node> Add_Stream_Normal_Info_2_Node(
    std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_normal_info_2", {}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Stream_Tail_Filter_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node = Add_Node("stream_tail_filter", {"In_1"}, {"Out_1"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  return node;
}

std::shared_ptr<Node> Add_Normal_Condition_Node(
    uint32_t times, std::shared_ptr<Configuration> config = nullptr) {
  auto node =
      Add_Node("normal-condition", {"In_1"}, {"Out_1", "Out_2"}, config);
  Sequence s1;
  auto node_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*node_fu, Process(testing::_)).Times(times).InSequence(s1);
  return node;
}

void NodeRunTest::TestAdd(std::string add_flowunit_name) {
  auto match_at_node = std::make_shared<Node>();
  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "3");
  auto config = configbuilder.Build();
  std::shared_ptr<Node> add_node;

  if (add_flowunit_name == "add") {
    add_node = Add_Add_Node(config);
  } else if (add_flowunit_name == "stream_add") {
    add_node = Add_Stream_Add_Node(config);
  }
  auto input_node = Add_Test_2_0_Node();

  auto device_ = flow_->GetDevice();
  auto input_map_1 =
      std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>();
  BuildDataQueue(match_at_node.get(), input_map_1, device_);

  auto input_map_2 =
      std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>();
  BuildDataQueue(match_at_node.get(), input_map_2, device_);

  auto add_output_port = add_node->GetOutputPort("Out_1");
  EXPECT_TRUE(add_output_port->ConnectPort(input_node->GetInputPort("In_1")));

  auto add_queue_1 = add_node->GetInputPort("In_1")->GetQueue();
  auto add_queue_2 = add_node->GetInputPort("In_2")->GetQueue();
  add_queue_1->PushBatch(&input_map_1["In_1"]);
  add_queue_2->PushBatch(&input_map_1["In_2"]);

  add_queue_1->PushBatch(&input_map_2["In_1"]);
  add_queue_2->PushBatch(&input_map_2["In_2"]);

  auto queue_1 = input_node->GetInputPort("In_1")->GetQueue();

  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(queue_1->Size(), 2);

  std::vector<std::shared_ptr<Buffer>> buffer_vecort_0;
  queue_1->PopBatch(&buffer_vecort_0);
  EXPECT_EQ(buffer_vecort_0[0]->GetBytes(), 40);
  EXPECT_EQ(buffer_vecort_0[1]->GetBytes(), 40);

  auto data_result = (int*)buffer_vecort_0[0]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result[i], 10 + 2 * i);
  }

  auto data_result_2 = (int*)buffer_vecort_0[1]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result_2[i], 10 + 2 * i);
  }
}

TEST_F(NodeTest, Init) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node = std::make_shared<Node>();
  node->SetFlowUnitInfo("test_2_inputs_2_outputs", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(node->Init({"In_1", "In_2"}, {"Out_1"}, config), STATUS_BADCONF);
  EXPECT_EQ(node->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config),
            STATUS_SUCCESS);
  EXPECT_EQ(node->GetInputNum(), 2);
  EXPECT_EQ(node->GetOutputNum(), 2);
  EXPECT_NE(node->GetInputPort("In_1"), nullptr);
  EXPECT_NE(node->GetInputPort("In_2"), nullptr);
  EXPECT_NE(node->GetOutputPort("Out_1"), nullptr);
  EXPECT_NE(node->GetOutputPort("Out_2"), nullptr);
  EXPECT_EQ(node->GetOutputPort("In_None"), nullptr);

  auto another_node = std::make_shared<Node>();
  another_node->SetFlowUnitInfo("test_2_0", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(another_node->Init({"In_1", "In_1", "In_2"}, {}, config),
            STATUS_SUCCESS);
  EXPECT_EQ(another_node->GetInputNum(), 2);
  EXPECT_EQ(another_node->GetOutputNum(), 0);
  EXPECT_NE(another_node->GetInputPort("In_1"), nullptr);
  EXPECT_EQ(another_node->Init({}, {}, config), STATUS_BADCONF);

  auto invalid_node = std::make_shared<Node>();
  invalid_node->SetFlowUnitInfo("invalid_test", "cpu", "0", flowunit_mgr_);
}

TEST_F(NodeTest, SendEvent) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node = std::make_shared<Node>();
  node->SetFlowUnitInfo("test_0_2", "cpu", "0", flowunit_mgr_);

  EXPECT_EQ(node->Init({}, {"Out_1", "Out_2"}, config), STATUS_OK);

  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  auto event_vector = std::vector<std::shared_ptr<FlowUnitInnerEvent>>();
  event_vector.push_back(event);
  EXPECT_EQ(node->SendBatchEvent(event_vector), STATUS_OK);
  FlowunitEventList events = nullptr;
  EXPECT_EQ(node->GetEventPort()->Recv(events), STATUS_OK);
  EXPECT_EQ(events->size(), 1);
  EXPECT_EQ(events->at(0), event);
}

TEST_F(NodeRecvTest, RecvEmpty) {
  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  EXPECT_TRUE(match_stream_data_list.empty());
}

TEST_F(NodeRecvTest, RecvMismatch) {
  node_->SetInputInOrder(true);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  auto origin_node2_input1 = node2_input1_mismatch_;
  input_queue_1->PushBatch(&node2_input1_mismatch_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  auto origin_node2_input2 = node2_input2_mismatch_;
  input_queue_2->PushBatch(&node2_input2_mismatch_);

  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_FAULT);
  ASSERT_EQ(match_stream_data_list.size(), 0);
}

TEST_F(NodeRecvTest, RecvNoOrder) {
  node_->SetInputInOrder(false);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  auto origin_node2_input1 = node2_input1_;
  input_queue_1->PushBatch(&node2_input1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  auto origin_node2_input2 = node2_input2_;
  input_queue_2->PushBatch(&node2_input2_);

  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  ASSERT_EQ(match_stream_data_list.size(), 2);
  auto& s1 = match_stream_data_list.front();
  auto& s2 = match_stream_data_list.back();
  ASSERT_EQ(s1->GetDataCount(), 1);
  ASSERT_EQ(s2->GetDataCount(), 1);

  auto s1_data = s1->GetBufferList();
  auto s2_data = s2->GetBufferList();
  ASSERT_EQ(s1_data->size(), 2);
  ASSERT_EQ(s2_data->size(), 2);
  auto s1_p1_bl = s1_data->at("In_1");
  auto s1_p2_bl = s1_data->at("In_2");
  ASSERT_EQ(s1_p1_bl.size(), 1);
  ASSERT_EQ(s1_p2_bl.size(), 1);
  EXPECT_EQ(s1_p1_bl.front(), origin_node2_input1.back());
  EXPECT_EQ(s1_p2_bl.front(), origin_node2_input2.back());
  auto s2_p1_bl = s2_data->at("In_1");
  auto s2_p2_bl = s2_data->at("In_2");
  ASSERT_EQ(s2_p1_bl.size(), 1);
  ASSERT_EQ(s2_p2_bl.size(), 1);
  EXPECT_EQ(s2_p1_bl.front(), origin_node2_input1.front());
  EXPECT_EQ(s2_p2_bl.front(), origin_node2_input2.front());

  EXPECT_EQ(0, input_queue_1->Size());
  EXPECT_EQ(0, input_queue_2->Size());
}

TEST_F(NodeRecvTest, RecvOrder) {
  node_->SetInputInOrder(true);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  auto origin_node2_input1 = node2_input1_;
  input_queue_1->PushBatch(&node2_input1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  auto origin_node2_input2 = node2_input2_;
  input_queue_2->PushBatch(&node2_input2_);

  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  ASSERT_EQ(match_stream_data_list.size(), 1);
  auto& s1 = match_stream_data_list.front();
  ASSERT_EQ(s1->GetDataCount(), 1);

  auto s1_data = s1->GetBufferList();
  ASSERT_EQ(s1_data->size(), 2);
  auto s1_p1_bl = s1_data->at("In_1");
  auto s1_p2_bl = s1_data->at("In_2");
  ASSERT_EQ(s1_p1_bl.size(), 1);
  ASSERT_EQ(s1_p2_bl.size(), 1);
  EXPECT_EQ(s1_p1_bl.front(), origin_node2_input1.front());
  EXPECT_EQ(s1_p2_bl.front(), origin_node2_input2.front());

  EXPECT_EQ(0, input_queue_1->Size());
  EXPECT_EQ(0, input_queue_2->Size());
}

TEST_F(NodeRecvTest, RecvGatherAll) {
  node_->SetInputInOrder(false);
  node_->SetInputGatherAll(true);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  auto origin_node2_input1 = node2_input1_;
  input_queue_1->PushBatch(&node2_input1_);
  input_queue_1->PushBatch(&node2_input1_end_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  auto origin_node2_input2 = node2_input2_;
  input_queue_2->PushBatch(&node2_input2_);
  input_queue_2->PushBatch(&node2_input2_end_);

  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  ASSERT_EQ(match_stream_data_list.size(), 1);
  auto& s1 = match_stream_data_list.front();
  ASSERT_EQ(s1->GetDataCount(), 2);

  auto s1_data = s1->GetBufferList();
  ASSERT_EQ(s1_data->size(), 2);
  auto s1_p1_bl = s1_data->at("In_1");
  auto s1_p2_bl = s1_data->at("In_2");
  ASSERT_EQ(s1_p1_bl.size(), 2);
  ASSERT_EQ(s1_p2_bl.size(), 2);
  EXPECT_EQ(s1_p1_bl.front(), origin_node2_input1.front());
  EXPECT_EQ(s1_p2_bl.front(), origin_node2_input2.front());
  EXPECT_TRUE(BufferManageView::GetIndexInfo(s1_p1_bl.back())->IsEndFlag());
  EXPECT_TRUE(BufferManageView::GetIndexInfo(s1_p2_bl.back())->IsEndFlag());

  EXPECT_EQ(0, input_queue_1->Size());
  EXPECT_EQ(0, input_queue_2->Size());
}

TEST_F(NodeRecvTest, RecvTwice) {
  node_->SetInputGatherAll(true);
  node_->SetInputInOrder(true);

  // push first
  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  auto origin_node2_input1 = node2_input1_;
  input_queue_1->PushBatch(&node2_input1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  auto origin_node2_input2 = node2_input2_;
  input_queue_2->PushBatch(&node2_input2_);
  input_queue_2->PushBatch(&node2_input2_end_);

  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  EXPECT_TRUE(match_stream_data_list.empty());

  // push again
  input_queue_1->PushBatch(&node2_input1_end_);

  EXPECT_EQ(
      node_->GenInputMatchStreamData(RunType::DATA, match_stream_data_list),
      STATUS_SUCCESS);
  ASSERT_EQ(match_stream_data_list.size(), 1);
  auto& s1 = match_stream_data_list.front();
  ASSERT_EQ(s1->GetDataCount(), 2);

  auto s1_data = s1->GetBufferList();
  ASSERT_EQ(s1_data->size(), 2);
  auto s1_p1_bl = s1_data->at("In_1");
  auto s1_p2_bl = s1_data->at("In_2");
  ASSERT_EQ(s1_p1_bl.size(), 2);
  ASSERT_EQ(s1_p2_bl.size(), 2);
  EXPECT_EQ(s1_p1_bl.front(), origin_node2_input1.front());
  EXPECT_EQ(s1_p2_bl.front(), origin_node2_input2.front());
  EXPECT_TRUE(BufferManageView::GetIndexInfo(s1_p1_bl.back())->IsEndFlag());
  EXPECT_TRUE(BufferManageView::GetIndexInfo(s1_p2_bl.back())->IsEndFlag());

  EXPECT_EQ(0, input_queue_1->Size());
  EXPECT_EQ(0, input_queue_2->Size());
}

TEST_F(NodeRunTest, NodeOutput) {
  auto first_node = Add_Test_0_2_Node();
  auto second_node = Add_Test_2_0_Node();

  auto first_output_port_1 = first_node->GetOutputPort("Out_1");
  auto first_output_port_2 = first_node->GetOutputPort("Out_2");
  EXPECT_TRUE(
      first_output_port_1->ConnectPort(second_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      first_output_port_2->ConnectPort(second_node->GetInputPort("In_2")));
  first_node->Run(DATA);

  auto second_input_queue_1 = second_node->GetInputPort("In_1")->GetQueue();
  auto second_input_queue_2 = second_node->GetInputPort("In_2")->GetQueue();

  EXPECT_EQ(second_input_queue_1->Size(), 2);
  EXPECT_EQ(second_input_queue_2->Size(), 2);

  std::vector<std::shared_ptr<Buffer>> p1_bl;
  std::vector<std::shared_ptr<Buffer>> p2_bl;
  second_input_queue_1->PopBatch(&p1_bl);
  second_input_queue_2->PopBatch(&p2_bl);
  EXPECT_EQ(p1_bl.size(), 2);
  EXPECT_EQ(p2_bl.size(), 2);
  auto p1_b1 = p1_bl.front();
  auto p1_b2 = p1_bl.back();
  auto p2_b1 = p2_bl.front();
  auto p2_b2 = p2_bl.back();
  EXPECT_TRUE(BufferManageView::GetIndexInfo(p1_b2)->IsEndFlag());
  EXPECT_TRUE(BufferManageView::GetIndexInfo(p2_b2)->IsEndFlag());
  EXPECT_EQ(p1_b1->GetBytes(), 40);
  EXPECT_EQ(p2_b1->GetBytes(), 40);
  auto p1_b1_ptr = (const int32_t*)p1_b1->ConstData();
  auto p2_b1_ptr = (const int32_t*)p2_b1->ConstData();
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(p1_b1_ptr[i], i);
    EXPECT_EQ(p2_b1_ptr[i], i + 10);
  }
}

TEST_F(NodeRunTest, AddRun) { TestAdd("add"); }

TEST_F(NodeRunTest, StreamAddRun) { TestAdd("stream_add"); }

TEST_F(NodeRunTest, GartherScatterRun) {
  auto output_node = Add_Test_Orgin_0_2_Node();
  auto condition_node = Add_Conditionn_Node();
  auto expand_node = Add_Expand_Normal_Node();
  auto collapse_node = Add_Collapse_Normal_Node();
  auto stream_add_node = Add_Stream_Add_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  EXPECT_TRUE(output_port_1->ConnectPort(condition_node->GetInputPort("In_1")));

  auto condition_port_1 = condition_node->GetOutputPort("Out_1");
  EXPECT_TRUE(condition_port_1->ConnectPort(expand_node->GetInputPort("In_1")));

  auto expand_port_1 = expand_node->GetOutputPort("Out_1");
  EXPECT_TRUE(expand_port_1->ConnectPort(collapse_node->GetInputPort("In_1")));

  auto condition_port_2 = condition_node->GetOutputPort("Out_2");
  EXPECT_TRUE(
      condition_port_2->ConnectPort(stream_add_node->GetInputPort("In_1")));

  auto collapse_port_1 = collapse_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      collapse_port_1->ConnectPort(stream_add_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      output_port_2->ConnectPort(stream_add_node->GetInputPort("In_2")));

  auto add_port = stream_add_node->GetOutputPort("Out_1");
  EXPECT_TRUE(add_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_add_node->Run(DATA), STATUS_SUCCESS);

  auto queue_1 = input_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> buffer_vector;
  queue_1->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 11);
  for (int i = 0; i < 10; i++) {
    auto data_result = (int*)buffer_vector[i]->ConstData();
    if (i % 2 == 0) {
      EXPECT_EQ(data_result[0], 20 + 6 * i);
    } else {
      EXPECT_EQ(data_result[0], 10 + 2 * i);
    }
  }

  auto end_flag = buffer_vector.back();
  EXPECT_TRUE(BufferManageView::GetIndexInfo(end_flag)->IsEndFlag());
}

TEST_F(NodeRunTest, NormalErrorThroughNormalCollaspe) {
  auto output_node = Add_Error_Start_Normal_Node();
  auto expand_node = Add_Expand_Normal_Node();
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto collapse_node = Add_Collapse_Normal_Node();
  auto input_node = Add_Test_1_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto expand_node_port = expand_node->GetOutputPort("Out_1");
  auto stream_add_port = simple_pass_node->GetOutputPort("Out_1");
  auto collapse_node_port = collapse_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_port_1->ConnectPort(expand_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      expand_node_port->ConnectPort(simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      stream_add_port->ConnectPort(collapse_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      collapse_node_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(input_node->Run(DATA), STATUS_STOP);
}

TEST_F(NodeRunTest, NormalErrorThroughStreamCollaspe) {
  auto output_node = Add_Error_Start_Normal_Node();
  auto expand_node = Add_Expand_Stream_Node();
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto collapse_node = Add_Collapse_Normal_Node();
  auto input_node = Add_Test_1_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto expand_node_port = expand_node->GetOutputPort("Out_1");
  auto stream_add_port = simple_pass_node->GetOutputPort("Out_1");
  auto collapse_node_port = collapse_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_port_1->ConnectPort(expand_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      expand_node_port->ConnectPort(simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      stream_add_port->ConnectPort(collapse_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      collapse_node_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(input_node->Run(DATA), STATUS_STOP);
}

TEST_F(NodeRunTest, StreamGartherScatterRun) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto gather_node = Add_Garther_Node();
  auto add_node = Add_Add_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto garther_output_port = gather_node->GetOutputPort("Out_1");
  auto add_output_port = add_node->GetOutputPort("Out_1");

  EXPECT_TRUE(output_port_1->ConnectPort(scatter_node->GetInputPort("In_1")));
  EXPECT_TRUE(output_port_2->ConnectPort(add_node->GetInputPort("In_2")));
  EXPECT_TRUE(
      scatter_output_port->ConnectPort(gather_node->GetInputPort("In_1")));
  EXPECT_TRUE(garther_output_port->ConnectPort(add_node->GetInputPort("In_1")));
  EXPECT_TRUE(add_output_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(EVENT), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> buffer_vector;
  auto queue_1 = gather_node->GetInputPort("In_1")->GetQueue();
  queue_1->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 12);
  auto s1 = BufferManageView::GetIndexInfo(buffer_vector[0])->GetStream().get();
  auto s2 = BufferManageView::GetIndexInfo(buffer_vector[1])->GetStream().get();
  EXPECT_EQ(s1, s2);
  for (int i = 0; i < 10; i++) {
    auto data_result = (int*)buffer_vector[i]->ConstData();
    EXPECT_EQ(data_result[0], i);
  }
  queue_1->PushBatch(&buffer_vector);
  buffer_vector.clear();

  std::vector<std::shared_ptr<Buffer>> buffer_vector_one;
  std::vector<std::shared_ptr<Buffer>> buffer_vector_two;
  EXPECT_EQ(gather_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(gather_node->Run(EVENT), STATUS_SUCCESS);
  auto queue_2 = add_node->GetInputPort("In_1")->GetQueue();
  auto queue_3 = add_node->GetInputPort("In_2")->GetQueue();
  queue_2->PopBatch(&buffer_vector_one);
  queue_3->PopBatch(&buffer_vector_two);
  EXPECT_EQ(buffer_vector_one.size(), 2);
  EXPECT_EQ(buffer_vector_two.size(), 2);
  EXPECT_EQ(buffer_vector_one[0]->GetBytes(), 40);
  EXPECT_EQ(buffer_vector_two[0]->GetBytes(), 40);

  auto data_result = (int*)buffer_vector_one[0]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result[i], i);
  }
  queue_2->PushBatch(&buffer_vector_one);
  queue_3->PushBatch(&buffer_vector_two);
  buffer_vector_two.clear();
  buffer_vector_one.clear();

  std::vector<std::shared_ptr<Buffer>> final_buffer_vector;
  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);
  auto queue_4 = input_node->GetInputPort("In_1")->GetQueue();
  queue_4->PopBatch(&final_buffer_vector);
  EXPECT_EQ(final_buffer_vector.size(), 2);
  EXPECT_EQ(final_buffer_vector[0]->GetBytes(), 40);
  auto add_data_result = (int*)final_buffer_vector[0]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(add_data_result[i], 10 + 2 * i);
  }
  final_buffer_vector.clear();
}

TEST_F(NodeRunTest, ConditionRun) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto condition_node = Add_Conditionn_Node();
  auto garther_node = Add_Garther_Node();
  auto add_node = Add_Add_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto condition_output_1_port = condition_node->GetOutputPort("Out_1");
  auto condition_output_2_port = condition_node->GetOutputPort("Out_2");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  auto add_output_port = add_node->GetOutputPort("Out_1");

  EXPECT_TRUE(output_port_1->ConnectPort(scatter_node->GetInputPort("In_1")));
  EXPECT_TRUE(output_port_2->ConnectPort(add_node->GetInputPort("In_2")));
  EXPECT_TRUE(
      scatter_output_port->ConnectPort(condition_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      condition_output_1_port->ConnectPort(garther_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      condition_output_2_port->ConnectPort(garther_node->GetInputPort("In_1")));
  EXPECT_TRUE(garther_output_port->ConnectPort(add_node->GetInputPort("In_1")));
  EXPECT_TRUE(add_output_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> buffer_vector;
  auto queue = garther_node->GetInputPort("In_1")->GetQueue();
  queue->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 14);  // contain 4 end_flag
  queue->PushBatch(&buffer_vector);
  buffer_vector.clear();

  EXPECT_EQ(garther_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(garther_node->Run(EVENT), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> add_vector_1;
  std::vector<std::shared_ptr<Buffer>> add_vector_2;
  auto add_queue_1 = add_node->GetInputPort("In_1")->GetQueue();
  auto add_queue_2 = add_node->GetInputPort("In_2")->GetQueue();
  add_queue_1->PopBatch(&add_vector_1);
  add_queue_2->PopBatch(&add_vector_2);
  EXPECT_EQ(add_vector_1.size(), 2);
  EXPECT_EQ(add_vector_2.size(), 2);
  add_queue_1->PushBatch(&add_vector_1);
  add_queue_2->PushBatch(&add_vector_2);
  add_vector_1.clear();
  add_vector_2.clear();

  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> final_buffer_vector;
  auto queue_4 = input_node->GetInputPort("In_1")->GetQueue();
  queue_4->PopBatch(&final_buffer_vector);
  EXPECT_EQ(final_buffer_vector.size(), 2);
  EXPECT_EQ(final_buffer_vector[0]->GetBytes(), 40);
  EXPECT_TRUE(
      BufferManageView::GetIndexInfo(final_buffer_vector[1])->IsEndFlag());
  auto add_data_result = (int*)final_buffer_vector[0]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(add_data_result[i], 10 + 2 * i);
  }
  final_buffer_vector.clear();
}

/*
   output_node ---> loop_node ---> end_node
                   |         |
                   |         |
                    <--------
*/

TEST_F(NodeRunTest, LoopRunBatchTwice) {
  ConfigurationBuilder loop_configbuilder;
  auto config = loop_configbuilder.Build();

  auto output_node = Add_Test_0_1_Batch_Node();
  output_node->SetPriority(0);
  auto loop_node = Add_Loop_Node(config);
  loop_node->SetPriority(1);
  auto end_node = Add_Test_1_0_Batch_Node();
  end_node->SetPriority(2);
  auto output_0_1_port = output_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_0_1_port->ConnectPort(loop_node->GetInputPort("In_1")));
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_port = loop_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_loop_port->ConnectPort(loop_node->GetInputPort("In_1")));
  auto output_loop_end_port = loop_node->GetOutputPort("Out_2");
  EXPECT_TRUE(
      output_loop_end_port->ConnectPort(end_node->GetInputPort("In_1")));
  input_ports = output_loop_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  input_ports = output_loop_end_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  for (int index = 0; index < 10; index++) {
    EXPECT_EQ(loop_node->Run(DATA), STATUS_SUCCESS);

    if (index == 4) {
      EXPECT_EQ(output_node->Open(), STATUS_SUCCESS);
      EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
      auto queue = loop_node->GetInputPort("In_1")->GetQueue();
      EXPECT_EQ(queue->Size(), 22);
    }
  }
  EXPECT_EQ(end_node->Run(DATA), STATUS_STOP);

  for (int index = 0; index < 6; index++) {
    EXPECT_EQ(loop_node->Run(DATA), STATUS_SUCCESS);
  }
  EXPECT_EQ(end_node->Run(DATA), STATUS_STOP);
}

/*
   output_node ---> loop_node ---> end_node
                   |         |
                   |         |
                    <--------
*/

TEST_F(NodeRunTest, LoopRunBatch) {
  ConfigurationBuilder loop_configbuilder;
  loop_configbuilder.AddProperty("queue_size", "1");
  auto config = loop_configbuilder.Build();

  auto output_node = Add_Test_0_1_Batch_Node();
  output_node->SetPriority(0);
  auto loop_node = Add_Loop_Node(config);
  loop_node->SetPriority(1);
  auto end_node = Add_Test_1_0_Batch_Node();
  end_node->SetPriority(2);
  auto output_0_1_port = output_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_0_1_port->ConnectPort(loop_node->GetInputPort("In_1")));
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_port = loop_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_loop_port->ConnectPort(loop_node->GetInputPort("In_1")));
  auto output_loop_end_port = loop_node->GetOutputPort("Out_2");
  EXPECT_TRUE(
      output_loop_end_port->ConnectPort(end_node->GetInputPort("In_1")));
  input_ports = output_loop_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  input_ports = output_loop_end_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  for (int index = 0; index < 11; index++) {
    for (int i = 0; i < 11; i++) {
      EXPECT_EQ(loop_node->Run(DATA), STATUS_SUCCESS);
    }
  }
  EXPECT_EQ(end_node->Run(DATA), STATUS_STOP);
}

/*
   output_node ---> loop_node ---> end_node
                   |         |
                   |         |
                  <--loop_end--
*/

TEST_F(NodeRunTest, LoopRunBatchMultiFlowUnit) {
  ConfigurationBuilder loop_configbuilder;
  loop_configbuilder.AddProperty("queue_size", "3");
  auto config = loop_configbuilder.Build();

  auto output_node = Add_Test_0_1_Batch_Node();
  output_node->SetPriority(0);
  auto loop_node = Add_Loop_Node(config);
  loop_node->SetPriority(1);
  auto loop_end_node = Add_Loop_End_Node();
  loop_end_node->SetPriority(2);
  auto end_node = Add_Test_1_0_Batch_Node();
  end_node->SetPriority(3);
  auto output_0_1_port = output_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_0_1_port->ConnectPort(loop_node->GetInputPort("In_1")));
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_output1_port = loop_node->GetOutputPort("Out_1");
  auto output_loop_output2_port = loop_node->GetOutputPort("Out_2");
  EXPECT_TRUE(output_loop_output1_port->ConnectPort(
      loop_end_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      output_loop_output2_port->ConnectPort(end_node->GetInputPort("In_1")));
  input_ports = output_loop_output1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  input_ports = output_loop_output2_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  auto output_loop_end_port = loop_end_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      output_loop_end_port->ConnectPort(loop_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  for (int index = 0; index < 10; index++) {
    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(loop_node->Run(DATA), STATUS_SUCCESS);
    }
    EXPECT_EQ(loop_end_node->Run(DATA), STATUS_SUCCESS);
  }
  EXPECT_EQ(end_node->Run(DATA), STATUS_STOP);
}

TEST_F(NodeRunTest, StreamInfo) {
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(15);
  auto stream_mid_node = Add_Stream_Mid_Node();
  auto stream_end_node = Add_Stream_End_Node(2);
  auto final_input_node = Add_Test_2_0_Node();

  auto start_info_port = stream_info_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_info_port->ConnectPort(stream_start_node->GetInputPort("In_1")));
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_output_port->ConnectPort(simple_pass_node->GetInputPort("In_1")));

  auto simple_pass_port = simple_pass_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      simple_pass_port->ConnectPort(stream_mid_node->GetInputPort("In_1")));

  auto mid_output_port = stream_mid_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      mid_output_port->ConnectPort(stream_end_node->GetInputPort("In_1")));
  auto end_output_port = stream_end_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      end_output_port->ConnectPort(final_input_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> start_buffer_vector;
  auto queue_2 = stream_mid_node->GetInputPort("In_1")->GetQueue();
  queue_2->PopBatch(&start_buffer_vector);
  EXPECT_EQ(start_buffer_vector.size(), 5);
  for (int i = 0; i < 5; i++) {
    auto data_result = (int*)start_buffer_vector[i]->ConstData();
    EXPECT_EQ(data_result[0], i);
  }
  auto stream =
      BufferManageView::GetIndexInfo(start_buffer_vector[0])->GetStream();
  auto data_meta = stream->GetStreamMeta();
  auto start_num =
      *(std::static_pointer_cast<int>(data_meta->GetMeta("start_index")).get());
  auto end_num =
      *(std::static_pointer_cast<int>(data_meta->GetMeta("end_index")).get());
  auto interval =
      *(std::static_pointer_cast<int>(data_meta->GetMeta("interval")).get());
  EXPECT_EQ(start_num, 0);
  EXPECT_EQ(end_num, 15);
  EXPECT_EQ(interval, 3);
  queue_2->PushBatch(&start_buffer_vector);
  start_buffer_vector.clear();

  EXPECT_EQ(stream_mid_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> mid_buffer_vector;
  auto queue_3 = stream_end_node->GetInputPort("In_1")->GetQueue();
  queue_3->PopBatch(&mid_buffer_vector);
  EXPECT_EQ(mid_buffer_vector.size(), 2);
  auto data_result_0 = (int*)mid_buffer_vector[0]->ConstData();
  auto data_result_1 = (int*)mid_buffer_vector[1]->ConstData();

  auto data_group_meta_1 = BufferManageView::GetIndexInfo(mid_buffer_vector[0])
                               ->GetStream()
                               ->GetStreamMeta();
  EXPECT_EQ(
      *std::static_pointer_cast<int>(data_group_meta_1->GetMeta("magic_num")),
      3343);
  EXPECT_EQ(data_result_0[0], 0);
  EXPECT_EQ(data_result_1[0], 3);
  queue_3->PushBatch(&mid_buffer_vector);
  mid_buffer_vector.clear();

  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
  auto queue_4 = final_input_node->GetInputPort("In_1")->GetQueue();
  EXPECT_EQ(queue_4->Size(), 0);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  queue_2->PopBatch(&start_buffer_vector);

  EXPECT_EQ(start_buffer_vector.size(), 5);
  for (int i = 0; i < 5; i++) {
    auto data_result = (int*)start_buffer_vector[i]->ConstData();
    EXPECT_EQ(data_result[0], i + 5);
  }

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  queue_2->PopBatch(&start_buffer_vector);
  EXPECT_EQ(start_buffer_vector.size(), 12);

  int base_v = 0;
  for (int i = 0; i < 12; i++) {
    if (BufferManageView::GetIndexInfo(start_buffer_vector[i])->IsEndFlag()) {
      continue;
    }
    auto data_result = (int*)start_buffer_vector[i]->ConstData();
    EXPECT_EQ(data_result[0], base_v + 5);
    ++base_v;
  }
  queue_2->PushBatch(&start_buffer_vector);
  start_buffer_vector.clear();

  EXPECT_EQ(stream_mid_node->Run(DATA), STATUS_SUCCESS);
  queue_3->PopBatch(&mid_buffer_vector);
  EXPECT_EQ(mid_buffer_vector.size(), 5);
  for (int i = 0; i < 3; i++) {
    auto data_result = (int*)mid_buffer_vector[i]->ConstData();
    EXPECT_EQ(data_result[0], 6 + i * 3);
  }
  queue_3->PushBatch(&mid_buffer_vector);
  mid_buffer_vector.clear();

  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(EVENT), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> end_buffer_vector;
  queue_4->PopBatch(&end_buffer_vector);
  EXPECT_EQ(end_buffer_vector.size(), 2);
  auto final_result = (int*)end_buffer_vector[0]->ConstData();
  EXPECT_EQ(final_result[0], 30);
}

void NodeRunTest::TestWrongAdd(std::string flowunit_name, Status run_status) {
  ConfigurationBuilder configbuilderflowunit;
  auto config_flowunit = configbuilderflowunit.Build();
  config_flowunit->SetProperty("need_check_output", true);
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();

  auto output_node = Add_Test_0_2_Node();
  auto wrong_add_node = std::make_shared<Node>();
  wrong_add_node->SetFlowUnitInfo(flowunit_name, "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(wrong_add_node->Init({"In_1", "In_2"}, {"Out_1"}, config_flowunit),
            STATUS_SUCCESS);
  EXPECT_EQ(wrong_add_node->Open(), STATUS_SUCCESS);

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  EXPECT_TRUE(output_port_1->ConnectPort(wrong_add_node->GetInputPort("In_1")));
  EXPECT_TRUE(output_port_2->ConnectPort(wrong_add_node->GetInputPort("In_2")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(wrong_add_node->Run(DATA), run_status);
}

TEST_F(NodeRunTest, Run_Normal_Count_InSame) {
  TestWrongAdd("wrong_add", STATUS_STOP);
}

TEST_F(NodeRunTest, Run_Normal_Count_InSame_2) {
  TestWrongAdd("wrong_add_2", STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Run_Collapse_Not_One) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto garther_node = Add_Garther_Gen_More_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_port_1->ConnectPort(scatter_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      scatter_output_port->ConnectPort(garther_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(garther_node->Run(DATA), STATUS_STOP);
}

TEST_F(NodeRunTest, CacheFull) {
  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("queue_size", "5");
  auto config = configbuilder.Build();
  auto start_node = Add_Test_Orgin_0_2_Node();
  auto pass_node = Add_Simple_Pass_Node(10);
  auto receive_node = Add_Test_2_0_Node(config);

  auto output_port_1 = start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_port_1->ConnectPort(pass_node->GetInputPort("In_1")));
  auto output_port_2 = start_node->GetOutputPort("Out_2");
  EXPECT_TRUE(output_port_2->ConnectPort(receive_node->GetInputPort("In_1")));
  auto add_output_port_1 = pass_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      add_output_port_1->ConnectPort(receive_node->GetInputPort("In_2")));

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);

  auto queue_1 = receive_node->GetInputPort("In_1")->GetQueue();
  auto queue_2 = receive_node->GetInputPort("In_2")->GetQueue();
  EXPECT_EQ(queue_1->Size(), 11);
  EXPECT_EQ(queue_2->Size(), 0);

  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(queue_1->Size(), 11);
  EXPECT_EQ(queue_2->Size(), 11);

  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(queue_1->Size(), 6);
  EXPECT_EQ(queue_2->Size(), 6);

  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_STOP);
  EXPECT_EQ(queue_1->Size(), 0);
  EXPECT_EQ(queue_2->Size(), 0);
}

// thread_pool has not implement set priority
TEST_F(NodeRunTest, DISABLED_RunPriority) {
  auto device_ = flow_->GetDevice();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "5");
  auto config = configbuilder.Build();
  auto run_node = std::make_shared<Node>();
  run_node->SetFlowUnitInfo("get_priority", "cpu", "0", flowunit_mgr_);
  auto print_node = std::make_shared<Node>();
  print_node->SetFlowUnitInfo("print", "cpu", "0", flowunit_mgr_);
  EXPECT_EQ(run_node->Init({"In_1"}, {"Out_1"}, config), STATUS_SUCCESS);
  EXPECT_EQ(run_node->Open(), STATUS_SUCCESS);
  EXPECT_EQ(print_node->Init({"In_1"}, {}, config), STATUS_SUCCESS);
  EXPECT_EQ(print_node->Open(), STATUS_SUCCESS);

  auto output_port_1 = run_node->GetOutputPort("Out_1");
  EXPECT_TRUE(output_port_1->ConnectPort(print_node->GetInputPort("In_1")));

  int32_t default_priority = 3;
  size_t data_size = 5;
  size_t buffer_size = 3;
  std::vector<std::shared_ptr<Buffer>> in_data(data_size * buffer_size,
                                               nullptr);
  for (size_t i = 0; i < buffer_size; ++i) {
    for (size_t j = 0; j < data_size; ++j) {
      auto buffer = std::make_shared<Buffer>(device_);
      buffer->Build(1 * sizeof(int));
      BufferManageView::SetPriority(buffer, default_priority + i);
      in_data[i * data_size + j] = buffer;
    }
  }

  auto in_queue = run_node->GetInputPort("In_1")->GetQueue();
  in_queue->PushBatch(&in_data);

  EXPECT_EQ(in_queue->Size(), data_size * buffer_size);
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(run_node->Run(DATA), STATUS_SUCCESS);

    auto out_queue = print_node->GetInputPort("In_1")->GetQueue();
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    out_queue->PopBatch(&buffer_vector);

    EXPECT_EQ(buffer_vector.size(), data_size);
    for (size_t i = 0; i < data_size; i++) {
      EXPECT_EQ(buffer_vector[i]->GetBytes(), 1 * sizeof(int));
      auto data_result = (int*)buffer_vector[i]->ConstData();
      if (i == data_size - 1) {
        EXPECT_EQ(data_result[i], 0);
      } else {
        EXPECT_EQ(data_result[i], default_priority + (buffer_size - 1 - i));
      }
    }
  }
}

TEST_F(NodeRunTest, Normal_Process_Error_Recieve_InVisible) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(5);
  auto simple_error_node = Add_Simple_Error_Node(25);
  // Invisible error end node do not execute process function
  auto error_end_node = Add_Error_End_Normal_Node(0);

  stream_info_node->SetName("stream_info_normal_node");
  stream_info_node->SetPriority(0);
  stream_start_node->SetName("stream_start_node");
  stream_start_node->SetPriority(1);
  simple_error_node->SetName("simple_error_node");
  simple_error_node->SetPriority(2);
  error_end_node->SetName("error_end_normal_node");
  error_end_node->SetPriority(3);
  EXPECT_EQ(error_end_node->IsExceptionVisible(), true);
  error_end_node->SetExceptionVisible(false);

  EXPECT_TRUE(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
      stream_start_node->GetInputPort("In_1")));
  EXPECT_TRUE(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
      simple_error_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
      error_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 5);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 27);

  EXPECT_EQ(error_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Process_Error_Recieve_Visible) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(5);
  auto simple_error_node = Add_Simple_Error_Node(25);
  auto error_end_node = Add_Error_End_Normal_Node(25);

  stream_info_node->SetName("stream_info_normal_node");
  stream_info_node->SetPriority(0);
  stream_start_node->SetName("stream_start_node");
  stream_start_node->SetPriority(1);
  simple_error_node->SetName("simple_error_node");
  simple_error_node->SetPriority(2);
  error_end_node->SetName("error_end_normal_node");
  error_end_node->SetPriority(3);
  error_end_node->SetExceptionVisible(true);

  EXPECT_TRUE(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
      stream_start_node->GetInputPort("In_1")));
  EXPECT_TRUE(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
      simple_error_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
      error_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 5);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 27);

  EXPECT_EQ(error_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Process_Error_Expand_Visible) {
  auto start_node = Add_Error_Start_Normal_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_TRUE(start_node->GetOutputPort("Out_1")->ConnectPort(
      expand_process_node->GetInputPort("In_1")));
  EXPECT_TRUE(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
      simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
      receive_error_node->GetInputPort("In_1")));

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 3);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Process_Error_Expand_Invisible) {
  auto start_node = Add_Error_Start_Normal_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(1);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  expand_process_node->SetExceptionVisible(true);

  EXPECT_TRUE(start_node->GetOutputPort("Out_1")->ConnectPort(
      expand_process_node->GetInputPort("In_1")));
  EXPECT_TRUE(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
      simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
      receive_error_node->GetInputPort("In_1")));

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recv_queue);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Process_Error_Collapse_Visible) {
  auto start_node = Add_Error_Start_Normal_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto collapse_node = Add_Normal_Collapse_Process_Node(1, false);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");
  collapse_node->SetExceptionVisible(true);

  EXPECT_TRUE(start_node->GetOutputPort("Out_1")->ConnectPort(
      expand_process_node->GetInputPort("In_1")));
  EXPECT_TRUE(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
      simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
      collapse_node->GetInputPort("In_1")));
  EXPECT_TRUE(collapse_node->GetOutputPort("Out_1")->ConnectPort(
      receive_node->GetInputPort("In_1")));

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 2);

  EXPECT_FALSE(error_buffer_vector[0]->HasError());

  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Process_Error_Collapse_Invisible) {
  auto start_node = Add_Error_Start_Normal_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto collapse_node = Add_Normal_Collapse_Process_Node2(1, 0);
  auto receive_node = Add_Stream_Process_Node({1, 1, 1});

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");
  receive_node->SetExceptionVisible(true);

  EXPECT_TRUE(start_node->GetOutputPort("Out_1")->ConnectPort(
      expand_process_node->GetInputPort("In_1")));
  EXPECT_TRUE(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
      simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
      collapse_node->GetInputPort("In_1")));
  EXPECT_TRUE(collapse_node->GetOutputPort("Out_1")->ConnectPort(
      receive_node->GetInputPort("In_1")));

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 2);

  EXPECT_TRUE(error_buffer_vector[1]->HasError());

  receive_queue->PushBatch(&error_buffer_vector);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Process_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_error_node = Add_Simple_Error_Node(10);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto recieve_node = Add_Stream_Process_Node({1, 15, 1});

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  simple_pass_node->SetName("simple_pass_node");
  recieve_node->SetName("recieve_node");
  recieve_node->SetExceptionVisible(true);

  EXPECT_TRUE(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
      stream_start_node->GetInputPort("In_1")));
  EXPECT_TRUE(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
      simple_error_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
      simple_pass_node->GetInputPort("In_1")));
  EXPECT_TRUE(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
      recieve_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> index_buffer_vector;
  auto queue = recieve_node->GetInputPort("In_1")->GetQueue();
  queue->PopBatch(&index_buffer_vector);

  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : index_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);

  queue->PushBatch(&index_buffer_vector);
  EXPECT_EQ(recieve_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(recieve_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Recv_InVisible_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(2);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto recieve_node = Add_Stream_Process_Node({1, 1, 1});

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  simple_pass_node->SetName("simple_pass_node");
  recieve_node->SetName("recieve_node");
  recieve_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                recieve_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  auto queue = recieve_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(queue, 1);
  EXPECT_EQ(recieve_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Recv_Visible_Error) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_error_node_cfg = builder.Build();
  simple_error_node_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto simple_error_node =
      Add_Stream_In_Process_Error_Node(2, simple_error_node_cfg);
  auto simple_pass_node = Add_Simple_Pass_Node(6);
  auto recieve_node_cfg = builder.Build();
  recieve_node_cfg->SetProperty<uint32_t>("batch_size", 6);
  auto recieve_node = Add_Stream_Process_Node({1, 1, 1}, recieve_node_cfg);

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  simple_pass_node->SetName("simple_pass_node");
  recieve_node->SetName("recieve_node");

  simple_pass_node->SetExceptionVisible(true);
  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                recieve_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  auto queue = recieve_node->GetInputPort("In_1")->GetQueue();
  EXPECT_EQ(queue->Size(), 6);
  EXPECT_EQ(recieve_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto simple_error_queue = simple_error_node->GetInputPort("In_1")->GetQueue();
  EXPECT_EQ(simple_error_queue->Size(), 5);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_queue->Size(), 10);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_DataPre_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(2);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  receive_node->SetName("receive_node");

  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 1);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  auto simple_error_queue = simple_error_node->GetInputPort("In_1")->GetQueue();
  EXPECT_EQ(simple_error_queue->Size(), 5);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Process_Error) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(2);
  auto simple_error_node_cfg = builder.Build();
  simple_error_node_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto simple_error_node = Add_Stream_Process_Error_Node(simple_error_node_cfg);
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  receive_node->SetName("receive_node");

  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 1);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  auto simple_error_queue = simple_error_node->GetInputPort("In_1")->GetQueue();
  EXPECT_EQ(simple_error_queue->Size(), 5);

  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Recv_Visible_Error) {
  auto error_start_node = Add_Error_Start_Node();
  auto simple_stream_node = Add_Stream_Process_Node({1, 1, 1});
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  error_start_node->SetName("error_start_node");
  simple_stream_node->SetName("simple_stream_node");
  receive_node->SetName("receive_node");
  simple_stream_node->SetExceptionVisible(true);
  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(error_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(error_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recieve_queue);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Recv_Invisible_Error) {
  auto error_start_node = Add_Error_Start_Node();
  auto simple_stream_node = Add_Stream_Process_Node({0, 0, 0});
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  error_start_node->SetName("error_start_node");
  simple_stream_node->SetName("simple_stream_node");
  receive_node->SetName("receive_node");

  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(error_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(error_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 1);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Send_Error) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);

  auto simple_stream_node_cfg = builder.Build();
  simple_stream_node_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto simple_stream_node =
      Add_Stream_Process_Node({1, 2, 1}, simple_stream_node_cfg);

  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_stream_node->SetName("simple_stream_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Expand_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_process_error_node = Add_Normal_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_process_error_node->SetName("expand_process_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");

  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Expand_Recieve_Invisible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 1);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Expand_Recieve_Visible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(1);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  expand_process_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recv_queue);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Normal_Expand_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto normal_start_node = Add_Normal_Expand_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  normal_start_node->SetName("normal_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                normal_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(normal_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(normal_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(normal_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(normal_start_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_DataPre_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_datapre_error_node = Add_Expand_Datapre_Error_Node();
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_datapre_error_node->SetName("expand_datapre_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_datapre_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_datapre_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_datapre_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto receive_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(receive_queue, 1);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_datapre_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_datapre_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_datapre_error_node->Run(EVENT), STATUS_SUCCESS);

  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  EXPECT_EQ(error_buffer_vector.size(), 3);
  for (uint32_t i = 0; i < 3; i++) {
    auto error = error_buffer_vector[i]->GetError();
    EXPECT_NE(error, nullptr);
  }
  receive_queue->PushBatch(&error_buffer_vector);

  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_process_error_node = Add_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_process_error_node->SetName("expand_process_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");

  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_Recieve_Invisible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 1);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_Recieve_Visible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(1);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  expand_process_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recv_queue);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_Recieve_Event_Error) {
  auto device_ = flow_->GetDevice();
  auto input_map_1 =
      std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>();
  BuildDataEventStart(input_map_1, device_);

  auto stream_start_node = Add_Stream_Start_Node(2);

  ConfigurationBuilder builder;
  auto simple_stream_node_cfg = builder.Build();
  simple_stream_node_cfg->SetProperty<uint32_t>("batch_size", 10);
  auto simple_stream_node =
      Add_Stream_Process_Node({1, 2, 1}, simple_stream_node_cfg);

  stream_start_node->SetName("stream_start_node");
  simple_stream_node->SetName("simple_stream_node");
  simple_stream_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_stream_node->GetInputPort("In_1")),
            true);

  auto start_queue_1 = stream_start_node->GetInputPort("In_1")->GetQueue();
  auto index_start_vector = input_map_1["In_1"];
  start_queue_1->PushBatch(&index_start_vector);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto input_map_2 =
      std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>();
  BuildDataEventStop(input_map_2);
  auto index_stop_vector = input_map_2["In_1"];
  start_queue_1->PushBatch(&index_stop_vector);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Expand_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->ConnectPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                simple_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Collapse_DataGroupPre_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(4);
  auto collapse_error_node = Add_Collapse_Datagrouppre_Error_Node();
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_error_node->SetName("collapse_error_node");
  receive_node->SetName("receive_error_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);

  auto queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(queue, 3);
}

TEST_F(NodeRunTest, DISABLED_Collapse_DataPre_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Expand_Process_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(16);
  auto collapse_error_node = Add_Collapse_DataPre_Error_Node(4);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_error_node->SetName("collapse_error_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Collapse_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Expand_Process_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(16);
  auto collapse_error_node = Add_Collapse_Process_Error_Node(4);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_error_node->SetName("receive_error_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(EVENT), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Stream_Collapse_Send_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Expand_Process_Node(2);
  auto simple_pass_node = Add_Simple_Pass_Node(8);
  auto collapse_node = Add_Collapse_Process_Node(2);
  auto stream_error_node = Add_Stream_Datapre_Error_Node();

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  stream_error_node->SetName("stream_error_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                stream_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, DISABLED_Stream_Collapse_Visible_Recv_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_error_node = Add_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto collapse_node = Add_Collapse_Process_Node(4);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_error_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");
  collapse_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_FALSE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Stream_Collapse_Invisible_Recv_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_error_node = Add_Expand_Process_Error_Node(2);
  auto simple_pass_node = Add_Simple_Pass_Node(4);
  auto collapse_node = Add_Collapse_Process_Node(1);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_error_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  EXPECT_FALSE(error_buffer_vector[0]->HasError());
  EXPECT_TRUE(error_buffer_vector[1]->HasError());
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Normal_Collapse_DataPre_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Normal_Expand_Process_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(16);
  auto collapse_error_node = Add_Normal_Collapse_Datapre_Error_Node(4);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_error_node->SetName("collapse_error_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Normal_Collapse_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_node = Add_Normal_Expand_Process_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(16);
  auto collapse_error_node = Add_Normal_Collapse_Process_Error_Node(4);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_error_node->SetName("collapse_error_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Normal_Collapse_Visible_Recv_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_error_node = Add_Normal_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto collapse_node = Add_Normal_Collapse_Process_Node(4, false);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_error_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");
  collapse_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_FALSE(error_buffer_vector[i]->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Normal_Collapse_Invisible_Recv_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_error_node = Add_Normal_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto collapse_node = Add_Normal_Collapse_Process_Node(3, false);
  auto receive_node = Add_Stream_Process_Node({0, 0, 0});

  start_node->SetName("start_node");
  expand_error_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  std::shared_ptr<FlowUnitError> error;
  for (auto& buffer : error_buffer_vector) {
    if (buffer->HasError()) {
      error = buffer->GetError();
    }
  }
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  EXPECT_FALSE(error_buffer_vector[0]->HasError());
  EXPECT_TRUE(error_buffer_vector[3]->HasError());
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, DISABLED_Normal_Collapse_Send_Error) {
  auto start_node = Add_Stream_Normal_Info_2_Node();
  auto expand_node = Add_Normal_Expand_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(15);
  auto collapse_node = Add_Normal_Collapse_Process_Node(2, true);
  auto stream_error_node = Add_Stream_Datapre_Error_Node();

  start_node->SetName("start_node");
  expand_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  stream_error_node->SetName("stream_error_node");

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->ConnectPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->ConnectPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->ConnectPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->ConnectPort(
                stream_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Completion_Unfinish_Normal_Data) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);

  auto stream_tail_filter_node_cfg = builder.Build();
  stream_tail_filter_node_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto stream_tail_filter_node =
      Add_Stream_Tail_Filter_Node(3, stream_tail_filter_node_cfg);

  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto stream_end_node = Add_Stream_End_Node(1);

  auto start_info_port = stream_info_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_info_port->ConnectPort(stream_start_node->GetInputPort("In_1")));
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(start_output_port->ConnectPort(
      stream_tail_filter_node->GetInputPort("In_1")));
  auto mid_output_port = stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      mid_output_port->ConnectPort(simple_pass_node->GetInputPort("In_1")));
  auto simple_pass_output_port = simple_pass_node->GetOutputPort("Out_1");
  EXPECT_TRUE(simple_pass_output_port->ConnectPort(
      stream_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Completion_Unfinish_Stream_Data) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);

  auto stream_tail_filter_node_cfg = builder.Build();
  stream_tail_filter_node_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto stream_tail_filter_node_1 =
      Add_Stream_Tail_Filter_Node(3, stream_tail_filter_node_cfg);

  auto stream_tail_filter_node2_cfg = builder.Build();
  stream_tail_filter_node2_cfg->SetProperty<uint32_t>("batch_size", 5);
  auto stream_tail_filter_node_2 =
      Add_Stream_Tail_Filter_Node(2, stream_tail_filter_node2_cfg);

  auto stream_end_node = Add_Stream_End_Node(1);

  auto start_info_port = stream_info_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_info_port->ConnectPort(stream_start_node->GetInputPort("In_1")));
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(start_output_port->ConnectPort(
      stream_tail_filter_node_1->GetInputPort("In_1")));
  auto mid_output_port = stream_tail_filter_node_1->GetOutputPort("Out_1");
  EXPECT_TRUE(mid_output_port->ConnectPort(
      stream_tail_filter_node_2->GetInputPort("In_1")));
  auto simple_pass_output_port =
      stream_tail_filter_node_2->GetOutputPort("Out_1");
  EXPECT_TRUE(simple_pass_output_port->ConnectPort(
      stream_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_1->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_2->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_1->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_2->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_1->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_2->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Completion_Unfinish_Expand_Collapse_Data) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);

  auto stream_tail_filter_node_cfg = builder.Build();
  stream_tail_filter_node_cfg->SetProperty<uint32_t>("batch_size", 10);
  auto stream_tail_filter_node =
      Add_Stream_Tail_Filter_Node(2, stream_tail_filter_node_cfg);

  auto expand_node = Add_Expand_Process_Node(10);
  auto collapse_node = Add_Collapse_Process_Node(10);
  auto stream_end_node = Add_Stream_End_Node(1);

  auto start_info_port = stream_info_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_info_port->ConnectPort(stream_start_node->GetInputPort("In_1")));
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(start_output_port->ConnectPort(
      stream_tail_filter_node->GetInputPort("In_1")));
  auto stream_tail_output_port =
      stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      stream_tail_output_port->ConnectPort(expand_node->GetInputPort("In_1")));
  auto expand_output_port = expand_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      expand_output_port->ConnectPort(collapse_node->GetInputPort("In_1")));
  auto collapse_output_port = collapse_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      collapse_output_port->ConnectPort(stream_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(expand_node->Run(EVENT), STATUS_SUCCESS);
  }
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(collapse_node->Run(EVENT), STATUS_SUCCESS);
  }
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Completion_Unfinish_Condition_Data) {
  ConfigurationBuilder builder;
  auto stream_info_node = Add_Stream_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);

  auto stream_tail_filter_node_cfg = builder.Build();
  stream_tail_filter_node_cfg->SetProperty<uint32_t>("batch_size", 10);
  auto stream_tail_filter_node =
      Add_Stream_Tail_Filter_Node(2, stream_tail_filter_node_cfg);

  auto condition_node = Add_Normal_Condition_Node(10);
  auto stream_end_node = Add_Stream_End_Node(1);

  auto start_info_port = stream_info_node->GetOutputPort("Out_1");
  EXPECT_TRUE(
      start_info_port->ConnectPort(stream_start_node->GetInputPort("In_1")));

  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_TRUE(start_output_port->ConnectPort(
      stream_tail_filter_node->GetInputPort("In_1")));

  auto stream_tail_output_port =
      stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_TRUE(stream_tail_output_port->ConnectPort(
      condition_node->GetInputPort("In_1")));

  auto condition_output_port_1 = condition_node->GetOutputPort("Out_1");
  EXPECT_TRUE(condition_output_port_1->ConnectPort(
      stream_end_node->GetInputPort("In_1")));
  auto condition_output_port_2 = condition_node->GetOutputPort("Out_2");
  EXPECT_TRUE(condition_output_port_2->ConnectPort(
      stream_end_node->GetInputPort("In_1")));

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Dynamic_Config) {
  ConfigurationBuilder configbuilderflowunit;
  auto config_flowunit = configbuilderflowunit.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto device_ = flow_->GetDevice();
  auto dynamic_config_node = std::make_shared<Node>();
  dynamic_config_node->SetFlowUnitInfo("dynamic_config", "cpu", "0",
                                       flowunit_mgr_);
  dynamic_config_node->SetSessionManager(&node_test_session_manager);
  auto dynamic_get_config_node_1 = std::make_shared<Node>();
  dynamic_get_config_node_1->SetFlowUnitInfo("dynamic_get_config", "cpu", "0",
                                             flowunit_mgr_);
  auto dynamic_get_config_node_2 = std::make_shared<Node>();
  dynamic_get_config_node_2->SetFlowUnitInfo("dynamic_get_config", "cpu", "0",
                                             flowunit_mgr_);
  auto dynamic_get_config_node_3 = std::make_shared<Node>();
  dynamic_get_config_node_3->SetFlowUnitInfo("dynamic_get_config_other", "cpu",
                                             "0", flowunit_mgr_);
  auto dynamic_get_config_node_4 = std::make_shared<Node>();
  dynamic_get_config_node_4->SetFlowUnitInfo("dynamic_get_config_other", "cpu",
                                             "0", flowunit_mgr_);
  dynamic_get_config_node_1->SetName("dynamic_get_config_1");
  dynamic_get_config_node_2->SetName("dynamic_get_config_2");

  EXPECT_EQ(dynamic_config_node->Init({}, {"Out_1"}, config_flowunit),
            STATUS_SUCCESS);
  EXPECT_EQ(dynamic_config_node->Open(), STATUS_SUCCESS);
  EXPECT_EQ(
      dynamic_get_config_node_1->Init({"In_1"}, {"Out_1"}, config_flowunit),
      STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_1->Open(), STATUS_SUCCESS);
  EXPECT_EQ(
      dynamic_get_config_node_2->Init({"In_1"}, {"Out_1"}, config_flowunit),
      STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_2->Open(), STATUS_SUCCESS);
  EXPECT_EQ(
      dynamic_get_config_node_3->Init({"In_1"}, {"Out_1"}, config_flowunit),
      STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_3->Open(), STATUS_SUCCESS);

  EXPECT_EQ(
      dynamic_get_config_node_4->Init({"In_1"}, {"Out_1"}, config_flowunit),
      STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_4->Open(), STATUS_SUCCESS);

  auto dynamic_config_port = dynamic_config_node->GetOutputPort("Out_1");
  EXPECT_TRUE(dynamic_config_port->ConnectPort(
      dynamic_get_config_node_1->GetInputPort("In_1")));

  auto dynamic_get_config_port_1 =
      dynamic_get_config_node_1->GetOutputPort("Out_1");
  EXPECT_TRUE(dynamic_get_config_port_1->ConnectPort(
      dynamic_get_config_node_2->GetInputPort("In_1")));

  auto dynamic_get_config_port_2 =
      dynamic_get_config_node_2->GetOutputPort("Out_1");
  EXPECT_TRUE(dynamic_get_config_port_2->ConnectPort(
      dynamic_get_config_node_3->GetInputPort("In_1")));

  auto dynamic_get_config_port_3 =
      dynamic_get_config_node_3->GetOutputPort("Out_1");
  EXPECT_TRUE(dynamic_get_config_port_3->ConnectPort(
      dynamic_get_config_node_4->GetInputPort("In_1")));

  EXPECT_EQ(dynamic_config_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_1->Run(DATA), STATUS_SUCCESS);
  auto queue_1 = dynamic_get_config_node_2->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> buffer_vector_1;
  queue_1->PopBatch(&buffer_vector_1);
  std::string test_1 = "";
  buffer_vector_1[0]->Get("test", test_1);
  EXPECT_EQ(test_1, "node.dynamic_get_config_1.test");
  queue_1->PushBatch(&buffer_vector_1);

  EXPECT_EQ(dynamic_get_config_node_2->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<Buffer>> buffer_vector_2;
  auto queue_2 = dynamic_get_config_node_3->GetInputPort("In_1")->GetQueue();
  queue_2->PopBatch(&buffer_vector_2);
  std::string test_2 = "";
  buffer_vector_2[0]->Get("test", test_2);
  EXPECT_EQ(test_2, "flowunit.dynamic_get_config.test");
  queue_2->PushBatch(&buffer_vector_2);

  EXPECT_EQ(dynamic_get_config_node_3->Run(DATA), STATUS_SUCCESS);
  auto queue_3 = dynamic_get_config_node_4->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<Buffer>> buffer_vector_3;
  queue_3->PopBatch(&buffer_vector_3);
  std::string test_3 = "";
  buffer_vector_3[0]->Get("test", test_3);
  EXPECT_EQ(test_3, "nodes.test");
  queue_3->PushBatch(&buffer_vector_3);
}

TEST_F(NodeRunTest, ConditionSwitchRun) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto switch_case_node = Add_Switch_Case_Node();
  auto garther_node = Add_Garther_Node();
  auto add_node = Add_Add_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto condition_output_1_port = switch_case_node->GetOutputPort("Out_1");
  auto condition_output_2_port = switch_case_node->GetOutputPort("Out_2");
  auto condition_output_3_port = switch_case_node->GetOutputPort("Out_3");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  auto add_output_port = add_node->GetOutputPort("Out_1");

  EXPECT_TRUE(output_port_1->ConnectPort(scatter_node->GetInputPort("In_1")));
  EXPECT_TRUE(output_port_2->ConnectPort(add_node->GetInputPort("In_2")));
  EXPECT_TRUE(
      scatter_output_port->ConnectPort(switch_case_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      condition_output_1_port->ConnectPort(garther_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      condition_output_2_port->ConnectPort(garther_node->GetInputPort("In_1")));
  EXPECT_TRUE(
      condition_output_3_port->ConnectPort(garther_node->GetInputPort("In_1")));
  EXPECT_TRUE(garther_output_port->ConnectPort(add_node->GetInputPort("In_1")));
  EXPECT_TRUE(add_output_port->ConnectPort(input_node->GetInputPort("In_1")));

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(switch_case_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> buffer_vector;
  auto queue = garther_node->GetInputPort("In_1")->GetQueue();
  queue->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 16);  // contain 4 end_flag
  queue->PushBatch(&buffer_vector);
  buffer_vector.clear();

  EXPECT_EQ(garther_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(garther_node->Run(EVENT), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> add_vector_1;
  std::vector<std::shared_ptr<Buffer>> add_vector_2;
  auto add_queue_1 = add_node->GetInputPort("In_1")->GetQueue();
  auto add_queue_2 = add_node->GetInputPort("In_2")->GetQueue();
  add_queue_1->PopBatch(&add_vector_1);
  add_queue_2->PopBatch(&add_vector_2);
  EXPECT_EQ(add_vector_1.size(), 2);
  EXPECT_EQ(add_vector_2.size(), 2);
  add_queue_1->PushBatch(&add_vector_1);
  add_queue_2->PushBatch(&add_vector_2);
  add_vector_1.clear();
  add_vector_2.clear();

  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<Buffer>> final_buffer_vector;
  auto queue_4 = input_node->GetInputPort("In_1")->GetQueue();
  queue_4->PopBatch(&final_buffer_vector);
  EXPECT_EQ(final_buffer_vector.size(), 2);
  EXPECT_EQ(final_buffer_vector[0]->GetBytes(), 40);
  EXPECT_TRUE(
      BufferManageView::GetIndexInfo(final_buffer_vector[1])->IsEndFlag());
  auto add_data_result = (int*)final_buffer_vector[0]->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(add_data_result[i], 10 + 2 * i);
  }
  final_buffer_vector.clear();
}

}  // namespace modelbox
