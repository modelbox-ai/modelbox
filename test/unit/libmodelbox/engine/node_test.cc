
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

namespace modelbox {

using ::testing::Sequence;

void BuildDataEventStart(
    std::unordered_map<std::string, std::shared_ptr<IndexBufferList>>&
        input_map,
    std::shared_ptr<Device> device,
    std::shared_ptr<VirtualStream> virtual_stream) {
  std::vector<std::shared_ptr<IndexBuffer>> index_bf_list_1(1);
  std::vector<int> data_1 = {0, 25, 3};
  auto buf_1 = std::make_shared<Buffer>(device);
  buf_1->Build(3 * sizeof(int));
  auto dev_data_1 = (int*)buf_1->MutableData();
  for (size_t i = 0; i < data_1.size(); ++i) {
    dev_data_1[i] = data_1[i];
  }
  index_bf_list_1[0] = std::make_shared<IndexBuffer>(buf_1);
  auto index_buffer_list = std::make_shared<IndexBufferList>(index_bf_list_1);
  auto session_ctx = std::make_shared<SessionContext>();
  auto session_content = std::make_shared<int>(1111);
  session_ctx->SetPrivate("session", session_content);
  virtual_stream->SetSessionContext(session_ctx);
  virtual_stream->LabelIndexBuffer(index_buffer_list);
  virtual_stream->Close();
  input_map.emplace("In_1", index_buffer_list);
}

void BuildDataEventStop(
    std::unordered_map<std::string, std::shared_ptr<IndexBufferList>>&
        input_map,
    std::shared_ptr<VirtualStream> virtual_stream) {
  std::vector<std::shared_ptr<IndexBuffer>> index_bf_list_1(1);

  auto buf_1 = std::make_shared<Buffer>();
  buf_1->Build(1 * sizeof(int));
  auto error = std::make_shared<FlowUnitError>("test error");
  buf_1->SetError(error);

  index_bf_list_1[0] = std::make_shared<IndexBuffer>(buf_1);

  auto last_bg = virtual_stream->GetLastBufferGroup()->GenerateSameLevelGroup();
  index_bf_list_1[0]->SetBufferGroup(last_bg);
  auto index_buffer_list = std::make_shared<IndexBufferList>(index_bf_list_1);
  auto error_index = index_buffer_list->GetDataErrorIndex();
  last_bg->GetStreamLevelGroup()->SetDataError(error_index, error);

  input_map.emplace("In_1", index_buffer_list);
}

void BuildDataQueue(
    std::unordered_map<std::string, std::vector<std::shared_ptr<IndexBuffer>>>&
        input_map,
    std::shared_ptr<Device> device) {
  std::vector<std::shared_ptr<IndexBuffer>> index_bf_list_1(1);
  std::vector<std::shared_ptr<IndexBuffer>> index_bf_list_2(1);

  std::vector<int> data_1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> data_2 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  auto buf_1 = std::make_shared<Buffer>(device);
  buf_1->Build(10 * sizeof(int));
  auto dev_data_1 = (int*)buf_1->MutableData();
  for (size_t i = 0; i < data_1.size(); ++i) {
    dev_data_1[i] = data_1[i];
  }
  index_bf_list_1[0] = std::make_shared<IndexBuffer>(buf_1);
  index_bf_list_1[0]->SetPriority(0);
  index_bf_list_1[0]->BindToRoot();

  auto buf_2 = std::make_shared<Buffer>(device);
  buf_2->Build(10 * sizeof(int));
  auto dev_data_2 = (int*)buf_2->MutableData();
  for (size_t i = 0; i < data_2.size(); ++i) {
    dev_data_2[i] = data_2[i];
  }
  index_bf_list_2[0] = std::make_shared<IndexBuffer>(buf_2);
  index_bf_list_1[0]->CopyMetaTo(index_bf_list_2[0]);
  index_bf_list_2[0]->SetPriority(0);

  input_map.emplace("In_1", index_bf_list_1);
  input_map.emplace("In_2", index_bf_list_2);
}

void CheckQueueHasDataError(std::shared_ptr<BufferQueue> queue,
                            uint32_t queue_size) {
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  queue->PopBatch(&error_buffer_vector);
  auto error_index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = error_index_buffer_list->GetDataError();
  EXPECT_EQ(error_index_buffer_list->GetBufferNum(), queue_size);
  EXPECT_NE(error, nullptr);
  queue->PushBatch(&error_buffer_vector);
}

void CheckQueueNotHasDataError(std::shared_ptr<BufferQueue> queue) {
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  queue->PopBatch(&error_buffer_vector);
  auto error_index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = error_index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  queue->PushBatch(&error_buffer_vector);
}

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
  std::shared_ptr<Node> node_;
  std::vector<std::shared_ptr<IndexBuffer>> root_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> root_vector_2_;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_2_;
  std::vector<std::shared_ptr<IndexBuffer>> other_buffer_vector_1_;
  std::vector<std::shared_ptr<IndexBuffer>> other_buffer_vector_2_;

  std::shared_ptr<IndexBuffer> root_buffer_;
  std::shared_ptr<IndexBuffer> buffer_01_;
  std::shared_ptr<IndexBuffer> buffer_02_;
  std::shared_ptr<IndexBuffer> buffer_03_;
  std::shared_ptr<IndexBuffer> buffer_11_;
  std::shared_ptr<IndexBuffer> buffer_12_;

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
    node_->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config);

    single_match_ = node_->GetSingleMatchCache()->GetReceiveBuffer();
    group_match_ = node_->GetStreamMatchCache()->GetStreamReceiveBuffer();

    root_buffer_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindToRoot();

    buffer_01_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindDownLevelTo(buffer_01_, true, false);

    buffer_02_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindDownLevelTo(buffer_02_, false, false);

    buffer_03_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindDownLevelTo(buffer_03_, false, true);

    buffer_11_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindDownLevelTo(buffer_11_, true, false);

    buffer_12_ = std::make_shared<IndexBuffer>();
    root_buffer_->BindDownLevelTo(buffer_12_, false, true);

    root_vector_1_.push_back(root_buffer_);
    root_vector_2_.push_back(root_buffer_);

    buffer_vector_1_.push_back(buffer_03_);
    buffer_vector_1_.push_back(buffer_01_);
    buffer_vector_1_.push_back(buffer_11_);
    buffer_vector_1_.push_back(buffer_12_);

    buffer_vector_2_.push_back(buffer_11_);
    buffer_vector_2_.push_back(buffer_02_);
    buffer_vector_2_.push_back(buffer_12_);
    buffer_vector_2_.push_back(buffer_03_);
    buffer_vector_2_.push_back(buffer_01_);

    other_buffer_vector_1_.push_back(buffer_02_);

    other_buffer_vector_2_.push_back(buffer_01_);
    other_buffer_vector_2_.push_back(buffer_03_);
    other_buffer_vector_2_.push_back(buffer_11_);
    other_buffer_vector_2_.push_back(buffer_12_);
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
  void TestWrongAdd(std::string add_flowunit_name);

 protected:
  std::shared_ptr<MockFlow> flow_;
  std::shared_ptr<VirtualStream> virtual_stream_;
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
    virtual_stream_ = std::make_shared<VirtualStream>(nullptr, 0);
  };
  virtual void TearDown() { flow_->Destroy(); };
};

std::shared_ptr<Node> Add_Node(
    std::string name, std::set<std::string> inputs,
    std::set<std::string> outputs,
    std::shared_ptr<Configuration> config = nullptr) {
  if (config == nullptr) {
    ConfigurationBuilder configbuilder;
    config = configbuilder.Build();
  }
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto node = std::make_shared<Node>(name, "cpu", "0", flowunit_mgr, nullptr);
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
  EXPECT_CALL(*stream_end_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, DataPre(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, Process(testing::_)).Times(times).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  EXPECT_CALL(*stream_end_fu, DataGroupPost(testing::_))
      .Times(1)
      .InSequence(s1);
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
  EXPECT_CALL(*node_fu, DataGroupPre(testing::_)).Times(1).InSequence(s1);
  for (uint32_t i = 0; i < times; i++) {
    EXPECT_CALL(*node_fu, DataPre(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, Process(testing::_)).Times(1).InSequence(s1);
    EXPECT_CALL(*node_fu, DataPost(testing::_)).Times(1).InSequence(s1);
  }
  EXPECT_CALL(*node_fu, DataGroupPost(testing::_)).Times(1).InSequence(s1);
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
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<IndexBuffer>>>();
  BuildDataQueue(input_map_1, device_);

  auto input_map_2 =
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<IndexBuffer>>>();
  BuildDataQueue(input_map_2, device_);

  auto add_output_port = add_node->GetOutputPort("Out_1");
  EXPECT_EQ(add_output_port->AddPort(input_node->GetInputPort("In_1")), true);

  auto add_queue_1 = add_node->GetInputPort("In_1")->GetQueue();
  auto add_queue_2 = add_node->GetInputPort("In_2")->GetQueue();
  add_queue_1->PushBatch(&input_map_1["In_1"]);
  add_queue_2->PushBatch(&input_map_1["In_2"]);

  add_queue_1->PushBatch(&input_map_2["In_1"]);
  add_queue_2->PushBatch(&input_map_2["In_2"]);

  auto queue_1 = input_node->GetInputPort("In_1")->GetQueue();

  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);

  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(queue_1->Size(), 2);

  std::vector<std::shared_ptr<IndexBuffer>> buffer_vecort_0;
  queue_1->PopBatch(&buffer_vecort_0);
  EXPECT_NE(buffer_vecort_0[0]->GetSameLevelGroup(),
            buffer_vecort_0[1]->GetSameLevelGroup());
  EXPECT_EQ(buffer_vecort_0[0]->GetBufferPtr()->GetBytes(), 40);
  EXPECT_EQ(buffer_vecort_0[1]->GetBufferPtr()->GetBytes(), 40);

  auto data_result = (int*)buffer_vecort_0[0]->GetBufferPtr()->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result[i], 10 + 2 * i);
  }

  auto data_result_2 = (int*)buffer_vecort_0[1]->GetBufferPtr()->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result_2[i], 10 + 2 * i);
  }
}

TEST_F(NodeTest, Init) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>("test_2_inputs_2_outputs", "cpu", "0",
                                      flowunit_mgr_, nullptr);
  EXPECT_EQ(node_->Init({"In_1", "In_2"}, {"Out_1"}, config), STATUS_BADCONF);
  EXPECT_EQ(node_->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config),
            STATUS_SUCCESS);
  EXPECT_EQ(node_->GetInputNum(), 2);
  EXPECT_EQ(node_->GetOutputNum(), 2);
  EXPECT_NE(node_->GetInputPort("In_1"), nullptr);
  EXPECT_NE(node_->GetInputPort("In_2"), nullptr);
  EXPECT_NE(node_->GetOutputPort("Out_1"), nullptr);
  EXPECT_NE(node_->GetOutputPort("Out_2"), nullptr);

  EXPECT_EQ(node_->GetOutputPort("In_None"), nullptr);

  auto another_node_ =
      std::make_shared<Node>("test_2_0", "cpu", "0", flowunit_mgr_, nullptr);
  EXPECT_EQ(another_node_->Init({"In_1", "In_1", "In_2"}, {}, config),
            STATUS_SUCCESS);
  EXPECT_EQ(another_node_->GetInputNum(), 2);
  EXPECT_EQ(another_node_->GetOutputNum(), 0);
  EXPECT_NE(another_node_->GetInputPort("In_1"), nullptr);

  auto invalid_node = std::make_shared<Node>("invalid_test", "cpu", "0",
                                             flowunit_mgr_, nullptr);
  EXPECT_EQ(another_node_->Init({}, {}, config), STATUS_BADCONF);
}

TEST_F(NodeTest, SendEvent) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ =
      std::make_shared<Node>("test_0_2", "cpu", "0", flowunit_mgr_, nullptr);

  EXPECT_EQ(node_->Init({}, {"Out_1", "Out_2"}, config), STATUS_OK);

  auto event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  auto event_vector = std::vector<std::shared_ptr<FlowUnitInnerEvent>>();
  event_vector.push_back(event);
  EXPECT_EQ(node_->SendBatchEvent(event_vector), STATUS_OK);
  FlowunitEventList events = nullptr;
  EXPECT_EQ(node_->GetEventPort()->Recv(events), STATUS_OK);
  EXPECT_EQ(events->size(), 1);
  EXPECT_EQ(events->at(0), event);
}

TEST_F(NodeTest, CreateInputBuffer) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>("test_2_inputs_2_outputs", "cpu", "0",
                                      flowunit_mgr_, nullptr);

  EXPECT_EQ(node_->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config),
            STATUS_SUCCESS);
  auto input_map = node_->CreateInputBuffer();
  EXPECT_EQ(input_map.size(), 2);
  EXPECT_NE(input_map.find("In_1"), input_map.end());
  EXPECT_NE(input_map.find("In_2"), input_map.end());
}

TEST_F(NodeTest, CreateOuputBuffer) {
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();
  auto node_ = std::make_shared<Node>("test_2_inputs_2_outputs", "cpu", "0",
                                      flowunit_mgr_, nullptr);

  EXPECT_EQ(node_->Init({"In_1", "In_2"}, {"Out_1", "Out_2"}, config),
            STATUS_SUCCESS);
  auto output_map = node_->CreateOutputBuffer();
  EXPECT_EQ(output_map.size(), 2);
  EXPECT_NE(output_map.find("Out_1"), output_map.end());
  EXPECT_NE(output_map.find("Out_2"), output_map.end());
}

TEST_F(NodeRecvTest, RecvEmpty) {
  auto port_1 = node_->GetInputPort("In_1");
  auto port_2 = node_->GetInputPort("In_2");
  auto input_buffer = node_->CreateInputBuffer();
  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_NE(input_buffer.end(), input_buffer.find("In_1"));
  EXPECT_NE(input_buffer.end(), input_buffer.find("In_2"));

  EXPECT_EQ(0, input_buffer["In_1"].size());
  EXPECT_EQ(0, input_buffer["In_2"].size());
}

TEST_F(NodeRecvTest, RecvRoot) {
  auto input_buffer = node_->CreateInputBuffer();
  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&root_vector_1_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(0, input_buffer["In_1"].size());
  EXPECT_EQ(0, input_buffer["In_2"].size());

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&root_vector_2_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(1, input_buffer["In_1"].size());
  EXPECT_EQ(1, input_buffer["In_2"].size());

  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(0)->GetSameLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(0)->GetSameLevelGroup());

  EXPECT_EQ(0, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
}

TEST_F(NodeRecvTest, RecvSingle) {
  auto input_buffer = node_->CreateInputBuffer();
  node_->SetInputOrder(false);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&buffer_vector_2_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(2, input_buffer["In_1"].size());
  EXPECT_EQ(2, input_buffer["In_2"].size());

  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(0)->GetSameLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(0)->GetSameLevelGroup());
  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(1)->GetSameLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(1)->GetSameLevelGroup());
  EXPECT_EQ(input_buffer["In_1"][1]->GetBuffer(0)->GetSameLevelGroup(),
            input_buffer["In_2"][1]->GetBuffer(0)->GetSameLevelGroup());
  EXPECT_EQ(input_buffer["In_1"][1]->GetBuffer(1)->GetSameLevelGroup(),
            input_buffer["In_2"][1]->GetBuffer(1)->GetSameLevelGroup());

  EXPECT_EQ(1, single_match_->size());
  EXPECT_EQ(0, group_match_->size());

  EXPECT_EQ(0, input_queue_1->Size());
  EXPECT_EQ(0, input_queue_2->Size());
}

TEST_F(NodeRecvTest, RecvOrder) {
  auto input_buffer = node_->CreateInputBuffer();
  node_->SetFlowType(STREAM);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&buffer_vector_2_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(2, input_buffer["In_1"].size());
  EXPECT_EQ(2, input_buffer["In_2"].size());
  auto first_size = input_buffer["In_1"][0]->GetBufferNum();
  auto second_size = input_buffer["In_1"][1]->GetBufferNum();
  for (uint32_t i = 0; i < first_size; i++) {
    EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(i)->GetSameLevelGroup(),
              input_buffer["In_2"][0]->GetBuffer(i)->GetSameLevelGroup());
    EXPECT_EQ(
        input_buffer["In_1"][0]->GetBuffer(i)->GetSameLevelGroup()->GetOrder(),
        i + 1);
  }

  for (uint32_t i = 0; i < second_size; i++) {
    EXPECT_EQ(input_buffer["In_1"][1]->GetBuffer(i)->GetSameLevelGroup(),
              input_buffer["In_2"][1]->GetBuffer(i)->GetSameLevelGroup());
    EXPECT_EQ(
        input_buffer["In_1"][1]->GetBuffer(i)->GetSameLevelGroup()->GetOrder(),
        i + 1);
  }

  EXPECT_EQ(1, single_match_->size());
  EXPECT_EQ(1, group_match_->size());
}

TEST_F(NodeRecvTest, RecvGroupOrder) {
  auto input_buffer = node_->CreateInputBuffer();
  node_->SetOutputType(COLLAPSE);
  node_->SetInputGatherAll(true);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&buffer_vector_2_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(1, input_buffer["In_1"].size());
  EXPECT_EQ(1, input_buffer["In_2"].size());

  EXPECT_EQ(
      input_buffer["In_1"][0]->GetBuffer(0)->GetSameLevelGroup()->GetOrder(),
      1);
  EXPECT_EQ(
      input_buffer["In_1"][0]->GetBuffer(1)->GetSameLevelGroup()->GetOrder(),
      2);

  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(0)->GetSameLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(0)->GetSameLevelGroup());
  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(1)->GetSameLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(1)->GetSameLevelGroup());

  EXPECT_EQ(input_buffer["In_1"][0]->GetBuffer(0)->GetStreamLevelGroup(),
            input_buffer["In_2"][0]->GetBuffer(1)->GetStreamLevelGroup());

  EXPECT_EQ(1, single_match_->size());
  EXPECT_EQ(1, group_match_->size());
}

TEST_F(NodeRecvTest, RecvTwice) {
  auto input_buffer = node_->CreateInputBuffer();
  node_->SetFlowType(STREAM);
  node_->SetOutputType(COLLAPSE);
  node_->SetInputGatherAll(true);

  auto port_1 = node_->GetInputPort("In_1");
  auto input_queue_1 = port_1->GetQueue();
  input_queue_1->PushBatch(&buffer_vector_1_);

  auto port_2 = node_->GetInputPort("In_2");
  auto input_queue_2 = port_2->GetQueue();
  input_queue_2->PushBatch(&buffer_vector_2_);

  EXPECT_EQ(node_->RecvDataQueue(&input_buffer), STATUS_SUCCESS);

  input_queue_1->PushBatch(&other_buffer_vector_1_);
  auto other_input_buffer = node_->CreateInputBuffer();
  EXPECT_EQ(node_->RecvDataQueue(&other_input_buffer), STATUS_SUCCESS);
  EXPECT_EQ(1, other_input_buffer["In_1"].size());
  EXPECT_EQ(1, other_input_buffer["In_2"].size());

  EXPECT_EQ(other_input_buffer["In_1"][0]->GetBuffer(0)->GetSameLevelGroup(),
            other_input_buffer["In_2"][0]->GetBuffer(0)->GetSameLevelGroup());
  EXPECT_EQ(other_input_buffer["In_1"][0]->GetBuffer(1)->GetSameLevelGroup(),
            other_input_buffer["In_2"][0]->GetBuffer(1)->GetSameLevelGroup());
  EXPECT_EQ(other_input_buffer["In_1"][0]->GetBuffer(2)->GetSameLevelGroup(),
            other_input_buffer["In_2"][0]->GetBuffer(2)->GetSameLevelGroup());

  EXPECT_EQ(other_input_buffer["In_1"][0]
                ->GetBuffer(0)
                ->GetSameLevelGroup()
                ->GetOrder(),
            1);
  EXPECT_EQ(other_input_buffer["In_1"][0]
                ->GetBuffer(1)
                ->GetSameLevelGroup()
                ->GetOrder(),
            2);
  EXPECT_EQ(other_input_buffer["In_1"][0]
                ->GetBuffer(2)
                ->GetSameLevelGroup()
                ->GetOrder(),
            3);

  EXPECT_EQ(0, single_match_->size());
  EXPECT_EQ(0, group_match_->size());
}

TEST_F(NodeRunTest, OnlyOneOutputRun) {
  auto output_node = Add_Test_0_2_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_port_1->AddPort(input_node->GetInputPort("In_1")), true);
  EXPECT_EQ(output_port_2->AddPort(input_node->GetInputPort("In_2")), true);
  output_node->Run(DATA);

  auto queue_1 = input_node->GetInputPort("In_1")->GetQueue();
  auto queue_2 = input_node->GetInputPort("In_2")->GetQueue();

  EXPECT_EQ(queue_1->Size(), 1);
  EXPECT_EQ(queue_2->Size(), 1);

  std::vector<std::shared_ptr<IndexBuffer>> buffer_vecort_0;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vecort_1;
  queue_1->PopBatch(&buffer_vecort_0);
  queue_2->PopBatch(&buffer_vecort_1);
  EXPECT_EQ(buffer_vecort_0[0]->GetSameLevelGroup(),
            buffer_vecort_1[0]->GetSameLevelGroup());
  EXPECT_EQ(buffer_vecort_0[0]->GetBufferPtr()->GetBytes(), 40);
  EXPECT_EQ(buffer_vecort_1[0]->GetBufferPtr()->GetBytes(), 40);
  auto data_result_0 = (int*)buffer_vecort_0[0]->GetBufferPtr()->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result_0[i], i);
  }
  auto data_result_1 = (int*)buffer_vecort_1[0]->GetBufferPtr()->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result_1[i], i + 10);
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
  EXPECT_EQ(output_port_1->AddPort(condition_node->GetInputPort("In_1")), true);
  auto condition_port_1 = condition_node->GetOutputPort("Out_1");
  EXPECT_EQ(condition_port_1->AddPort(expand_node->GetInputPort("In_1")), true);
  auto expand_port_1 = expand_node->GetOutputPort("Out_1");
  EXPECT_EQ(expand_port_1->AddPort(collapse_node->GetInputPort("In_1")), true);
  auto condition_port_2 = condition_node->GetOutputPort("Out_2");
  EXPECT_EQ(condition_port_2->AddPort(stream_add_node->GetInputPort("In_1")),
            true);
  auto collapse_port_1 = collapse_node->GetOutputPort("Out_1");
  EXPECT_EQ(collapse_port_1->AddPort(stream_add_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(output_port_2->AddPort(stream_add_node->GetInputPort("In_2")),
            true);
  auto add_port = stream_add_node->GetOutputPort("Out_1");
  EXPECT_EQ(add_port->AddPort(input_node->GetInputPort("In_1")), true);

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_add_node->Run(DATA), STATUS_SUCCESS);

  auto queue_1 = input_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
  queue_1->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 10);
  for (int i = 0; i < 10; i++) {
    auto data_result = (int*)buffer_vector[i]->GetBufferPtr()->ConstData();
    if (i % 2 == 0) {
      EXPECT_EQ(data_result[0], 20 + 6 * i);
    } else {
      EXPECT_EQ(data_result[0], 10 + 2 * i);
    }
  }
  queue_1->PushBatch(&buffer_vector);
  buffer_vector.clear();
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
  EXPECT_EQ(output_port_1->AddPort(expand_node->GetInputPort("In_1")), true);
  EXPECT_EQ(expand_node_port->AddPort(simple_pass_node->GetInputPort("In_1")), true);
  EXPECT_EQ(stream_add_port->AddPort(collapse_node->GetInputPort("In_1")), true);
  EXPECT_EQ(collapse_node_port->AddPort(input_node->GetInputPort("In_1")), true);

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(input_node->Run(DATA), STATUS_SUCCESS);

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
  EXPECT_EQ(output_port_1->AddPort(expand_node->GetInputPort("In_1")), true);
  EXPECT_EQ(expand_node_port->AddPort(simple_pass_node->GetInputPort("In_1")), true);
  EXPECT_EQ(stream_add_port->AddPort(collapse_node->GetInputPort("In_1")), true);
  EXPECT_EQ(collapse_node_port->AddPort(input_node->GetInputPort("In_1")), true);

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(input_node->Run(DATA), STATUS_SUCCESS);

}

TEST_F(NodeRunTest, StreamGartherScatterRun) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto garther_node = Add_Garther_Node();
  auto add_node = Add_Add_Node();
  auto input_node = Add_Test_2_0_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  auto add_output_port = add_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_port_1->AddPort(scatter_node->GetInputPort("In_1")), true);
  EXPECT_EQ(output_port_2->AddPort(add_node->GetInputPort("In_2")), true);
  EXPECT_EQ(scatter_output_port->AddPort(garther_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(garther_output_port->AddPort(add_node->GetInputPort("In_1")), true);
  EXPECT_EQ(add_output_port->AddPort(input_node->GetInputPort("In_1")), true);

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
  auto queue_1 = garther_node->GetInputPort("In_1")->GetQueue();
  queue_1->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 10);
  EXPECT_NE(buffer_vector[0]->GetSameLevelGroup(),
            buffer_vector[1]->GetSameLevelGroup());
  for (int i = 0; i < 10; i++) {
    auto data_result = (int*)buffer_vector[i]->GetBufferPtr()->ConstData();
    EXPECT_EQ(data_result[0], i);
  }
  queue_1->PushBatch(&buffer_vector);
  buffer_vector.clear();

  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_one;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_two;
  EXPECT_EQ(garther_node->Run(DATA), STATUS_SUCCESS);
  auto queue_2 = add_node->GetInputPort("In_1")->GetQueue();
  auto queue_3 = add_node->GetInputPort("In_2")->GetQueue();
  queue_2->PopBatch(&buffer_vector_one);
  queue_3->PopBatch(&buffer_vector_two);
  EXPECT_EQ(buffer_vector_one.size(), 1);
  EXPECT_EQ(buffer_vector_two.size(), 1);
  EXPECT_EQ(buffer_vector_one[0]->GetSameLevelGroup(),
            buffer_vector_two[0]->GetSameLevelGroup());
  EXPECT_EQ(buffer_vector_two[0]->GetBufferPtr()->GetBytes(), 40);
  EXPECT_EQ(buffer_vector_one[0]->GetBufferPtr()->GetBytes(), 40);

  auto data_result = (int*)buffer_vector_one[0]->GetBufferPtr()->ConstData();
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(data_result[i], i);
  }
  queue_2->PushBatch(&buffer_vector_one);
  queue_3->PushBatch(&buffer_vector_two);
  buffer_vector_two.clear();
  buffer_vector_one.clear();

  std::vector<std::shared_ptr<IndexBuffer>> final_buffer_vector;
  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);
  auto queue_4 = input_node->GetInputPort("In_1")->GetQueue();
  queue_4->PopBatch(&final_buffer_vector);
  EXPECT_EQ(final_buffer_vector.size(), 1);
  EXPECT_EQ(final_buffer_vector[0]->GetBufferPtr()->GetBytes(), 40);
  auto add_data_result =
      (int*)final_buffer_vector[0]->GetBufferPtr()->ConstData();
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

  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto condition_output_1_port = condition_node->GetOutputPort("Out_1");
  auto condition_output_2_port = condition_node->GetOutputPort("Out_2");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  auto add_output_port = add_node->GetOutputPort("Out_1");
  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_port_1->AddPort(scatter_node->GetInputPort("In_1")), true);
  EXPECT_EQ(output_port_2->AddPort(add_node->GetInputPort("In_2")), true);
  EXPECT_EQ(scatter_output_port->AddPort(condition_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(
      condition_output_1_port->AddPort(garther_node->GetInputPort("In_1")),
      true);
  EXPECT_EQ(
      condition_output_2_port->AddPort(garther_node->GetInputPort("In_1")),
      true);
  EXPECT_EQ(garther_output_port->AddPort(add_node->GetInputPort("In_1")), true);
  EXPECT_EQ(add_output_port->AddPort(input_node->GetInputPort("In_1")), true);

  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(scatter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
  auto queue = garther_node->GetInputPort("In_1")->GetQueue();
  queue->PopBatch(&buffer_vector);
  EXPECT_EQ(buffer_vector.size(), 10);
  for (uint32_t i = 1; i < 10; i++) {
    EXPECT_EQ(buffer_vector[0]->GetStreamLevelGroup(),
              buffer_vector[i]->GetStreamLevelGroup());
  }
  queue->PushBatch(&buffer_vector);
  buffer_vector.clear();

  EXPECT_EQ(garther_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<IndexBuffer>> add_vector_1;
  std::vector<std::shared_ptr<IndexBuffer>> add_vector_2;
  auto add_queue_1 = add_node->GetInputPort("In_1")->GetQueue();
  auto add_queue_2 = add_node->GetInputPort("In_2")->GetQueue();
  add_queue_1->PopBatch(&add_vector_1);
  add_queue_2->PopBatch(&add_vector_2);
  EXPECT_EQ(add_vector_1.size(), 1);
  EXPECT_EQ(add_vector_2.size(), 1);

  EXPECT_EQ(add_vector_1[0]->GetStreamLevelGroup(),
            add_vector_2[0]->GetStreamLevelGroup());

  add_queue_1->PushBatch(&add_vector_1);
  add_queue_2->PushBatch(&add_vector_2);
  add_vector_1.clear();
  add_vector_2.clear();

  EXPECT_EQ(add_node->Run(DATA), STATUS_SUCCESS);

  std::vector<std::shared_ptr<IndexBuffer>> final_buffer_vector;
  auto queue_4 = input_node->GetInputPort("In_1")->GetQueue();
  queue_4->PopBatch(&final_buffer_vector);
  EXPECT_EQ(final_buffer_vector.size(), 1);
  EXPECT_EQ(final_buffer_vector[0]->GetBufferPtr()->GetBytes(), 40);
  auto add_data_result =
      (int*)final_buffer_vector[0]->GetBufferPtr()->ConstData();
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
  EXPECT_EQ(output_0_1_port->AddPort(loop_node->GetInputPort("In_1")), true);
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_port = loop_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_loop_port->AddPort(loop_node->GetInputPort("In_1")), true);
  auto output_loop_end_port = loop_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_loop_end_port->AddPort(end_node->GetInputPort("In_1")),
            true);
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
      EXPECT_EQ(queue->Size(), 20);
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
  EXPECT_EQ(output_0_1_port->AddPort(loop_node->GetInputPort("In_1")), true);
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_port = loop_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_loop_port->AddPort(loop_node->GetInputPort("In_1")), true);
  auto output_loop_end_port = loop_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_loop_end_port->AddPort(end_node->GetInputPort("In_1")),
            true);
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
    for (int i = 0; i < 10; i++) {
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
  EXPECT_EQ(output_0_1_port->AddPort(loop_node->GetInputPort("In_1")), true);
  auto input_ports = output_0_1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(0);
  }

  auto output_loop_output1_port = loop_node->GetOutputPort("Out_1");
  auto output_loop_output2_port = loop_node->GetOutputPort("Out_2");
  EXPECT_EQ(
      output_loop_output1_port->AddPort(loop_end_node->GetInputPort("In_1")),
      true);
  EXPECT_EQ(output_loop_output2_port->AddPort(end_node->GetInputPort("In_1")),
            true);
  input_ports = output_loop_output1_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  input_ports = output_loop_output2_port->GetConnectInPort();
  for (auto& input_port : input_ports) {
    input_port->SetPriority(1);
  }

  auto output_loop_end_port = loop_end_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_loop_end_port->AddPort(loop_node->GetInputPort("In_1")),
            true);

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
  EXPECT_EQ(start_info_port->AddPort(stream_start_node->GetInputPort("In_1")),
            true);
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_EQ(start_output_port->AddPort(simple_pass_node->GetInputPort("In_1")),
            true);

  auto simple_pass_port = simple_pass_node->GetOutputPort("Out_1");
  EXPECT_EQ(simple_pass_port->AddPort(stream_mid_node->GetInputPort("In_1")),
            true);

  auto mid_output_port = stream_mid_node->GetOutputPort("Out_1");
  EXPECT_EQ(mid_output_port->AddPort(stream_end_node->GetInputPort("In_1")),
            true);
  auto end_output_port = stream_end_node->GetOutputPort("Out_1");
  EXPECT_EQ(end_output_port->AddPort(final_input_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<IndexBuffer>> start_buffer_vector;
  auto queue_2 = stream_mid_node->GetInputPort("In_1")->GetQueue();
  queue_2->PopBatch(&start_buffer_vector);
  EXPECT_EQ(start_buffer_vector.size(), 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(start_buffer_vector[0]->GetStreamLevelGroup(),
              start_buffer_vector[i]->GetStreamLevelGroup());
    auto data_result =
        (int*)start_buffer_vector[i]->GetBufferPtr()->ConstData();
    EXPECT_EQ(data_result[0], i);
  }
  auto data_meta = start_buffer_vector[0]->GetDataMeta();

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
  std::vector<std::shared_ptr<IndexBuffer>> mid_buffer_vector;
  auto queue_3 = stream_end_node->GetInputPort("In_1")->GetQueue();
  queue_3->PopBatch(&mid_buffer_vector);
  EXPECT_EQ(mid_buffer_vector.size(), 2);
  EXPECT_EQ(mid_buffer_vector[0]->GetStreamLevelGroup(),
            mid_buffer_vector[1]->GetStreamLevelGroup());
  auto data_result_0 = (int*)mid_buffer_vector[0]->GetBufferPtr()->ConstData();
  auto data_result_1 = (int*)mid_buffer_vector[1]->GetBufferPtr()->ConstData();

  auto data_group_meta_1 = mid_buffer_vector[0]->GetGroupDataMeta();
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
    EXPECT_EQ(start_buffer_vector[0]->GetStreamLevelGroup(),
              start_buffer_vector[i]->GetStreamLevelGroup());
    auto data_result =
        (int*)start_buffer_vector[i]->GetBufferPtr()->ConstData();
    EXPECT_EQ(data_result[0], i + 5);
  }

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  queue_2->PopBatch(&start_buffer_vector);
  EXPECT_EQ(start_buffer_vector.size(), 10);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(start_buffer_vector[0]->GetStreamLevelGroup(),
              start_buffer_vector[i]->GetStreamLevelGroup());
    auto data_result =
        (int*)start_buffer_vector[i]->GetBufferPtr()->ConstData();
    EXPECT_EQ(data_result[0], i + 5);
  }
  queue_2->PushBatch(&start_buffer_vector);
  start_buffer_vector.clear();

  EXPECT_EQ(stream_mid_node->Run(DATA), STATUS_SUCCESS);
  queue_3->PopBatch(&mid_buffer_vector);
  EXPECT_EQ(mid_buffer_vector.size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mid_buffer_vector[0]->GetStreamLevelGroup(),
              mid_buffer_vector[i]->GetStreamLevelGroup());
    auto data_result = (int*)mid_buffer_vector[i]->GetBufferPtr()->ConstData();
    EXPECT_EQ(data_result[0], 6 + i * 3);
  }
  queue_3->PushBatch(&mid_buffer_vector);
  mid_buffer_vector.clear();

  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<IndexBuffer>> end_buffer_vector;
  queue_4->PopBatch(&end_buffer_vector);
  EXPECT_EQ(end_buffer_vector.size(), 1);
  auto final_result = (int*)end_buffer_vector[0]->GetBufferPtr()->ConstData();
  EXPECT_EQ(final_result[0], 30);
}

void NodeRunTest::TestWrongAdd(std::string flowunit_name) {
  ConfigurationBuilder configbuilderflowunit;
  auto config_flowunit = configbuilderflowunit.Build();
  config_flowunit->SetProperty("need_check_output", true);
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();

  auto output_node = Add_Test_0_2_Node();
  auto wrong_add_node =
      std::make_shared<Node>(flowunit_name, "cpu", "0", flowunit_mgr_, nullptr);
  EXPECT_EQ(wrong_add_node->Init({"In_1", "In_2"}, {"Out_1"}, config_flowunit),
            STATUS_SUCCESS);
  EXPECT_EQ(wrong_add_node->Open(), STATUS_SUCCESS);

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_port_1->AddPort(wrong_add_node->GetInputPort("In_1")), true);
  EXPECT_EQ(output_port_2->AddPort(wrong_add_node->GetInputPort("In_2")), true);
  EXPECT_EQ(output_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(wrong_add_node->Run(DATA), STATUS_STOP);
}

TEST_F(NodeRunTest, Run_Normal_Count_InSame) { TestWrongAdd("wrong_add"); }

TEST_F(NodeRunTest, Run_Normal_Count_InSame_2) { TestWrongAdd("wrong_add_2"); }

TEST_F(NodeRunTest, Run_Collapse_Not_One) {
  auto output_node = Add_Test_0_2_Node();
  auto scatter_node = Add_Scatter_Node();
  auto garther_node = Add_Garther_Gen_More_Node();

  auto output_port_1 = output_node->GetOutputPort("Out_1");
  auto output_port_2 = output_node->GetOutputPort("Out_2");
  auto scatter_output_port = scatter_node->GetOutputPort("Out_1");
  auto garther_output_port = garther_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_port_1->AddPort(scatter_node->GetInputPort("In_1")), true);
  EXPECT_EQ(scatter_output_port->AddPort(garther_node->GetInputPort("In_1")),
            true);

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
  EXPECT_EQ(output_port_1->AddPort(pass_node->GetInputPort("In_1")), true);
  auto output_port_2 = start_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_port_1->AddPort(receive_node->GetInputPort("In_1")), true);
  auto add_output_port_1 = pass_node->GetOutputPort("Out_1");
  EXPECT_EQ(add_output_port_1->AddPort(receive_node->GetInputPort("In_2")),
            true);

  auto Start_Receive = [receive_node, pass_node] {
    auto queue_1 = receive_node->GetInputPort("In_1")->GetQueue();
    auto queue_2 = receive_node->GetInputPort("In_2")->GetQueue();

    sleep(1);
    EXPECT_EQ(queue_1->Size(), 10);
    EXPECT_EQ(queue_2->Size(), 0);
    EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);

    EXPECT_EQ(pass_node->Run(DATA), STATUS_SUCCESS);

    EXPECT_EQ(queue_1->Size(), 5);
    EXPECT_EQ(queue_2->Size(), 10);

    EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
    EXPECT_EQ(queue_1->Size(), 5);
    EXPECT_EQ(queue_2->Size(), 5);

    EXPECT_EQ(receive_node->Run(DATA), STATUS_STOP);
    EXPECT_EQ(queue_1->Size(), 0);
    EXPECT_EQ(queue_2->Size(), 0);
  };
  auto recieve_thread = std::make_shared<std::thread>(Start_Receive);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);

  recieve_thread->join();
}

TEST_F(NodeRunTest, EnlargeCache) {
  ConfigurationBuilder test_2_0_configbuilder;
  test_2_0_configbuilder.AddProperty("queue_size", "5");
  auto config = test_2_0_configbuilder.Build();

  auto start_node = Add_Test_Orgin_0_2_Node();
  auto condition_node = Add_Half_Condition_Node();
  auto pass_node = Add_Simple_Pass_Node(5);
  auto receive_node = Add_Test_2_0_Node(config);

  auto output_port_1 = start_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_port_1->AddPort(condition_node->GetInputPort("In_1")), true);
  auto condition_output_port_1 = condition_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      condition_output_port_1->AddPort(receive_node->GetInputPort("In_1")),
      true);
  auto condition_output_port_2 = condition_node->GetOutputPort("Out_2");
  EXPECT_EQ(condition_output_port_2->AddPort(pass_node->GetInputPort("In_1")),
            true);
  auto pass_output_port = pass_node->GetOutputPort("Out_1");
  EXPECT_EQ(pass_output_port->AddPort(receive_node->GetInputPort("In_1")),
            true);
  auto output_port_2 = start_node->GetOutputPort("Out_2");
  EXPECT_EQ(output_port_2->AddPort(receive_node->GetInputPort("In_2")), true);

  auto Start_Receive = [receive_node, pass_node] {
    auto queue_1 = receive_node->GetInputPort("In_1")->GetQueue();
    auto queue_2 = receive_node->GetInputPort("In_2")->GetQueue();

    sleep(1);
    EXPECT_EQ(receive_node->GetSingleMatchCache()->GetLimitCount(), 5);
    EXPECT_EQ(queue_1->Size(), 5);
    EXPECT_EQ(queue_2->Size(), 10);

    EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
    EXPECT_EQ(receive_node->GetSingleMatchCache()->GetLimitCount(), 10);
    EXPECT_EQ(queue_1->Size(), 0);
    EXPECT_EQ(queue_2->Size(), 5);

    EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
    EXPECT_EQ(receive_node->GetSingleMatchCache()->GetLimitCount(), 10);
    EXPECT_EQ(queue_1->Size(), 0);
    EXPECT_EQ(queue_2->Size(), 0);

    EXPECT_EQ(pass_node->Run(DATA), STATUS_SUCCESS);
    EXPECT_EQ(receive_node->Run(DATA), STATUS_STOP);
    EXPECT_EQ(receive_node->GetSingleMatchCache()->GetLimitCount(), 5);
    EXPECT_EQ(queue_1->Size(), 0);
    EXPECT_EQ(queue_2->Size(), 0);
  };
  auto recieve_thread = std::make_shared<std::thread>(Start_Receive);
  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);

  recieve_thread->join();
}

// thread_pool has not implement set priority
TEST_F(NodeRunTest, DISABLED_RunPriority) {
  auto device_ = flow_->GetDevice();
  auto flowunit_mgr_ = FlowUnitManager::GetInstance();

  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty("batch_size", "5");
  auto config = configbuilder.Build();
  auto run_node = std::make_shared<Node>("get_priority", "cpu", "0",
                                         flowunit_mgr_, nullptr);
  auto print_node =
      std::make_shared<Node>("print", "cpu", "0", flowunit_mgr_, nullptr);
  EXPECT_EQ(run_node->Init({"In_1"}, {"Out_1"}, config), STATUS_SUCCESS);
  EXPECT_EQ(run_node->Open(), STATUS_SUCCESS);
  EXPECT_EQ(print_node->Init({"In_1"}, {}, config), STATUS_SUCCESS);
  EXPECT_EQ(print_node->Open(), STATUS_SUCCESS);

  auto output_port_1 = run_node->GetOutputPort("Out_1");
  EXPECT_EQ(output_port_1->AddPort(print_node->GetInputPort("In_1")), true);

  int32_t default_priority = 3;
  size_t data_size = 5;
  size_t buffer_size = 3;
  std::vector<std::shared_ptr<IndexBuffer>> in_data(data_size * buffer_size,
                                                    nullptr);
  for (size_t i = 0; i < buffer_size; ++i) {
    for (size_t j = 0; j < data_size; ++j) {
      auto buffer = std::make_shared<Buffer>(device_);
      buffer->Build(1 * sizeof(int));
      auto indexbuffer = std::make_shared<IndexBuffer>(buffer);
      indexbuffer->BindToRoot();
      indexbuffer->SetPriority(default_priority + i);
      in_data[i * data_size + j] = indexbuffer;
    }
  }

  auto in_queue = run_node->GetInputPort("In_1")->GetQueue();
  in_queue->PushBatch(&in_data);

  EXPECT_EQ(in_queue->Size(), data_size * buffer_size);
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(run_node->Run(DATA), STATUS_SUCCESS);

    auto out_queue = print_node->GetInputPort("In_1")->GetQueue();
    std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
    out_queue->PopBatch(&buffer_vector);

    EXPECT_EQ(buffer_vector.size(), data_size);
    for (size_t i = 0; i < data_size; i++) {
      EXPECT_EQ(buffer_vector[i]->GetBufferPtr()->GetBytes(), 1 * sizeof(int));
      auto data_result = (int*)buffer_vector[i]->GetBufferPtr()->ConstData();
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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
                error_end_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 5);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 25);

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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
                error_end_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  auto recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 5);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);

  recieve_queue = error_end_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 25);

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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 1);
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 1);

  EXPECT_FALSE(error_buffer_vector[0]->GetBufferPtr()->HasError());

  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Process_Error_Collapse_Invisible) {
  auto start_node = Add_Error_Start_Normal_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto collapse_node = Add_Normal_Collapse_Process_Node(0, false);
  auto receive_node = Add_Stream_Process_Node({1, 1, 1});

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_node");
  simple_pass_node->SetName("simple_pass_node");
  collapse_node->SetName("collapse_node");
  receive_node->SetName("receive_node");
  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 1);

  EXPECT_TRUE(error_buffer_vector[0]->GetBufferPtr()->HasError());

  receive_queue->PushBatch(&error_buffer_vector);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Process_Error) {
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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                recieve_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<IndexBuffer>> index_buffer_vector;
  auto queue = recieve_node->GetInputPort("In_1")->GetQueue();
  queue->PopBatch(&index_buffer_vector);

  auto index_buffer_list =
      std::make_shared<IndexBufferList>(index_buffer_vector);
  auto error = index_buffer_list->GetDataError();
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

TEST_F(NodeRunTest, Normal_Recv_InVisible_Error) {
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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Normal_Recv_Visible_Error) {
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
  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Normal_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_DataPre_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(2);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_error_node->SetName("simple_error_node");
  receive_node->SetName("receive_node");

  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Process_Error) {
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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_error_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Recv_Visible_Error) {
  auto error_start_node = Add_Error_Start_Node();
  auto simple_stream_node = Add_Stream_Process_Node({1, 1, 1});
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  error_start_node->SetName("error_start_node");
  simple_stream_node->SetName("simple_stream_node");
  receive_node->SetName("receive_node");
  simple_stream_node->SetExceptionVisible(true);
  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(error_start_node->GetOutputPort("Out_1")->AddPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(error_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recieve_queue);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Recv_Invisible_Error) {
  auto error_start_node = Add_Error_Start_Node();
  auto simple_stream_node = Add_Stream_Process_Node({0, 0, 0});
  auto receive_node = Add_Collapse_Recieve_Error_Node(1);

  error_start_node->SetName("error_start_node");
  simple_stream_node->SetName("simple_stream_node");
  receive_node->SetName("receive_node");

  receive_node->SetExceptionVisible(true);

  EXPECT_EQ(error_start_node->GetOutputPort("Out_1")->AddPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(error_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto recieve_queue = receive_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recieve_queue, 1);
  EXPECT_EQ(receive_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Send_Error) {
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

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_stream_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_stream_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Normal_Expand_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_process_error_node = Add_Normal_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_process_error_node->SetName("expand_process_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");

  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_error_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Expand_Recieve_Invisible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 1);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Expand_Recieve_Visible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Normal_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(1);
  auto receive_error_node = Add_Normal_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  expand_process_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recv_queue);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Normal_Expand_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto normal_start_node = Add_Normal_Expand_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  normal_start_node->SetName("normal_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                normal_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(normal_start_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Expand_DataPre_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_datapre_error_node = Add_Expand_Datapre_Error_Node();
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_datapre_error_node->SetName("expand_datapre_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_datapre_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_datapre_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  EXPECT_EQ(error_buffer_vector.size(), 3);
  for (uint32_t i = 0; i < 3; i++) {
    std::vector<std::shared_ptr<IndexBuffer>> single_buffer_vector;
    single_buffer_vector.push_back(error_buffer_vector[i]);
    auto index_buffer_list =
        std::make_shared<IndexBufferList>(single_buffer_vector);
    auto error = index_buffer_list->GetDataError();
    EXPECT_NE(error, nullptr);
  }
  receive_queue->PushBatch(&error_buffer_vector);

  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(receive_error_node->Run(EVENT), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Expand_Process_Error) {
  auto start_node = Add_Normal_Start_Node();
  auto expand_process_error_node = Add_Expand_Process_Error_Node(4);
  auto simple_pass_node = Add_Simple_Pass_Node(12);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(4);

  start_node->SetName("start_node");
  expand_process_error_node->SetName("expand_process_error_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");

  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Expand_Recieve_Invisible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Expand_Process_Node(0);
  auto simple_pass_node = Add_Simple_Pass_Node(0);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  receive_error_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueHasDataError(recv_queue, 1);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Expand_Recieve_Visible_Error) {
  auto start_node = Add_Error_Start_Node();
  auto expand_process_node = Add_Expand_Process_Node(1);
  auto simple_pass_node = Add_Simple_Pass_Node(1);
  auto receive_error_node = Add_Collapse_Recieve_Error_Node(1);

  start_node->SetName("start_node");
  expand_process_node->SetName("expand_process_node");
  simple_pass_node->SetName("simple_pass_node");
  receive_error_node->SetName("receive_error_node");
  expand_process_node->SetExceptionVisible(true);

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_process_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_process_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                receive_error_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_process_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  auto recv_queue = receive_error_node->GetInputPort("In_1")->GetQueue();
  CheckQueueNotHasDataError(recv_queue);
  EXPECT_EQ(receive_error_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Expand_Recieve_Event_Error) {
  auto device_ = flow_->GetDevice();
  auto input_map_1 =
      std::unordered_map<std::string, std::shared_ptr<IndexBufferList>>();
  BuildDataEventStart(input_map_1, device_, virtual_stream_);

  auto stream_start_node = Add_Stream_Start_Node(2);

  ConfigurationBuilder builder;
  auto simple_stream_node_cfg = builder.Build();
  simple_stream_node_cfg->SetProperty<uint32_t>("batch_size", 10);
  auto simple_stream_node =
      Add_Stream_Process_Node({1, 2, 1}, simple_stream_node_cfg);

  stream_start_node->SetName("stream_start_node");
  simple_stream_node->SetName("simple_stream_node");
  simple_stream_node->SetExceptionVisible(true);

  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_stream_node->GetInputPort("In_1")),
            true);

  auto start_queue_1 = stream_start_node->GetInputPort("In_1")->GetQueue();
  auto index_start_vector = input_map_1["In_1"]->GetIndexBufferVector();
  start_queue_1->PushBatch(&index_start_vector);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);

  auto input_map_2 =
      std::unordered_map<std::string, std::shared_ptr<IndexBufferList>>();
  BuildDataEventStop(input_map_2, virtual_stream_);
  auto index_stop_vector = input_map_2["In_1"]->GetIndexBufferVector();
  start_queue_1->PushBatch(&index_stop_vector);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(simple_stream_node->Run(DATA), STATUS_SUCCESS);
}

TEST_F(NodeRunTest, Stream_Expand_Send_Error) {
  auto stream_info_node = Add_Stream_Normal_Info_Node();
  auto stream_start_node = Add_Stream_Start_Node(3);
  auto simple_pass_node = Add_Simple_Pass_Node(10);
  auto simple_error_node = Add_Stream_Datapre_Error_Node();

  stream_info_node->SetName("stream_info_node");
  stream_start_node->SetName("stream_start_node");
  simple_pass_node->SetName("simple_pass_node");
  simple_error_node->SetName("simple_error_node");

  EXPECT_EQ(stream_info_node->GetOutputPort("Out_1")->AddPort(
                stream_start_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(stream_start_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Collapse_DataGroupPre_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Collapse_DataPre_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->AddPort(
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
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Collapse_Process_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->AddPort(
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
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Stream_Collapse_Send_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
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

TEST_F(NodeRunTest, Stream_Collapse_Visible_Recv_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
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
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_FALSE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Stream_Collapse_Invisible_Recv_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
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
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  EXPECT_FALSE(error_buffer_vector[0]->GetBufferPtr()->HasError());
  EXPECT_TRUE(error_buffer_vector[1]->GetBufferPtr()->HasError());
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Collapse_DataPre_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Collapse_Process_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_error_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_error_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_TRUE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Collapse_Visible_Recv_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_FALSE(error_buffer_vector[i]->GetBufferPtr()->HasError());
  }
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Collapse_Invisible_Recv_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_error_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_error_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
                receive_node->GetInputPort("In_1")),
            true);

  EXPECT_EQ(start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(expand_error_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(collapse_node->Run(DATA), STATUS_SUCCESS);

  auto receive_queue = receive_node->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> error_buffer_vector;
  receive_queue->PopBatch(&error_buffer_vector);
  auto index_buffer_list =
      std::make_shared<IndexBufferList>(error_buffer_vector);
  auto error = index_buffer_list->GetDataError();
  EXPECT_NE(error, nullptr);
  EXPECT_EQ(error_buffer_vector.size(), 4);
  EXPECT_FALSE(error_buffer_vector[0]->GetBufferPtr()->HasError());
  EXPECT_TRUE(error_buffer_vector[3]->GetBufferPtr()->HasError());
  receive_queue->PushBatch(&error_buffer_vector);
}

TEST_F(NodeRunTest, Normal_Collapse_Send_Error) {
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

  EXPECT_EQ(start_node->GetOutputPort("Out_1")->AddPort(
                expand_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(expand_node->GetOutputPort("Out_1")->AddPort(
                simple_pass_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(simple_pass_node->GetOutputPort("Out_1")->AddPort(
                collapse_node->GetInputPort("In_1")),
            true);
  EXPECT_EQ(collapse_node->GetOutputPort("Out_1")->AddPort(
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
  EXPECT_EQ(start_info_port->AddPort(stream_start_node->GetInputPort("In_1")),
            true);
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      start_output_port->AddPort(stream_tail_filter_node->GetInputPort("In_1")),
      true);
  auto mid_output_port = stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_EQ(mid_output_port->AddPort(simple_pass_node->GetInputPort("In_1")),
            true);
  auto simple_pass_output_port = simple_pass_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      simple_pass_output_port->AddPort(stream_end_node->GetInputPort("In_1")),
      true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);

  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(simple_pass_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);

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
  EXPECT_EQ(start_info_port->AddPort(stream_start_node->GetInputPort("In_1")),
            true);
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_EQ(start_output_port->AddPort(
                stream_tail_filter_node_1->GetInputPort("In_1")),
            true);
  auto mid_output_port = stream_tail_filter_node_1->GetOutputPort("Out_1");
  EXPECT_EQ(
      mid_output_port->AddPort(stream_tail_filter_node_2->GetInputPort("In_1")),
      true);
  auto simple_pass_output_port =
      stream_tail_filter_node_2->GetOutputPort("Out_1");
  EXPECT_EQ(
      simple_pass_output_port->AddPort(stream_end_node->GetInputPort("In_1")),
      true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_1->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_2->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_1->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node_2->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
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
  EXPECT_EQ(start_info_port->AddPort(stream_start_node->GetInputPort("In_1")),
            true);
  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      start_output_port->AddPort(stream_tail_filter_node->GetInputPort("In_1")),
      true);
  auto stream_tail_output_port =
      stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_EQ(stream_tail_output_port->AddPort(expand_node->GetInputPort("In_1")),
            true);
  auto expand_output_port = expand_node->GetOutputPort("Out_1");
  EXPECT_EQ(expand_output_port->AddPort(collapse_node->GetInputPort("In_1")),
            true);
  auto collapse_output_port = collapse_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      collapse_output_port->AddPort(stream_end_node->GetInputPort("In_1")),
      true);

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
  EXPECT_EQ(start_info_port->AddPort(stream_start_node->GetInputPort("In_1")),
            true);

  auto start_output_port = stream_start_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      start_output_port->AddPort(stream_tail_filter_node->GetInputPort("In_1")),
      true);
  auto stream_tail_output_port =
      stream_tail_filter_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      stream_tail_output_port->AddPort(condition_node->GetInputPort("In_1")),
      true);
  auto condition_output_port_1 = condition_node->GetOutputPort("Out_1");
  EXPECT_EQ(
      condition_output_port_1->AddPort(stream_end_node->GetInputPort("In_1")),
      true);
  auto condition_output_port_2 = condition_node->GetOutputPort("Out_2");
  EXPECT_EQ(
      condition_output_port_2->AddPort(stream_end_node->GetInputPort("In_1")),
      true);

  EXPECT_EQ(stream_info_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_start_node->Run(EVENT), STATUS_SUCCESS);
  EXPECT_EQ(stream_tail_filter_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(condition_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(stream_end_node->Run(DATA), STATUS_SUCCESS);
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
  auto dynamic_config_node = std::make_shared<Node>(
      "dynamic_config", "cpu", "0", flowunit_mgr_, nullptr);
  auto dynamic_get_config_node_1 = std::make_shared<Node>(
      "dynamic_get_config", "cpu", "0", flowunit_mgr_, nullptr);
  auto dynamic_get_config_node_2 = std::make_shared<Node>(
      "dynamic_get_config", "cpu", "0", flowunit_mgr_, nullptr);
  auto dynamic_get_config_node_3 = std::make_shared<Node>(
      "dynamic_get_config_other", "cpu", "0", flowunit_mgr_, nullptr);
  auto dynamic_get_config_node_4 = std::make_shared<Node>(
      "dynamic_get_config_other", "cpu", "0", flowunit_mgr_, nullptr);
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
  EXPECT_EQ(dynamic_config_port->AddPort(
                dynamic_get_config_node_1->GetInputPort("In_1")),
            true);

  auto dynamic_get_config_port_1 =
      dynamic_get_config_node_1->GetOutputPort("Out_1");
  EXPECT_EQ(dynamic_get_config_port_1->AddPort(
                dynamic_get_config_node_2->GetInputPort("In_1")),
            true);

  auto dynamic_get_config_port_2 =
      dynamic_get_config_node_2->GetOutputPort("Out_1");
  EXPECT_EQ(dynamic_get_config_port_2->AddPort(
                dynamic_get_config_node_3->GetInputPort("In_1")),
            true);

  auto dynamic_get_config_port_3 =
      dynamic_get_config_node_3->GetOutputPort("Out_1");
  EXPECT_EQ(dynamic_get_config_port_3->AddPort(
                dynamic_get_config_node_4->GetInputPort("In_1")),
            true);
  EXPECT_EQ(dynamic_config_node->Run(DATA), STATUS_SUCCESS);
  EXPECT_EQ(dynamic_get_config_node_1->Run(DATA), STATUS_SUCCESS);
  auto queue_1 = dynamic_get_config_node_2->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_1;
  queue_1->PopBatch(&buffer_vector_1);
  std::string test_1 = "";
  buffer_vector_1[0]->GetBufferPtr()->Get("test", test_1);
  EXPECT_EQ(test_1, "node.dynamic_get_config_1.test");
  queue_1->PushBatch(&buffer_vector_1);

  EXPECT_EQ(dynamic_get_config_node_2->Run(DATA), STATUS_SUCCESS);
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_2;
  auto queue_2 = dynamic_get_config_node_3->GetInputPort("In_1")->GetQueue();
  queue_2->PopBatch(&buffer_vector_2);
  std::string test_2 = "";
  buffer_vector_2[0]->GetBufferPtr()->Get("test", test_2);
  EXPECT_EQ(test_2, "flowunit.dynamic_get_config.test");
  queue_2->PushBatch(&buffer_vector_2);

  EXPECT_EQ(dynamic_get_config_node_3->Run(DATA), STATUS_SUCCESS);
  auto queue_3 = dynamic_get_config_node_4->GetInputPort("In_1")->GetQueue();
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector_3;
  queue_3->PopBatch(&buffer_vector_3);
  std::string test_3 = "";
  buffer_vector_3[0]->GetBufferPtr()->Get("test", test_3);
  EXPECT_EQ(test_3, "nodes.test");
  queue_3->PushBatch(&buffer_vector_3);
}

}  // namespace modelbox
