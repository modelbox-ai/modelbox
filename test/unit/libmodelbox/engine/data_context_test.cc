
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

#include "modelbox/data_context.h"

#include "gtest/gtest.h"
#include "modelbox/node.h"
#include "modelbox/port.h"

namespace modelbox {

class TestNode : public Node {
 public:
  void InitIO(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names) {
    ConfigurationBuilder builder;
    auto config = builder.Build();
    NodeBase::Init(input_port_names, output_port_names, config);
  }
};

class DataContextTest : public testing::Test {
 public:
  DataContextTest() {}
  virtual ~DataContextTest() {}

 protected:
  virtual void SetUp() {
    session_ = std::make_shared<Session>(nullptr);
    stream_ = std::make_shared<Stream>(session_);
    end_stream_ = std::make_shared<Stream>(session_);
    root_index_info_ = std::make_shared<BufferIndexInfo>();
    root_index_info_->SetIndex(0);
    root_end_index_info_ = std::make_shared<BufferIndexInfo>();
    root_end_index_info_->SetIndex(1);
    root_end_index_info_->MarkAsEndFlag();
    node_ = std::make_shared<TestNode>();
    node_->InitIO(in_port_names_, out_port_names_);
  };
  virtual void TearDown(){};

  std::shared_ptr<PortDataMap> BuildData(size_t data_count, bool has_end,
                                         bool expand_from_end = false) {
    if (expand_from_end) {
      data_count = 1;
    }
    auto data = std::make_shared<PortDataMap>();
    for (auto& port_name : in_port_names_) {
      auto& port_data_list = (*data)[port_name];
      for (size_t i = 1; i <= data_count; ++i) {
        auto buffer = std::make_shared<Buffer>();
        auto index = BufferManageView::GetIndexInfo(buffer);
        auto inherit_info = std::make_shared<BufferInheritInfo>();
        inherit_info->SetType(BufferProcessType::EXPAND);
        if (!expand_from_end) {
          inherit_info->SetInheritFrom(root_index_info_);
          index->SetStream(stream_);
          index->SetIndex(stream_->GetBufferCount());
          stream_->IncreaseBufferCount();
        } else {
          inherit_info->SetInheritFrom(root_end_index_info_);
          index->SetStream(end_stream_);
          index->SetIndex(0);
          end_stream_->IncreaseBufferCount();
        }
        index->SetInheritInfo(inherit_info);
        port_data_list.push_back(buffer);
        if (i == data_count && has_end) {
          index->MarkAsEndFlag();
        }
      }
    }

    return data;
  }

  void ProcessData(FlowUnitDataContext* data_ctx, BufferProcessType type,
                   size_t expect_input_count, size_t output_count) {
    auto process_info = std::make_shared<BufferProcessInfo>();
    process_info->SetType(type);

    for (auto& input_name : in_port_names_) {
      auto inputs = data_ctx->Input(input_name);
      ASSERT_EQ(inputs->Size(), expect_input_count);
      std::list<std::shared_ptr<BufferIndexInfo>> input_index_list;
      for (auto& input : *inputs) {
        input_index_list.push_back(BufferManageView::GetIndexInfo(input));
      }

      process_info->SetParentBuffers(input_name, std::move(input_index_list));
    }

    auto output_map = data_ctx->Output();
    ASSERT_NE(output_map, nullptr);
    for (auto& output_name : out_port_names_) {
      auto output_list = std::make_shared<BufferList>();
      (*output_map)[output_name] = output_list;
      for (size_t i = 0; i < output_count; ++i) {
        auto buffer = std::make_shared<Buffer>();
        output_list->PushBack(buffer);
        auto index = BufferManageView::GetIndexInfo(output_list->Back());
        index->SetProcessInfo(process_info);
      }
    }
  }

  std::shared_ptr<Session> session_;
  std::shared_ptr<Stream> stream_;
  std::shared_ptr<Stream> end_stream_;
  std::shared_ptr<BufferIndexInfo> root_index_info_;
  std::shared_ptr<BufferIndexInfo> root_end_index_info_;

  std::set<std::string> in_port_names_{"in_1"};
  std::set<std::string> out_port_names_{"out_1"};
  std::shared_ptr<TestNode> node_;
};

TEST_F(DataContextTest, NormalTest) {
  node_->SetFlowType(FlowType::NORMAL);
  auto data = BuildData(10, true);

  NormalFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* recv data */
  // write data
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ProcessData(&data_ctx, BufferProcessType::ORIGIN, 9, 9);

  data_ctx.SetStatus(STATUS_SUCCESS);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 10);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
}

TEST_F(DataContextTest, StreamTest) {
  node_->SetFlowType(FlowType::STREAM);
  auto data = BuildData(10, true);

  StreamFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* recv data */
  // write data
  data_ctx.WriteInputData(data);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ProcessData(&data_ctx, BufferProcessType::ORIGIN, 9, 9);

  data_ctx.SetStatus(STATUS_SUCCESS);
  // post process
  data_ctx.PostProcess();
  EXPECT_TRUE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 10);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
}

TEST_F(DataContextTest, StreamTest2) {
  node_->SetFlowType(FlowType::STREAM);

  auto data_ctx = std::make_shared<StreamFlowUnitDataContext>(
      node_.get(), nullptr, session_);
  session_->AddStateListener(data_ctx);
  /* 1. recv data and continue generate */
  // write data
  auto data = BuildData(1, false);
  data_ctx->WriteInputData(data);
  EXPECT_TRUE(data_ctx->IsDataPre());
  // process
  ASSERT_FALSE(data_ctx->IsSkippable());
  ProcessData(data_ctx.get(), BufferProcessType::ORIGIN, 1, 1);

  data_ctx->SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx->SetStatus(STATUS_CONTINUE);

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);  // event to continue process
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx->PostProcess();
  EXPECT_FALSE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx->ClearData();
  ASSERT_FALSE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_CONTINUE);
  /* 2. recv end flag */
  // write data
  data = BuildData(1, true);
  data_ctx->WriteInputData(data);
  EXPECT_FALSE(data_ctx->IsDataPre());
  // process
  ASSERT_TRUE(data_ctx->IsSkippable());
  // post process
  data_ctx->PostProcess();
  EXPECT_FALSE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_TRUE(out_data.begin()->second.empty());
  data_ctx->ClearData();
  ASSERT_FALSE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_CONTINUE);
  /* 3. recv event and delay to send event */
  // set event
  data_ctx->SetEvent(user_event);
  EXPECT_FALSE(data_ctx->IsDataPre());
  // process
  ASSERT_FALSE(data_ctx->IsSkippable());
  ASSERT_EQ(data_ctx->Event(), user_event);

  auto output_map = data_ctx->Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx->SetStatus(STATUS_CONTINUE);  // delay event, only return continue
  // post process
  data_ctx->PostProcess();
  EXPECT_FALSE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx->ClearData();
  ASSERT_FALSE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_CONTINUE);
  /* 4. close session */
  session_->Close();
  // check event
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);  // event to continue process
  process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto finish_event = process_event->GetUserEvent();
  // triger finish event
  data_ctx->SetEvent(finish_event);
  EXPECT_FALSE(data_ctx->IsDataPre());
  // process
  ASSERT_TRUE(data_ctx->IsSkippable());
  ASSERT_EQ(data_ctx->Event(), finish_event);
  // post process
  data_ctx->PostProcess();
  EXPECT_TRUE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx->ClearData();
  ASSERT_TRUE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_CONTINUE);
}

TEST_F(DataContextTest, StreamTest_SendEventOutOfNodeRun) {
  node_->SetFlowType(FlowType::STREAM);

  auto data_ctx = std::make_shared<StreamFlowUnitDataContext>(
      node_.get(), nullptr, session_);
  session_->AddStateListener(data_ctx);
  /* 1. recv data and continue generate */
  // write data
  auto data = BuildData(1, false);
  data_ctx->WriteInputData(data);
  EXPECT_TRUE(data_ctx->IsDataPre());
  // process
  ASSERT_FALSE(data_ctx->IsSkippable());
  ProcessData(data_ctx.get(), BufferProcessType::ORIGIN, 1, 1);

  data_ctx->SetStatus(STATUS_CONTINUE);  // event will send out of node run

  // post process
  data_ctx->PostProcess();
  EXPECT_FALSE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx->ClearData();
  ASSERT_FALSE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_CONTINUE);
  /* 2. send delay event */
  data_ctx->SendEvent(std::make_shared<FlowUnitEvent>());

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);  // event to continue process
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  /* 3. recv event */
  // set event
  data_ctx->SetEvent(user_event);
  EXPECT_FALSE(data_ctx->IsDataPre());
  // process
  ASSERT_FALSE(data_ctx->IsSkippable());
  ASSERT_EQ(data_ctx->Event(), user_event);

  auto output_map = data_ctx->Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx->SetStatus(STATUS_SUCCESS);
  // post process
  ASSERT_EQ(data_ctx->PostProcess(), STATUS_SUCCESS);
  EXPECT_FALSE(data_ctx->IsDataPost());
  data_ctx->UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx->PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx->ClearData();
  ASSERT_FALSE(data_ctx->IsFinished());
  ASSERT_EQ(data_ctx->GetStatus(), STATUS_SUCCESS);
}

TEST_F(DataContextTest, NormalExpandTest) {
  node_->SetOutputType(FlowOutputType::EXPAND);
  node_->SetFlowType(FlowType::NORMAL);

  NormalExpandFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv data and continue expand */
  // write data
  auto data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  ProcessData(&data_ctx, BufferProcessType::EXPAND, 1, 1);

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);  // event to continue expand current buffer
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 2. recv event and close session before user send event*/
  // set event
  data_ctx.SetEvent(user_event);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ASSERT_EQ(data_ctx.Event(), user_event);

  auto output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  session_->Close();
  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 0);  // session close, send user event failed
  process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(),
            2);  // expand end, include data and end flag
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
}

TEST_F(DataContextTest, StreamExpandTest) {
  node_->SetOutputType(FlowOutputType::EXPAND);
  node_->SetFlowType(FlowType::STREAM);

  StreamExpandFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv data and continue expand */
  // write data
  auto data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  ProcessData(&data_ctx, BufferProcessType::EXPAND, 1, 1);

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);  // event to continue expand current buffer
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  auto expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // should not expand next
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 2. recv event and close session before user send event*/
  // set event
  data_ctx.SetEvent(user_event);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ASSERT_EQ(data_ctx.Event(), user_event);

  auto output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  session_->Close();
  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 0);  // session close, send user event failed
  process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  user_event = process_event->GetUserEvent();
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event,
            nullptr);  // end flag not received, should not send expand next
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(),
            2);  // expand end, include data and end flag
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 3. recv end flag and expand */
  // write data
  data = BuildData(1, true);
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_TRUE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // no data to expand
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  ASSERT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());  // data ctx process end
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
}

TEST_F(DataContextTest, StreamExpandTest2) {
  node_->SetOutputType(FlowOutputType::EXPAND);
  node_->SetFlowType(FlowType::STREAM);

  StreamExpandFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv data and continue expand */
  // write data
  auto data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  auto output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event =
      process_event->GetUserEvent();  // event to continue expand cur buffer
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  auto expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // should not expand next
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 2. recv event and close session after user send event*/
  // set event
  data_ctx.SetEvent(user_event);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ASSERT_EQ(data_ctx.Event(), user_event);

  output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  session_->Close();
  data_ctx.SetStatus(STATUS_CONTINUE);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);
  process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  user_event =
      process_event->GetUserEvent();  // event to continue expand current buffer
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // should not expand next
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 3. recv end flag */
  // write data
  data = BuildData(1, true);
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event,
            nullptr);  // should not send expand next, buffer[0] is still expand
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_TRUE(out_data.begin()->second.empty());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 4. recv event */
  // set event
  data_ctx.SetEvent(user_event);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ASSERT_EQ(data_ctx.Event(), user_event);

  output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  session_->Close();
  data_ctx.SetStatus(STATUS_CONTINUE);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 0);  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();  // expand end flag
  ASSERT_NE(expand_event, nullptr);
  EXPECT_EQ(expand_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_NEXT_STREAM);
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 2);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 5. expand end flag */
  // expand end flag buffer
  data_ctx.ExpandNextBuffer();
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_TRUE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // no data to expand
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  ASSERT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());  // data ctx process end
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
}

TEST_F(DataContextTest, StreamExpandTest3) {
  node_->SetOutputType(FlowOutputType::EXPAND);
  node_->SetFlowType(FlowType::STREAM);

  StreamExpandFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv data and continue expand */
  // write data
  auto data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  auto output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  auto output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SendEvent(std::make_shared<FlowUnitEvent>());
  data_ctx.SetStatus(STATUS_CONTINUE);

  FlowunitEventList event_list;
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 1);
  auto process_event = event_list->front();
  ASSERT_NE(process_event, nullptr);
  auto user_event =
      process_event->GetUserEvent();  // event to continue expand cur buffer
  ASSERT_EQ(process_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_UNFINISH_DATA);
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  auto expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // should not expand next
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 2. recv data2 */
  // write data2
  data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event,
            nullptr);  // should not send expand next, data1 is still expand
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_TRUE(out_data.begin()->second.empty());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_CONTINUE);
  /* 3. recv event and stop continue*/
  // set event
  data_ctx.SetEvent(user_event);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ASSERT_EQ(data_ctx.Event(), user_event);

  output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SetStatus(STATUS_SUCCESS);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_NE(expand_event, nullptr);  // expand data2
  EXPECT_EQ(expand_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_NEXT_STREAM);
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 2);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
  /* 4. recv data3 */
  // write data3
  data = BuildData(1, false);
  data_ctx.WriteInputData(data);
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_TRUE(event_list->empty());  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event,
            nullptr);  // should not send expand next, data2 event has been sent
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_TRUE(out_data.begin()->second.empty());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
  /* 5. expand data2 */
  // expand data2
  data_ctx.ExpandNextBuffer();
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SetStatus(STATUS_SUCCESS);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 0);  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();  // expand data3
  ASSERT_NE(expand_event, nullptr);
  EXPECT_EQ(expand_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::EXPAND_NEXT_STREAM);
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 2);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
  /* 6. expand data3 */
  // expand data3
  data_ctx.ExpandNextBuffer();
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());

  output_map = data_ctx.Output();
  ASSERT_NE(output_map, nullptr);
  output_list = std::make_shared<BufferList>();
  (*output_map)["out_1"] = output_list;
  output_list->PushBack(std::make_shared<Buffer>());

  data_ctx.SetStatus(STATUS_SUCCESS);

  event_list->clear();
  node_->GetEventPort()->Recv(event_list);
  ASSERT_EQ(event_list->size(), 0);  // no user event
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // expand event
  expand_event = data_ctx.GenerateSendEvent();
  ASSERT_EQ(expand_event, nullptr);  // no data to expand
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 2);
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.back());
  EXPECT_TRUE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
}

TEST_F(DataContextTest, NormalCollapseTest) {
  node_->SetOutputType(FlowOutputType::COLLAPSE);
  node_->SetFlowType(FlowType::STREAM);

  NormalCollapseFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv data */
  auto data = BuildData(10, true);
  // write data
  data_ctx.WriteInputData(data);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ProcessData(&data_ctx, BufferProcessType::COLLAPSE, 9, 1);

  data_ctx.SetStatus(STATUS_SUCCESS);
  // post process
  data_ctx.PostProcess();
  EXPECT_TRUE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);  // no end flag
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
}

TEST_F(DataContextTest, StreamCollapseTest) {
  node_->SetOutputType(FlowOutputType::COLLAPSE);
  node_->SetFlowType(FlowType::STREAM);

  StreamCollapseFlowUnitDataContext data_ctx(node_.get(), nullptr, session_);
  /* 1. recv sub stream1 and stream2 */
  auto stream1 = BuildData(10, true);
  auto stream2 = BuildData(1, true, true);
  // write data
  data_ctx.WriteInputData(stream1);
  data_ctx.WriteInputData(stream2);
  EXPECT_TRUE(data_ctx.IsDataPre());
  // process
  ASSERT_FALSE(data_ctx.IsSkippable());
  ProcessData(&data_ctx, BufferProcessType::COLLAPSE, 9, 1);

  data_ctx.SetStatus(STATUS_SUCCESS);
  // post process
  data_ctx.PostProcess();
  EXPECT_TRUE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // collapse event
  auto collapse_event = data_ctx.GenerateSendEvent();  // collapse next stream
  ASSERT_NE(collapse_event, nullptr);
  EXPECT_EQ(collapse_event->GetEventCode(),
            FlowUnitInnerEvent::EventCode::COLLAPSE_NEXT_STREAM);
  // check output and clear
  PortDataMap out_data;
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);  // no end flag
  auto out_index =
      BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_FALSE(out_index->IsEndFlag());
  EXPECT_EQ(out_index->GetIndex(), 0);
  data_ctx.ClearData();
  ASSERT_FALSE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
  /* 2. recv sub stream2 */
  // collapse event
  data_ctx.CollapseNextStream();
  EXPECT_FALSE(data_ctx.IsDataPre());
  // process
  ASSERT_TRUE(data_ctx.IsSkippable());
  // post process
  data_ctx.PostProcess();
  EXPECT_FALSE(data_ctx.IsDataPost());
  data_ctx.UpdateProcessState();
  // check output and clear
  out_data.clear();
  data_ctx.PopOutputData(out_data);
  ASSERT_EQ(out_data.size(), 1);
  ASSERT_EQ(out_data.begin()->second.size(), 1);  // end flag
  out_index = BufferManageView::GetIndexInfo(out_data.begin()->second.front());
  EXPECT_TRUE(out_index->IsEndFlag());
  EXPECT_EQ(out_index->GetIndex(), 1);
  data_ctx.ClearData();
  ASSERT_TRUE(data_ctx.IsFinished());
  ASSERT_EQ(data_ctx.GetStatus(), STATUS_SUCCESS);
}

}  // namespace modelbox
