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

#include "modelbox/match_stream.h"

#include <functional>
#include <future>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "modelbox/session.h"

namespace modelbox {
class MatchStreamTest : public testing::Test {
 protected:
  void SetUp() override {
    in_port1_ = std::make_shared<InPort2>("a", nullptr);
    in_port2_ = std::make_shared<InPort2>("b", nullptr);
    data_ports_.push_back(in_port1_);
    data_ports_.push_back(in_port2_);

    // prepare data
    auto root_buffer_index = std::make_shared<BufferIndexInfo>();
    auto session = std::make_shared<Session>(nullptr);

    auto stream1 = std::make_shared<Stream>(session);
    buffer1_ = std::make_shared<Buffer>();
    auto buffer1_index = std::make_shared<BufferIndexInfo>();
    auto buffer1_inherit = std::make_shared<BufferInheritInfo>();
    buffer1_inherit->SetInheritFrom(root_buffer_index);
    buffer1_inherit->SetType(BufferProcessType::EXPAND);
    buffer1_index->SetInheritInfo(buffer1_inherit);
    buffer1_index->SetStream(stream1);
    buffer1_index->SetIndex(0);
    BufferManageView::SetIndexInfo(buffer1_, buffer1_index);

    auto stream2 = std::make_shared<Stream>(session);
    buffer2_ = std::make_shared<Buffer>();
    auto buffer2_index = std::make_shared<BufferIndexInfo>();
    auto buffer2_inherit = std::make_shared<BufferInheritInfo>();
    buffer2_inherit->SetInheritFrom(root_buffer_index);
    buffer2_inherit->SetType(BufferProcessType::EXPAND);
    buffer2_index->SetInheritInfo(buffer2_inherit);
    buffer2_index->SetStream(stream2);
    buffer2_index->SetIndex(0);
    BufferManageView::SetIndexInfo(buffer2_, buffer2_index);

    // push data
    in_port1_->GetQueue()->Push(buffer1_);
    in_port2_->GetQueue()->Push(buffer2_);
  }

  std::shared_ptr<Buffer> buffer1_;
  std::shared_ptr<Buffer> buffer2_;
  std::shared_ptr<InPort2> in_port1_;
  std::shared_ptr<InPort2> in_port2_;
  std::vector<std::shared_ptr<InPort2>> data_ports_;
};

TEST_F(MatchStreamTest, InputMatchStreamManagerTest) {
  // run
  InputMatchStreamManager input_match_stream_mgr("test", 32, 2);
  auto ret = input_match_stream_mgr.LoadData(data_ports_);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  std::list<std::shared_ptr<MatchStreamData>> match_stream_list;
  ret = input_match_stream_mgr.GenMatchStreamData(match_stream_list);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  ASSERT_EQ(match_stream_list.size(), 1);
  auto match_stream = match_stream_list.front();
  ASSERT_EQ(match_stream->GetDataCount(), 1);
  auto data_map = match_stream->GetBufferList();
  ASSERT_EQ(data_map->size(), 2);
  ASSERT_EQ(data_map->at("a").size(), 1);
  ASSERT_EQ(data_map->at("a").size(), 1);
  EXPECT_EQ(data_map->at("a").front(), buffer1_);
  EXPECT_EQ(data_map->at("b").front(), buffer2_);
}

TEST_F(MatchStreamTest, OutputMatchStream) {
  std::set<std::string> output_port_names{"a", "b"};
  OutputMatchStreamManager output_match_stream_mgr(
      "test", std::move(output_port_names));

  auto buffer1_index = BufferManageView::GetIndexInfo(buffer1_);
  auto buffer2_index = BufferManageView::GetIndexInfo(buffer2_);

  auto out_buffer1 = std::make_shared<Buffer>();
  auto out_index = std::make_shared<BufferIndexInfo>();
  auto out_inherit = std::make_shared<BufferInheritInfo>();
  out_inherit->SetInheritFrom(buffer1_index);
  out_inherit->SetType(BufferProcessType::EXPAND);
  auto out_process = std::make_shared<BufferProcessInfo>();
  out_process->SetParentBuffers("a", {buffer1_index});
  out_process->SetParentBuffers("b", {buffer2_index});
  out_index->SetInheritInfo(out_inherit);
  out_index->SetProcessInfo(out_process);
  BufferManageView::SetIndexInfo(out_buffer1, out_index);

  auto out_buffer2 = std::make_shared<Buffer>();
  auto out_index2 = std::make_shared<BufferIndexInfo>();
  auto out_inherit2 = std::make_shared<BufferInheritInfo>();
  out_inherit2->SetInheritFrom(buffer1_index);
  out_inherit2->SetType(BufferProcessType::EXPAND);
  auto out_process2 = std::make_shared<BufferProcessInfo>();
  out_process2->SetParentBuffers("a", {buffer1_index});
  out_process2->SetParentBuffers("b", {buffer2_index});
  out_index2->SetInheritInfo(out_inherit2);
  out_index2->SetProcessInfo(out_process2);
  BufferManageView::SetIndexInfo(out_buffer2, out_index2);

  std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      output_data;
  output_data["a"].push_back(out_buffer1);
  output_data["b"].push_back(out_buffer2);
  std::unordered_map<std::string, std::shared_ptr<DataMeta>> port_stream_meta;
  auto ret = output_match_stream_mgr.UpdateStreamInfo(
      output_data, port_stream_meta, nullptr);
  ASSERT_EQ(ret, STATUS_SUCCESS);
  ASSERT_EQ(output_match_stream_mgr.GetOutputStreamCount(), 1);
  EXPECT_NE(out_index->GetStream(), nullptr);
  EXPECT_EQ(out_index->GetIndex(), 0);
  EXPECT_NE(out_index2->GetStream(), nullptr);
  EXPECT_EQ(out_index2->GetIndex(), 0);
}
}  // namespace modelbox