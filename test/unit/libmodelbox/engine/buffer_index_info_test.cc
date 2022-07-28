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

#include "modelbox/buffer_index_info.h"

#include <functional>
#include <future>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"

namespace modelbox {
class BufferIndexInfoTest : public testing::Test {};

TEST_F(BufferIndexInfoTest, IndexInfoTest) {
  auto index_info = std::make_shared<BufferIndexInfo>();
  index_info->SetIndex(0);
  EXPECT_TRUE(index_info->IsFirstBufferInStream());
  EXPECT_FALSE(index_info->IsEndFlag());
  EXPECT_FALSE(index_info->IsPlaceholder());
  index_info->SetIndex(1);
  EXPECT_FALSE(index_info->IsFirstBufferInStream());
  EXPECT_FALSE(index_info->IsEndFlag());
  EXPECT_FALSE(index_info->IsPlaceholder());
  index_info = std::make_shared<BufferIndexInfo>();
  index_info->SetIndex(1);
  index_info->MarkAsEndFlag();
  EXPECT_FALSE(index_info->IsFirstBufferInStream());
  EXPECT_TRUE(index_info->IsEndFlag());
  EXPECT_FALSE(index_info->IsPlaceholder());
  index_info = std::make_shared<BufferIndexInfo>();
  index_info->SetIndex(1);
  index_info->MarkAsPlaceholder();
  EXPECT_FALSE(index_info->IsFirstBufferInStream());
  EXPECT_FALSE(index_info->IsEndFlag());
  EXPECT_TRUE(index_info->IsPlaceholder());
}

TEST_F(BufferIndexInfoTest, ProcessInfoTest) {
  auto process_info = std::make_shared<BufferProcessInfo>();
  auto a_buffer = std::make_shared<BufferIndexInfo>();
  auto b_buffer = std::make_shared<BufferIndexInfo>();
  process_info->SetParentBuffers("a", {a_buffer});
  process_info->SetParentBuffers("b", {b_buffer});
  const auto &buffers = process_info->GetParentBuffers();
  ASSERT_EQ(buffers.size(), 2);
  ASSERT_EQ(buffers.at("a").size(), 1);
  ASSERT_EQ(buffers.at("b").size(), 1);
  EXPECT_EQ(buffers.at("a").front(), a_buffer);
  EXPECT_EQ(buffers.at("b").front(), b_buffer);
  EXPECT_EQ(buffers.begin()->second.front(), a_buffer);  // test read order

  EXPECT_EQ(process_info->GetType(), BufferProcessType::ORIGIN);
  process_info->SetType(BufferProcessType::EXPAND);
  EXPECT_EQ(process_info->GetType(), BufferProcessType::EXPAND);
}

TEST_F(BufferIndexInfoTest, InheritInfoTest) {
  auto inherit_info = std::make_shared<BufferInheritInfo>();
  EXPECT_EQ(inherit_info->GetType(), BufferProcessType::EXPAND);
  inherit_info->SetType(BufferProcessType::CONDITION_START);
  EXPECT_EQ(inherit_info->GetType(), BufferProcessType::CONDITION_START);

  auto root_index_info = std::make_shared<BufferIndexInfo>();
  EXPECT_EQ(inherit_info->GetDeepth(), 0);
  inherit_info->SetInheritFrom(root_index_info);
  EXPECT_EQ(inherit_info->GetDeepth(), 0);
  EXPECT_EQ(inherit_info->GetInheritFrom(), root_index_info);

  auto second_index_info = std::make_shared<BufferIndexInfo>();
  second_index_info->SetInheritInfo(inherit_info);
  auto inherit_info2 = std::make_shared<BufferInheritInfo>();
  EXPECT_EQ(inherit_info2->GetDeepth(), 0);
  inherit_info2->SetInheritFrom(second_index_info);
  EXPECT_EQ(inherit_info2->GetDeepth(), 1);
  EXPECT_EQ(inherit_info2->GetInheritFrom(), second_index_info);
}

TEST_F(BufferIndexInfoTest, BufferManageViewTest) {
  auto buffer = std::make_shared<Buffer>();
  EXPECT_EQ(BufferManageView::GetPriority(buffer), 0);
  BufferManageView::SetPriority(buffer, 1);
  EXPECT_EQ(BufferManageView::GetPriority(buffer), 1);

  EXPECT_NE(BufferManageView::GetIndexInfo(buffer), nullptr);
  EXPECT_EQ(BufferManageView::GetFirstParentBuffer(buffer), nullptr);

  auto index_info = std::make_shared<BufferIndexInfo>();
  BufferManageView::SetIndexInfo(buffer, index_info);
  EXPECT_EQ(BufferManageView::GetIndexInfo(buffer), index_info);
  EXPECT_EQ(BufferManageView::GetFirstParentBuffer(buffer), nullptr);

  auto in_buffer = std::make_shared<Buffer>();
  auto in_buffer_index = std::make_shared<BufferIndexInfo>();
  BufferManageView::SetIndexInfo(in_buffer, in_buffer_index);
  auto in_buffer2 = std::make_shared<Buffer>();
  auto in_buffer2_index = std::make_shared<BufferIndexInfo>();
  BufferManageView::SetIndexInfo(in_buffer2, in_buffer2_index);
  std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      input_data;
  input_data["a"].push_back(in_buffer);
  input_data["a"].push_back(in_buffer2);
  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list;
  BufferManageView::GenProcessInfo<std::vector<std::shared_ptr<Buffer>>>(
      input_data, 2,
      [](const std::vector<std::shared_ptr<Buffer>> &container, size_t idx) {
        return container[idx];
      },
      process_info_list, false);
  ASSERT_EQ(process_info_list.size(), 2);
  ASSERT_EQ(process_info_list.front()->GetParentBuffers().size(), 1);
  ASSERT_EQ(
      process_info_list.front()->GetParentBuffers().begin()->second.size(), 1);
  ASSERT_EQ(
      process_info_list.front()->GetParentBuffers().begin()->second.front(),
      in_buffer_index);

  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list2;
  BufferManageView::GenProcessInfo<std::vector<std::shared_ptr<Buffer>>>(
      input_data, 2,
      [](const std::vector<std::shared_ptr<Buffer>> &container, size_t idx) {
        return container[idx];
      },
      process_info_list2, true);
  ASSERT_EQ(process_info_list2.size(), 1);
  ASSERT_EQ(process_info_list2.front()->GetParentBuffers().size(), 1);
  ASSERT_EQ(
      process_info_list2.front()->GetParentBuffers().begin()->second.size(), 2);
  ASSERT_EQ(
      process_info_list2.front()->GetParentBuffers().begin()->second.front(),
      in_buffer_index);
}

}  // namespace modelbox