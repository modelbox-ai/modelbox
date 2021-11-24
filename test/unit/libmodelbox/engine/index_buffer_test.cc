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


#include "modelbox/index_buffer.h"

#include "gtest/gtest.h"
namespace modelbox {
class BufferGroupTest : public testing::Test {
 public:
  BufferGroupTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){

  };
};

class IndexBufferTest : public testing::Test {
 public:
  IndexBufferTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){

  };
};

class IndexBufferListTest : public testing::Test {
 public:
  IndexBufferListTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){

  };
};

TEST_F(BufferGroupTest, AddOneSubSingleGroup) {
  auto root = std::make_shared<BufferGroup>();

  auto group_0 = root->AddSubGroup(true, true);
  uint32_t* root_sum = nullptr;
  EXPECT_EQ(root->GetOrder(), 1);
  EXPECT_EQ(root->GetPortId(), 0);
  EXPECT_EQ(root->GetGroupSum(root_sum), STATUS_NOTFOUND);
  EXPECT_EQ(root->GetGroup(), nullptr);
  EXPECT_EQ(group_0->IsStartGroup(), true);
  EXPECT_EQ(group_0->IsEndGroup(), true);

  uint32_t group_0_sum;
  EXPECT_EQ(group_0->GetOrder(), 1);
  EXPECT_EQ(group_0->GetPortId(), 0);
  EXPECT_EQ(group_0->GetGroupSum(&group_0_sum), STATUS_SUCCESS);
  EXPECT_EQ(1, group_0_sum);
  EXPECT_NE(group_0->GetGroup(), nullptr);
  EXPECT_EQ(group_0->IsStartGroup(), true);
  EXPECT_EQ(group_0->IsEndGroup(), true);
}

TEST_F(BufferGroupTest, AddOneSubMultiGroup) {
  auto root = std::make_shared<BufferGroup>();

  auto group_0 = root->AddSubGroup(true, false);
  uint32_t group_0_sum;
  EXPECT_EQ(group_0->GetOrder(), 1);
  EXPECT_EQ(group_0->GetPortId(), 0);
  EXPECT_EQ(group_0->GetGroupSum(&group_0_sum), STATUS_NOTFOUND);
  EXPECT_NE(group_0->GetGroup(), nullptr);
  EXPECT_EQ(group_0->IsStartGroup(), true);
  EXPECT_EQ(group_0->IsEndGroup(), false);

  auto group_1 = root->AddSubGroup(false, false);
  uint32_t group_1_sum;
  EXPECT_EQ(group_1->GetOrder(), 2);
  EXPECT_EQ(group_1->GetPortId(), 0);
  EXPECT_EQ(group_1->GetGroupSum(&group_1_sum), STATUS_NOTFOUND);
  EXPECT_NE(group_1->GetGroup(), nullptr);
  EXPECT_EQ(group_1->IsStartGroup(), false);
  EXPECT_EQ(group_1->IsEndGroup(), false);

  auto group_2 = root->AddSubGroup(false, true);
  uint32_t group_2_sum;
  EXPECT_EQ(group_2->GetOrder(), 3);
  EXPECT_EQ(group_2->GetPortId(), 0);
  EXPECT_EQ(group_2->GetGroupSum(&group_2_sum), STATUS_SUCCESS);
  EXPECT_EQ(3, group_2_sum);
  EXPECT_NE(group_2->GetGroup(), nullptr);
  EXPECT_EQ(group_2->IsStartGroup(), false);
  EXPECT_EQ(group_2->IsEndGroup(), true);

  EXPECT_EQ(group_1->GetGroup(), group_2->GetGroup());
  EXPECT_EQ(group_0->GetGroup(), group_1->GetGroup());
}

TEST_F(BufferGroupTest, AddInvalidSubGroup) {
  auto root = std::make_shared<BufferGroup>();

  auto invalid_group_0 = root->AddSubGroup(false, false);
  EXPECT_EQ(invalid_group_0, nullptr);

  auto invalid_group_1 = root->AddSubGroup(false, true);
  EXPECT_EQ(invalid_group_1, nullptr);
}

TEST_F(BufferGroupTest, AddMultiSubGroup) {
  auto root = std::make_shared<BufferGroup>();

  auto group_01 = root->AddSubGroup(true, false);
  auto group_02 = root->AddSubGroup(false, false);
  auto group_03 = root->AddSubGroup(false, true);
  uint32_t group_01_sum;
  EXPECT_EQ(group_01->GetOrder(), 1);
  EXPECT_EQ(group_01->GetPortId(), 0);
  EXPECT_EQ(group_01->GetGroupSum(&group_01_sum), STATUS_SUCCESS);
  EXPECT_EQ(3, group_01_sum);
  EXPECT_NE(group_01->GetGroup(), nullptr);
  EXPECT_EQ(group_01->IsStartGroup(), true);
  EXPECT_EQ(group_01->IsEndGroup(), false);

  auto group_11 = root->AddSubGroup(true, false);
  auto group_12 = root->AddSubGroup(false, true);
  uint32_t group_11_sum;
  EXPECT_EQ(group_11->GetOrder(), 1);
  EXPECT_EQ(group_11->GetPortId(), 1);
  EXPECT_EQ(group_11->GetGroupSum(&group_11_sum), STATUS_SUCCESS);
  EXPECT_EQ(2, group_11_sum);
  EXPECT_NE(group_11->GetGroup(), nullptr);
  EXPECT_EQ(group_11->IsStartGroup(), true);
  EXPECT_EQ(group_11->IsEndGroup(), false);
}

TEST_F(BufferGroupTest, GenNewBufferGroup) {
  auto root = std::make_shared<BufferGroup>();
  auto buffer_group_1 = root->GenerateSameLevelGroup();
  auto buffer_group_2 = root->GenerateSameLevelGroup();

  EXPECT_NE(buffer_group_1, nullptr);
  EXPECT_NE(buffer_group_2, nullptr);
  EXPECT_EQ(buffer_group_1->GetOneLevelGroup(), root);
  EXPECT_EQ(buffer_group_2->GetOneLevelGroup(), root);
}

TEST_F(IndexBufferTest, BindToRoot) {
  auto buffer_0 = std::make_shared<IndexBuffer>();
  EXPECT_EQ(buffer_0->GetSameLevelGroup(), nullptr);
  EXPECT_EQ(buffer_0->BindToRoot(), true);
  EXPECT_NE(buffer_0->GetSameLevelGroup(), nullptr);

  auto buffer_1 = std::make_shared<IndexBuffer>();
  EXPECT_EQ(buffer_1->BindToRoot(), true);
  EXPECT_NE(buffer_1->GetSameLevelGroup(), nullptr);
  EXPECT_NE(buffer_0->GetSameLevelGroup(), buffer_1->GetSameLevelGroup());
}

TEST_F(IndexBufferTest, CopyMetaTo) {
  auto buffer_0 = std::make_shared<IndexBuffer>();
  auto buffer_1 = std::make_shared<IndexBuffer>();
  EXPECT_EQ(buffer_0->CopyMetaTo(buffer_1), false);
  EXPECT_EQ(buffer_0->BindToRoot(), true);
  EXPECT_EQ(buffer_0->CopyMetaTo(buffer_1), true);
  EXPECT_EQ(buffer_0->GetSameLevelGroup(), buffer_1->GetSameLevelGroup());
}

TEST_F(IndexBufferTest, GenDownLevelBuffer) {
  auto buffer_0 = std::make_shared<IndexBuffer>();
  auto buffer_00 = std::make_shared<IndexBuffer>();
  auto buffer_01 = std::make_shared<IndexBuffer>();
  auto buffer_02 = std::make_shared<IndexBuffer>();
  auto buffer_03 = std::make_shared<IndexBuffer>();
  EXPECT_EQ(buffer_0->BindToRoot(), true);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_00, false, false), false);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_00, true, false), true);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_01, true, false), false);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_01, false, false), true);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_02, false, false), true);
  EXPECT_EQ(buffer_0->BindDownLevelTo(buffer_03, false, true), true);

  EXPECT_NE(buffer_00->GetStreamLevelGroup(), nullptr);
  EXPECT_NE(buffer_01->GetStreamLevelGroup(), nullptr);

  EXPECT_EQ(buffer_00->GetStreamLevelGroup(), buffer_01->GetStreamLevelGroup());
  EXPECT_EQ(buffer_00->GetSameLevelGroup()->GetOrder(), 1);
  uint32_t group_sum;
  EXPECT_EQ(buffer_00->GetSameLevelGroup()->GetGroupSum(&group_sum),
            STATUS_SUCCESS);
  EXPECT_EQ(4, group_sum);
}

TEST_F(IndexBufferTest, GenUpLevelBuffer) {
  auto root_buffer = std::make_shared<IndexBuffer>();
  auto buffer_00 = std::make_shared<IndexBuffer>();
  auto buffer_01 = std::make_shared<IndexBuffer>();
  auto buffer_02 = std::make_shared<IndexBuffer>();
  auto buffer_03 = std::make_shared<IndexBuffer>();

  EXPECT_EQ(root_buffer->BindToRoot(), true);
  EXPECT_EQ(root_buffer->BindDownLevelTo(buffer_00, true, false), true);
  EXPECT_EQ(root_buffer->BindDownLevelTo(buffer_01, false, false), true);
  EXPECT_EQ(root_buffer->BindDownLevelTo(buffer_02, false, false), true);
  EXPECT_EQ(root_buffer->BindDownLevelTo(buffer_03, false, true), true);

  auto uplevel_buffer = std::make_shared<IndexBuffer>();
  buffer_03->BindUpLevelTo(uplevel_buffer);
  EXPECT_EQ(uplevel_buffer->GetSameLevelGroup(),
            root_buffer->GetSameLevelGroup());
}

TEST_F(IndexBufferListTest, GetBufferPtrList) {
  auto buffer_0 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto buffer_1 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto placeholder_buffer = std::make_shared<IndexBuffer>();
  placeholder_buffer->MarkAsPlaceholder();
  auto buffer_2 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto index_buffer_list = std::make_shared<IndexBufferList>();
  index_buffer_list->PushBack(buffer_0);
  index_buffer_list->PushBack(buffer_1);
  index_buffer_list->PushBack(placeholder_buffer);
  index_buffer_list->PushBack(buffer_2);
  auto buffer_list = index_buffer_list->GetBufferPtrList();
  EXPECT_EQ(buffer_list.size(), 3);
}

TEST_F(IndexBufferListTest, GetPlaceholderPos) {
  auto buffer_0 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto buffer_1 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto placeholder_buffer = std::make_shared<IndexBuffer>();
  placeholder_buffer->MarkAsPlaceholder();
  auto buffer_2 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto index_buffer_list = std::make_shared<IndexBufferList>();
  index_buffer_list->PushBack(buffer_0);
  index_buffer_list->PushBack(buffer_1);
  index_buffer_list->PushBack(placeholder_buffer);
  index_buffer_list->PushBack(buffer_2);
  auto pos_set = index_buffer_list->GetPlaceholderPos();
  EXPECT_EQ(pos_set.size(), 1);
  auto pos_iter = pos_set.find(2);
  EXPECT_NE(pos_iter, pos_set.end());
}

TEST_F(IndexBufferListTest, BackfillEmptyBuffer) {
  auto index_buffer_list = std::make_shared<IndexBufferList>();
  index_buffer_list->BackfillBuffer(std::set<uint32_t>{2});
  EXPECT_EQ(index_buffer_list->GetBufferNum(), 0);

  index_buffer_list->BackfillBuffer(std::set<uint32_t>{0});
  EXPECT_EQ(index_buffer_list->GetBufferNum(), 1);
  EXPECT_TRUE(index_buffer_list->GetBuffer(0)->IsPlaceholder());

  index_buffer_list->BackfillBuffer(std::set<uint32_t>{1});
  EXPECT_EQ(index_buffer_list->GetBufferNum(), 2);
  EXPECT_TRUE(index_buffer_list->GetBuffer(1)->IsPlaceholder());
}

TEST_F(IndexBufferListTest, BackfillBuffer) {
  auto buffer_0 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto buffer_1 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto buffer_2 = std::make_shared<IndexBuffer>(std::make_shared<Buffer>());
  auto index_buffer_list = std::make_shared<IndexBufferList>();
  index_buffer_list->PushBack(buffer_0);
  index_buffer_list->PushBack(buffer_1);
  index_buffer_list->PushBack(buffer_2);
  index_buffer_list->BackfillBuffer(std::set<uint32_t>{2});
  EXPECT_EQ(index_buffer_list->GetBufferNum(), 4);
  EXPECT_TRUE(index_buffer_list->GetBuffer(2)->IsPlaceholder());
}

}  // namespace modelbox
