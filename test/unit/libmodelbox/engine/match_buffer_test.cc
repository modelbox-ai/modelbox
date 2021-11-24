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


#include "modelbox/match_buffer.h"

#include "gtest/gtest.h"

namespace modelbox {

class MatchBufferTest : public testing::Test {
 public:
  MatchBufferTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){

  };
};

TEST_F(MatchBufferTest, IsMatch) {
  auto buffer = std::make_shared<IndexBuffer>();
  buffer->BindToRoot();

  auto match_buffer_0 = std::make_shared<MatchBuffer>(-1);
  match_buffer_0->SetBuffer("name_1", buffer);
  EXPECT_EQ(false, match_buffer_0->IsMatch());

  auto match_buffer_1 = std::make_shared<MatchBuffer>(1);
  match_buffer_1->SetBuffer("name_1", buffer);
  EXPECT_EQ(true, match_buffer_1->IsMatch());

  auto match_buffer_2 = std::make_shared<MatchBuffer>(2);
  match_buffer_2->SetBuffer("name_1", buffer);
  EXPECT_EQ(false, match_buffer_2->IsMatch());

  auto buffer_1 = std::make_shared<IndexBuffer>();
  auto match_buffer_3 = std::make_shared<MatchBuffer>(1);
  match_buffer_3->SetBuffer("name_1", buffer_1);
  EXPECT_EQ(false, match_buffer_3->IsMatch());
}

TEST_F(MatchBufferTest, GetOrder) {
  auto root_buffer = std::make_shared<IndexBuffer>();
  root_buffer->BindToRoot();

  auto buffer_1 = std::make_shared<IndexBuffer>();
  auto buffer_2 = std::make_shared<IndexBuffer>();
  auto buffer_3 = std::make_shared<IndexBuffer>();

  root_buffer->BindDownLevelTo(buffer_1, true, false);
  root_buffer->BindDownLevelTo(buffer_2, false, false);
  root_buffer->BindDownLevelTo(buffer_3, false, true);

  auto match_buffer_1 = std::make_shared<MatchBuffer>(1);
  match_buffer_1->SetBuffer("name_1", buffer_1);
  auto match_buffer_2 = std::make_shared<MatchBuffer>(1);
  match_buffer_2->SetBuffer("name_1", buffer_2);
  auto match_buffer_3 = std::make_shared<MatchBuffer>(1);
  match_buffer_3->SetBuffer("name_1", buffer_3);
  EXPECT_EQ(1, match_buffer_1->GetOrder());
  EXPECT_EQ(2, match_buffer_2->GetOrder());
  EXPECT_EQ(3, match_buffer_3->GetOrder());
}

TEST_F(MatchBufferTest, Sort) {
  auto root_buffer = std::make_shared<IndexBuffer>();
  root_buffer->BindToRoot();

  auto buffer_1 = std::make_shared<IndexBuffer>();
  auto buffer_2 = std::make_shared<IndexBuffer>();
  auto buffer_3 = std::make_shared<IndexBuffer>();

  root_buffer->BindDownLevelTo(buffer_1, true, false);
  root_buffer->BindDownLevelTo(buffer_2, false, false);
  root_buffer->BindDownLevelTo(buffer_3, false, true);

  std::vector<std::shared_ptr<MatchBuffer>> sort_vector;
  auto match_buffer_1 = std::make_shared<MatchBuffer>(1);
  match_buffer_1->SetBuffer("name_1", buffer_3);
  sort_vector.push_back(match_buffer_1);
  auto match_buffer_2 = std::make_shared<MatchBuffer>(1);
  match_buffer_2->SetBuffer("name_1", buffer_1);
  sort_vector.push_back(match_buffer_2);
  auto match_buffer_3 = std::make_shared<MatchBuffer>(1);
  match_buffer_3->SetBuffer("name_1", buffer_2);
  sort_vector.push_back(match_buffer_3);

  std::sort(sort_vector.begin(), sort_vector.end(),
            [](std::shared_ptr<MatchBuffer> a, std::shared_ptr<MatchBuffer> b) {
              return (a->GetOrder() < b->GetOrder());
            });
  EXPECT_EQ(1, sort_vector[0]->GetOrder());
  EXPECT_EQ(2, sort_vector[1]->GetOrder());
  EXPECT_EQ(3, sort_vector[2]->GetOrder());
}

}  // namespace modelbox