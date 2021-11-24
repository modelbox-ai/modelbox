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


#include "modelbox/output_buffer.h"

#include "gtest/gtest.h"
namespace modelbox {

class OutputRingsTest : public testing::Test {
 public:
  OutputRingsTest() {}

 protected:
  std::shared_ptr<OriginDataMap> legal_output_map_;
  std::shared_ptr<OriginDataMap> legal_condition_output_map_;
  std::shared_ptr<OriginDataMap> legal_empty_output_map_;
  std::shared_ptr<OriginDataMap> illegal_output_map_1_;
  std::shared_ptr<OriginDataMap> illegal_output_map_2_;
  std::shared_ptr<OriginDataMap> illegal_output_map_3_;
  virtual void SetUp() {
    legal_output_map_ = std::make_shared<OriginDataMap>();
    legal_condition_output_map_ = std::make_shared<OriginDataMap>();
    legal_empty_output_map_ = std::make_shared<OriginDataMap>();
    illegal_output_map_1_ = std::make_shared<OriginDataMap>();
    illegal_output_map_2_ = std::make_shared<OriginDataMap>();
    illegal_output_map_3_ = std::make_shared<OriginDataMap>();

    std::vector<std::shared_ptr<Buffer>> zero_buffer_vector;

    std::vector<std::shared_ptr<Buffer>> null_buffer_vector;
    null_buffer_vector.push_back(nullptr);

    std::vector<std::shared_ptr<Buffer>> three_buffer_vector;
    three_buffer_vector.push_back(std::make_shared<Buffer>());
    three_buffer_vector.push_back(std::make_shared<Buffer>());
    three_buffer_vector.push_back(std::make_shared<Buffer>());

    std::vector<std::shared_ptr<Buffer>> four_buffer_vector;
    four_buffer_vector.push_back(std::make_shared<Buffer>());
    four_buffer_vector.push_back(std::make_shared<Buffer>());
    four_buffer_vector.push_back(std::make_shared<Buffer>());
    four_buffer_vector.push_back(std::make_shared<Buffer>());

    auto buffer_null_output = std::make_shared<BufferList>(null_buffer_vector);
    auto buffer_0_output = std::make_shared<BufferList>(zero_buffer_vector);
    auto buffer_3_output = std::make_shared<BufferList>(three_buffer_vector);
    auto buffer_4_output = std::make_shared<BufferList>(four_buffer_vector);

    legal_output_map_->emplace("output_1", buffer_3_output);
    legal_output_map_->emplace("output_2", buffer_3_output);
    legal_output_map_->emplace("output_3", buffer_3_output);

    legal_condition_output_map_->emplace("output_1", buffer_null_output);
    legal_condition_output_map_->emplace("output_2", buffer_3_output);
    legal_condition_output_map_->emplace("output_3", buffer_null_output);

    legal_empty_output_map_->emplace("output_1", buffer_0_output);
    legal_empty_output_map_->emplace("output_2", buffer_0_output);
    legal_empty_output_map_->emplace("output_3", buffer_0_output);

    illegal_output_map_1_->emplace("output_1", buffer_3_output);
    illegal_output_map_1_->emplace("output_2", buffer_0_output);
    illegal_output_map_1_->emplace("output_3", buffer_3_output);

    illegal_output_map_2_->emplace("output_1", buffer_3_output);
    illegal_output_map_2_->emplace("output_2", buffer_4_output);
    illegal_output_map_2_->emplace("output_3", buffer_3_output);

    illegal_output_map_3_->emplace("output_1", buffer_0_output);
    illegal_output_map_3_->emplace("output_2", buffer_3_output);
    illegal_output_map_3_->emplace("output_3", buffer_3_output);
  };
  virtual void TearDown(){};
};

TEST_F(OutputRingsTest, Init) {
  auto legal_output = OutputRings(*(legal_output_map_.get()));
  EXPECT_EQ(legal_output.IsValid(), STATUS_SUCCESS);

  auto legal_condition_output =
      OutputRings(*(legal_condition_output_map_.get()));
  EXPECT_EQ(legal_condition_output.IsValid(), STATUS_SUCCESS);

  auto legal_empty_output = OutputRings(*(legal_empty_output_map_.get()));
  EXPECT_EQ(legal_empty_output.IsValid(), STATUS_SUCCESS);

  auto illegal_output_1 = OutputRings(*(illegal_output_map_1_.get()));
  EXPECT_EQ(illegal_output_1.IsValid(), STATUS_INVALID);

  auto illegal_output_2 = OutputRings(*(illegal_output_map_2_.get()));
  EXPECT_EQ(illegal_output_2.IsValid(), STATUS_INVALID);

  auto illegal_output_3 = OutputRings(*(illegal_output_map_3_.get()));
  EXPECT_EQ(illegal_output_3.IsValid(), STATUS_INVALID);
}

TEST_F(OutputRingsTest, GetOneBufferList) {
  auto legal_output = OutputRings(*(legal_output_map_.get()));
  legal_output.IsValid();
  EXPECT_NE(legal_output.GetOneBufferList(), nullptr);
}

TEST_F(OutputRingsTest, BroadcastMetaToRing) {
  auto legal_output = OutputRings(*(legal_output_map_.get()));
  legal_output.IsValid();
  auto buffer_list = legal_output.GetOneBufferList();

  for (uint32_t j = 0; j < buffer_list->GetBufferNum(); j++) {
    buffer_list->GetBuffer(j)->BindToRoot();
  }
  EXPECT_EQ(legal_output.BroadcastMetaToAll(), STATUS_SUCCESS);

  auto legal_condition_output =
      OutputRings(*(legal_condition_output_map_.get()));
  legal_condition_output.IsValid();
  auto condition_buffer_list = legal_condition_output.GetOneBufferList();

  for (uint32_t j = 0; j < condition_buffer_list->GetBufferNum(); j++) {
    buffer_list->GetBuffer(j)->BindToRoot();
  }
  EXPECT_EQ(legal_condition_output.BroadcastMetaToAll(), STATUS_SUCCESS);
}

TEST_F(OutputRingsTest, AppendOutputMap) {
  auto legal_output = OutputRings(*(legal_output_map_.get()));
  legal_output.IsValid();
  auto buffer_list = legal_output.GetOneBufferList();

  for (uint32_t j = 0; j < buffer_list->GetBufferNum(); j++) {
    buffer_list->GetBuffer(j)->BindToRoot();
  }
  EXPECT_EQ(legal_output.BroadcastMetaToAll(), STATUS_SUCCESS);

  OutputIndexBuffer map;
  std::vector<std::shared_ptr<IndexBuffer>> vector_1;
  std::vector<std::shared_ptr<IndexBuffer>> vector_2;
  std::vector<std::shared_ptr<IndexBuffer>> vector_3;
  map.emplace("output_1", vector_1);
  map.emplace("output_2", vector_1);
  map.emplace("output_3", vector_1);

  EXPECT_EQ(legal_output.AppendOutputMap(&map), STATUS_SUCCESS);
  EXPECT_EQ(map["output_1"].size(), 3);
  EXPECT_EQ(map["output_2"].size(), 3);
  EXPECT_EQ(map["output_3"].size(), 3);

  EXPECT_EQ(map["output_1"][0]->GetSameLevelGroup(),
            map["output_2"][0]->GetSameLevelGroup());
  EXPECT_EQ(map["output_2"][0]->GetSameLevelGroup(),
            map["output_3"][0]->GetSameLevelGroup());
}

}  // namespace modelbox