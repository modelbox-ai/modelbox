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


#include "modelbox/buffer_type.h"
#include "gtest/gtest.h"
namespace modelbox {
class BufferTypeTest : public testing::Test {
 public:
  BufferTypeTest() = default;

 protected:
  void SetUp() override {
    auto *tree = BufferTypeTree::GetInstance();
    tree->AddRootType("raw");
  };
  void TearDown() override {
    auto *tree = BufferTypeTree::GetInstance();
    tree->RemoveType("raw");
  };
};

TEST_F(BufferTypeTest, AddRootType) {
  auto *tree = BufferTypeTree::GetInstance();
  EXPECT_EQ(true,tree->AddRootType("raw"));
  EXPECT_EQ(false,tree->AddRootType("unknow_node"));
  EXPECT_EQ(true,tree->RemoveType("raw"));
  EXPECT_EQ(true,tree->AddRootType("raw"));
  EXPECT_EQ("raw",tree->GetType("raw")->GetType());
}

TEST_F(BufferTypeTest, AddType) {
  auto *tree = BufferTypeTree::GetInstance();
  EXPECT_EQ(true, tree->AddType("tensor", "raw"));
  EXPECT_EQ("tensor", tree->GetType("tensor")->GetType());
  EXPECT_EQ(false, tree->AddType("tensor", "wrong_node"));

  EXPECT_EQ(false, tree->AddType("other_tensor", "wrong_node"));
  EXPECT_EQ(nullptr, tree->GetType("other_tensor"));
  tree->AddType("other_tensor", "raw");
  EXPECT_EQ(false, tree->AddType("other_tensor", "tensor"));
}

TEST_F(BufferTypeTest, IsCompatible) {
  auto *tree = BufferTypeTree::GetInstance();
  tree->AddType("other_tensor", "raw");
  tree->AddType("tensor", "raw");
  tree->AddType("nhwc_tensor", "tensor");
  EXPECT_EQ(true, tree->IsCompatible("nhwc_tensor","tensor"));
  EXPECT_EQ(true, tree->IsCompatible("nhwc_tensor","raw"));
  EXPECT_EQ(true, tree->IsCompatible("nhwc_tensor","nhwc_tensor"));
  EXPECT_EQ(false, tree->IsCompatible("nhwc_tensor","other_tensor"));
  EXPECT_EQ(false, tree->IsCompatible("nhwc_tensor","unknow_node"));
  EXPECT_EQ(false, tree->IsCompatible("unknow_node","raw"));
}

TEST_F(BufferTypeTest, RemoveType) {
  auto *tree = BufferTypeTree::GetInstance();
  EXPECT_EQ(false, tree->RemoveType("unknow_format"));
  tree->AddType("other_tensor", "raw");
  tree->AddType("tensor", "raw");
  tree->AddType("nhwc_tensor", "tensor");
  EXPECT_EQ(false, tree->RemoveType("unknow_format"));
  EXPECT_EQ(true, tree->RemoveType("tensor"));
  EXPECT_EQ(nullptr, tree->GetType("nhwc_tensor"));
  EXPECT_EQ(nullptr, tree->GetType("tensor"));
  EXPECT_EQ(1, tree->GetType("raw")->GetChildrenType().size());
  EXPECT_EQ("other_tensor",
            tree->GetType("raw")->GetChildrenType()[0]->GetType());
}

}  // namespace modelbox