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

#include "modelbox/flow_graph_desc.h"

#include "gtest/gtest.h"
#include "mockflow.h"

namespace modelbox {

class FlowGraphDescTest : public testing::Test {};

TEST_F(FlowGraphDescTest, AddInputOutput) {
  auto flow_graph_desc = std::make_shared<FlowGraphDesc>();
  auto input1 = flow_graph_desc->AddInput("input1");
  ASSERT_NE(input1, nullptr);
  EXPECT_EQ(input1->GetNodeName(), "input1");
  flow_graph_desc->AddOutput("output1", input1);
  auto input_exist = flow_graph_desc->AddInput("input1");
  EXPECT_EQ(input_exist, nullptr);
}

TEST_F(FlowGraphDescTest, AddNode) {
  auto flow_graph_desc = std::make_shared<FlowGraphDesc>();
  flow_graph_desc->SetBatchSize(8);
  flow_graph_desc->SetQueueSize(32);
  flow_graph_desc->SetSkipDefaultDrivers(true);
  flow_graph_desc->SetDriversDir({"/test/"});

  auto input1 = flow_graph_desc->AddInput("input1");
  ASSERT_NE(input1, nullptr);

  auto node1 = flow_graph_desc->AddNode(
      "fu", "cpu", {"image_width=100", "image_height=100"}, input1);
  ASSERT_NE(node1, nullptr);
  EXPECT_EQ(node1->GetNodeName(), "fu");

  auto node1_port1 = (*node1)[1];
  ASSERT_NE(node1_port1, nullptr);
  EXPECT_EQ(node1_port1->GetNodeName(), "fu");
  EXPECT_FALSE(node1_port1->IsDescribeInName());
  EXPECT_EQ(node1_port1->GetPortIdx(), 1);

  auto node1_portx = (*node1)["x"];
  ASSERT_NE(node1_portx, nullptr);
  EXPECT_EQ(node1_portx->GetNodeName(), "fu");
  EXPECT_TRUE(node1_portx->IsDescribeInName());
  EXPECT_EQ(node1_portx->GetPortName(), "x");

  auto node2 = flow_graph_desc->AddNode(
      "fu", "cpu", {"image_width=100", "image_height=100"}, node1);
  ASSERT_NE(node2, nullptr);
  EXPECT_EQ(node2->GetNodeName(), "fu2");
  node2->SetNodeName("custom_fu");
  EXPECT_EQ(node2->GetNodeName(), "custom_fu");

  auto node3 = flow_graph_desc->AddNode(
      "fu", "cpu", {"image_width=100", "image_height=100"},
      {{"in1", (*node1)[0]}, {"in2", (*node2)["x"]}});
  ASSERT_NE(node3, nullptr);

  flow_graph_desc->AddOutput("output1", node3);
}

TEST_F(FlowGraphDescTest, AddFunction) {
  auto flow_graph_desc = std::make_shared<FlowGraphDesc>();
  auto func_node = flow_graph_desc->AddFunction(
      [](const std::shared_ptr<DataContext>& data_ctx) -> Status {
        return STATUS_OK;
      },
      {"in1", "in2"}, {"out"}, nullptr);
  EXPECT_EQ(func_node, nullptr);

  func_node = flow_graph_desc->AddFunction(
      [](const std::shared_ptr<DataContext>& data_ctx) -> Status {
        return STATUS_OK;
      },
      {"in1", "in2"}, {"out"}, {{"in2", nullptr}});
  ASSERT_NE(func_node, nullptr);
  func_node->SetNodeName("my_function");
  auto port0 = (*func_node)[0];
  ASSERT_NE(port0, nullptr);
  EXPECT_FALSE(port0->IsDescribeInName());
  EXPECT_EQ(port0->GetNodeName(), "my_function");
  EXPECT_EQ(port0->GetPortIdx(), 0);

  port0 = (*func_node)["out"];
  ASSERT_NE(port0, nullptr);
  EXPECT_TRUE(port0->IsDescribeInName());
  EXPECT_EQ(port0->GetNodeName(), "my_function");
  EXPECT_EQ(port0->GetPortName(), "out");

  func_node = flow_graph_desc->AddFunction(
      [](const std::shared_ptr<DataContext>& data_ctx) -> Status {
        return STATUS_OK;
      },
      {"in1"}, {"out"}, {{"in2", nullptr}});
  EXPECT_NE(func_node, nullptr);
}

}  // namespace modelbox