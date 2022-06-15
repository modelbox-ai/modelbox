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
  flow_graph_desc->Init();
  auto input1 = flow_graph_desc->AddInput("input1");
  flow_graph_desc->AddOutput("output1", input1);
  EXPECT_EQ(flow_graph_desc->GetStatus(), STATUS_SUCCESS);
  auto input_exist = flow_graph_desc->AddInput("input1");
  EXPECT_EQ(input_exist, nullptr);
  EXPECT_NE(flow_graph_desc->GetStatus(), STATUS_OK);
}

TEST_F(FlowGraphDescTest, AddNode) {
  auto flow_cfg = std::make_shared<FlowConfig>();
  flow_cfg->SetBatchSize(8);
  flow_cfg->SetQueueSize(32);

  auto flow_graph_desc = std::make_shared<FlowGraphDesc>();
  flow_graph_desc->Init(flow_cfg);
  auto input1 = flow_graph_desc->AddInput("input1");
  auto not_exist = flow_graph_desc->AddNode(
      "not_exist", "cpu", {"image_width=100", "image_height=100"}, input1);
  EXPECT_EQ(not_exist, nullptr);
  flow_graph_desc->AddOutput("output1", not_exist);
  EXPECT_EQ(flow_graph_desc->GetStatus(), STATUS_FAULT);
}

TEST_F(FlowGraphDescTest, AddFunction) {
  auto flow_graph_desc = std::make_shared<FlowGraphDesc>();
  flow_graph_desc->Init();
  auto func_node = flow_graph_desc->AddFunction(
      [](std::shared_ptr<DataContext> ctx) -> Status { return STATUS_OK; },
      {"in1", "in2"}, {"out"}, nullptr);
  EXPECT_EQ(func_node, nullptr);
  EXPECT_EQ(flow_graph_desc->GetStatus(), STATUS_FAULT);

  func_node = flow_graph_desc->AddFunction(
      [](std::shared_ptr<DataContext> ctx) -> Status { return STATUS_OK; },
      {"in1", "in2"}, {"out"}, {{"in2", nullptr}});
  func_node->SetNodeName("my_function");
  EXPECT_NE(func_node, nullptr);
  auto port0 = (*func_node)[0];
  ASSERT_NE(port0, nullptr);
  EXPECT_EQ(port0->GetNodeName(), "my_function");
  EXPECT_EQ(port0->GetPortName(), "out");
  EXPECT_EQ((*func_node)[1], nullptr);
  port0 = (*func_node)["out"];
  ASSERT_NE(port0, nullptr);
  EXPECT_EQ(port0->GetNodeName(), "my_function");
  EXPECT_EQ(port0->GetPortName(), "out");
  EXPECT_EQ((*func_node)["out2"], nullptr);
  EXPECT_EQ(flow_graph_desc->GetStatus(), STATUS_FAULT);

  func_node = flow_graph_desc->AddFunction(
      [](std::shared_ptr<DataContext> ctx) -> Status { return STATUS_OK; },
      {"in1"}, {"out"}, {{"in2", nullptr}});
  EXPECT_EQ(func_node, nullptr);
  EXPECT_EQ(flow_graph_desc->GetStatus(), STATUS_FAULT);
}

TEST_F(FlowGraphDescTest, NotInit) {
  auto graph_desc = std::make_shared<FlowGraphDesc>();
  auto input = graph_desc->AddInput("123");
  EXPECT_EQ(input, nullptr);
  auto node = graph_desc->AddNode("flowunit", "cpu");
  EXPECT_EQ(node, nullptr);
  graph_desc->AddOutput("321", input);
  EXPECT_EQ(graph_desc->GetStatus(), STATUS_FAULT);
}

}  // namespace modelbox