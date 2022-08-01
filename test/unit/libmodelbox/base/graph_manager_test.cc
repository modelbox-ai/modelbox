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


#include <dlfcn.h>
#include <poll.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include "modelbox/base/driver.h"
#include "modelbox/base/graph_manager.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "graph_conf_mockgraphconf/graph_conf_mockgraphconf.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

namespace modelbox {

Status SaveConfigFile(std::string &name, std::string &value) {
  std::ofstream fp(name);
  fp << value;
  fp << std::endl;
  fp.close();
  return STATUS_OK;
}

Status RemoveFile(std::string &name) {
  remove(name.c_str());
  return STATUS_OK;
}

class GraphManagerTest : public testing::Test {
 public:
  GraphManagerTest() = default;
  MockDriverCtl ctl_;
  std::shared_ptr<modelbox::Configuration> config_;
  std::string conf_file_name_;
  std::string conf_file_value_;

 protected:
  void SetUp() override {
    std::shared_ptr<Drivers> drivers_ = Drivers::GetInstance();

    modelbox::DriverDesc desc;
    desc.SetClass("DRIVER-GRAPHCONF");
    desc.SetType("GRAPHVIZ");
    desc.SetName("GRAPHCONF-GRAPHVIZ");
    desc.SetDescription("graph config parse graphviz");
    desc.SetVersion("0.1.0");
    std::string file_path_device =
        std::string(TEST_LIB_DIR) + "/libmodelbox-graphconf-graphviz.so";
    desc.SetFilePath(file_path_device);

    ctl_.AddMockDriverGraphConf("graphviz", "", desc);

    drivers_->Scan(TEST_LIB_DIR, "libmodelbox-graphconf-graphviz.so");

    ConfigurationBuilder builder;
    config_ = builder.Build();

    conf_file_value_ =
        "digraph demo {                               \
          bgcolor=\"beige\"                           \
          node [shape=\"record\", height=.1]          \
          node0[label=\"<f0> | <f1> G | <f2>\"]       \
          node1[label=\"<f0> | <f1> E | <f2>\"]       \
          node2[label=\"<f0> | <f1> B | <f2>\"]       \
          node0:f0 -> node1:f1                        \
          node0:f2 -> node2:f1                        \
      }";

    conf_file_name_ = std::string(TEST_DATA_DIR) + "/test_graph.gv";
    SaveConfigFile(conf_file_name_, conf_file_value_);
  };

  void TearDown() override {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    drivers->Clear();
    RemoveFile(conf_file_name_);
  };
};

TEST_F(GraphManagerTest, ResolveStr) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconf", conf_file_value_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);
  EXPECT_NE(graphvizconf, nullptr);

  auto gcgraph = graphvizconf->Resolve();
  EXPECT_NE(gcgraph, nullptr);

  auto graph_configuration = gcgraph->GetConfiguration();
  auto graph_bgcolor = graph_configuration->GetString("bgcolor");
  MBLOG_INFO << "bgcolor : " << graph_bgcolor;
  EXPECT_EQ(graph_bgcolor, "beige");
  auto graph_error = graph_configuration->GetString("error");
  EXPECT_EQ(graph_error, "");
};

TEST_F(GraphManagerTest, NodeStr) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconf", conf_file_value_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);

  auto gcgraph = graphvizconf->Resolve();

  auto node0 = gcgraph->GetNode("node0");
  EXPECT_NE(node0, nullptr);

  auto outputPort = node0->GetOutputPorts();
  std::string node0_output_port0 = *outputPort->begin();
  MBLOG_INFO << "node0 outputPort0 : " << node0_output_port0;
  EXPECT_EQ(node0_output_port0, "f0");

  std::string node0_output_port1 = *(++outputPort->begin());
  MBLOG_INFO << "node0 outputPort1 : " << node0_output_port1;
  EXPECT_EQ(node0_output_port1, "f2");

  auto node1 = gcgraph->GetNode("node1");
  EXPECT_NE(node1, nullptr);

  auto node1_configuration = node1->GetConfiguration();
  auto node1_height = node1_configuration->GetString("height");
  EXPECT_EQ(node1_height, ".1");
  auto node1_label = node1_configuration->GetString("label");
  EXPECT_EQ(node1_label, "<f0> | <f1> E | <f2>");
  auto node1_shape = node1_configuration->GetString("shape");
  EXPECT_EQ(node1_shape, "record");
  auto node1_error = node1_configuration->GetString("error");
  EXPECT_EQ(node1_error, "");

  auto node2 = gcgraph->GetNode("node2");
  EXPECT_NE(node2, nullptr);

  auto inputPorts = node2->GetInputPorts();
  std::string node2_input_port0 = *inputPorts->begin();
  MBLOG_INFO << "node2 inputPort0 : " << node2_input_port0;
  EXPECT_EQ(node2_input_port0, "f1");

  auto err_node = gcgraph->GetNode("err_node");
  EXPECT_EQ(err_node, nullptr);

  auto all_nodes = gcgraph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 3);

  auto root_graph = gcgraph->GetNode("node0")->GetRootGraph();
  EXPECT_EQ(root_graph, gcgraph);
};

TEST_F(GraphManagerTest, EdgeStr) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconf", conf_file_value_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);

  auto gcgraph = graphvizconf->Resolve();

  auto edge1 = gcgraph->GetEdge("node0:f0-node1:f1");
  EXPECT_NE(edge1, nullptr);

  auto edge1_configuration = edge1->GetConfiguration();
  auto edge1_headport = edge1_configuration->GetString("headport");
  EXPECT_EQ(edge1_headport, "f1");
  auto edge1_tailport = edge1_configuration->GetString("tailport");
  EXPECT_EQ(edge1_tailport, "f0");
  auto edge1_error = edge1_configuration->GetString("error");
  EXPECT_EQ(edge1_error, "");

  auto head_node = edge1->GetHeadNode();
  auto node0 = gcgraph->GetNode("node0");
  EXPECT_EQ(head_node, node0);

  auto tail_node = edge1->GetTailNode();
  auto node1 = gcgraph->GetNode("node1");
  EXPECT_EQ(tail_node, node1);

  auto edge2 = gcgraph->GetEdge("node0:f2-node2:f1");
  EXPECT_NE(edge2, nullptr);

  auto edge3 = gcgraph->GetEdge("node1-node2");
  EXPECT_EQ(edge3, nullptr);

  auto all_edges = gcgraph->GetAllEdges();
  EXPECT_EQ(all_edges.size(), 2);
};

TEST_F(GraphManagerTest, SubgraphStr){

};

TEST_F(GraphManagerTest, ResolveFile) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconffilepath", conf_file_name_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);

  auto gcgraph = graphvizconf->Resolve();
};

TEST_F(GraphManagerTest, NodeFile) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconffilepath", conf_file_name_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);

  auto gcgraph = graphvizconf->Resolve();
  ASSERT_NE(gcgraph, nullptr);

  auto node0 = gcgraph->GetNode("node0");
  EXPECT_NE(node0, nullptr);

  auto node1 = gcgraph->GetNode("node1");
  EXPECT_NE(node1, nullptr);

  auto node2 = gcgraph->GetNode("node2");
  EXPECT_NE(node2, nullptr);

  auto err_node = gcgraph->GetNode("err_node");
  EXPECT_EQ(err_node, nullptr);

  auto all_nodes = gcgraph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 3);

  auto root_graph = gcgraph->GetNode("node0")->GetRootGraph();
  EXPECT_EQ(root_graph, gcgraph);
};

TEST_F(GraphManagerTest, EdgeFile) {
  config_->SetProperty("graph.format", "graphviz");
  config_->SetProperty("graph.graphconffilepath", conf_file_name_);

  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();

  auto ret = graphconf_mgr.Initialize(drivers, config_);
  EXPECT_EQ(ret, STATUS_OK);

  auto graphvizconf = graphconf_mgr.LoadGraphConfig(config_);

  auto gcgraph = graphvizconf->Resolve();

  auto edge1 = gcgraph->GetEdge("node0:f0-node1:f1");
  EXPECT_NE(edge1, nullptr);

  auto head_node = edge1->GetHeadNode();
  auto node0 = gcgraph->GetNode("node0");
  EXPECT_EQ(head_node, node0);

  auto tail_node = edge1->GetTailNode();
  auto node1 = gcgraph->GetNode("node1");
  EXPECT_EQ(tail_node, node1);

  auto edge2 = gcgraph->GetEdge("node0:f2-node2:f1");
  EXPECT_NE(edge2, nullptr);

  auto edge3 = gcgraph->GetEdge("node1-node2");
  EXPECT_EQ(edge3, nullptr);

  auto all_edges = gcgraph->GetAllEdges();
  EXPECT_EQ(all_edges.size(), 2);
};

TEST_F(GraphManagerTest, SubgraphFile){

};

}  // namespace modelbox