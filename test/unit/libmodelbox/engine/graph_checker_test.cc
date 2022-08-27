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

#include "modelbox/graph_checker.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "graph_conf_mockgraphconf/graph_conf_mockgraphconf.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/log.h"

namespace modelbox {

class GraphCheckerTest : public testing::Test {
 public:
  GraphCheckerTest() = default;

 protected:
  void SetUp() override {
    flow_ = std::make_shared<MockFlow>();

    {
      auto mock_desc = GenerateFlowunitDesc("test_0_1", {}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("test_0_2", {}, {"Out_1", "Out_2"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_3_0", {"In_1", "In_2", "In_3"}, {});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("test_2_0", {"In_1", "In_2"}, {});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("test_1_0", {"In_1"}, {});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_1_1_normal", {"In_1"}, {"Out_1"});
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("test_1_1", {"In_1"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      mock_desc->SetStreamSameCount(true);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_1_1_same_name", {"In_1"}, {"In_1"});
      mock_desc->SetFlowType(STREAM);
      mock_desc->SetStreamSameCount(true);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("stream_1_1", {"In_1"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      mock_desc->SetStreamSameCount(false);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("condition_1_3", {"In_1"},
                                            {"Out_1", "Out_2", "Out_3"});
      mock_desc->SetConditionType(IF_ELSE);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("condition_1_2", {"In_1"}, {"Out_1", "Out_2"});
      mock_desc->SetConditionType(IF_ELSE);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    // expand and collapse
    {
      auto mock_desc =
          GenerateFlowunitDesc("collapse_1_1", {"In_1"}, {"Out_1"});
      mock_desc->SetOutputType(COLLAPSE);
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("collapse_2_1", {"In_1", "In_2"}, {"Out_1"});
      mock_desc->SetOutputType(COLLAPSE);
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("expand_1_1", {"In_1"}, {"Out_1"});
      mock_desc->SetOutputType(EXPAND);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("expand_1_2", {"In_1"}, {"Out_1", "Out_2"});
      mock_desc->SetOutputType(EXPAND);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("expand_2_2", {"In_1", "In_2"},
                                            {"Out_1", "Out_2"});
      mock_desc->SetOutputType(EXPAND);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_1_2", {"In_1"}, {"Out_1", "Out_2"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc("test_1_3", {"In_1"},
                                            {"Out_1", "Out_2", "Out_3"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_1_2_normal", {"In_1"}, {"Out_1", "Out_2"});
      mock_desc->SetFlowType(NORMAL);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_3_1", {"In_1", "In_2", "In_3"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_2_1", {"In_1", "In_2"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc(
          "test_4_1", {"In_1", "In_2", "In_3", "In_4"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_loop", {"In_1"}, {"Out_1", "Out_2"});
      mock_desc->SetLoopType(LOOP);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc = GenerateFlowunitDesc(
          "test_loop_invalid", {"In_1", "In_2"}, {"Out_1", "Out_2"});
      mock_desc->SetLoopType(LOOP);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    {
      auto mock_desc =
          GenerateFlowunitDesc("test_1_1_stream", {"In_1"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      auto mock_funcitons = std::make_shared<MockFunctionCollection>();
      flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
    }

    flow_->Init(false);
  }

  void TearDown() override {
    auto flowunit_mgr = FlowUnitManager::GetInstance();
    flowunit_mgr->Clear();
    auto device_mgr = DeviceManager::GetInstance();
    device_mgr->Clear();
    auto drivers = Drivers::GetInstance();
    drivers->Clear();
  }

  void BuildGcGraph(const std::shared_ptr<Configuration> &config,
                    std::shared_ptr<GCGraph> &gcgraph) {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();

    auto device_mgr = DeviceManager::GetInstance();
    device_mgr->Initialize(drivers, config);

    auto flowunit_mgr = FlowUnitManager::GetInstance();
    flowunit_mgr->Initialize(drivers, device_mgr, config);

    GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();
    graphconf_mgr.Initialize(drivers, config);
    auto graphvizconf = graphconf_mgr.LoadGraphConfig(config);
    gcgraph = graphvizconf->Resolve();
  }

  std::shared_ptr<Graph> InitGraph(
      const std::shared_ptr<Configuration> &config) {
    auto device_mgr = DeviceManager::GetInstance();
    auto flowunit_mgr = FlowUnitManager::GetInstance();
    auto graph = std::make_shared<Graph>();
    graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
    return graph;
  }

  Status BuildGraph(const std::shared_ptr<Configuration> &config,
                    std::shared_ptr<Graph> &graph) {
    std::shared_ptr<GCGraph> gcgraph;
    BuildGcGraph(config, gcgraph);
    if (!gcgraph) {
      return STATUS_BADCONF;
    }

    graph = InitGraph(config);
    return graph->Build(gcgraph);
  }

  std::shared_ptr<Node> CastNode(const std::shared_ptr<NodeBase> &node) {
    return std::dynamic_pointer_cast<Node>(node);
  }

  void TestGraph(const std::string &graph, const Status &status) {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("graph.format", "graphviz");
    config->SetProperty("graph.graphconf", graph);
    std::shared_ptr<Graph> mb_graph;
    EXPECT_EQ(BuildGraph(config, mb_graph), status);
  }

 private:
  std::shared_ptr<MockFlow> flow_;
};

TEST_F(GraphCheckerTest, VirtualNode_NormalFlow) {
  std::string conf_file_value =
      R"(
        digraph demo {
          input1[type=input]
          output1[type=output]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          input1 -> b:In_1
          b:Out_1 -> output1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, VirtualNode_MatchAtVirtualInput) {
  std::string conf_file_value =
      R"(
        digraph demo {
          input1[type=input]
          input2[type=input]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          input1 -> b:In_1
          input2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, VirtualNode_MatchMultiInputOutput) {
  std::string conf_file_value =
      R"(
        digraph demo {
          input1[type=input]
          input2[type=input]
          output1[type=output]
          output2[type=output]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          input1 -> b:In_1
          input2 -> c:In_1
          b:Out_1 -> output1
          c:Out_1 -> output2
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}
/*
  a --> b --> d
    |         |
    |         |
    c --------
*/

TEST_F(GraphCheckerTest, SinglePortMatch_SingleOutPortLinkMultiInPort) {
  std::string conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_1 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

/*
  a --> b --> d
    |       |
    |       |
    c ------
*/

TEST_F(GraphCheckerTest, SinglePortNotMatch_SingleOutPortLinkSingleInPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_1 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

/*
  a --> b --> d
  |           |
  |           |
  c ----------
*/

TEST_F(GraphCheckerTest, MuliPortMatch_MultiOutPortLinkMultiInPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

/*
  a --> b --> d
  |         |
  |         |
  c --------
*/

TEST_F(GraphCheckerTest, MuliPortNotMatch_MultiOutPortLinkSingleInPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionMatch_OneInPortThreeOutPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_3, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          b:Out_3 -> e:In_1
          c:Out_1 -> f:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_OutConditionInMultiPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> e:In_2
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
          c:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_MutiConditionInSinglePort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
          c:Out_1 -> e:In_1
          c:Out_2 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotMatch_AllOutPortLinkDifferenceInPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_3, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          b:Out_3 -> e:In_1
          c:Out_1 -> f:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotMatch_MultiOutPortLinkInPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_3_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_1 -> d:In_1
          b:Out_2 -> e:In_1
          c:Out_1 -> f:In_1
          d:Out_1 -> f:In_2
          e:Out_1 -> f:In_3
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotMatch_SinglePortConditionNotMatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_3, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> c:In_2
          b:Out_3 -> d:In_1
          c:Out_1 -> d:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionMatch_SinglePortMatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_1 -> c:In_2
          b:Out_2 -> d:In_1
          c:Out_1 -> d:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, LoopMatch_LoopSelf) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1 
          b:Out_1 -> b:In_1
          b:Out_2 -> c:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, LoopMatch_LoopHasNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> d:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, LoopNotMatch_OverHierarchyLink) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_2_normal, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> d:In_1
          c:Out_2 -> d:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_NormalFlow) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_OnlyExpand) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_OnlyCollapse) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_OverMatchArch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          f[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
          g:Out_1 -> h:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_ExpandInMatchArch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          f[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
          g:Out_1 -> h:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_ExpandIsMatchNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> e:In_1
          d:Out_1 -> e:In_2
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest,
       ExpandCollapseMatch_MultiOutputExpandDirectConnectCollapse) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> d:In_1
          b:Out_2 -> d:In_2
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_CollapseIsMatchNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1 
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_CollapseIsMatchNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_CollapseInMatchArch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest,
       ExpandCollapseNotMatch_CollapseInMatchArch_SinglePathMatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_OneExpandMultiCollapse) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseMatch_MultiArch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0] 
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> e:In_1
          d:Out_1 -> f:In_1
          e:Out_1 -> g:In_1
          f:Out_1 -> g:In_2
          g:Out_1 -> h:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_OverHierarchyLink_FromOutToIn) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0, label="<Out_1>"] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"] 
          c[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1> "]
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0, label="<In_1>"] 
          e[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_2
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ExpandCollapseNotMatch_OverHierarchyLink_FromInToOut) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1> "]
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0, label="<In_1>"] 
          e[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_2
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotMatch_OverHierarchyLink_FromOutToIn) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_2
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> e:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotMatch_OverHierarchyLink_FromInToOut) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> e:In_1
          c:Out_2 -> f:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotAddition_MultiConditionLinkSameOut) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> e:In_1
          c:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, ConditionNotAddition_EndifAndInOtherMultiPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          a:Out_2 -> d:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_2
          c:Out_1 -> d:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_EndifAndCollapseInOnePort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> f:In_1
          d:Out_2 -> e:In_1
          e:Out_1 -> f:In_2
          e:Out_2 -> f:In_2
          f:Out_1 -> h:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionAddition_ConditionInExpandCollapse) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0] 
          e[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_2
          c:Out_1 -> d:In_1
          c:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, BranchCollapseMatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0] 
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          c:Out_1 -> e:In_2
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_SinglePortLinkMultiPortThroughNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          httpserver_sync_receive[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          param_analysis[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0] 
          my_nv_image_decoder[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0] 
          image_resolution_judge[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          color_tranpose_1[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          padding[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          normalize[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          face_detetc_infer[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0]
          face_detect_post[type=flowunit, flowunit=test_3_1, device=cpu, deviceid=0]
          face_condition[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          httpserver_sync_receive:Out_1 -> param_analysis:In_1
          param_analysis:Out_1 -> my_nv_image_decoder:In_1
          param_analysis:Out_2 -> image_resolution_judge:In_1
          my_nv_image_decoder:Out_1 -> image_resolution_judge:In_1
          image_resolution_judge:Out_1 -> face_detect_post:In_1
          image_resolution_judge:Out_1 -> color_tranpose_1:In_1
          color_tranpose_1:Out_1 -> padding:In_1
          padding:Out_1 -> normalize:In_1
          normalize:Out_1 -> face_detetc_infer:In_1
          face_detetc_infer:Out_1 -> face_detect_post:In_2
          face_detetc_infer:Out_2 -> face_detect_post:In_3
          face_detect_post:Out_1 -> face_condition:In_1
          image_resolution_judge:Out_2 -> face_condition:In_1
          face_condition:Out_1 -> g:In_1
          face_condition:Out_2 -> g:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_EndIfNodeIsAlsoCondition) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          begin[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          a[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          begin:Out_1 -> a:In_1
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          d:Out_2 -> f:In_1
          e:Out_1 -> f:In_1
          f:Out_1 -> end:In_1
          a:Out_2 -> end:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_EndIfNodeIsAlsoExpand) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          begin[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          a[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=collapse_2_1, device=cpu, deviceid=0]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          begin:Out_1 -> a:In_1
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          d:Out_2 -> f:In_1
          e:Out_1 -> f:In_2
          f:Out_1 -> end:In_1
          a:Out_2 -> end:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, ConditionMatch_EndIfNodeIsAlsoCollapse) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          begin[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          a[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          aa[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          begin:Out_1 -> a:In_1
          a:Out_1 -> aa:In_1
          aa:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> end:In_1
          a:Out_2 -> end:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, MultiNotMatch_MultiExpandSingleCollapseInBranch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          begin[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          a[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          begin:Out_1 -> a:In_1
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_BADCONF);
}

TEST_F(GraphCheckerTest, Bicycle) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_1_3, device=cpu, deviceid=0]
          i[type=flowunit, flowunit=test_3_1, device=cpu, deviceid=0]
          j[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          k[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          l[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          m[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0]
          n[type=flowunit, flowunit=expand_2_2, device=cpu, deviceid=0]
          o[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          p[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          q[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          r[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          s[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          t[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          u[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          v[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          w[type=flowunit, flowunit=test_1_3, device=cpu, deviceid=0]
          x[type=flowunit, flowunit=test_3_1, device=cpu, deviceid=0]
          y[type=flowunit, flowunit=test_4_1, device=cpu, deviceid=0]
          z[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          out[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c: In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
          e:Out_1 -> j:In_2
          e:Out_1 -> t:In_1
          e:Out_1 -> k:In_1
          e:Out_1 -> y:In_4
          f:Out_1 -> g:In_1
          g:Out_1 -> h:In_1
          h:Out_1 -> i:In_1
          h:Out_2 -> i:In_2
          h:Out_3 -> i:In_3
          i:Out_1 -> j:In_1
          j:Out_1 -> k:In_2
          j:Out_1 -> y:In_3
          t:Out_1 -> y:In_2
          t:Out_2 -> u:In_1
          u:Out_1 -> v:In_1
          v:Out_1 -> w:In_1
          w:Out_1 -> x:In_1
          w:Out_2 -> x:In_2
          w:Out_3 -> x:In_3
          x:Out_1 -> y:In_2
          k:Out_1 -> l:In_1
          l:Out_1 -> m:In_1
          l:Out_2 -> y:In_1
          m:Out_1 -> n:In_1
          m:Out_2 -> n:In_2
          n:Out_1 -> o:In_1
          n:Out_2 -> o:In_2
          o:Out_1 -> p:In_1
          p:Out_1 -> q:In_1
          q:Out_1 -> r:In_1
          r:Out_1 -> s:In_1
          s:Out_1 -> y:In_1
          y:Out_1 -> z:In_1
          z:Out_1 -> out:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, Park) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_1_3, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=test_3_1, device=cpu, deviceid=0]
          i[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          j[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          k[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          d:Out_2 -> i:In_1
          e:Out_1 -> f:In_1
          f:Out_1 -> g:In_1
          g:Out_1 -> h:In_1
          g:Out_2 -> h:In_2
          g:Out_3 -> h:In_3
          h:Out_1 -> i:In_1
          b:Out_1 -> i:In_2
          i:Out_1 -> j:In_1
          b:Out_1 -> j:In_2
          j:Out_1 -> k:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, Road) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_3, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_3_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          i[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          j[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          k[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          l[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          m[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          n[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          o[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          p[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          q[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          r[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          s[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          t[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          u[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          v[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          w[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          x[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          y[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          z[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          aa[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          bb[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          cc[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          dd[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          ee[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          ff[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          gg[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]


          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
          e:Out_2 -> f:In_2
          e:Out_3 -> f:In_3
          f:Out_1 -> g:In_1
          c:Out_1 -> g:In_2
          g:Out_1 -> h:In_1
          h:Out_1 -> i:In_1
          h:Out_1 -> n:In_1
          h:Out_2 -> o:In_1
          i:Out_1 -> j:In_1
          i:Out_2 -> j:In_2
          j:Out_1 -> k:In_1
          k:Out_1 -> l:In_1
          l:Out_1 -> m:In_1
          m:Out_1 -> n:In_2
          n:Out_1 -> o:In_1
          o:Out_1 -> p:In_1
          o:Out_1 -> v:In_1
          o:Out_2 -> w:In_1
          p:Out_1 -> q:In_1
          p:Out_2 -> q:In_2
          q:Out_1 -> r:In_1
          r:Out_1 -> s:In_1
          p:Out_2 -> t:In_1
          s:Out_1 -> t:In_2
          t:Out_1 -> u:In_1
          u:Out_1 -> v:In_2
          v:Out_1 -> w:In_1
          w:Out_1 -> x:In_1
          w:Out_1 -> dd:In_1
          w:Out_2 -> ee:In_1
          x:Out_1 -> y:In_1
          x:Out_2 -> y:In_2
          y:Out_1 -> z:In_1
          z:Out_1 -> aa:In_1
          aa:Out_1 -> bb:In_1
          bb:Out_1 -> cc:In_1
          cc:Out_1 -> dd:In_2
          dd:Out_1 -> ee:In_1
          b:Out_2 -> ff:In_1
          ee:Out_1 -> ff:In_1
          ff:Out_1 -> gg:In_1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, NodeHasSameNameInInputOutputPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0]
          b[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          c[type=flowunit, flowunit=expand_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          f[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          g[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          h[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0]
          i[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0]
          j[type=flowunit, flowunit=test_1_1_same_name, device=cpu, deviceid=0]
          output1[type=output]

          a:Out_1->b:In_1
          b:Out_1->c:In_1
          b:Out_1->i:In_1
          b:Out_2->j:In_1
          c:Out_1->d:In_1
          c:Out_2->d:In_2
          c:Out_1->g:In_1
          d:Out_1->e:In_1
          e:Out_1->f:In_1
          f:Out_1->g:In_2
          g:Out_1->h:In_1
          h:Out_1->i:In_2
          i:Out_1->j:In_1
          j:In_1->output1
        }
      )";

  TestGraph(conf_file_value, STATUS_OK);
}

TEST_F(GraphCheckerTest, GetSetMatchNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0] 
          b[type=flowunit, flowunit=expand_1_1, device=cpu, deviceid=0] 
          c[type=flowunit, flowunit=condition_1_2, device=cpu, deviceid=0]
          d[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0]
          e[type=flowunit, flowunit=collapse_1_1, device=cpu, deviceid=0] 
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  std::shared_ptr<Graph> graph;
  EXPECT_TRUE(BuildGraph(config, graph) == STATUS_OK);
  EXPECT_EQ(CastNode(graph->GetNode("a"))->GetMatchNode(), nullptr);
  EXPECT_EQ(CastNode(graph->GetNode("b"))->GetMatchNode(), nullptr);
  EXPECT_EQ(CastNode(graph->GetNode("c"))->GetMatchNode(), nullptr);
  EXPECT_EQ(CastNode(graph->GetNode("d"))->GetMatchNode(), graph->GetNode("c"));
  EXPECT_EQ(CastNode(graph->GetNode("e"))->GetMatchNode(), graph->GetNode("b"));
  EXPECT_EQ(CastNode(graph->GetNode("f"))->GetMatchNode(), nullptr);
}

}  // namespace modelbox