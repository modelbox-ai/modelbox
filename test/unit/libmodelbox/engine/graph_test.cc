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

#include "modelbox/graph.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "graph_conf_mockgraphconf/graph_conf_mockgraphconf.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/log.h"

using ::testing::_;
namespace modelbox {
class GraphTest : public testing::Test {
 public:
  GraphTest() {}

 protected:
  virtual void SetUp() {
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
      auto mock_desc = GenerateFlowunitDesc("stream_1_1", {"In_1"}, {"Out_1"});
      mock_desc->SetFlowType(STREAM);
      mock_desc->SetStreamSameCount(false);
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

    {
      auto mock_desc =
          GenerateFlowunitDesc("collapse_1_1", {"In_1"}, {"Out_1"});
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
          GenerateFlowunitDesc("test_1_2", {"In_1"}, {"Out_1", "Out_2"});
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

  virtual void TearDown() {
    auto device_mgr = DeviceManager::GetInstance();
    device_mgr->Clear();
    auto flowunit_mgr = FlowUnitManager::GetInstance();
    flowunit_mgr->Clear();
    auto drivers = Drivers::GetInstance();
    drivers->Clear();
  }

  Status BuildGraph(std::shared_ptr<Configuration> config,
                    std::shared_ptr<GCGraph>* gcgraph_out = nullptr) {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();

    auto device_mgr = DeviceManager::GetInstance();
    device_mgr->Initialize(drivers, config);

    auto flowunit_mgr = FlowUnitManager::GetInstance();
    flowunit_mgr->Initialize(drivers, device_mgr, config);

    GraphConfigManager graphconf_mgr = GraphConfigManager::GetInstance();
    graphconf_mgr.Initialize(drivers, config);
    auto graphvizconf = graphconf_mgr.LoadGraphConfig(config);
    auto gcgraph = graphvizconf->Resolve();
    if (gcgraph_out) {
      *gcgraph_out = gcgraph;
    }

    auto graph = std::make_shared<Graph>();
    graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);

    return graph->Build(gcgraph);
  }

 private:
  std::shared_ptr<MockFlow> flow_;
};

/*
      ---->b---->
     /           \
    a             d---->e---->f
     \           /
      ---->c---->
*/
TEST_F(GraphTest, BuildGraph) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0, label="<Out_1> | <Out_2>"]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          d[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>"]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) == STATUS_OK);
}

/*
      ---->b---->    e-->
     /           \       \
    a             d------>f
     \           /
      ---->c---->
*/
TEST_F(GraphTest, BuildGraph_IsolatedPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0, label="<Out_1> | <Out_2>"]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          d[type=flowunit, flowunit=test_2_1, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>"]
          e[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
          d:Out_1 -> f:In_1
          e:Out_1 -> f:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) != STATUS_OK);
}

/*
     ---->b----
    /          \
    a           --->d   e-->f
    \          /
     ---->c----
*/
TEST_F(GraphTest, BuildGraph_IsolatedNode) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0, label="<Out_1> | <Out_2>"]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          e[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_1
          e:Out_1 -> f:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) != STATUS_OK);
}

/*
       -------------
      /             \
     ---->b---->     \
    /           \     \
   a             d---->e---->f
    \           /
     ---->c---->
*/
TEST_F(GraphTest, BuildGraph_Topology) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_2, device=cpu, deviceid=0, label="<Out_1> | <Out_2>"]
          b[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          d[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          e[type=flowunit, flowunit=test_1_2, device=cpu, deviceid=0, label="<In_1> | <Out_1> | <Out_1>"]
          f[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> f:In_1
          e:Out_2 -> b:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) != STATUS_OK);
}

/*
      a
      |
  --> b --> end
 |    |
  <-- c
*/

TEST_F(GraphTest, BuildGraph_SingleLoop) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1>|<Out_1>|<Out_2>"]
          c[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> end:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  auto status = BuildGraph(config);
  MBLOG_ERROR << status.WrapErrormsgs();
  EXPECT_TRUE(status == STATUS_OK);
}

/*
      a            --> end
      |           |
  --> b --> d --> e -->
 |    |           |    |
  <-- c             <--
*/

TEST_F(GraphTest, DISABLED_BuildGraph_DoubleLoop) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1>|<Out_1>|<Out_2>"]
          c[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          d[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1>| <Out_1>"]
          e[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1> |<Out_1>|<Out_2>"]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> e:In_1
          e:Out_2 -> end:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) == STATUS_OK);
}

/*     ------> end
      |     f --------->
      |     |          |
a --> b --> c --> d    |
      |     |    |     |
      |      <-- e     |
      <----------------
*/

TEST_F(GraphTest, DISABLED_BuildGraph_LoopInLoop) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1>|<Out_1>|<Out_2>"]
          c[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1> | <Out_1>|<Out_2>"]
          d[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1>| <Out_1>"]
          e[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1>|<Out_1>"]
          f[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1>|<Out_1>"]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          e:Out_1 -> c:In_1
          c:Out_2 -> f:In_1
          f:Out_1 -> b:In_1
          b:Out_2 -> end:In_1

        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) == STATUS_OK);
}

TEST_F(GraphTest, BuildGraph_LoopInputOutputInvalid) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          a2[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          b[type=flowunit, flowunit=test_loop_invalid, device=cpu, deviceid=0, label="<In_1>|<In_2>|<Out_1>|<Out_2>"]
          c[type=flowunit, flowunit=test_1_1_normal, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a1:Out_1 -> b:In_1
          a2:Out_1 -> b:In_2
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> end:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_TRUE(BuildGraph(config) == STATUS_FAULT);
}

TEST_F(GraphTest, BuildGraph_StreamInLoop) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]
          b[type=flowunit, flowunit=test_loop, device=cpu, deviceid=0, label="<In_1>|<Out_1>|<Out_2>"]
          c[type=flowunit, flowunit=test_1_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          end[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]
          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> b:In_1
          b:Out_2 -> end:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  auto status = BuildGraph(config);
  MBLOG_ERROR << status.WrapErrormsgs();
  EXPECT_TRUE(status == STATUS_FAULT);
}

TEST_F(GraphTest, OrphanCheck) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          orphan[flowunit=test_1_1]
          a[flowunit=test_0_1]
          b[flowunit=test_1_0]
          a:Out_1 -> b:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, SkipOrphan) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          orphan[flowunit=test_1_1]
          a[flowunit=test_0_1]
          b[flowunit=test_1_0]
          a:Out_1 -> b:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_OK);
}

TEST_F(GraphTest, DefaultGraphConfig) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          queue_size=8
          batch_size=8
          a[flowunit=test_0_1]
          b[flowunit=test_1_1, batch_size=16]
          c[flowunit=test_1_0, queue_size=16]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  std::shared_ptr<modelbox::GCGraph> gcgraph;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config, &gcgraph), STATUS_OK);

  auto config_a = gcgraph->GetNode("a")->GetConfiguration();
  auto config_b = gcgraph->GetNode("b")->GetConfiguration();
  auto config_c = gcgraph->GetNode("c")->GetConfiguration();

  EXPECT_EQ(config_a->GetUint32("queue_size", 0), 8);
  EXPECT_EQ(config_a->GetUint32("batch_size", 0), 8);
  EXPECT_EQ(config_b->GetUint32("queue_size", 0), 8);
  EXPECT_EQ(config_b->GetUint32("batch_size", 0), 16);
  EXPECT_EQ(config_c->GetUint32("queue_size", 0), 16);
  EXPECT_EQ(config_c->GetUint32("batch_size", 0), 8);
}

TEST_F(GraphTest, InputStreamUnmatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_2]
          b[flowunit=test_1_1]
          c[flowunit=stream_1_1]
          d[flowunit=test_2_0]

          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2

        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_OK);
}

TEST_F(GraphTest, InputStreamCollapseRoot) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=collapse_1_1]
          c[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, InputStreamCollapseUnmatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_2]
          b[flowunit=expand_1_1]
          c[flowunit=test_1_1]
          d[flowunit=test_2_0]

          a:Out_1 -> b:In_1
          a:Out_2 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> d:In_2
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, InputStreamConditionUnmatch) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=test_1_0]
          d[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> d:In_1
          a:Out_1 -> c:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, InputStreamConditionOne) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_1 -> c:In_1
          b:Out_2 -> c:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_OK);
}

TEST_F(GraphTest, InputStreamConditionOne1) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=test_1_1]
          d[flowunit=test_1_1]
          e[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          b:Out_1 -> d:In_1
          c:Out_1 -> e:In_1
          d:Out_1 -> e:In_1
          b:Out_2 -> e:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, InputConditionConnectWrongPort) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=condition_1_2]
          d[flowunit=test_1_0]
          e[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          a:Out_1 -> c:In_1
          b:Out_1 -> d:In_1
          b:Out_2 -> e:In_1
          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_BADCONF);
}

TEST_F(GraphTest, InputConditionNotHasSameCount) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=stream_1_1]
          d[flowunit=test_1_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          b:Out_2 -> d:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_OK);
}

TEST_F(GraphTest, SucessConditionGraph) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=condition_1_2]
          c[flowunit=condition_1_2]
          d[flowunit=test_1_1]
          e[flowunit=test_1_1]
          f[flowunit=test_1_1]
          g[flowunit=test_1_0]

          a:Out_1 -> b:In_1

          b:Out_1 -> c:In_1
          b:Out_2 -> g:In_1

          c:Out_1 -> d:In_1
          c:Out_2 -> e:In_1

          d:Out_1  -> f:In_1
          e:Out_1  -> f:In_1

          f:Out_1 -> g:In_1
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_SUCCESS);
}

TEST_F(GraphTest, SucessExpandCollapseGraph) {
  const auto *conf_file_value =
      R"(
        digraph demo {
          a[flowunit=test_0_1]
          b[flowunit=expand_1_1]
          c[flowunit=test_1_1]
          d[flowunit=collapse_1_1]
          e[flowunit=test_2_0]

          a:Out_1 -> b:In_1
          b:Out_1 -> c:In_1
          c:Out_1 -> d:In_1
          d:Out_1 -> e:In_1
          a:Out_1 -> e:In_2
        }
      )";

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.strict", false);
  config->SetProperty("graph.graphconf", conf_file_value);
  EXPECT_EQ(BuildGraph(config), STATUS_SUCCESS);
}

TEST_F(GraphTest, BuildGraphFromArray) {
  const char* conf_file_value[] = {
      "digraph demo {", "    a[flowunit=test_0_1, a=x, c=x]",
      "    b[flowunit=test_1_0]", "    a:Out_1 -> b:In_1", "}"};

  std::vector<std::string> graph_config;
  for (size_t i = 0; i < sizeof(conf_file_value) / sizeof(char*); i++) {
    graph_config.push_back(conf_file_value[i]);
  }

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  config->SetProperty("graph.format", "graphviz");
  config->SetProperty("graph.graphconf", graph_config);
  EXPECT_TRUE(BuildGraph(config) == STATUS_OK);
}

}  // namespace modelbox
