
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

#include <atomic>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <thread>

#include "engine/scheduler/flow_scheduler.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockflow.h"
#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "modelbox/graph.h"
#include "modelbox/node.h"

namespace modelbox {
using ::testing::Sequence;
class FlowTest : public testing::Test {
 public:
  FlowTest() = default;

 protected:
  std::shared_ptr<MockFlow> flow_;

  void SetUp() override {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init();
  };

  void TearDown() override { flow_->Destroy(); };
};

class MockNode : public Node {
 public:
  MOCK_METHOD1(Run, Status(RunType type));
};

static SessionManager g_test_session_manager;

TEST_F(FlowTest, All) {
  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<Node> node_a = nullptr;
  std::shared_ptr<Node> node_b = nullptr;
  std::shared_ptr<Node> node_c = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_a = std::make_shared<Node>();
    node_a->SetFlowUnitInfo("listen", "cpu", "0", flowunit_mgr);
    node_a->SetName("gendata");
    node_a->Init({}, {"Out_1", "Out_2"}, config);
    node_a->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_a));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_b = std::make_shared<Node>();
    node_b->SetFlowUnitInfo("add", "cpu", "0", flowunit_mgr);
    node_b->SetName("addop");
    node_b->Init({"In_1", "In_2"}, {"Out_1"}, config);
    node_b->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_b));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("max_count", 50);

    node_c = std::make_shared<Node>();
    node_c->SetFlowUnitInfo("check_print", "cpu", "0", flowunit_mgr);
    node_c->SetName("check_print");
    node_c->Init({"IN1", "IN2", "IN3"}, {}, config);
    node_c->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_c));
  }

  graph->AddLink(node_a->GetName(), "Out_1", node_b->GetName(), "In_1");
  graph->AddLink(node_a->GetName(), "Out_2", node_b->GetName(), "In_2");
  graph->AddLink(node_a->GetName(), "Out_1", node_c->GetName(), "IN1");
  graph->AddLink(node_a->GetName(), "Out_2", node_c->GetName(), "IN2");
  graph->AddLink(node_b->GetName(), "Out_1", node_c->GetName(), "IN3");

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  graph->RunAsync();

  Status retval;
  graph->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);
}

TEST_F(FlowTest, PortEnlargeQueue) {
  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<Node> start_node = nullptr;
  std::shared_ptr<Node> condition_node = nullptr;
  std::shared_ptr<Node> simple_pass_node = nullptr;
  std::shared_ptr<Node> receive_node = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    start_node = std::make_shared<Node>();
    start_node->SetFlowUnitInfo("test_orgin_0_2", "cpu", "0", flowunit_mgr);
    start_node->SetName("test_orgin_0_2");
    auto status = start_node->Init({}, {"Out_1", "Out_2"}, config);
    EXPECT_EQ(status, STATUS_OK);
    start_node->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(start_node));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    condition_node = std::make_shared<Node>();
    condition_node->SetFlowUnitInfo("half-condition", "cpu", "0", flowunit_mgr);
    condition_node->SetName("half-condition");
    auto status = condition_node->Init({"In_1"}, {"Out_1", "Out_2"}, config);
    EXPECT_EQ(status, STATUS_OK);
    condition_node->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(condition_node));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    simple_pass_node = std::make_shared<Node>();
    simple_pass_node->SetFlowUnitInfo("simple_pass", "cpu", "0", flowunit_mgr);
    simple_pass_node->SetName("simple_pass");
    auto status = simple_pass_node->Init({"In_1"}, {"Out_1"}, config);
    EXPECT_EQ(status, STATUS_OK);
    simple_pass_node->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(simple_pass_node));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("queue_size", "5");

    receive_node = std::make_shared<Node>();
    receive_node->SetFlowUnitInfo("test_2_0", "cpu", "0", flowunit_mgr);
    receive_node->SetName("receive");
    auto status = receive_node->Init({"In_1", "In_2"}, {}, config);
    EXPECT_EQ(status, STATUS_OK);
    receive_node->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(receive_node));
  }

  graph->AddLink(start_node->GetName(), "Out_1", condition_node->GetName(),
                 "In_1");
  graph->AddLink(condition_node->GetName(), "Out_1", receive_node->GetName(),
                 "In_1");
  graph->AddLink(condition_node->GetName(), "Out_2",
                 simple_pass_node->GetName(), "In_1");
  graph->AddLink(simple_pass_node->GetName(), "Out_1", receive_node->GetName(),
                 "In_1");
  graph->AddLink(start_node->GetName(), "Out_2", receive_node->GetName(),
                 "In_2");

  Sequence s1;
  auto pass_fu = std::dynamic_pointer_cast<MockFlowUnit>(
      simple_pass_node->GetFlowUnitGroup()->GetExecutorUnit());
  EXPECT_CALL(*pass_fu, Process(testing::_)).Times(5).InSequence(s1);

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  graph->RunAsync();

  Status retval;
  graph->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);
}

TEST_F(FlowTest, TensorList_All) {
  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<Node> node_a = nullptr;
  std::shared_ptr<Node> node_b = nullptr;
  std::shared_ptr<Node> node_c = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_a = std::make_shared<Node>();
    node_a->SetFlowUnitInfo("listen", "cpu", "0", flowunit_mgr);
    node_a->SetName("gendata");
    node_a->Init({}, {"Out_1", "Out_2"}, config);
    node_a->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_a));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_b = std::make_shared<Node>();
    node_b->SetFlowUnitInfo("tensorlist_test_1", "cpu", "0", flowunit_mgr);
    node_b->SetName("tensorlist_test_1");
    node_b->Init({"IN1"}, {"OUT1"}, config);
    node_b->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_b));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_c = std::make_shared<Node>();
    node_c->SetFlowUnitInfo("check_tensorlist_test_1", "cpu", "0",
                            flowunit_mgr);
    node_c->SetName("check_tensorlist_test_1");
    node_c->Init({"IN1", "IN2"}, {}, config);
    node_c->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_c));
  }

  graph->AddLink(node_a->GetName(), "Out_1", node_b->GetName(), "IN1");
  graph->AddLink(node_a->GetName(), "Out_2", node_c->GetName(), "IN1");
  graph->AddLink(node_b->GetName(), "OUT1", node_c->GetName(), "IN2");

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  graph->RunAsync();

  Status retval;
  graph->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);
}

TEST_F(FlowTest, FAILED_ALL) {
  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<MockNode> node_a = nullptr;
  std::shared_ptr<Node> node_b = nullptr;
  std::shared_ptr<Node> node_c = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_a = std::make_shared<MockNode>();
    node_a->SetFlowUnitInfo("listen", "cpu", "0", flowunit_mgr);
    EXPECT_CALL(*node_a, Run(testing::_))
        .WillRepeatedly(testing::Return(STATUS_FAULT));
    node_a->SetName("gendata");
    node_a->Init({}, {"Out_1", "Out_2"}, config);
    node_a->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_a));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();

    node_b = std::make_shared<Node>();
    node_b->SetFlowUnitInfo("add", "cpu", "0", flowunit_mgr);
    node_b->SetName("addop");
    node_b->Init({"In_1", "In_2"}, {"Out_1"}, config);
    node_b->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_b));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("max_count", 50);

    node_c = std::make_shared<Node>();
    node_c->SetFlowUnitInfo("check_print", "cpu", "0", flowunit_mgr);
    node_c->SetName("check_print");
    node_c->Init({"IN1", "IN2", "IN3"}, {}, config);
    node_c->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_c));
  }

  graph->AddLink(node_a->GetName(), "Out_1", node_b->GetName(), "In_1");
  graph->AddLink(node_a->GetName(), "Out_2", node_b->GetName(), "In_2");
  graph->AddLink(node_a->GetName(), "Out_1", node_c->GetName(), "IN1");
  graph->AddLink(node_a->GetName(), "Out_2", node_c->GetName(), "IN2");
  graph->AddLink(node_b->GetName(), "Out_1", node_c->GetName(), "IN3");

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  graph->RunAsync();

  Status retval;
  graph->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_FAULT);
}

TEST_F(FlowTest, ConfigJson) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = ["digraph demo {",
          "listen[type=flowunit, flowunit=listen, device=cpu, deviceid=0, label=\"<Out_1> | <Out_2>\"]",
          "tensorlist_test_2[type=flowunit, flowunit=tensorlist_test_2, device=cpu, deviceid=0, label=\"<IN1> | <IN2> | <OUT1>\", batch_size=4]",
          "check_tensorlist_test_2[type=flowunit, flowunit=check_tensorlist_test_2, device=cpu, deviceid=0, label=\"<IN1> | <IN2> | <IN3>\", batch_size=6]",
          "listen:Out_1 -> tensorlist_test_2:IN1",
          "listen:Out_2 -> tensorlist_test_2:IN2",
          "listen:Out_1 -> check_tensorlist_test_2:IN1",
          "listen:Out_2 -> check_tensorlist_test_2:IN2",
          "tensorlist_test_2:OUT1 -> check_tensorlist_test_2:IN3",
        "}"
        ]
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  std::string config_file_path = std::string(TEST_WORKING_DIR) + "/test.json";
  std::string json_data;
  auto ret = TomlToJson(toml_content, &json_data, true);
  ASSERT_TRUE(ret);
  MBLOG_INFO << json_data;
  std::ofstream ofs(config_file_path);
  EXPECT_TRUE(ofs.is_open());
  ofs.write(json_data.data(), json_data.size());
  ofs.flush();
  ofs.close();
  Defer {
    auto rmret = remove(config_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  };

  auto flow = std::make_shared<Flow>();
  ret = flow->Init(config_file_path);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();

  Status retval;
  flow->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);

  flow->Stop();
}

TEST_F(FlowTest, Extern_Config) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          listen[type=flowunit, flowunit=listen, device=cpu, deviceid=0, label="<Out_1> | <Out_2>"]             
          add[type=flowunit, flowunit=add, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>"] 
          check_print[type=flowunit, flowunit=check_print, device=cpu, deviceid=0, label="<IN1> | <IN2> | <IN3>" , max_count=50]                                
          listen:Out_1 -> add:In_1
          listen:Out_2 -> add:In_2
          listen:Out_1 -> check_print:IN1
          listen:Out_2 -> check_print:IN2
          add:Out_1 -> check_print:IN3                                                                             
        }'''
    format = "graphviz"
  )";

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init("graph", toml_content);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();

  Status retval;
  flow->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);

  flow->Stop();
}

TEST_F(FlowTest, DISABLED_Perf) {
  auto graph = std::make_shared<Graph>();
  auto gc = std::make_shared<GCGraph>();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();

  std::shared_ptr<Node> node_a = nullptr;
  std::shared_ptr<Node> node_b = nullptr;
  std::shared_ptr<Node> node_c = nullptr;

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("interval_time", 0);
    config->SetProperty("queue_size", 1024);

    node_a = std::make_shared<Node>();
    node_a->SetFlowUnitInfo("listen", "cpu", "0", flowunit_mgr);
    node_a->SetName("gendata");
    node_a->Init({}, {"Out_1", "Out_2"}, config);
    node_a->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_a));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("queue_size", 1024);

    node_b = std::make_shared<Node>();
    node_b->SetFlowUnitInfo("add", "cpu", "0", flowunit_mgr);
    node_b->SetName("add");
    node_b->Init({"In_1", "In_2"}, {"Out_1"}, config);
    node_b->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_b));
  }

  {
    ConfigurationBuilder configbuilder;
    auto config = configbuilder.Build();
    config->SetProperty("max_count", INT64_MAX);
    config->SetProperty("queue_size", 1024);

    node_c = std::make_shared<Node>();
    node_c->SetFlowUnitInfo("check_print", "cpu", "0", flowunit_mgr);
    node_c->SetName("check_print");
    node_c->Init({"IN1", "IN2", "IN3"}, {}, config);
    node_c->SetSessionManager(&g_test_session_manager);
    EXPECT_TRUE(graph->AddNode(node_c));
  }

  graph->AddLink(node_a->GetName(), "Out_1", node_b->GetName(), "In_1");
  graph->AddLink(node_a->GetName(), "Out_2", node_b->GetName(), "In_2");
  graph->AddLink(node_a->GetName(), "Out_1", node_c->GetName(), "IN1");
  graph->AddLink(node_a->GetName(), "Out_2", node_c->GetName(), "IN2");
  graph->AddLink(node_b->GetName(), "Out_1", node_c->GetName(), "IN3");

  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();
  graph->Initialize(flowunit_mgr, device_mgr, nullptr, config);
  EXPECT_TRUE(graph->Build(gc) == STATUS_OK);
  graph->RunAsync();

  Status retval;
  graph->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);
}

TEST_F(FlowTest, Statistics) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          statistic_test[type=flowunit, flowunit=statistic_test, device=cpu, deviceid=0, label="<IN1> | <OUT1>"]
          output[type=output]

          input -> statistic_test:IN1
          statistic_test:OUT1 -> output
        }'''
    format = "graphviz"
  )";

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init("graph", toml_content);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();

  auto external = flow->CreateExternalDataMap();
  auto input_buffer = external->CreateBufferList();
  input_buffer->Build({1});
  auto session_ctx = external->GetSessionContext();
  auto session_id = session_ctx->GetSessionId();
  auto graph_id = flow->GetGraphId();

  auto profiler = flow->GetProfiler();
  auto statistics = Statistics::GetGlobalItem();
  Defer {Statistics::ReleaseGlobalItem();};
  std::atomic<std::uint32_t> change_notify_count = {0};
  std::atomic<std::uint32_t> timer_notify_count = {0};
  const std::string path_pattern = "flow.*.*.statistic_test.test_key";
  auto change_notify_cfg = std::make_shared<StatisticsNotifyCfg>(
      path_pattern,
      [&change_notify_count, graph_id, session_id](
          const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
        MBLOG_INFO << "Change notify [" << msg->path_ << "]";
        EXPECT_EQ(msg->path_, "flow." + graph_id + "." + session_id +
                                  ".statistic_test.test_key");
        EXPECT_TRUE(msg->value_->IsInt32());
        EXPECT_FALSE(msg->value_->IsString());
        int32_t test_val = 0;
        auto ret = msg->value_->GetInt32(test_val);
        EXPECT_TRUE(ret);
        EXPECT_EQ(test_val, 1);
        ++change_notify_count;
      },
      std::set<StatisticsNotifyType>{StatisticsNotifyType::CREATE,
                                     StatisticsNotifyType::CHANGE});
  statistics->RegisterNotify(change_notify_cfg);

  auto delete_notify_cfg = std::make_shared<StatisticsNotifyCfg>(
      path_pattern,
      [session_id](
          const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
        MBLOG_INFO << "Delete notify [" << msg->path_ << "]";
        EXPECT_EQ(msg->type_, StatisticsNotifyType::DELETE);
      },
      StatisticsNotifyType::DELETE);
  statistics->RegisterNotify(delete_notify_cfg);

  auto timer_notify_cfg = std::make_shared<StatisticsNotifyCfg>(
      path_pattern,
      [&timer_notify_count, graph_id, session_id](
          const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
        ++timer_notify_count;
      });
  timer_notify_cfg->SetNotifyTimer(100, 100);
  statistics->RegisterNotify(timer_notify_cfg);

  EXPECT_NE(session_ctx, nullptr);
  external->Send("input", input_buffer);
  OutputBufferList output_buffer;
  external->Recv(output_buffer);
  external->Close();

  flow->Stop();

  auto item = statistics->GetItem("flow." + graph_id + "." + session_id +
                                  ".statistic_test.test_key");
  ASSERT_NE(item, nullptr);
  EXPECT_TRUE(item->GetValue()->IsInt32());
  int32_t test_val = 0;
  auto b_ret = item->GetValue()->GetInt32(test_val);
  EXPECT_TRUE(b_ret);
  EXPECT_EQ(test_val, 1);

  // change_val == create_val, only notify once
  EXPECT_EQ(change_notify_count, 1);
  EXPECT_GE(timer_notify_count, 0);  // minimum timer notify interval is 60s
  statistics->UnRegisterNotify(change_notify_cfg);
  statistics->UnRegisterNotify(delete_notify_cfg);
  statistics->UnRegisterNotify(timer_notify_cfg);
}

TEST_F(FlowTest, LoopGraph_All) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          data_input[type=flowunit, flowunit=test_0_1_batch_thread, device=cpu, deviceid=0, label= "<Out_1>", interval_time = 10000]             
          loop[type=flowunit, flowunit=loop, device=cpu, deviceid=0, label="<In_1> | <Out_1> | <Out_2>"] 
          data_output[type=flowunit, flowunit=test_1_0_batch_thread, device=cpu, deviceid=0, label="<In_1>"]                                
          data_input:Out_1 -> loop:In_1
          loop:Out_1 -> loop:In_1
          loop:Out_2 -> data_output:In_1                                                                        
        }'''
    format = "graphviz"
  )";

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init("graph", toml_content);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();

  Status retval;
  flow->Wait(0, &retval);
  EXPECT_EQ(retval, STATUS_STOP);

  flow->Stop();
}

TEST_F(FlowTest, NormalError) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          error_input[type=flowunit, flowunit=error_start_normal, device=cpu, deviceid=0, label= "<Out_1>"]             
          error_output[type=flowunit, flowunit=error_end_normal, device=cpu, deviceid=0, label="<In_1>"]                                
          error_input:Out_1 -> error_output:In_1
        }'''
    format = "graphviz"
  )";

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init("graph", toml_content);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();

  Status retval;
  flow->Wait(1000 * 5, &retval);
  EXPECT_EQ(retval, STATUS_SUCCESS);

  flow->Stop();
}

}  // namespace modelbox
