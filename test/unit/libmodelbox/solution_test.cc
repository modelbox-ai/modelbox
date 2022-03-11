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

#include <sys/stat.h>

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
#include "modelbox/external_data_simple.h"
#include "modelbox/graph.h"
#include "modelbox/node.h"
#include "modelbox/solution.h"

namespace modelbox {
using ::testing::Sequence;
class SolutionTest : public testing::Test {
 public:
  SolutionTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

std::shared_ptr<MockFlow> RunGraph(const std::string &solution_name,
                                   const std::string &toml_content) {
  std::string config_file_path =
      std::string(TEST_DATA_DIR) + "/solution_test.toml";
  struct stat buffer;
  if (stat(config_file_path.c_str(), &buffer) == 0) {
    remove(config_file_path.c_str());
  }
  std::ofstream solution_test_toml(config_file_path);
  EXPECT_TRUE(solution_test_toml.is_open());
  solution_test_toml.write(toml_content.data(), toml_content.size());
  solution_test_toml.flush();
  solution_test_toml.close();
  Defer {
    auto rmret = remove(config_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  };

  Solution solution(solution_name);
  solution.SetSolutionDir(TEST_DATA_DIR);
  auto flow = std::make_shared<MockFlow>();
  flow->Init();
  flow->BuildAndRun(solution);
  return flow;
}

static void TestRunGraph(const std::string &solution_name,
                         const std::string &toml_content) {
  auto mock_flow = RunGraph(solution_name, toml_content);
  auto flow = mock_flow->GetFlow();
  flow->Wait(1000);
  flow->Stop();
}

static void TestExternalData_Send(const std::string &solution_name,
                                  const std::string &toml_content,
                                  const std::string input_port_name) {
  auto mock_flow = RunGraph(solution_name, toml_content);
  auto flow = mock_flow->GetFlow();
  char data[3] = {1, 2, 3};
  if (!input_port_name.empty()) {
    auto data_map = flow->CreateExternalDataMap();
    auto external_data_simple = std::make_shared<ExternalDataSimple>(data_map);
    std::shared_ptr<Buffer> buffer = nullptr;
    auto status = external_data_simple->PushData(input_port_name, data, 3, {});
    EXPECT_EQ(status, STATUS_SUCCESS);
  }
  flow->Wait(1000);
}

static void TestExternalData_Recv(const std::string &solution_name,
                                  const std::string &toml_content,
                                  std::vector<std::string> out_ports) {
  auto mock_flow = RunGraph(solution_name, toml_content);
  auto flow = mock_flow->GetFlow();
  auto data_map = flow->CreateExternalDataMap();
  auto external_data_simple = std::make_shared<ExternalDataSimple>(data_map);
  char data[3] = {1, 2, 3};
  auto status = external_data_simple->PushData("input1", data, 3, {});
  EXPECT_EQ(status, STATUS_SUCCESS);

  if (!out_ports.empty()) {
    for (auto out_port : out_ports) {
      std::shared_ptr<Buffer> buffer = nullptr;
      external_data_simple->GetResult(out_port, buffer);

      EXPECT_GT(buffer->GetBytes(), 0);
    }
  }

  flow->Wait(1000);
}

TEST_F(SolutionTest, Solution_Function) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [flow]
    name = "reencoder"
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " + R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start,
          device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu,
          deviceid=0, label="<In_1> | <Out_1>"] stream_end[type=flowunit,
          flowunit=virtual_stream_end, device=cpu, deviceid=0,
          label="<In_1>"]

          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->stream_end:In_1

        }'''
    format = "graphviz"
  )";

  TestRunGraph("reencoder", toml_content);
}

TEST_F(SolutionTest, RecvData) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [flow]
    name = "reencoder"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph reencoder {
        input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          add_1[type=flowunit, flowunit=add_1, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          
          input1 ->add_1:In_1
          add_1:Out_1->output1
        }"""
    format = "graphviz"
  )";

  TestExternalData_Recv("reencoder", toml_content, {"output1"});
}

TEST_F(SolutionTest, SendData) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [flow]
    name = "reencoder"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph reencoder {
     input[type=input, device=cpu,deviceid=0] 
     stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
     stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
     stream_end[type=flowunit, flowunit=virtual_stream_end, device=cpu, deviceid=0, label="<In_1>"]
          
     input ->stream_start:In_1
     stream_start:Out_1 ->stream_mid:In_1
     stream_mid:Out_1->stream_end:In_1
     }"""
    format = "graphviz"
  )";

  TestExternalData_Send("reencoder", toml_content, "input");
}

}  // namespace modelbox