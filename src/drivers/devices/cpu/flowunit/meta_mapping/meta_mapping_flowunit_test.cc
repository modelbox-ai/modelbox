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


#include "meta_mapping_flowunit.h"

#include <functional>
#include <future>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class MetaMappingFlowUnitTest : public testing::Test {
 public:
  MetaMappingFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();
  std::shared_ptr<MockFlow> RunDriverFlow(const std::string &rules);

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> MetaMappingFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<MockFlow> MetaMappingFlowUnitTest::RunDriverFlow(
    const std::string &rules) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input, device=cpu, deviceid=0]
          meta_mapping[type=flowunit, flowunit=buff_meta_mapping, device=cpu, deviceid=0, label="<output_data>", src_meta="src", dest_meta="dest", rules=")" +
                             rules + R"("]
          output[type=output, deveice=cpu, deviceid=0]
          input -> meta_mapping:in_data
          meta_mapping:out_data -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("InitUnit", toml_content, -1);
  return driver_flow;
}

Status MetaMappingFlowUnitTest::AddMockFlowUnit() { return STATUS_OK; }

TEST_F(MetaMappingFlowUnitTest, NameMapping) {
  auto driver_flow = RunDriverFlow("");

  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({1});
  auto buffer = buffer_list->At(0);
  buffer->Set("src", (int32_t)123);
  ext_data->Send("input", buffer_list);
  modelbox::OutputBufferList output_buffer_map;
  ext_data->Recv(output_buffer_map);
  EXPECT_EQ(output_buffer_map.size(), 1);
  auto output_buffer_list = output_buffer_map["output"];
  EXPECT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  int32_t dest_val;
  EXPECT_TRUE(output_buffer->Get("dest", dest_val));
  EXPECT_EQ(dest_val, 123);
  ext_data->Shutdown();

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(MetaMappingFlowUnitTest, Int32Mapping) {
  auto driver_flow = RunDriverFlow("1=2,3=4,5=6");

  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({1, 1, 1, 1});
  auto buffer = buffer_list->At(0);
  buffer->Set("src", (int32_t)1);
  buffer->Set("expect", (int32_t)2);
  buffer = buffer_list->At(1);
  buffer->Set("src", (int32_t)3);
  buffer->Set("expect", (int32_t)4);
  buffer = buffer_list->At(2);
  buffer->Set("src", (int32_t)5);
  buffer->Set("expect", (int32_t)6);
  buffer = buffer_list->At(3);
  buffer->Set("src", (int32_t)333);
  buffer->Set("expect", (int32_t)333);
  ext_data->Send("input", buffer_list);
  modelbox::OutputBufferList output_buffer_map;
  ext_data->Recv(output_buffer_map);
  EXPECT_EQ(output_buffer_map.size(), 1);
  auto output_buffer_list = output_buffer_map["output"];
  EXPECT_EQ(output_buffer_list->Size(), 4);
  auto output_buffer1 = output_buffer_list->At(0);
  auto output_buffer2 = output_buffer_list->At(1);
  auto output_buffer3 = output_buffer_list->At(2);
  auto output_buffer4 = output_buffer_list->At(3);
  int32_t dest_val;
  int32_t expect_val;
  EXPECT_TRUE(output_buffer1->Get("dest", dest_val));
  EXPECT_TRUE(output_buffer1->Get("expect", expect_val));
  EXPECT_EQ(dest_val, expect_val);
  EXPECT_TRUE(output_buffer2->Get("dest", dest_val));
  EXPECT_TRUE(output_buffer2->Get("expect", expect_val));
  EXPECT_EQ(dest_val, expect_val);
  EXPECT_TRUE(output_buffer3->Get("dest", dest_val));
  EXPECT_TRUE(output_buffer3->Get("expect", expect_val));
  EXPECT_EQ(dest_val, expect_val);
  EXPECT_TRUE(output_buffer4->Get("dest", dest_val));
  EXPECT_TRUE(output_buffer4->Get("expect", expect_val));
  EXPECT_EQ(dest_val, expect_val);
  ext_data->Shutdown();

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(MetaMappingFlowUnitTest, StringMapping) {
  auto driver_flow = RunDriverFlow("face=dis1|dis2");

  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({1});
  auto buffer = buffer_list->At(0);
  buffer->Set("src", std::string("face"));
  ext_data->Send("input", buffer_list);
  modelbox::OutputBufferList output_buffer_map;
  ext_data->Recv(output_buffer_map);
  EXPECT_EQ(output_buffer_map.size(), 1);
  auto output_buffer_list = output_buffer_map["output"];
  EXPECT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  std::string dest_val;
  EXPECT_TRUE(output_buffer->Get("dest", dest_val));
  EXPECT_EQ(dest_val, "dis1|dis2");
  ext_data->Shutdown();

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox