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

#include "data_source_generator.h"

#include <functional>
#include <thread>

#include "common/mock_cert.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/iam_auth.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {

class DataSourceGeneratorFlowUnitTest : public testing::Test {
 public:
  DataSourceGeneratorFlowUnitTest()
      : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override { auto ret = AddMockFlowUnit(); };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();
  std::shared_ptr<MockFlow> RunDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> DataSourceGeneratorFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<MockFlow> DataSourceGeneratorFlowUnitTest::RunDriverFlow() {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          data_source_gengerator[type=flowunit, flowunit=data_source_generator, device=cpu, deviceid=0, source_type="url", url="http://0.0.0.0:8080/video", url_type="file"]
          data_source_parser[type=flowunit, flowunit=data_source_parser, device=cpu, deviceid=0]
          data_source_parser_checker[type=flowunit, flowunit=data_source_parser_checker, device=cpu, deviceid=0]
          data_source_gengerator:out_data -> data_source_parser:in_data
          data_source_parser:out_video_url -> data_source_parser_checker:stream_meta
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret =
      driver_flow->BuildAndRun("data_source_gengerator", toml_content, -1);

  return driver_flow;
}

Status DataSourceGeneratorFlowUnitTest::AddMockFlowUnit() {
  auto mock_desc =
      GenerateFlowunitDesc("data_source_parser_checker", {"stream_meta"}, {});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](const std::shared_ptr<DataContext>& data_ctx,
          const std::shared_ptr<MockFlowUnit>& mock_flowunit) -> Status {
    auto stream_meta = data_ctx->GetInputMeta("stream_meta");
    EXPECT_NE(stream_meta, nullptr);
    if (!stream_meta) {
      return modelbox::STATUS_SUCCESS;
    }

    auto source_url = std::static_pointer_cast<std::string>(
        stream_meta->GetMeta("source_url"));
    EXPECT_NE(source_url, nullptr);
    if (source_url != nullptr) {
      EXPECT_FALSE(source_url->empty());
      EXPECT_EQ(*source_url, "http://0.0.0.0:8080/video");
    }
    return modelbox::STATUS_SUCCESS;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
  return STATUS_OK;
}

TEST_F(DataSourceGeneratorFlowUnitTest, UrlInputTest) {
  auto driver_flow = RunDriverFlow();
  driver_flow->GetFlow()->Wait(3 * 1000);
}
}  // namespace modelbox