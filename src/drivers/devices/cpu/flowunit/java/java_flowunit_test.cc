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


#include <pybind11/embed.h>

#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {
class JavaFlowUnitTest : public testing::Test {
 public:
  JavaFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 protected:
  void SetUp() override {}

  void TearDown() override { driver_flow_->Clear(); };

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> JavaFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(JavaFlowUnitTest, DISABLED_Init) {
  auto op_dir = test_data_dir + "/java_op";
  std::string toml_content = R"(
    [driver]
    dir=[")" + test_lib_dir + "\",\"" +
                             op_dir + "\"]\n    " +
                             R"(
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                                                                                                                    
}}'''
format = "graphviz"

  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("JavaFlowUnit", toml_content, 0);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox