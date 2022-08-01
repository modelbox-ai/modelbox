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

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "common/tensorflow_inference/tensorflow_inference_mock.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"

using ::testing::_;
using namespace tensorflow_inference;

namespace modelbox {
class InferenceTensorflowCpuFlowUnitTest : public testing::Test {
 public:
  InferenceTensorflowCpuFlowUnitTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  void SetUp() override {
    auto version = GetTFVersion();

    if (SUPPORT_TF_VERSION.find(version) == SUPPORT_TF_VERSION.end()) {
      version_suitable_ = false;
      MBLOG_INFO << "the version is " << version
                 << ", not in support version, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit(driver_flow_);
    EXPECT_EQ(ret, STATUS_OK);

    SetUpTomlFiles(version);
  }

  void TearDown() override {
    if (!version_suitable_) {
      GTEST_SKIP();
    }

    RemoveFiles();
    driver_flow_->Clear();
  };

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_toml_file = "virtual_tfcpu_test.toml";

  std::string tensorflow_cpu_path, dest_toml_file;

 private:
  void SetUpTomlFiles(const std::string &version);
  void RemoveFiles();

  std::shared_ptr<DriverFlowTest> driver_flow_;
  bool version_suitable_{true};
};

void InferenceTensorflowCpuFlowUnitTest::RemoveFiles() {
  auto ret = remove(dest_toml_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorflow_cpu_path.c_str());
  EXPECT_EQ(ret, 0);
}

void InferenceTensorflowCpuFlowUnitTest::SetUpTomlFiles(
    const std::string &version) {
  const std::string src_file_dir = test_assets + "/tensorflow/" + version;

  const std::string src_file_pb_toml = test_data_dir + "/" + test_toml_file;

  tensorflow_cpu_path = test_data_dir + "/tensorflow_cpu";
  auto mkdir_ret = mkdir(tensorflow_cpu_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  dest_toml_file = tensorflow_cpu_path + "/" + test_toml_file;
  auto status = ReplaceVersion(src_file_pb_toml, dest_toml_file, version);
  EXPECT_EQ(status, STATUS_OK);
}

std::shared_ptr<DriverFlowTest>
InferenceTensorflowCpuFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(InferenceTensorflowCpuFlowUnitTest, RunUnitBatch) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/tensorflow_cpu\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1_batch[type=flowunit, flowunit=test_0_1_batch, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=inference, device=cpu, deviceid=0, label="<input> | <output>", batch_size=10]
          test_1_0_batch[type=flowunit, flowunit=test_1_0_batch, device=cpu, deviceid=0, label="<In_1>", batch_size=10]  
                                  
          test_0_1_batch:Out_1 -> inference:input
          inference:output -> test_1_0_batch:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content, 99999);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox
