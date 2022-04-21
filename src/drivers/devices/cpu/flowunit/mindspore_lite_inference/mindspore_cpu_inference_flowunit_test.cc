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

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "mindspore_inference_flowunit_test.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {
class InferenceMindSporeCPUFlowUnitTest : public testing::Test {
 public:
  InferenceMindSporeCPUFlowUnitTest()
      : mindspore_flow_(std::make_shared<InferenceMindSporeFlowUnitTest>()) {}

 protected:
  virtual void SetUp() {
    auto ret = mindspore_flow_->Init();
    EXPECT_EQ(ret, STATUS_OK);

    const std::string src_file =
        test_assets + "/mindspore_inference/" + test_model_file;
    const std::string src_toml = test_data_dir + "/" + test_toml_file;
    mindspore_inference_path = test_data_dir + "/mindspore_inference";
    mkdir(mindspore_inference_path.c_str(), 0700);
    dest_model_file = mindspore_inference_path + "/" + test_model_file;
    dest_toml_file = mindspore_inference_path + "/" + test_toml_file;
    CopyFile(src_file, dest_model_file, true);
    CopyFile(src_toml, dest_toml_file, true);
    const std::string src_file_en =
        test_assets + "/mindspore_inference/" + test_model_file_en;
    const std::string src_toml_en = test_data_dir + "/" + test_toml_file_en;
    dest_model_file_en = mindspore_inference_path + "/" + test_model_file_en;
    dest_toml_file_en = mindspore_inference_path + "/" + test_toml_file_en;
    CopyFile(src_file_en, dest_model_file_en, true);
    CopyFile(src_toml_en, dest_toml_file_en, true);
  }

  virtual void TearDown() {
    remove(dest_model_file.c_str());
    remove(dest_toml_file.c_str());
    remove(dest_model_file_en.c_str());
    remove(dest_toml_file_en.c_str());
    remove(mindspore_inference_path.c_str());

    mindspore_flow_ = nullptr;
  };

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_model_file = "tensor_add.mindir",
                    test_toml_file = "virtual_mindspore_infer_test.toml",
                    test_model_file_en = "tensor_add_en.mindir",
                    test_toml_file_en = "virtual_mindspore_infer_test_en.toml";

  std::string mindspore_inference_path, dest_model_file, dest_toml_file,
      dest_model_file_en, dest_toml_file_en;

  std::shared_ptr<InferenceMindSporeFlowUnitTest> mindspore_flow_;
};

TEST_F(InferenceMindSporeCPUFlowUnitTest, RunUnit) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_ms_infer_data[type=flowunit, flowunit=prepare_ms_infer_data, device=cpu, deviceid=0]             
          mindspore_inference[type=flowunit, flowunit=mindspore_inference, device=cpu, deviceid=0, batch_size=2]
          check_ms_infer_result[type=flowunit, flowunit=check_ms_infer_result, device=cpu, deviceid=0, batch_size=2]  
                                  
          prepare_ms_infer_data:out1 -> mindspore_inference:x_
          prepare_ms_infer_data:out2 -> mindspore_inference:y_
          mindspore_inference:"Default/Add-op3"-> check_ms_infer_result:in
        }'''
    format = "graphviz"
  )";
  auto ret = mindspore_flow_->Run("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(InferenceMindSporeCPUFlowUnitTest, RunUnitEncrypt) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_ms_infer_data[type=flowunit, flowunit=prepare_ms_infer_data, device=cpu, deviceid=0]             
          mindspore_inference[type=flowunit, flowunit=mindspore_inference_encrypt, device=cpu, deviceid=0, batch_size=2]
          check_ms_infer_result[type=flowunit, flowunit=check_ms_infer_result, device=cpu, deviceid=0, batch_size=2]  
                                  
          prepare_ms_infer_data:out1 -> mindspore_inference:x_
          prepare_ms_infer_data:out2 -> mindspore_inference:y_
          mindspore_inference:"Default/Add-op3" -> check_ms_infer_result:in
        }'''
    format = "graphviz"
  )";
  auto ret = mindspore_flow_->Run("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox
