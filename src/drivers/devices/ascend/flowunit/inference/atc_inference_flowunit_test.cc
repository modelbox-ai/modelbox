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


#include <dsmi_common_interface.h>

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {
class InferenceAscendFlowUnitTest : public testing::Test {
 public:
  InferenceAscendFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  virtual void SetUp() {
    // Test ascend runtime
    int32_t count = 0;
    auto dsmi_ret = dsmi_get_device_count(&count);
    if (dsmi_ret != 0) {
      MBLOG_INFO << "no ascend device, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);

    const std::string src_file =
        test_assets + "/atc_inference/" + test_model_file;
    const std::string src_toml = test_data_dir + "/" + test_toml_file;
    atc_inference_path = test_data_dir + "/atc_inference";
    mkdir(atc_inference_path.c_str(), 0700);
    dest_model_file = atc_inference_path + "/" + test_model_file;
    dest_toml_file = atc_inference_path + "/" + test_toml_file;
    CopyFile(src_file, dest_model_file, true);
    CopyFile(src_toml, dest_toml_file, true);
    const std::string src_file_en =
        test_assets + "/atc_inference/" + test_model_file_en;
    const std::string src_toml_en = test_data_dir + "/" + test_toml_file_en;
    dest_model_file_en = atc_inference_path + "/" + test_model_file_en;
    dest_toml_file_en = atc_inference_path + "/" + test_toml_file_en;
    CopyFile(src_file_en, dest_model_file_en, true);
    CopyFile(src_toml_en, dest_toml_file_en, true);
  }

  virtual void TearDown() {
    remove(dest_model_file.c_str());
    remove(dest_toml_file.c_str());
    remove(dest_model_file_en.c_str());
    remove(dest_toml_file_en.c_str());
    remove(atc_inference_path.c_str());

    driver_flow_ = nullptr;
  };

  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_model_file = "2d_2048_w_stage1_pad0.om",
                    test_toml_file = "virtual_atc_infer_test.toml",
                    test_model_file_en = "2d_2048_w_stage1_pad0_en.om",
                    test_toml_file_en = "virtual_atc_infer_test_en.toml";

  std::string atc_inference_path, dest_model_file, dest_toml_file,
      dest_model_file_en, dest_toml_file_en;

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

Status InferenceAscendFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc = GenerateFlowunitDesc("prepare_infer_data", {}, {"out"});
    mock_desc->SetFlowType(STREAM);
    auto open_func =
        [=](const std::shared_ptr<modelbox::Configuration>& flow_option,
            std::shared_ptr<MockFlowUnit> mock_flowunit) {
          auto ext_data = mock_flowunit->CreateExternalData();
          if (!ext_data) {
            MBLOG_ERROR << "can not get external data.";
          }

          auto buffer_list = ext_data->CreateBufferList();
          buffer_list->Build({10});

          auto status = ext_data->Send(buffer_list);
          if (!status) {
            MBLOG_ERROR << "external data send buffer list failed:" << status;
          }

          status = ext_data->Close();
          if (!status) {
            MBLOG_ERROR << "external data close failed:" << status;
          }

          return modelbox::STATUS_OK;
        };

    auto process_func = [=](std::shared_ptr<DataContext> op_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
      MBLOG_INFO << "prepare_infer_data Process";
      auto output_buf_1 = op_ctx->Output("out");
      const size_t len = 2048;
      std::vector<size_t> shape_vector(1, len * sizeof(float));
      modelbox::ModelBoxDataType type = MODELBOX_FLOAT;
      output_buf_1->Build(shape_vector);
      output_buf_1->Set("type", type);
      std::vector<size_t> shape{len};
      output_buf_1->Set("shape", shape);
      auto dev_data = (float*)(output_buf_1->MutableData());
      for (size_t i = 0; i < output_buf_1->Size(); ++i) {
        for (size_t j = 0; j < len; ++j) {
          dev_data[i * len + j] = 0.0;
        }
      }

      return modelbox::STATUS_OK;
    };

    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterOpenFunc(open_func);
    mock_functions->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_functions->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }
  {
    auto mock_desc = GenerateFlowunitDesc("check_infer_result", {"in"}, {});
    mock_desc->SetFlowType(STREAM);
    auto process_func = [=](std::shared_ptr<DataContext> op_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
      std::shared_ptr<BufferList> input_bufs = op_ctx->Input("in");
      EXPECT_EQ(input_bufs->Size(), 1);
      std::vector<size_t> input_shape;
      auto result = input_bufs->At(0)->Get("shape", input_shape);
      EXPECT_TRUE(result);
      EXPECT_EQ(input_shape.size(), 4);
      EXPECT_EQ(input_shape[0], 1);
      EXPECT_EQ(input_shape[1], 256);
      EXPECT_EQ(input_shape[2], 1);
      EXPECT_EQ(input_shape[3], 2048);

      auto ptr = (const float*)input_bufs->ConstData();
      for (size_t i = 0; i < 200; ++i) {
        EXPECT_TRUE(std::abs(ptr[i]) < 1e-7);
      }

      return modelbox::STATUS_OK;
    };

    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_functions->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  return STATUS_OK;
}

std::shared_ptr<MockFlow> InferenceAscendFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(InferenceAscendFlowUnitTest, RunUnit) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_infer_data[type=flowunit, flowunit=prepare_infer_data, device=cpu, deviceid=0, label="<out>"]             
          atc_inference[type=flowunit, flowunit=atc_inference, device=ascend, deviceid=0, label="<input> | <output:0>", batch_size=1]
          check_infer_result[type=flowunit, flowunit=check_infer_result, device=cpu, deviceid=0, label="<in>", batch_size=1]  
                                  
          prepare_infer_data:out -> atc_inference:input
          atc_inference:output:0 -> check_infer_result:in
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("RunUnit", toml_content, 3 * 1000);
}

TEST_F(InferenceAscendFlowUnitTest, RunUnitEncrypt) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_infer_data[type=flowunit, flowunit=prepare_infer_data, device=cpu, deviceid=0, label="<out>"]             
          atc_inference[type=flowunit, flowunit=acl_inference_encrypt, device=ascend, deviceid=0, label="<input> | <output:0>", batch_size=1]
          check_infer_result[type=flowunit, flowunit=check_infer_result, device=cpu, deviceid=0, label="<in>", batch_size=1]  
                                  
          prepare_infer_data:out -> atc_inference:input
          atc_inference:output:0 -> check_infer_result:in
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("RunUnit", toml_content, 3 * 1000);
}

}  // namespace modelbox