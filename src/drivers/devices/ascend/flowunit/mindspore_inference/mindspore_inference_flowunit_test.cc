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
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {
class InferenceMindSporeFlowUnitTest : public testing::Test {
 public:
  InferenceMindSporeFlowUnitTest()
      : driver_flow_(std::make_shared<MockFlow>()) {}

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

    driver_flow_ = nullptr;
  };

  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_model_file = "tensor_add.mindir",
                    test_toml_file = "virtual_mindspore_infer_test.toml",
                    test_model_file_en = "tensor_add_en.mindir",
                    test_toml_file_en = "virtual_mindspore_infer_test_en.toml";

  std::string mindspore_inference_path, dest_model_file, dest_toml_file,
      dest_model_file_en, dest_toml_file_en;

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

Status InferenceMindSporeFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("prepare_ms_infer_data", {}, {"out1", "out2"});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(2);
    auto open_func = [=](const std::shared_ptr<Configuration>& opts,
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
      MBLOG_INFO << "prepare_ms_infer_data "
                 << "Process";
      auto output_buf_1 = op_ctx->Output("out1");
      auto output_buf_2 = op_ctx->Output("out2");
      const size_t len = 2;
      std::vector<size_t> shape_vector(2, len * sizeof(float));
      modelbox::ModelBoxDataType type = MODELBOX_FLOAT;

      output_buf_1->Build(shape_vector);
      output_buf_1->Set("type", type);
      std::vector<size_t> shape{len, 2};
      output_buf_1->Set("shape", shape);
      auto dev_data1 = (float*)(output_buf_1->MutableData());
      MBLOG_INFO << "output_buf_1.size: " << output_buf_1->Size();
      float val = 1.0;
      for (size_t i = 0; i < output_buf_1->Size(); ++i) {
        for (size_t j = 0; j < len; ++j) {
          dev_data1[i * len + j] = val;
          val += 1.0;
        }
      }

      output_buf_2->Build(shape_vector);
      output_buf_2->Set("type", type);
      output_buf_2->Set("shape", shape);
      auto dev_data2 = (float*)(output_buf_2->MutableData());
      val = 2.0;
      for (size_t i = 0; i < output_buf_2->Size(); ++i) {
        for (size_t j = 0; j < len; ++j) {
          dev_data2[i * len + j] = val;
          val += 1.0;
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
    auto mock_desc = GenerateFlowunitDesc("check_ms_infer_result", {"in"}, {});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(2);
    auto data_post_func = [=](std::shared_ptr<DataContext> op_ctx,
                              std::shared_ptr<MockFlowUnit> mock_flowunit) {
      MBLOG_INFO << "check_ms_infer_result "
                 << "DataPost";
      return modelbox::STATUS_STOP;
    };

    auto process_func = [=](std::shared_ptr<DataContext> op_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
      std::shared_ptr<BufferList> input_bufs = op_ctx->Input("in");
      EXPECT_EQ(input_bufs->Size(), 2);
      std::vector<int64_t> input_shape;
      auto result = input_bufs->At(0)->Get("shape", input_shape);
      EXPECT_TRUE(result);
      EXPECT_EQ(input_shape.size(), 2);
      EXPECT_EQ(input_shape[0], 2);
      EXPECT_EQ(input_shape[1], 2);

      auto ptr = (const float*)input_bufs->ConstData();
      float val = 3.0;
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE((std::abs(ptr[i]) - val) < 1e-7);
        val += 2.0;
      }

      return modelbox::STATUS_OK;
    };

    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterDataPostFunc(data_post_func);
    mock_functions->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_functions->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  return STATUS_OK;
}

std::shared_ptr<MockFlow> InferenceMindSporeFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

// wait for mindspore update change AclEnvGuard shared_ptr into weak_ptr
TEST_F(InferenceMindSporeFlowUnitTest, DISABLED_RunUnit) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_ms_infer_data[type=flowunit, flowunit=prepare_ms_infer_data, device=cpu, deviceid=0]             
          mindspore_inference[type=flowunit, flowunit=mindspore_inference, device=ascend, deviceid=0, batch_size=2]
          check_ms_infer_result[type=flowunit, flowunit=check_ms_infer_result, device=cpu, deviceid=0, batch_size=2]  
                                  
          prepare_ms_infer_data:out1 -> mindspore_inference:x_
          prepare_ms_infer_data:out2 -> mindspore_inference:y_
          mindspore_inference:output_0_trans_Cast_2_0 -> check_ms_infer_result:in
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(InferenceMindSporeFlowUnitTest, DISABLED_RunUnitEncrypt) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          prepare_ms_infer_data[type=flowunit, flowunit=prepare_ms_infer_data, device=cpu, deviceid=0]             
          mindspore_inference[type=flowunit, flowunit=mindspore_inference_encrypt, device=ascend, deviceid=0, batch_size=2]
          check_ms_infer_result[type=flowunit, flowunit=check_ms_infer_result, device=cpu, deviceid=0, batch_size=2]  
                                  
          prepare_ms_infer_data:out1 -> mindspore_inference:x_
          prepare_ms_infer_data:out2 -> mindspore_inference:y_
          mindspore_inference:output_0_trans_Cast_2_0 -> check_ms_infer_result:in
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox
