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


#include <cuda_runtime.h>
#include <dlfcn.h>

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

namespace modelbox {
class TorchInferenceFlowUnitTest : public testing::Test {
 public:
  TorchInferenceFlowUnitTest()
      : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    AddMockFlowUnit();
    SetUpTomlFiles();
  }

  void TearDown() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      GTEST_SKIP();
    }

    RemoveFiles();
    driver_flow_ = nullptr;
  };
  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_pt_file = "pytorch_example.pt",
                    test_pt_file_en = "pytorch_example_en.pt",
                    test_pt_2_output_file = "pytorch_example_2.pt",
                    test_toml_file = "virtual_torch_test.toml",
                    test_toml_file_en = "virtual_torch_test_encryt.toml",
                    test_toml_2_output_file = "virtual_torch_test_2.toml";
  std::string torch_model_path, dest_pt_file, dest_toml_file;
  std::string dest_pt_2_output_file, dest_toml_2_output_file;
  std::string dest_pt_file_en, dest_toml_file_en;

 private:
  void AddMockFlowUnit();
  void Register_Test_0_1_Flowunit();
  void Register_Test_1_0_Flowunit();
  void Register_Test_2_0_Flowunit();
  void SetUpTomlFiles();
  void RemoveFiles();

  std::shared_ptr<MockFlow> driver_flow_;
};

void TorchInferenceFlowUnitTest::RemoveFiles() {
  auto ret = remove(dest_toml_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_pt_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_pt_2_output_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_toml_2_output_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_toml_file_en.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_pt_file_en.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(torch_model_path.c_str());
  EXPECT_EQ(ret, 0);
}

void TorchInferenceFlowUnitTest::SetUpTomlFiles() {
  const std::string src_file_dir = test_assets + "/torch";
  const std::string src_pt_file = src_file_dir + "/" + test_pt_file;
  const std::string src_file_pt_toml = test_data_dir + "/" + test_toml_file;
  const std::string src_pt_2_output_file =
      src_file_dir + "/" + test_pt_2_output_file;
  const std::string src_file_pt_2_output_toml =
      test_data_dir + "/" + test_toml_2_output_file;
  const std::string src_pt_file_en = src_file_dir + "/" + test_pt_file_en;
  const std::string src_file_pt_toml_en =
      test_data_dir + "/" + test_toml_file_en;

  torch_model_path = test_data_dir + "/torch";
  auto mkdir_ret = mkdir(torch_model_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  dest_pt_file = torch_model_path + "/" + test_pt_file;
  auto status = CopyFile(src_pt_file, dest_pt_file, 0);
  EXPECT_EQ(status, STATUS_OK);

  dest_toml_file = torch_model_path + "/" + test_toml_file;
  status = CopyFile(src_file_pt_toml, dest_toml_file, 0);
  EXPECT_EQ(status, STATUS_OK);

  dest_pt_2_output_file = torch_model_path + "/" + test_pt_2_output_file;
  status = CopyFile(src_pt_2_output_file, dest_pt_2_output_file, 0);
  EXPECT_EQ(status, STATUS_OK);

  dest_toml_2_output_file = torch_model_path + "/" + test_toml_2_output_file;
  status = CopyFile(src_file_pt_2_output_toml, dest_toml_2_output_file, 0);
  EXPECT_EQ(status, STATUS_OK);

  dest_pt_file_en = torch_model_path + "/" + test_pt_file_en;
  status = CopyFile(src_pt_file_en, dest_pt_file_en, 0);
  EXPECT_EQ(status, STATUS_OK);

  dest_toml_file_en = torch_model_path + "/" + test_toml_file_en;
  status = CopyFile(src_file_pt_toml_en, dest_toml_file_en, 0);
  EXPECT_EQ(status, STATUS_OK);
}

void TorchInferenceFlowUnitTest::Register_Test_0_1_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_0_1", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto open_func =
      [=](const std::shared_ptr<modelbox::Configuration> &flow_option,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;
    auto spt = mock_flowunit_wp.lock();
    auto ext_data = spt->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({10 * sizeof(int)});
    auto data = (int *)buffer_list->MutableData();
    for (size_t i = 0; i < 10; i++) {
      data[i] = i;
    }

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

  auto process_func =
      [=](std::shared_ptr<DataContext> op_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_buf_1 = op_ctx->Output("Out_1");
    std::vector<size_t> shape_vector(1, 200 * sizeof(float));
    modelbox::ModelBoxDataType type = MODELBOX_FLOAT;
    output_buf_1->Build(shape_vector);
    output_buf_1->Set("type", type);
    std::vector<size_t> shape{20, 10};
    output_buf_1->Set("shape", shape);
    auto dev_data = (float *)(output_buf_1->MutableData());
    float num = 1.0;
    for (size_t i = 0; i < output_buf_1->Size(); ++i) {
      for (size_t j = 0; j < 200; ++j) {
        dev_data[i * 200 + j] = num;
      }
    }

    MBLOG_DEBUG << output_buf_1->GetBytes();
    MBLOG_DEBUG << "test_0_1 gen data, 0" << output_buf_1->Size();

    return modelbox::STATUS_OK;
  };

  auto mock_functions = std::make_shared<MockFunctionCollection>();
  mock_functions->RegisterOpenFunc(open_func);
  mock_functions->RegisterProcessFunc(process_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
};

void TorchInferenceFlowUnitTest::Register_Test_1_0_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_1_0", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);

  auto post_func = [=](std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MBLOG_INFO << "test_1_0 "
               << "DataPost";
    return modelbox::STATUS_STOP;
  };

  auto process_func = [=](std::shared_ptr<DataContext> op_ctx,
                          std::shared_ptr<MockFlowUnit> mock_flowunit) {
    std::shared_ptr<BufferList> input_bufs = op_ctx->Input("In_1");
    EXPECT_EQ(input_bufs->Size(), 1);
    std::vector<size_t> shape_vector{10, 10};
    std::vector<size_t> input_shape;
    auto result = input_bufs->At(0)->Get("shape", input_shape);
    EXPECT_TRUE(result);
    EXPECT_EQ(input_shape, shape_vector);

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data =
          static_cast<const float *>(input_bufs->ConstBufferData(i));
      MBLOG_DEBUG << "index: " << i;
      for (size_t j = 0; j < 100; j += 10) {
        MBLOG_DEBUG << input_data[j];
      }

      EXPECT_NEAR(input_data[0], 9.3490, 1e-4);
      EXPECT_NEAR(input_data[10], 7.3774, 1e-4);
      EXPECT_NEAR(input_data[20], 10.6521, 1e-4);
      EXPECT_NEAR(input_data[30], 8.6493, 1e-4);
      EXPECT_NEAR(input_data[40], 8.0384, 1e-4);
      EXPECT_NEAR(input_data[50], 9.2835, 1e-4);
      EXPECT_NEAR(input_data[60], 11.0915, 1e-4);
      EXPECT_NEAR(input_data[70], 10.5014, 1e-4);
      EXPECT_NEAR(input_data[80], 12.0796, 1e-4);
      EXPECT_NEAR(input_data[90], 11.1132, 1e-4);
    }
    return modelbox::STATUS_OK;
  };

  auto mock_functions = std::make_shared<MockFunctionCollection>();
  mock_functions->RegisterDataPostFunc(post_func);
  mock_functions->RegisterProcessFunc(process_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
};

void TorchInferenceFlowUnitTest::Register_Test_2_0_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_2_0", {"In_1", "In_2"}, {});
  mock_desc->SetFlowType(STREAM);

  auto post_func = [=](std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MBLOG_INFO << "test_2_0 "
               << "DataPost";
    return modelbox::STATUS_STOP;
  };

  auto process_func = [=](std::shared_ptr<DataContext> op_ctx,
                          std::shared_ptr<MockFlowUnit> mock_flowunit) {
    std::shared_ptr<BufferList> input_bufs = op_ctx->Input("In_1");
    EXPECT_EQ(input_bufs->Size(), 1);
    std::vector<size_t> shape_vector{10, 10};
    std::vector<size_t> input_shape;
    auto result = input_bufs->At(0)->Get("shape", input_shape);
    EXPECT_TRUE(result);
    EXPECT_EQ(input_shape, shape_vector);

    auto input_bufs_2 = op_ctx->Input("In_2");
    std::vector<size_t> input_shape_2;
    result = input_bufs_2->At(0)->Get("shape", input_shape_2);
    EXPECT_TRUE(result);
    std::vector<size_t> shape_vector_2{20, 10};
    EXPECT_EQ(input_shape_2, shape_vector_2);
    EXPECT_EQ(static_cast<const float *>(input_bufs_2->ConstBufferData(0))[0],
              1.0);

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data =
          static_cast<const float *>(input_bufs->ConstBufferData(i));
      MBLOG_DEBUG << "index: " << i;
      for (size_t j = 0; j < 100; j += 10) {
        MBLOG_DEBUG << input_data[j];
      }

      EXPECT_NEAR(input_data[0], 10.2705, 1e-4);
      EXPECT_NEAR(input_data[10], 9.03664, 1e-4);
      EXPECT_NEAR(input_data[20], 9.106, 1e-4);
      EXPECT_NEAR(input_data[30], 10.322, 1e-4);
      EXPECT_NEAR(input_data[40], 10.6219, 1e-4);
      EXPECT_NEAR(input_data[50], 10.5359, 1e-4);
      EXPECT_NEAR(input_data[60], 10.6534, 1e-4);
      EXPECT_NEAR(input_data[70], 10.1457, 1e-4);
      EXPECT_NEAR(input_data[80], 7.73253, 1e-4);
      EXPECT_NEAR(input_data[90], 8.46899, 1e-4);
    }
    return modelbox::STATUS_OK;
  };

  auto mock_functions = std::make_shared<MockFunctionCollection>();
  mock_functions->RegisterDataPostFunc(post_func);
  mock_functions->RegisterProcessFunc(process_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
};

void TorchInferenceFlowUnitTest::AddMockFlowUnit() {
  Register_Test_0_1_Flowunit();
  Register_Test_1_0_Flowunit();
  Register_Test_2_0_Flowunit();
}

std::shared_ptr<MockFlow> TorchInferenceFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(TorchInferenceFlowUnitTest, RunUnitSingleOutput) {
  std::string toml_content = R"(
    [log]
    level = "DEBUG"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/torch\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=torch, device=cuda, deviceid=0, label="<input> | <output>", skip_first_dim=true]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(TorchInferenceFlowUnitTest, RunUnitSingleOutputEncrypt) {
  std::string toml_content = R"(
    [log]
    level = "DEBUG"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/torch\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=torch_encrypt, device=cuda, deviceid=0, label="<input> | <output>", skip_first_dim=true]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(TorchInferenceFlowUnitTest, RunUnitMutiOutput) {
  std::string toml_content = R"(
    [log]
    level = "DEBUG"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/torch\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=torch_2, device=cuda, deviceid=0, label="<input> | <output>", skip_first_dim=true]
          test_2_0[type=flowunit, flowunit=test_2_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output1 -> test_2_0:In_1 
          inference:output2 -> test_2_0:In_2                                                                 
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox