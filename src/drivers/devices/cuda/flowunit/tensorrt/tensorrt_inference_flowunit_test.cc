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

using ::testing::_;

namespace modelbox {
class TensorRTFlowUnitTest : public testing::Test {
 public:
  TensorRTFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  void SetUp() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    SetUpTomlFile();
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  }

  void TearDown() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      GTEST_SKIP();
    }

    RemoveTomlFile();

    driver_flow_->Clear();
  };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_onnx_file = "model.onnx",
                    test_toml_file = "virtual_tensorrt_test.toml",
                    test_onnx_file_en = "model_en.onnx",
                    test_toml_file_en = "virtual_tensorrt_encrypt_test.toml",
                    test_plugin_toml_file = "virtual_plugin_tensorrt_test.toml";

  std::string tensorrt_path, dest_model_file, dest_toml_file;
  std::string tensorrt_path_en, dest_model_file_en, dest_toml_file_en;
  std::string tensorrt_plugin_path, dest_plugin_model_file,
      dest_plugin_toml_file;

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
  void SetUpTomlFile();
  void RemoveTomlFile();
};

void TensorRTFlowUnitTest::SetUpTomlFile() {
  const std::string src_file = test_assets + "/tensorrt/" + test_onnx_file;
  const std::string src_toml = test_data_dir + "/" + test_toml_file;
  const std::string src_file_en = test_assets + "/tensorrt/" + test_onnx_file_en;
  const std::string src_toml_en = test_data_dir + "/" + test_toml_file_en;
  const std::string src_plugin_toml =
      test_data_dir + "/" + test_plugin_toml_file;

  tensorrt_path = test_data_dir + "/tensorrt";
  auto mkdir_ret = mkdir(tensorrt_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);
  dest_model_file = tensorrt_path + "/" + test_onnx_file;
  dest_toml_file = tensorrt_path + "/" + test_toml_file;
  auto status = CopyFile(src_file, dest_model_file, 0);
  EXPECT_EQ(status, STATUS_OK);
  status = CopyFile(src_toml, dest_toml_file, 0);
  EXPECT_EQ(status, STATUS_OK);

  tensorrt_path_en = test_data_dir + "/tensorrt_encrypt";
  mkdir_ret = mkdir(tensorrt_path_en.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);
  dest_model_file_en = tensorrt_path_en + "/" + test_onnx_file_en;
  dest_toml_file_en = tensorrt_path_en + "/" + test_toml_file_en;
  status = CopyFile(src_file_en, dest_model_file_en, 0);
  EXPECT_EQ(status, STATUS_OK);
  status = CopyFile(src_toml_en, dest_toml_file_en, 0);
  EXPECT_EQ(status, STATUS_OK);

  tensorrt_plugin_path = test_data_dir + "/tensorrt_plugin";
  mkdir_ret = mkdir(tensorrt_plugin_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  dest_plugin_model_file = tensorrt_plugin_path + "/" + test_onnx_file;
  dest_plugin_toml_file = tensorrt_plugin_path + "/" + test_plugin_toml_file;
  status = CopyFile(src_file, dest_plugin_model_file, 0);
  EXPECT_EQ(status, STATUS_OK);
  status = CopyFile(src_plugin_toml, dest_plugin_toml_file, 0);
  EXPECT_EQ(status, STATUS_OK);
}

void TensorRTFlowUnitTest::RemoveTomlFile() {
  auto ret = remove(dest_model_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_toml_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorrt_path.c_str());
  EXPECT_EQ(ret, 0);

  ret = remove(dest_model_file_en.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_toml_file_en.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorrt_path_en.c_str());
  EXPECT_EQ(ret, 0);

  ret = remove(dest_plugin_model_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(dest_plugin_toml_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorrt_plugin_path.c_str());
  EXPECT_EQ(ret, 0);
}

Status TensorRTFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_0_1");
    desc_flowunit.SetDescription("The test input data, 0 inputs 1 output");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_0_1.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_0_1");
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1"));
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              auto spt = mock_flowunit_wp.lock();
              auto ext_data = spt->CreateExternalData();
              if (!ext_data) {
                MBLOG_ERROR << "can not get external data.";
              }

              auto buffer_list = ext_data->CreateBufferList();
              buffer_list->Build({10 * sizeof(int)});
              auto* data = (int*)buffer_list->MutableData();
              for (size_t i = 0; i < 10; i++) {
                data[i] = i;
              }

              auto status = ext_data->Send(buffer_list);
              if (!status) {
                MBLOG_ERROR << "external data send buffer list failed:"
                            << status;
              }

              status = ext_data->Close();
              if (!status) {
                MBLOG_ERROR << "external data close failed:" << status;
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](const std::shared_ptr<DataContext>& data_ctx) {
              MBLOG_INFO << "test_0_1 "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](const std::shared_ptr<DataContext>& data_ctx) {
              MBLOG_INFO << "test_0_1 "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](const std::shared_ptr<DataContext>& op_ctx) {
              auto output_buf_1 = op_ctx->Output("Out_1");
              std::vector<size_t> shape_vector(1, 784 * sizeof(float));
              modelbox::ModelBoxDataType type = MODELBOX_FLOAT;
              output_buf_1->Build(shape_vector);
              output_buf_1->Set("type", type);
              std::vector<size_t> shape{784};
              output_buf_1->Set("shape", shape);
              auto* dev_data = (float*)(output_buf_1->MutableData());
              for (size_t i = 0; i < output_buf_1->Size(); ++i) {
                for (size_t j = 0; j < 784; ++j) {
                  dev_data[i * 784 + j] = 0.0;
                }
              }

              MBLOG_DEBUG << output_buf_1->GetBytes();
              MBLOG_DEBUG << "test_0_1 gen data, 0" << output_buf_1->Size();

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_0_1", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_1_0");
    desc_flowunit.SetDescription("The test output data, 1 input 0 outputs");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_1_0.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_1_0");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
    mock_flowunit_desc->SetFlowType(STREAM);
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](const std::shared_ptr<DataContext>& data_ctx) {
              MBLOG_INFO << "test_1_0 "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](const std::shared_ptr<DataContext>& data_ctx) {
              MBLOG_INFO << "test_1_0 "
                         << "DataPost";
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](const std::shared_ptr<DataContext>& op_ctx) {
              std::shared_ptr<BufferList> input_bufs = op_ctx->Input("In_1");
              EXPECT_EQ(input_bufs->Size(), 1);
              std::vector<size_t> shape_vector{10};
              std::vector<size_t> input_shape;
              auto result = input_bufs->At(0)->Get("shape", input_shape);
              EXPECT_TRUE(result);

              const auto* input_data =
                  static_cast<const float*>(input_bufs->ConstBufferData(0));
              for (int i = 0; i < 10; ++i) {
                MBLOG_DEBUG << input_data[i];
              }

              EXPECT_NEAR(input_data[0], 0.0356422, 1e-6);
              EXPECT_NEAR(input_data[1], 0.0931573, 1e-6);
              EXPECT_NEAR(input_data[2], 0.0815316, 1e-6);
              EXPECT_NEAR(input_data[3], 0.0455169, 1e-6);
              EXPECT_NEAR(input_data[4], 0.0595113, 1e-6);
              EXPECT_NEAR(input_data[5], 0.4212710, 1e-6);
              EXPECT_NEAR(input_data[6], 0.051922, 1e-6);
              EXPECT_NEAR(input_data[7], 0.160296, 1e-6);
              EXPECT_NEAR(input_data[8], 0.00811869, 1e-6);
              EXPECT_NEAR(input_data[9], 0.0430332, 1e-6);
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_1_0", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}  // namespace modelbox

std::shared_ptr<DriverFlowTest> TensorRTFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(TensorRTFlowUnitTest, RunUnitSingle) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          tensorrt[type=flowunit, flowunit=tensorrt, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> tensorrt:"input:0"
          tensorrt:"output:0" -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(TensorRTFlowUnitTest, RunUnitPlugin) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          tensorrt[type=flowunit, flowunit=tensorrt_plugin, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> tensorrt:"input:0"
          tensorrt:"output:0" -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunPlugin", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(TensorRTFlowUnitTest, RunUnitSingleEncrypt) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          tensorrt[type=flowunit, flowunit=tensorrt_encrypt, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> tensorrt:"input:0"
          tensorrt:"output:0" -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnitSingleEncrypt", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

// TODO test batch inference
// TODO test quantize

}  // namespace modelbox
