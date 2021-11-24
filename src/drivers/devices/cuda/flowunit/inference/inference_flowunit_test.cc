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

using ::testing::_;

static std::set<std::string> SUPPORT_TF_VERSION = {"1.13.1", "1.15.0"};

namespace modelbox {
class InferenceCudaFlowUnitTest : public testing::Test {
 public:
  InferenceCudaFlowUnitTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  virtual void SetUp() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    auto version = GetTFVersion();

    if (SUPPORT_TF_VERSION.find(version) == SUPPORT_TF_VERSION.end()) {
      MBLOG_INFO << "the version is " << version
                 << ", not in support version, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);

    SetUpTomlFiles(version);
  }

  virtual void TearDown() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      GTEST_SKIP();
    }

    auto version = GetTFVersion();
    if (SUPPORT_TF_VERSION.find(version) == SUPPORT_TF_VERSION.end()) {
      GTEST_SKIP();
    }

    RemoveFiles();
    driver_flow_->Clear();
  };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_toml_file = "virtual_tfgpu_test.toml",
                    test_toml_plugin_file = "virtual_tfgpu_plugin_test.toml",
                    test_toml_save_model_file =
                        "virtual_tfgpu_save_model_test.toml";
  std::string tensorflow_cuda_path, dest_toml_file;
  std::string tensorflow_cuda_plugin_path, dest_toml_plugin_file;
  std::string tensorflow_cuda_save_model_path, dest_toml_save_model_file;

 private:
  std::string GetTFVersion();
  Status AddMockFlowUnit();
  void Register_Test_0_1_Batch_Flowunit(std::shared_ptr<MockDriverCtl> &ctl);
  void Register_Test_1_0_Batch_Flowunit(std::shared_ptr<MockDriverCtl> &ctl);
  void Register_Test_0_1_Flowunit(std::shared_ptr<MockDriverCtl> &ctl);
  void Register_Test_1_0_Flowunit(std::shared_ptr<MockDriverCtl> &ctl);
  void SetUpTomlFiles(const std::string &version);
  modelbox::Status ReplaceVersion(const std::string &src, const std::string &dest,
                                const std::string &version);
  void RemoveFiles();

  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::string InferenceCudaFlowUnitTest::GetTFVersion() {
  std::string ans = "";
  void *handler = dlopen(MODELBOX_TF_SO_PATH, RTLD_LAZY | RTLD_NODELETE);
  if (handler == nullptr) {
    MBLOG_ERROR << "dlopen error: " << dlerror();
    return ans;
  }

  Defer { dlclose(handler); };
  typedef const char *(*TF_Version)();
  TF_Version func = nullptr;

  func = (TF_Version)dlsym(handler, "TF_Version");
  if (func == nullptr) {
    MBLOG_ERROR << "dlsym TF_Version failed, " << dlerror();
    return ans;
  }

  ans = std::string(func());
  return ans;
}

modelbox::Status InferenceCudaFlowUnitTest::ReplaceVersion(
    const std::string &src, const std::string &dest,
    const std::string &version) {
  if (access(dest.c_str(), F_OK) == 0) {
    return modelbox::STATUS_FAULT;
  }

  std::ifstream src_file(src, std::ios::binary);
  std::ofstream dst_file(dest, std::ios::binary | std::ios::trunc);

  if (src_file.fail() || dst_file.fail()) {
    return modelbox::STATUS_FAULT;
  }

  std::string line;
  std::string tf_version = "TF_VERSION";

  while (std::getline(src_file, line)) {
    auto pos = line.find(tf_version);
    if (pos != std::string::npos) {
      line.replace(pos, tf_version.size(), version);
    }
    dst_file << line << "\n";
  }

  src_file.close();
  if (dst_file.fail()) {
    dst_file.close();
    remove(dest.c_str());
    return modelbox::STATUS_FAULT;
  }
  dst_file.close();

  return modelbox::STATUS_OK;
}

void InferenceCudaFlowUnitTest::RemoveFiles() {
  auto ret = remove(dest_toml_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorflow_cuda_path.c_str());
  EXPECT_EQ(ret, 0);

  ret = remove(dest_toml_plugin_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorflow_cuda_plugin_path.c_str());
  EXPECT_EQ(ret, 0);

  ret = remove(dest_toml_save_model_file.c_str());
  EXPECT_EQ(ret, 0);
  ret = remove(tensorflow_cuda_save_model_path.c_str());
  EXPECT_EQ(ret, 0);
}

void InferenceCudaFlowUnitTest::SetUpTomlFiles(const std::string &version) {
  const std::string src_file_dir = test_assets + "/tensorflow_cuda/" + version;

  const std::string src_file_pb_toml = test_data_dir + "/" + test_toml_file;
  const std::string src_plugin_toml =
      test_data_dir + "/" + test_toml_plugin_file;
  const std::string src_save_model_toml =
      test_data_dir + "/" + test_toml_save_model_file;

  tensorflow_cuda_path = test_data_dir + "/tensorflow_cuda";
  auto mkdir_ret = mkdir(tensorflow_cuda_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  tensorflow_cuda_plugin_path = test_data_dir + "/tensorflow_cuda_plugin";
  mkdir_ret = mkdir(tensorflow_cuda_plugin_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  tensorflow_cuda_save_model_path =
      test_data_dir + "/tensorflow_cuda_save_model";
  mkdir_ret = mkdir(tensorflow_cuda_save_model_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  dest_toml_file = tensorflow_cuda_path + "/" + test_toml_file;
  auto status = ReplaceVersion(src_file_pb_toml, dest_toml_file, version);
  EXPECT_EQ(status, STATUS_OK);

  dest_toml_plugin_file =
      tensorflow_cuda_plugin_path + "/" + test_toml_plugin_file;
  status = ReplaceVersion(src_plugin_toml, dest_toml_plugin_file, version);
  EXPECT_EQ(status, STATUS_OK);

  dest_toml_save_model_file =
      tensorflow_cuda_save_model_path + "/" + test_toml_save_model_file;
  status =
      ReplaceVersion(src_save_model_toml, dest_toml_save_model_file, version);
  EXPECT_EQ(status, STATUS_OK);
}

void InferenceCudaFlowUnitTest::Register_Test_0_1_Batch_Flowunit(
    std::shared_ptr<MockDriverCtl> &ctl) {
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("DRIVER-FLOWUNIT");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("test_0_1_batch");
  desc_flowunit.SetDescription("The test input batch data, 0 inputs 1 output");
  desc_flowunit.SetVersion("1.0.0");
  std::string file_path_flowunit =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_0_1_batch.so";
  desc_flowunit.SetFilePath(file_path_flowunit);
  auto mock_flowunit = std::make_shared<MockFlowUnit>();
  auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
  mock_flowunit_desc->SetFlowUnitName("test_0_1_batch");
  mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1"));
  mock_flowunit_desc->SetFlowType(STREAM);
  mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
  std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
  mock_flowunit_wp = mock_flowunit;

  EXPECT_CALL(*mock_flowunit, Open(_))
      .WillRepeatedly(testing::Invoke(
          [=](const std::shared_ptr<modelbox::Configuration> &flow_option) {
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
          }));

  EXPECT_CALL(*mock_flowunit, DataPre(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_0_1_batch "
                       << "DataPre";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPost(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_0_1_batch "
                       << "DataPost";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit,
              Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
      .WillRepeatedly(testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
        auto output_buf_1 = op_ctx->Output("Out_1");
        std::vector<size_t> shape_vector(10, 8 * sizeof(float));
        modelbox::ModelBoxDataType type = MODELBOX_FLOAT;
        output_buf_1->Build(shape_vector);
        output_buf_1->Set("type", type);
        std::vector<size_t> shape{8};
        output_buf_1->Set("shape", shape);
        auto dev_data = (float *)(output_buf_1->MutableData());
        float num;
        for (size_t i = 0; i < output_buf_1->Size(); ++i) {
          num = 1.0;
          for (size_t j = 0; j < 8; ++j) {
            dev_data[i * 8 + j] = num;
            num += 1.0;
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
  ctl->AddMockDriverFlowUnit("test_0_1_batch", "cpu", desc_flowunit,
                             std::string(TEST_DRIVER_DIR));
};

void InferenceCudaFlowUnitTest::Register_Test_1_0_Batch_Flowunit(
    std::shared_ptr<MockDriverCtl> &ctl) {
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("DRIVER-FLOWUNIT");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("test_1_0_batch");
  desc_flowunit.SetDescription("The test output batch data, 1 input 0 outputs");
  desc_flowunit.SetVersion("1.0.0");
  std::string file_path_flowunit =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_1_0_batch.so";
  desc_flowunit.SetFilePath(file_path_flowunit);
  auto mock_flowunit = std::make_shared<MockFlowUnit>();
  auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
  mock_flowunit_desc->SetFlowUnitName("test_1_0_batch");
  mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
  mock_flowunit_desc->SetFlowType(STREAM);
  mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
  std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
  mock_flowunit_wp = mock_flowunit;

  EXPECT_CALL(*mock_flowunit, Open(_))
      .WillRepeatedly(testing::Invoke(
          [=](const std::shared_ptr<modelbox::Configuration> &flow_option) {
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPre(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_1_0_batch "
                       << "DataPre";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPost(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_1_0_batch "
                       << "DataPost";
            return modelbox::STATUS_STOP;
          }));

  EXPECT_CALL(*mock_flowunit,
              Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
      .WillRepeatedly(testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
        std::shared_ptr<BufferList> input_bufs = op_ctx->Input("In_1");
        EXPECT_EQ(input_bufs->Size(), 10);
        std::vector<size_t> shape_vector{8};
        std::vector<size_t> input_shape;
        auto result = input_bufs->At(0)->Get("shape", input_shape);
        EXPECT_TRUE(result);
        EXPECT_EQ(input_shape, shape_vector);

        for (size_t i = 0; i < input_bufs->Size(); ++i) {
          auto input_data =
              static_cast<const float *>(input_bufs->ConstBufferData(i));
          MBLOG_DEBUG << "index: " << i;
          for (size_t j = 0; j < input_shape[0]; ++j) {
            MBLOG_DEBUG << input_data[j];
          }

          EXPECT_NEAR(input_data[0], 1.05097, 1e-5);
          EXPECT_NEAR(input_data[1], 1.30058, 1e-5);
          EXPECT_NEAR(input_data[2], 1.55019, 1e-5);
          EXPECT_NEAR(input_data[3], 1.7998, 1e-5);
          EXPECT_NEAR(input_data[4], 2.0494, 1e-5);
          EXPECT_NEAR(input_data[5], 2.29901, 1e-5);
          EXPECT_NEAR(input_data[6], 2.54862, 1e-5);
          EXPECT_NEAR(input_data[7], 2.79823, 1e-5);
        }
        return modelbox::STATUS_OK;
      }));

  EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
    return modelbox::STATUS_OK;
  }));
  desc_flowunit.SetMockFlowUnit(mock_flowunit);
  ctl->AddMockDriverFlowUnit("test_1_0_batch", "cpu", desc_flowunit,
                             std::string(TEST_DRIVER_DIR));
};

void InferenceCudaFlowUnitTest::Register_Test_0_1_Flowunit(
    std::shared_ptr<MockDriverCtl> &ctl) {
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
  mock_flowunit_desc->SetFlowType(STREAM);
  mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
  std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
  mock_flowunit_wp = mock_flowunit;

  EXPECT_CALL(*mock_flowunit, Open(_))
      .WillRepeatedly(testing::Invoke(
          [=](const std::shared_ptr<modelbox::Configuration> &flow_option) {
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
          }));

  EXPECT_CALL(*mock_flowunit, DataPre(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_0_1 "
                       << "DataPre";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPost(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_0_1 "
                       << "DataPost";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit,
              Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
      .WillRepeatedly(testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
        auto output_buf_1 = op_ctx->Output("Out_1");
        std::vector<size_t> shape_vector(1, 8 * sizeof(float));
        modelbox::ModelBoxDataType type = MODELBOX_FLOAT;
        output_buf_1->Build(shape_vector);
        output_buf_1->Set("type", type);
        std::vector<size_t> shape{8};
        output_buf_1->Set("shape", shape);
        auto dev_data = (float *)(output_buf_1->MutableData());
        float num = 1.0;
        for (size_t i = 0; i < output_buf_1->Size(); ++i) {
          for (size_t j = 0; j < 8; ++j) {
            dev_data[i * 8 + j] = num;
            num += 1.0;
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
  ctl->AddMockDriverFlowUnit("test_0_1", "cpu", desc_flowunit,
                             std::string(TEST_DRIVER_DIR));
};

void InferenceCudaFlowUnitTest::Register_Test_1_0_Flowunit(
    std::shared_ptr<MockDriverCtl> &ctl) {
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
          [=](const std::shared_ptr<modelbox::Configuration> &flow_option) {
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPre(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_1_0 "
                       << "DataPre";
            return modelbox::STATUS_OK;
          }));

  EXPECT_CALL(*mock_flowunit, DataPost(_))
      .WillRepeatedly(
          testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
            MBLOG_INFO << "test_1_0 "
                       << "DataPost";
            return modelbox::STATUS_STOP;
          }));

  EXPECT_CALL(*mock_flowunit,
              Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
      .WillRepeatedly(testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
        std::shared_ptr<BufferList> input_bufs = op_ctx->Input("In_1");
        EXPECT_EQ(input_bufs->Size(), 1);
        std::vector<size_t> shape_vector{8};
        std::vector<size_t> input_shape;
        auto result = input_bufs->At(0)->Get("shape", input_shape);
        EXPECT_TRUE(result);
        EXPECT_EQ(input_shape, shape_vector);

        for (size_t i = 0; i < input_bufs->Size(); ++i) {
          auto input_data =
              static_cast<const float *>(input_bufs->ConstBufferData(i));
          MBLOG_DEBUG << "index: " << i;
          for (size_t j = 0; j < input_shape[0]; ++j) {
            MBLOG_DEBUG << input_data[j];
          }

          EXPECT_NEAR(input_data[0], 1.05097, 1e-5);
          EXPECT_NEAR(input_data[1], 1.30058, 1e-5);
          EXPECT_NEAR(input_data[2], 1.55019, 1e-5);
          EXPECT_NEAR(input_data[3], 1.7998, 1e-5);
          EXPECT_NEAR(input_data[4], 2.0494, 1e-5);
          EXPECT_NEAR(input_data[5], 2.29901, 1e-5);
          EXPECT_NEAR(input_data[6], 2.54862, 1e-5);
          EXPECT_NEAR(input_data[7], 2.79823, 1e-5);
        }
        return modelbox::STATUS_OK;
      }));

  EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
    return modelbox::STATUS_OK;
  }));
  desc_flowunit.SetMockFlowUnit(mock_flowunit);
  ctl->AddMockDriverFlowUnit("test_1_0", "cpu", desc_flowunit,
                             std::string(TEST_DRIVER_DIR));
};

Status InferenceCudaFlowUnitTest::AddMockFlowUnit() {
  auto ctl = driver_flow_->GetMockFlowCtl();
  Register_Test_0_1_Batch_Flowunit(ctl);
  Register_Test_1_0_Batch_Flowunit(ctl);
  Register_Test_0_1_Flowunit(ctl);
  Register_Test_1_0_Flowunit(ctl);
  return STATUS_OK;
}

std::shared_ptr<DriverFlowTest> InferenceCudaFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(InferenceCudaFlowUnitTest, RunUnitBatch) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/tensorflow_cuda\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1_batch[type=flowunit, flowunit=test_0_1_batch, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=inference, device=cuda, deviceid=0, label="<input> | <output>", batch_size=10]
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

TEST_F(InferenceCudaFlowUnitTest, RunUnitSingle) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "/tensorflow_cuda\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=inference, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content, 99999);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(InferenceCudaFlowUnitTest, RunPlugin) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir +
                             "/tensorflow_cuda_plugin\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=inference_plugin, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunPlugin", toml_content, 99999);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(InferenceCudaFlowUnitTest, RunSaveModel) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir +
                             "/tensorflow_cuda_save_model\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1[type=flowunit, flowunit=test_0_1, device=cpu, deviceid=0, label="<Out_1>"]             
          inference[type=flowunit, flowunit=inference_save_model, device=cuda, deviceid=0, label="<input> | <output>"]
          test_1_0[type=flowunit, flowunit=test_1_0, device=cpu, deviceid=0, label="<In_1>"]                          
          test_0_1:Out_1 -> inference:input
          inference:output -> test_1_0:In_1                                                                  
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunSaveModel", toml_content, 99999);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox