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
#include <cuda_runtime.h>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;

namespace modelbox {
class MeanGpuFlowUnitTest : public testing::Test {
 public:
  MeanGpuFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  void SetUp() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  }

  void TearDown() override { driver_flow_->Clear(); };

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

Status MeanGpuFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();
  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_mean_0");
    desc_flowunit.SetDescription("The test input data, 0 inputs 1 output");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_mean_0.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_mean_0");
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
              auto data = (int*)buffer_list->MutableData();
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
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_DEBUG << "test_mean_0 "
                          << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_DEBUG << "test_mean_0 "
                          << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              auto output_buf_1 = op_ctx->Output("Out_1");
              std::vector<size_t> data_1_shape = {5 * 4 * 3 * sizeof(uint8_t)};
              output_buf_1->Build(data_1_shape);
              auto dev_data_1 =
                  static_cast<uint8_t*>(output_buf_1->At(0)->MutableData());
              for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 5; j++) {
                  for (size_t k = 0; k < 4; k++) {
                    dev_data_1[i * 20 + j * 4 + k] = static_cast<uint8_t>(100);
                  }
                }
              }

              std::vector<size_t> shape{4, 5, 3};
              output_buf_1->Set("shape", shape);
              output_buf_1->Set("type", ModelBoxDataType::MODELBOX_UINT8);

              MBLOG_DEBUG << "test_mean_0 gen data, 0"
                          << output_buf_1->GetBytes();

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));

    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_mean_0", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_mean_1");
    desc_flowunit.SetDescription("The test output data, 1 input 0 outputs");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_mean_1.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_mean_1");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
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
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_DEBUG << "test_mean_1 "
                          << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_DEBUG << "test_mean_1 "
                          << "DataPost";
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              auto input_bufs = op_ctx->Input("In_1");
              EXPECT_EQ(input_bufs->Size(), 1);
              for (size_t i = 0; i < input_bufs->Size(); ++i) {
                auto input_buf = input_bufs->At(i);
                std::vector<size_t> shape;
                input_buf->Get("shape", shape);
                size_t width = shape[1];
                size_t height = shape[0];
                EXPECT_EQ(width, 5);
                EXPECT_EQ(height, 4);

                const auto in_data =
                    static_cast<const float*>(input_buf->ConstData());
                for (size_t c = 0; c < 3; c++) {
                  for (size_t j = 0; j < width; j++) {
                    for (size_t k = 0; k < height; k++) {
                      float data = in_data[c * width * height + j * height + k];
                      if (c == 0) {
                        EXPECT_NEAR(data, 100, 0.0001);
                      } else if (c == 1) {
                        EXPECT_NEAR(data, 90, 0.0001);
                      } else {
                        EXPECT_NEAR(data, 80, 0.0001);
                      }
                    }
                  }
                }
              }

              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_mean_1", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}

std::shared_ptr<DriverFlowTest> MeanGpuFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(MeanGpuFlowUnitTest, RunUnit) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          test_mean_0[type=flowunit, flowunit=test_mean_0, device=cpu,deviceid=0, label="<Out_1>"] 
          mean[type=flowunit, flowunit=mean, device=cuda, deviceid=0, label="<in_data> | <out_data>", mean="0.0,10.0,20.0"]
          test_mean_1[type=flowunit, flowunit=test_mean_1, device=cpu, deviceid=0, label="<In_1>"] 

          test_mean_0:Out_1 -> mean:in_data
          mean:out_data -> test_mean_1:In_1
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("RunUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);
}

}  // namespace modelbox