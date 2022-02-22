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


#include <securec.h>

#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
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
class NppiResizeFlowUnitTest : public testing::Test {
 public:
  NppiResizeFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  virtual void SetUp() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  virtual void TearDown() { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> NppiResizeFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status NppiResizeFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();
  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("p3_test_0_1_resize");
    desc_flowunit.SetDescription("the test in 0 out 1");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-p3_test_0_1_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("p3_test_0_1_resize");
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1"));
    mock_flowunit_desc->SetFlowType(STREAM);
    mock_flowunit_desc->SetMaxBatchSize(16);
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

              std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";

              auto output_buf = ext_data->CreateBufferList();
              modelbox::TensorList output_tensor_list(output_buf);
              output_tensor_list.BuildFromHost<uchar>(
                  {1, {gimg_path.size() + 1}}, (void*)gimg_path.data(),
                  gimg_path.size() + 1);

              auto status = ext_data->Send(output_buf);
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
              MBLOG_INFO << "p3_test_0_1_resize "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "p3_test_0_1_resize "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "p3_test_0_1_resize process";
              auto output_buf = data_ctx->Output("Out_1");
              auto external = data_ctx->External();
              std::string gimg_path =
                  std::string((char*)(*external)[0]->ConstData());

              cv::Mat gimg_data = cv::imread(gimg_path.c_str());

              MBLOG_INFO << "gimage col " << gimg_data.cols << "  grow "
                         << gimg_data.rows
                         << " gchannel:" << gimg_data.channels();

              size_t gcols = gimg_data.cols;
              size_t grows = gimg_data.rows;
              size_t gchannels = gimg_data.channels();

              uint32_t batch_size = 5;
              std::vector<size_t> shape_vector(
                  batch_size,
                  modelbox::Volume({grows, gcols, gchannels}) * sizeof(uchar));
              output_buf->Build(shape_vector);
              for (size_t i = 0; i < batch_size; ++i) {
                std::string img_path = std::string(TEST_ASSETS) + "/test.jpg";
                cv::Mat img_data = cv::imread(img_path.c_str());
                std::vector<cv::Mat> vecChannels;
                cv::split(img_data, vecChannels);
                cv::Mat mergeImg;
                cv::merge(vecChannels, mergeImg);
                MBLOG_DEBUG << "image col " << img_data.cols << "  row "
                            << img_data.rows
                            << " channel:" << img_data.channels();

                int32_t cols = img_data.cols;
                int32_t rows = img_data.rows;
                int32_t channels = img_data.channels();

                output_buf->At(i)->Set("width", cols);
                output_buf->At(i)->Set("height", rows);
                output_buf->At(i)->Set("channel", channels);

                auto output_data =
                    static_cast<uchar*>(output_buf->MutableBufferData(i));
                for (int32_t j = 0; j < channels; j++) {
                  cv::Mat tmpMat = vecChannels.at(j);
                  memcpy_s(
                      output_data + (tmpMat.total() * tmpMat.elemSize() * j),
                      output_buf->At(i)->GetBytes() / channels, tmpMat.data,
                      tmpMat.total() * tmpMat.elemSize());
                }
              }

              MBLOG_INFO << "test_0_1 gen data finish";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("p3_test_0_1_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("p3_test_1_0_resize");
    desc_flowunit.SetDescription("the test in 1 out 0");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-p3_test_1_0_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("p3_test_1_0_resize");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
    mock_flowunit_desc->SetFlowType(STREAM);
    mock_flowunit_desc->SetMaxBatchSize(16);
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
              MBLOG_INFO << "p3_test_1_0_resize "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "p3_test_1_0_resize "
                         << "DataPost";
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              MBLOG_INFO << "p3_test_1_0_resize process";
              auto input_buf = op_ctx->Input("In_1");
              int32_t cols = 0, rows = 0, channels = 0;
              for (size_t i = 0; i < input_buf->Size(); i++) {
                input_buf->At(i)->Get("width", cols);
                input_buf->At(i)->Get("height", rows);
                input_buf->At(i)->Get("channel", channels);
                MBLOG_INFO << "image col " << cols << "  row " << rows
                           << " channel:" << channels;
                auto input_data =
                    static_cast<const uchar*>(input_buf->ConstBufferData(i));

                cv::Mat img_data(cv::Size(cols, rows), CV_8UC3);

                std::vector<cv::Mat> vecChannelsDest;
                for (int32_t j = 0; j < channels; j++) {
                  cv::Mat tmp(cv::Size(cols, rows), CV_8UC1);
                  memcpy_s(tmp.data, cols * rows, input_data + cols * rows * j,
                           input_buf->At(i)->GetBytes() / channels);
                  vecChannelsDest.push_back(tmp);
                }
                cv::merge(vecChannelsDest, img_data);

                std::string name = std::string(TEST_DATA_DIR) + "/test" +
                                   std::to_string(i) + ".jpg";
                cv::imwrite(name.c_str(), img_data);
              }
              MBLOG_DEBUG << "p3_test_1_0_resize process data finish";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("p3_test_1_0_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("c3r_test_0_1_resize");
    desc_flowunit.SetDescription("the test in 0 out 1");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-c3r_test_0_1_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("c3r_test_0_1_resize");
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1"));
    mock_flowunit_desc->SetFlowType(STREAM);
    mock_flowunit_desc->SetMaxBatchSize(16);
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

              std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";

              auto output_buf = ext_data->CreateBufferList();
              modelbox::TensorList output_tensor_list(output_buf);
              output_tensor_list.BuildFromHost<uchar>(
                  {1, {gimg_path.size() + 1}}, (void*)gimg_path.data(),
                  gimg_path.size() + 1);

              auto status = ext_data->Send(output_buf);
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
              MBLOG_INFO << "stream_info "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_info "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              auto output_bufs = data_ctx->Output("Out_1");
              auto external = data_ctx->External();
              std::string gimg_path =
                  std::string((char*)(*external)[0]->ConstData());

              cv::Mat gimg_data = cv::imread(gimg_path.c_str());

              MBLOG_INFO << "gimage col " << gimg_data.cols << "  grow "
                         << gimg_data.rows
                         << " gchannel:" << gimg_data.channels();

              long unsigned int gcols = gimg_data.cols;
              long unsigned int grows = gimg_data.rows;
              long unsigned int gchannels = gimg_data.channels();

              uint32_t batch_size = 5;
              std::vector<size_t> shape;
              for (size_t i = 0; i < batch_size; i++) {
                shape.push_back(grows * gcols * gchannels * sizeof(uchar));
              }

              output_bufs->Build(shape);

              for (size_t i = 0; i < 5; ++i) {
                std::string img_path = gimg_path;
                cv::Mat img_data = cv::imread(img_path.c_str());
                MBLOG_INFO << "image col " << img_data.cols << "  row "
                           << img_data.rows
                           << " channel:" << img_data.channels();

                int32_t cols = img_data.cols;
                int32_t rows = img_data.rows;
                int32_t channels = img_data.channels();

                output_bufs->At(i)->Set("width", cols);
                output_bufs->At(i)->Set("height", rows);
                output_bufs->At(i)->Set("channel", channels);

                auto output_data = output_bufs->At(i)->MutableData();
                memcpy_s(output_data, output_bufs->At(i)->GetBytes(),
                         img_data.data, img_data.total() * img_data.elemSize());
              }
              MBLOG_INFO << "test_0_1 gen data finish";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("c3r_test_0_1_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("c3r_test_1_0_resize");
    desc_flowunit.SetDescription("the test in 1 out 0");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-c3r_test_1_0_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("c3r_test_1_0_resize");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
    mock_flowunit_desc->SetFlowType(STREAM);
    mock_flowunit_desc->SetMaxBatchSize(16);
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
              MBLOG_INFO << "stream_info "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_info "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              MBLOG_INFO << "c3r_test_1_0_resize process";
              auto input = op_ctx->Input("In_1");

              for (size_t i = 0; i < input->Size(); i++) {
                int32_t cols;
                int32_t rows;
                int32_t channels;
                input->At(i)->Get("width", cols);
                input->At(i)->Get("height", rows);
                input->At(i)->Get("channel", channels);
                auto input_data = input->At(i)->ConstData();

                cv::Mat img_data(cv::Size(cols, rows), CV_8UC3);
                memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                         input_data, input->At(i)->GetBytes());

                std::string name = std::string(TEST_DATA_DIR) + "/test" +
                                   std::to_string(i) + ".jpg";
                cv::imwrite(name.c_str(), img_data);
              }

              MBLOG_INFO << "c3r_test_1_0_resize process data finish";
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("c3r_test_1_0_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}

TEST_F(NppiResizeFlowUnitTest, TestC3r) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          c3r_test_0_1_resize[type=flowunit, flowunit=c3r_test_0_1_resize,
          device=cpu, deviceid=0, label="<Out_1>",batch_size=5]

          nppi_resize[type=flowunit, flowunit=resize, device=cuda,
          deviceid=0, label="<in_image> | <out_image>", width=128, height=128,
          method="u8c3r", batch_size=5]

          c3r_test_1_0_resize[type=flowunit, flowunit=c3r_test_1_0_resize,
          device=cpu, deviceid=0, label="<In_1>",batch_size=5]

          c3r_test_0_1_resize:Out_1 -> nppi_resize:in_image
          nppi_resize:out_image -> c3r_test_1_0_resize:In_1
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("TestC3r", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);

  std::vector<std::string> filePath;
  ListFiles(std::string(TEST_DATA_DIR), "*", &filePath);
  for (auto& elem : filePath) {
    MBLOG_INFO << "filePath: " << elem;
  }

  for (size_t i = 0; i < 5; ++i) {
    std::string expected_file_path =
        std::string(TEST_ASSETS) + "/nppi_resize_128x128_result.jpg";
    cv::Mat expected_img = cv::imread(expected_file_path);

    std::string resize_result_file_path =
        std::string(TEST_DATA_DIR) + "/test" + std::to_string(i) + ".jpg";
    cv::Mat resize_result_img = cv::imread(resize_result_file_path);

    int result_data_size =
        resize_result_img.total() * resize_result_img.elemSize();
    int expected_data_size = expected_img.total() * expected_img.elemSize();
    EXPECT_EQ(result_data_size, expected_data_size);

    int ret =
        memcmp(resize_result_img.data, expected_img.data, result_data_size);
    EXPECT_EQ(ret, 0);

    auto rmret = remove(resize_result_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  }
}

}  // namespace modelbox