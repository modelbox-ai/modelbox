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

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;

namespace modelbox {
class CVResizeFlowUnitTest : public testing::Test {
 public:
  CVResizeFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  virtual void TearDown() { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> CVResizeFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status CVResizeFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();
  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_0_1_resize");
    desc_flowunit.SetDescription("the test in 0 out 1");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_0_1_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_0_1_resize");
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
                auto err_msg = "can not get external data.";
                modelbox::Status ret = {modelbox::STATUS_NODATA, err_msg};
                MBLOG_ERROR << err_msg;
                return ret;
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
                return status;
              }

              status = ext_data->Close();
              if (!status) {
                MBLOG_ERROR << "external data close failed:" << status;
                return status;
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
              std::vector<size_t> shape_vector(
                  batch_size,
                  modelbox::Volume({grows, gcols, gchannels}) * sizeof(uchar));
              output_bufs->Build(shape_vector);

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

                auto output_data =
                    static_cast<uchar*>(output_bufs->MutableBufferData(i));
                memcpy_s(output_data, output_bufs->At(i)->GetBytes(),
                         img_data.data, img_data.total() * img_data.elemSize());
              }
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_0_1_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_1_0_resize");
    desc_flowunit.SetDescription("the test in 1 out 0");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-test_1_0_resize.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_1_0_resize");
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
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              MBLOG_INFO << "test_1_0_resize process";
              auto input_buf = op_ctx->Input("In_1");
              int32_t cols;
              int32_t rows;
              int32_t channels;

              for (size_t i = 0; i < input_buf->Size(); i++) {
                input_buf->At(i)->Get("width", cols);
                input_buf->At(i)->Get("height", rows);
                input_buf->At(i)->Get("channel", channels);
                auto input_data =
                    static_cast<const uchar*>(input_buf->ConstBufferData(i));

                cv::Mat img_data(cv::Size(cols, rows), CV_8UC3);
                memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                         input_data, input_buf->At(i)->GetBytes());

                std::string name = std::string(TEST_DATA_DIR) + "/test" +
                                   std::to_string(i) + ".jpg";
                cv::imwrite(name.c_str(), img_data);
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_1_0_resize", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }
  return STATUS_OK;
}

TEST_F(CVResizeFlowUnitTest, InitUnit) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1_resize[type=flowunit, flowunit=test_0_1_resize, device=cpu, deviceid=0, label="<Out_1>"]
          cv_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <out_image>", width=128, height=128, interpolation="inter_nearest", batch_size=5]
          test_1_0_resize[type=flowunit, flowunit=test_1_0_resize, device=cpu, deviceid=0, label="<In_1>",batch_size=5]                                
          test_0_1_resize:Out_1 -> cv_resize:in_image 
          cv_resize:out_image -> test_1_0_resize:In_1                                                                      
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("InitUnit", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);

  std::vector<std::string> filePath;
  ListFiles(std::string(TEST_DATA_DIR), "*", &filePath);
  for (auto& elem : filePath) {
    MBLOG_DEBUG << "filePath: " << elem;
  }

  for (size_t i = 0; i < 5; ++i) {
    std::string expected_file_path =
        std::string(TEST_ASSETS) + "/cpu_resize_128x128_result.jpg";
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