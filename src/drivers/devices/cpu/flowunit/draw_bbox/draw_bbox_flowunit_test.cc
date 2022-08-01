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

#include "draw_bbox_flowunit.h"

#include <securec.h>

#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"

using ::testing::_;

namespace modelbox {
class DrawBBoxFlowUnitTest : public testing::Test {
 public:
  DrawBBoxFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> DrawBBoxFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status DrawBBoxFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_0_1_draw_bbox");
    desc_flowunit.SetDescription("the test in 0 out 1");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-test_0_1_draw_bbox.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_0_1_draw_bbox");
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1"));
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_2"));
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
              Status ret;
              if (!ext_data) {
                MBLOG_ERROR << "can not get external data.";
                ret = {STATUS_FAULT};
                return ret;
              }

              auto buffer_list = ext_data->CreateBufferList();
              buffer_list->Build({10 * sizeof(int)});
              auto *data = (int *)buffer_list->MutableData();
              for (size_t i = 0; i < 10; i++) {
                data[i] = i;
              }

              auto status = ext_data->Send(buffer_list);
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

              MBLOG_INFO << "send event test_0_1_draw_bbox";

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

              std::vector<size_t> shape;
              shape.push_back(sizeof(BBox) * 2);
              shape.push_back(sizeof(BBox) * 2);
              shape.push_back(sizeof(BBox) * 2);
              shape.push_back(sizeof(BBox) * 2);
              shape.push_back(sizeof(BBox) * 2);

              output_bufs->Build(shape);

              for (size_t i = 0; i < 5; ++i) {
                auto *output_data = output_bufs->MutableBufferData(i);
                std::shared_ptr<BBox> b1 = std::make_shared<BBox>();
                b1->w = 20;
                b1->h = 20;
                b1->x = 20;
                b1->y = 20;
                b1->category = 1;
                b1->score = 0.95;

                memcpy_s(output_data, sizeof(BBox), b1.get(), sizeof(BBox));

                std::shared_ptr<BBox> b2 = std::make_shared<BBox>();
                b2->w = 10;
                b2->h = 30;
                b2->x = 60;
                b2->y = 60;
                b2->score = 0.9;
                b2->category = 0;

                memcpy_s((char*)output_data + sizeof(BBox), sizeof(BBox),
                         b2.get(), sizeof(BBox));
              }

              // get image data
              auto output2_bufs = data_ctx->Output("Out_2");

              std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";

              MBLOG_INFO << "images path: " << gimg_path;
              cv::Mat img_data = cv::imread(gimg_path);

              MBLOG_INFO << "gimage col " << img_data.cols << "  grow "
                         << img_data.rows
                         << " gchannel:" << img_data.channels();

              int32_t gcols = img_data.cols;
              int32_t grows = img_data.rows;
              int32_t gchannels = img_data.channels();

              std::vector<size_t> shape2;
              shape2.push_back(img_data.total() * img_data.elemSize());
              shape2.push_back(img_data.total() * img_data.elemSize());
              shape2.push_back(img_data.total() * img_data.elemSize());
              shape2.push_back(img_data.total() * img_data.elemSize());
              shape2.push_back(img_data.total() * img_data.elemSize());
              MBLOG_INFO << "build" << img_data.total() * img_data.elemSize();
              output2_bufs->Build(shape2);

              for (size_t i = 0; i < 5; ++i) {
                std::string img_path = gimg_path;
                MBLOG_DEBUG << "image col " << img_data.cols << "  row "
                            << img_data.rows
                            << " channel:" << img_data.channels();

                output2_bufs->At(i)->Set("width", gcols);
                output2_bufs->At(i)->Set("height", grows);
                output2_bufs->At(i)->Set("channel", gchannels);

                auto *output2_data = output2_bufs->MutableBufferData(i);
                memcpy_s(output2_data, output2_bufs->At(i)->GetBytes(),
                         img_data.data, img_data.total() * img_data.elemSize());
              }

              MBLOG_INFO << "finsish test_0_1_draw_bbox";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_0_1_draw_bbox", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("test_1_0_draw_bbox");
    desc_flowunit.SetDescription("the test in 1 out 0");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) +
        "/libmodelbox-unit-cpu-test_1_0_draw_bbox.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("test_1_0_draw_bbox");
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
              MBLOG_INFO << "test_1_0_draw_bbox process";
              auto input = op_ctx->Input("In_1");

              for (size_t i = 0; i < input->Size(); i++) {
                int32_t width = 0;
                int32_t height = 0;
                int32_t channel = 0;
                input->At(i)->Get("width", width);
                input->At(i)->Get("height", height);
                input->At(i)->Get("channel", channel);

                MBLOG_DEBUG << "w:" << width << ",h:" << height
                            << ",c:" << channel;
                MBLOG_DEBUG << input->At(i)->GetBytes();

                cv::Mat img_data(height, width, CV_8UC3);
                memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                         input->ConstBufferData(i), input->At(i)->GetBytes());

                std::string name = std::string(TEST_DATA_DIR) + "/test" +
                                   std::to_string(i) + ".jpg";
                MBLOG_DEBUG << name;
                cv::imwrite(name, img_data);
              }
              MBLOG_INFO << "finish test_1_0_draw_bbox process";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("test_1_0_draw_bbox", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }
  return STATUS_OK;
}

TEST_F(DrawBBoxFlowUnitTest, InitUnit) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                            
          test_0_1_draw_bbox[type=flowunit, flowunit=test_0_1_draw_bbox, device=cpu, deviceid=0, label="<Out_1> | <Out_2>", batch_size=5]
          draw_bbox[type=flowunit, flowunit=draw_bbox, device=cpu, deviceid=0, label="<in_image> | <in_region> | <out_image>", batch_size=5]
          test_1_0_draw_bbox[type=flowunit, flowunit=test_1_0_draw_bbox, device=cpu, deviceid=0, label="<In_1>", batch_size=5]      

          test_0_1_draw_bbox:Out_1 -> draw_bbox:in_region 
          test_0_1_draw_bbox:Out_2 -> draw_bbox:in_image
          draw_bbox:out_image -> test_1_0_draw_bbox:In_1                                                                      
        }'''
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("InitUnit", toml_content, 3 * 1000);
  EXPECT_EQ(ret, STATUS_STOP);

  for (size_t i = 0; i < 5; ++i) {
    std::string name =
        std::string(TEST_DATA_DIR) + "/test" + std::to_string(i) + ".jpg";
    auto rmret = remove(name.c_str());
    EXPECT_EQ(rmret, 0);
  }
}

}  // namespace modelbox