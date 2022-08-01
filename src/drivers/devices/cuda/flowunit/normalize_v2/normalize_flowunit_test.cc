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
class NormalizeV2FlowUnitTest : public testing::Test {
 public:
  NormalizeV2FlowUnitTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}

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
  };

  void TearDown() override { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> NormalizeV2FlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status NormalizeV2FlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();
  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("copy");
    desc_flowunit.SetDescription("just copy data flowunit on CPU");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-copy.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("copy");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("input"));
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("output"));
    mock_flowunit_desc->SetFlowType(modelbox::NORMAL);
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
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              auto input = op_ctx->Input("input");
              auto output = op_ctx->Output("output");

              for (size_t i = 0; i < input->Size(); ++i) {
                output->PushBack(input->At(i));
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));

    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("copy", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}

TEST_F(NormalizeV2FlowUnitTest, NormalizeV2Test) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input1[type=input]   
          normalize_v2[type=flowunit, flowunit=image_preprocess, device=cuda, deviceid=0, label="<in_image> | <out_data>", output_layout="hwc", mean="0.0, 0.0, 0.0", standard_deviation_inverse="1.0, 1.0, 1.0"]
          normalize_v2_chw[type=flowunit, flowunit=image_preprocess, device=cuda, deviceid=0, label="<in_image> | <out_data>", output_layout="chw", mean="0.0, 0.0, 0.0", standard_deviation_inverse="1.0, 1.0, 1.0"]
          copy[type=flowunit, flowunit=copy, device=cpu, deviceid=0, label="<input> | <output>"]
          copy_chw[type=flowunit, flowunit=copy, device=cpu, deviceid=0, label="<input> | <output>"]
          output_hwc[type=output]
          output_chw[type=output]      

          input1 -> normalize_v2:in_image
          input1 -> normalize_v2_chw:in_image
          normalize_v2:out_data -> copy:input
          normalize_v2_chw:out_data -> copy_chw:input
          copy:output -> output_hwc
          copy_chw:output -> output_chw
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("InitUnit", toml_content, -1);
  auto flow = driver_flow->GetFlow();

  auto ext_data = flow->CreateExternalDataMap();

  {
    std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";
    cv::Mat bgr_img = cv::imread(gimg_path);

    cv::Mat bgr_img_float, bgr_img_float_chw;
    bgr_img.convertTo(bgr_img_float, CV_32FC3);
    bgr_img.convertTo(bgr_img_float_chw, CV_32FC3);

    int height = bgr_img_float.rows;
    int width = bgr_img_float.cols;
    int channel = bgr_img_float.channels();

    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int dstIdx = c * height * width + h * width + w;
          int srcIdx = h * width * channel + w * channel + c;
          *((float*)((void*)bgr_img_float_chw.data) + dstIdx) =
              *(((float*)((void*)bgr_img_float.data)) + srcIdx);
        }
      }
    }

    std::vector<std::string> output_name({"output_hwc", "output_chw"});
    std::vector<cv::Mat> opencv_out_check({bgr_img_float, bgr_img_float_chw});

    auto color_bl = ext_data->CreateBufferList();
    size_t img_size = bgr_img.total() * bgr_img.elemSize();
    color_bl->BuildFromHost({img_size}, bgr_img.data, img_size);
    // HWC
    color_bl->Set("shape", std::vector<size_t>(
                               {static_cast<size_t>(bgr_img.rows),
                                static_cast<size_t>(bgr_img.cols),
                                static_cast<size_t>(bgr_img.channels())}));
    color_bl->Set("layout", std::string("hwc"));
    color_bl->Set("type", ModelBoxDataType::MODELBOX_UINT8);
    color_bl->Set("pix_fmt", "bgr");

    auto status = ext_data->Send("input1", color_bl);
    EXPECT_EQ(status, STATUS_OK);

    OutputBufferList map_buffer_list;

    status = ext_data->Recv(map_buffer_list);
    EXPECT_EQ(status, STATUS_OK);

    for (size_t j = 0; j < output_name.size(); j++) {
      auto buffer_list = map_buffer_list[output_name[j]];
      EXPECT_EQ(buffer_list->Size(), 1);
      EXPECT_EQ(buffer_list->GetBytes(),
                opencv_out_check[j].total() * opencv_out_check[j].elemSize());
      ModelBoxDataType type = MODELBOX_TYPE_INVALID;
      buffer_list->At(0)->Get("type", type);
      EXPECT_EQ(type, ModelBoxDataType::MODELBOX_FLOAT);
      auto* opencv_data = (float*)opencv_out_check[j].data;
      const auto* out_data = (float*)(buffer_list->ConstBufferData(0));
      size_t count = buffer_list->GetBytes() / sizeof(float);
      for (size_t k = 0; k < count; ++k) {
        EXPECT_TRUE(*(out_data + k) - *(opencv_data + k) < 0.00000001);
        EXPECT_TRUE(*(opencv_data + k) - *(out_data + k) < 0.00000001);
      }
    }
  }

  MBLOG_INFO << "Send Shutdown";
  auto status = ext_data->Shutdown();
  EXPECT_EQ(status, STATUS_OK);

  flow->Wait(1000);
}  // namespace modelbox

}  // namespace modelbox