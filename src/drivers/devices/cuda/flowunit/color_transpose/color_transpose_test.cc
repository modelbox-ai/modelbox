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
#include <opencv2/imgproc/types_c.h>
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
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {
class ColorTransposeFlowUnitTest : public testing::Test {
 public:
  ColorTransposeFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    driver_flow_->Init(false);
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> ColorTransposeFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status ColorTransposeFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc = GenerateFlowunitDesc("copy", {"input"}, {"output"});
    auto process_func =
        [=](std::shared_ptr<DataContext> op_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto input = op_ctx->Input("input");
      auto output = op_ctx->Output("output");
      for (size_t i = 0; i < input->Size(); ++i) {
        output->PushBack(input->At(i));
      }
      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }
  return STATUS_OK;
}

TEST_F(ColorTransposeFlowUnitTest, ColorTransposeTest) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input1[type=input]   
          color_transpose_gray[type=flowunit, flowunit=color_convert, device=cuda deviceid=0, label="<in_image> | <out_image>", out_pix_fmt="gray"]
          color_transpose_rgb[type=flowunit, flowunit=color_convert, device=cuda deviceid=0, label="<in_image> | <out_image>", out_pix_fmt="rgb"]
          color_transpose_bgr[type=flowunit, flowunit=color_convert, device=cuda deviceid=0, label="<in_image> | <out_image>", out_pix_fmt="bgr"]
          copy_gray[type=flowunit, flowunit=copy, device=cpu, deviceid=0, label="<input> | <output>"]
          copy_rgb[type=flowunit, flowunit=copy, device=cpu, deviceid=0, label="<input> | <output>"]
          copy_bgr[type=flowunit, flowunit=copy, device=cpu, deviceid=0, label="<input> | <output>"]
          output_gray[type=output]
          output_rgb[type=output]
          output_bgr[type=output]         

          input1 -> color_transpose_gray:in_image
          input1 -> color_transpose_rgb:in_image
          input1 -> color_transpose_bgr:in_image
          color_transpose_gray:out_image -> copy_gray:input
          color_transpose_rgb:out_image -> copy_rgb:input
          color_transpose_bgr:out_image -> copy_bgr:input
          copy_gray:output -> output_gray
          copy_rgb:output -> output_rgb
          copy_bgr:output -> output_bgr
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("ColorTransposeTest", toml_content, -1);
  auto flow = driver_flow->GetFlow();

  {
    std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";
    cv::Mat bgr_img, gray_img, rgb_img;
    bgr_img = cv::imread(gimg_path);

    cv::cvtColor(bgr_img, rgb_img, CV_BGR2RGB);
    cv::cvtColor(rgb_img, gray_img, CV_RGB2GRAY);

    auto ext_data = flow->CreateExternalDataMap();
    GTEST_ASSERT_NE(ext_data, nullptr);

    std::vector<std::string> pix_fmt_list({"bgr", "rgb", "gray"});
    std::vector<cv::Mat> img_list({bgr_img, rgb_img, gray_img});
    std::vector<std::string> output_name(
        {"output_bgr", "output_rgb", "output_gray"});
    for (size_t i = 0; i < pix_fmt_list.size(); ++i) {
      // TODO don't skip GRAY
      if (i == 2) {
        break;
      }

      auto color_bl = ext_data->CreateBufferList();
      size_t img_size = img_list[i].total() * img_list[i].elemSize();
      color_bl->BuildFromHost({img_size}, img_list[i].data, img_size);
      // HWC
      color_bl->Set(
          "shape",
          std::vector<size_t>({static_cast<size_t>(img_list[i].rows),
                               static_cast<size_t>(img_list[i].cols),
                               static_cast<size_t>(img_list[i].channels())}));
      color_bl->Set("layout", std::string("hwc"));
      color_bl->Set("type", ModelBoxDataType::MODELBOX_UINT8);
      color_bl->Set("pix_fmt", pix_fmt_list[i]);

      auto status = ext_data->Send("input1", color_bl);
      EXPECT_EQ(status, STATUS_OK);

      OutputBufferList map_buffer_list;

      status = ext_data->Recv(map_buffer_list);
      EXPECT_EQ(status, STATUS_OK);

      auto host_device = color_bl->GetDevice();

      for (size_t j = 0; j < output_name.size(); j++) {
        auto buffer_list = map_buffer_list[output_name[j]];
        EXPECT_EQ(buffer_list->Size(), 1);
        EXPECT_EQ(buffer_list->GetBytes(),
                  img_list[j].total() * img_list[j].elemSize());
        std::string out_pix_fmt;
        buffer_list->At(0)->Get("pix_fmt", out_pix_fmt);
        EXPECT_EQ(out_pix_fmt, pix_fmt_list[j]);
        auto* opencv_data = (uint8_t*)img_list[j].data;

        auto* out_data = (uint8_t*)(buffer_list->ConstBufferData(0));
        for (size_t k = 0; k < buffer_list->GetBytes(); ++k) {
          // TODO don't skip GRAY
          if (j == 2) {
            break;
          }

          EXPECT_EQ(*(out_data + k), *(opencv_data + k));
        }
      }
    }

    auto status = ext_data->Shutdown();
    EXPECT_EQ(status, STATUS_OK);
  }

  flow->Wait(1000);
}

}  // namespace modelbox