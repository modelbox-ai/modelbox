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


#include "padding_flowunit.h"
#include <securec.h>

#include <functional>
#include <future>
#include <random>
#include <thread>
#include <opencv2/opencv.hpp>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
  
class PaddingFlowUnitTest : public testing::Test {
 public:
  PaddingFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}
 protected:
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> PaddingFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
};

TEST_F(PaddingFlowUnitTest, TestPaddingImage) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          padding[type=flowunit, flowunit=padding, device=cpu, deviceid=0, label="<in_image> | <out_image>",
          image_width=200, image_height=100, vertical_align=top, horizontal_align=center, padding_data="0, 255, 0"]

          input -> padding:in_image
          padding:out_image -> output
        }'''
    format = "graphviz"
  )";


  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("TestPaddingImage", toml_content, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  auto img = cv::imread(std::string(TEST_ASSETS) + "/test.jpg");
  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto input_buffer_list = extern_data->CreateBufferList();
  input_buffer_list->Build({img.total() * img.elemSize()});
  auto input_buffer = input_buffer_list->At(0);
  input_buffer->Set("width", img.cols);
  input_buffer->Set("height", img.rows);
  input_buffer->Set("pix_fmt", std::string("bgr"));
  auto e_ret = memcpy_s(input_buffer->MutableData(), input_buffer->GetBytes(),
                        img.data, img.total() * img.elemSize());
  EXPECT_EQ(e_ret, 0);
  auto status = extern_data->Send("input", input_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  cv::Mat out_img(cv::Size(200, 100), CV_8UC3, output_buffer->MutableData());
  auto expected_img = cv::imread(std::string(TEST_ASSETS) + "/padding_200x100_result.png");
  ASSERT_EQ(expected_img.cols, out_img.cols);
  ASSERT_EQ(expected_img.rows, out_img.rows );

  for (int32_t y = 0; y < expected_img.rows; ++y) {
    for (int32_t x = 0; x < expected_img.cols; ++x) {
      auto expected_pix = expected_img.at<cv::Vec3b>(y, x);
      auto pix = out_img.at<cv::Vec3b>(y, x);
      ASSERT_EQ(expected_pix[0], pix[0]);
      ASSERT_EQ(expected_pix[1], pix[1]);
      ASSERT_EQ(expected_pix[2], pix[2]);
    }
  }

  driver_flow->GetFlow()->Wait(3 * 1000);
}


}  // namespace modelbox
