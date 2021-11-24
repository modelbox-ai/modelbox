#/*
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

#include <acl/acl_rt.h>
#include <dsmi_common_interface.h>
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
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {

Status yuvI420ToNV12(uint8_t *in_data, int32_t w, int32_t h,
                     uint8_t *out_data) {
  int size = w * h;
  auto ret = memcpy_s(out_data, size, in_data, size);
  if (ret != 0) {
    MBLOG_ERROR << "copy Y data to out data failed, datasize = " << size;
    return modelbox::STATUS_FAULT;
  }
  for (int i = 0, j = 0; i < w * h / 4; i++, j += 2) {
    auto ret_u = memcpy_s(out_data + size + j, 1, in_data + size + i, 1);
    auto ret_v =
        memcpy_s(out_data + size + j + 1, 1, in_data + i + size * 5 / 4, 1);
    if (ret_u != 0 || ret_v != 0) {
      MBLOG_ERROR << "copy u/v data to out data failed";
      return modelbox::STATUS_FAULT;
    }
  }
  return modelbox::STATUS_SUCCESS;
}

class AscendPaddingFlowUnitTest : public testing::Test {
 public:
  AscendPaddingFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  virtual void SetUp() {
    // Test ascend runtime
    int32_t count = 0;
    auto dsmi_ret = dsmi_get_device_count(&count);
    if (dsmi_ret != 0) {
      MBLOG_INFO << "no ascend device, skip test suit";
      GTEST_SKIP();
    }
  }

  virtual void TearDown() { driver_flow_ = nullptr; };

  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> AscendPaddingFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(AscendPaddingFlowUnitTest, TestPaddingImage) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          padding[type=flowunit, flowunit=padding, device=ascend, deviceid=0, label="<in_image> | <out_image>",image_width=208, image_height=100,
          vertical_align=top, horizontal_align=center, padding_data = "255,255,0"]

          input -> padding:in_image
          padding:out_image -> output
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("TestPaddingImage", toml_content, 10);

  auto img = cv::imread(std::string(TEST_ASSETS) + "/test.jpg");
  cv::Mat I420data;
  cv::cvtColor(img, I420data, cv::COLOR_RGB2YUV_I420);
  int size = img.cols * img.rows * 3 / 2;
  cv::Mat nv12_data(img.cols * 3 / 2, img.rows, CV_8UC1);

  auto convert_status = yuvI420ToNV12(I420data.data, img.cols, img.rows, nv12_data.data);
  ASSERT_EQ(convert_status, modelbox::STATUS_SUCCESS);
  
  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();

  auto in_img_buffer_list = extern_data->CreateBufferList();
  in_img_buffer_list->Build({img.total() * img.elemSize() / 2});
  auto in_img_buffer = in_img_buffer_list->At(0);
  in_img_buffer->Set("width", img.cols);
  in_img_buffer->Set("height", img.rows);
  in_img_buffer->Set("pix_fmt", std::string("nv12"));
  auto e_ret = memcpy_s(in_img_buffer->MutableData(), in_img_buffer->GetBytes(),
                        nv12_data.data, size);
  EXPECT_EQ(e_ret, 0);

  auto status = extern_data->Send("input", in_img_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);

  auto output_buffer = output_buffer_list->At(0);
  ASSERT_EQ(output_buffer->GetBytes(), 208 * 100 * 3 / 2);

  cv::Mat yuv_out_img(100 * 3 / 2, 208, CV_8UC1);
  auto acl_ret = aclrtSetDevice(0);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);

  acl_ret = aclrtMemcpy(yuv_out_img.data, output_buffer->GetBytes(),
                        output_buffer->ConstData(), output_buffer->GetBytes(),
                        aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);

  cv::Mat bgr_out_img;
  cv::cvtColor(yuv_out_img, bgr_out_img, cv::COLOR_YUV2BGR_NV12);

  auto expected_img =
      cv::imread(std::string(TEST_ASSETS) + "/ascend_padding.png");
  ASSERT_EQ(expected_img.cols, bgr_out_img.cols);
  ASSERT_EQ(expected_img.rows, bgr_out_img.rows);

  for (int32_t y = 0; y < expected_img.rows; ++y) {
    for (int32_t x = 0; x < expected_img.cols; ++x) {
      auto expected_pix = expected_img.at<cv::Vec3b>(y, x);
      auto pix = bgr_out_img.at<cv::Vec3b>(y, x);
      ASSERT_EQ(expected_pix[0], pix[0]);
      ASSERT_EQ(expected_pix[1], pix[1]);
      ASSERT_EQ(expected_pix[2], pix[2]);
    }
  }

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox