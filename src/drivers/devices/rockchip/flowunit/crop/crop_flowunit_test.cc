/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#include <opencv2/opencv.hpp>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "modelbox/device/rockchip/rockchip_api.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class RockchipCropFlowUnitTest : public testing::Test {
 public:
  RockchipCropFlowUnitTest()
      : crop_driver_flow_(std::make_shared<DriverFlowTest>()),
        jpeg_decode_(std::make_shared<modelbox::MppJpegDecode>()) {}

 protected:
  virtual void SetUp() override {
    auto ret = jpeg_decode_->Init();
    if (ret != modelbox::STATUS_OK) {
      MBLOG_INFO << "no rockchip device, skip test suit";
      GTEST_SKIP();
    }

    MBLOG_INFO << "jpeg_decode:" << ret;
  }

  virtual void TearDown() override { crop_driver_flow_ = nullptr; };

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<DriverFlowTest> crop_driver_flow_;
  std::shared_ptr<modelbox::MppJpegDecode> jpeg_decode_;
};

std::shared_ptr<DriverFlowTest> RockchipCropFlowUnitTest::GetDriverFlow() {
  return crop_driver_flow_;
}

TEST_F(RockchipCropFlowUnitTest, RunUnit) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input1[type=input]
          input2[type=input]
          output[type=output]
          crop[type=flowunit, flowunit=crop, device=rockchip, deviceid=0]

          input1 -> crop:in_image
          input2 -> crop:in_region
          crop:out_image -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("RunUnit", toml_content, 10);

  auto img = cv::imread(std::string(TEST_ASSETS) + "/test.jpg");
  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  // in img
  auto in_img_buffer_list = extern_data->CreateBufferList();
  in_img_buffer_list->Build({img.total() * img.elemSize()});
  auto in_img_buffer = in_img_buffer_list->At(0);
  in_img_buffer->Set("width", img.cols);
  in_img_buffer->Set("height", img.rows);
  in_img_buffer->Set("width_stride", img.cols * 3);
  in_img_buffer->Set("height_stride", img.rows);
  in_img_buffer->Set("pix_fmt", std::string("bgr"));
  auto e_ret = memcpy_s(in_img_buffer->MutableData(), in_img_buffer->GetBytes(),
                        img.data, img.total() * img.elemSize());
  EXPECT_EQ(e_ret, 0);
  auto status = extern_data->Send("input1", in_img_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  // in box
  auto in_box_buffer_list = extern_data->CreateBufferList();
  in_box_buffer_list->Build({sizeof(int32_t) * 4});
  auto in_box_buffer = in_box_buffer_list->At(0);
  auto *data_ptr = (int32_t *)in_box_buffer->MutableData();
  data_ptr[0] = 30;
  data_ptr[1] = 0;
  data_ptr[2] = 128;
  data_ptr[3] = 128;
  status = extern_data->Send("input2", in_box_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  // check output
  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);

  int32_t out_width = 0;
  int32_t out_height = 0;
  int32_t out_width_stride = 0;
  int32_t out_height_stride = 0;
  std::string out_pix_fmt;
  output_buffer->Get("width", out_width);
  output_buffer->Get("height", out_height);
  output_buffer->Get("pix_fmt", out_pix_fmt);
  output_buffer->Get("width_stride", out_width_stride);
  output_buffer->Get("height_stride", out_height_stride);
  ASSERT_EQ(out_width, 128);
  ASSERT_EQ(out_height, 128);
  ASSERT_EQ(out_pix_fmt, std::string("bgr"));
  ASSERT_EQ(out_width_stride, 128 * 3);
  ASSERT_EQ(out_height_stride, 128);

  int32_t total_out_size = 128 * 128 * 3;
  std::shared_ptr<unsigned char> out_img_buf(
      new (std::nothrow) unsigned char[total_out_size],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_img_buf.get(), total_out_size, 0, total_out_size);
  EXPECT_EQ(e_ret, 0);

  auto *mpp_buffer = (MppBuffer)output_buffer->ConstData();

  auto *rgbsrc = (uint8_t *)mpp_buffer_get_ptr(mpp_buffer);
  auto *rgbdst = (uint8_t *)out_img_buf.get();

  // copy to memory
  for (int i = 0; i < out_height; i++) {
    e_ret = memcpy_s(rgbdst, out_width * 3, rgbsrc, out_width * 3);
    EXPECT_EQ(e_ret, 0);
    rgbsrc += out_width * 3;
    rgbdst += out_width * 3;
  }

  std::string out_file_name = std::string(TEST_ASSETS) + "/rockchip_crop_bgr";
  struct stat out_statbuf = {0};
  stat(out_file_name.c_str(), &out_statbuf);
  EXPECT_EQ(out_statbuf.st_size, total_out_size);

  FILE *fp_out = fopen(out_file_name.c_str(), "rb");
  ASSERT_NE(fp_out, nullptr);

  std::shared_ptr<unsigned char> out_file_img_buf(
      new (std::nothrow) unsigned char[out_statbuf.st_size],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_file_img_buf.get(), out_statbuf.st_size, 0,
                   out_statbuf.st_size);
  EXPECT_EQ(e_ret, 0);

  auto out_size = fread(out_file_img_buf.get(), 1, out_statbuf.st_size, fp_out);
  EXPECT_EQ(out_size, total_out_size);
  fclose(fp_out);

  // cmp memory
  EXPECT_EQ(
      memcmp(out_img_buf.get(), out_file_img_buf.get(), out_statbuf.st_size),
      0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox