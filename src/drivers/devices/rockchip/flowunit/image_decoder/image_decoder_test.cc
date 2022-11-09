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

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/crypto.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "modelbox/device/rockchip/rockchip_api.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class RockchipImageDecoderFlowUnitTest : public testing::Test {
 public:
  RockchipImageDecoderFlowUnitTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()),
        jpeg_decode_(std::make_shared<modelbox::MppJpegDecode>()) {}

 protected:
  void SetUp() override {
    auto ret = jpeg_decode_->Init();
    if (ret != modelbox::STATUS_OK) {
      MBLOG_INFO << "no rockchip device, skip test suit";
      GTEST_SKIP();
    }

    MBLOG_INFO << "jpeg_decode:" << ret;
  }

  void TearDown() override { driver_flow_ = nullptr; };

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<DriverFlowTest> driver_flow_;
  std::shared_ptr<modelbox::MppJpegDecode> jpeg_decode_;
};

std::shared_ptr<DriverFlowTest>
RockchipImageDecoderFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(RockchipImageDecoderFlowUnitTest, DecodeTest) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          image_decoder[type=flowunit, flowunit=image_decoder, device=rockchip, deviceid=0, batch_size=3]
          input -> image_decoder:in_encoded_image
          image_decoder:out_image -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("DecodeTest", toml_content, -1);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  MBLOG_INFO << toml_content;

  auto in_file_name = std::string(TEST_ASSETS) + "/test.jpg";

  // load file
  FILE *fp_in_image = fopen(in_file_name.c_str(), "rb");
  ASSERT_NE(fp_in_image, nullptr);

  struct stat in_image_statbuf = {0};
  stat(in_file_name.c_str(), &in_image_statbuf);
  EXPECT_EQ(in_image_statbuf.st_size > 0, true);

  auto img = cv::imread(in_file_name);
  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto in_img_buffer_list = extern_data->CreateBufferList();
  in_img_buffer_list->Build({(size_t)in_image_statbuf.st_size});
  auto in_img_buffer = in_img_buffer_list->At(0);

  auto in_image_size = fread(in_img_buffer->MutableData(), 1,
                             in_image_statbuf.st_size, fp_in_image);

  EXPECT_EQ(in_image_size, in_image_statbuf.st_size);
  fclose(fp_in_image);

  in_img_buffer->Set("pix_fmt", std::string("bgr"));

  int32_t total_out_size = img.cols * img.rows * 3;

  auto status = extern_data->Send("input", in_img_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  // check output
  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);

  ASSERT_EQ(output_buffer->GetBytes(), total_out_size);

  auto *mpp_buffer = (MppBuffer)output_buffer->ConstData();

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
  ASSERT_EQ(out_width, img.cols);
  ASSERT_EQ(out_height, img.rows);
  ASSERT_EQ(out_pix_fmt, std::string("bgr"));
  ASSERT_EQ(out_width_stride, img.cols * 3);
  ASSERT_EQ(out_height_stride, img.rows);

  std::shared_ptr<unsigned char> out_img_buf(
      new (std::nothrow) unsigned char[total_out_size],
      std::default_delete<unsigned char[]>());
  auto e_ret = memset_s(out_img_buf.get(), total_out_size, 0, total_out_size);
  EXPECT_EQ(e_ret, EOK);

  auto *rgbsrc = (uint8_t *)mpp_buffer_get_ptr(mpp_buffer);
  auto *rgbdst = (uint8_t *)out_img_buf.get();

  // copy to memory
  for (int i = 0; i < out_height; i++) {
    e_ret = memcpy_s(rgbdst, out_width * 3, rgbsrc, out_width * 3);
    EXPECT_EQ(e_ret, 0);
    rgbsrc += out_width * 3;
    rgbdst += out_width * 3;
  }

  auto out_file_name = std::string(TEST_ASSETS) + "/rockchip_decoder_test_bgr";
  FILE *fp_out_image = fopen(out_file_name.c_str(), "rb");
  ASSERT_NE(fp_out_image, nullptr);

  struct stat out_image_statbuf = {0};
  stat(out_file_name.c_str(), &out_image_statbuf);
  EXPECT_EQ(out_image_statbuf.st_size, total_out_size);

  std::shared_ptr<unsigned char> out_img_file_buf(
      new (std::nothrow) unsigned char[total_out_size],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_img_file_buf.get(), total_out_size, 0, total_out_size);
  EXPECT_EQ(e_ret, 0);

  auto out_image_size =
      fread(out_img_file_buf.get(), 1, total_out_size, fp_out_image);
  EXPECT_EQ(out_image_size, total_out_size);
  fclose(fp_out_image);

  // cmp memory
  EXPECT_EQ(memcmp(out_img_buf.get(), out_img_file_buf.get(), total_out_size),
            0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(RockchipImageDecoderFlowUnitTest, DecodeBase64Test) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          base64_decoder[type=flowunit, flowunit=base64_decoder, device=cpu, deviceid=0, batch_size=3, data_format=json, key=image_base64]
          image_decoder[type=flowunit, flowunit=image_decoder, device=rockchip, deviceid=0, batch_size=3, key=image_base64, pix_fmt=bgr]
          input -> base64_decoder:in_data
          base64_decoder:out_data -> image_decoder:in_encoded_image
          image_decoder:out_image -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("DecodeTest", toml_content, 10);

  MBLOG_INFO << toml_content;

  auto in_file_name = std::string(TEST_ASSETS) + "/test.jpg";

  // load file
  FILE *fp_in_image = fopen(in_file_name.c_str(), "rb");
  ASSERT_NE(fp_in_image, nullptr);

  struct stat in_image_statbuf = {0};
  stat(in_file_name.c_str(), &in_image_statbuf);
  EXPECT_EQ(in_image_statbuf.st_size > 0, true);

  auto img = cv::imread(in_file_name);
  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto in_img_buffer_list = extern_data->CreateBufferList();

  std::shared_ptr<unsigned char> in_img_file_buf(
      new (std::nothrow) unsigned char[in_image_statbuf.st_size],
      std::default_delete<unsigned char[]>());
  auto e_ret = memset_s(in_img_file_buf.get(), in_image_statbuf.st_size, 0,
                        in_image_statbuf.st_size);
  EXPECT_EQ(e_ret, EOK);

  auto in_image_size =
      fread(in_img_file_buf.get(), 1, in_image_statbuf.st_size, fp_in_image);
  EXPECT_EQ(in_image_size, in_image_statbuf.st_size);
  fclose(fp_in_image);

  std::string base64_image;
  auto en_ret = modelbox::Base64Encode(in_img_file_buf.get(),
                                       in_image_statbuf.st_size, &base64_image);
  EXPECT_EQ(en_ret, STATUS_OK);

  nlohmann::json base64_image_json;
  base64_image_json["image_base64"] = base64_image;
  std::string base64_image_json_str = base64_image_json.dump();

  in_img_buffer_list->Build({(size_t)base64_image_json_str.size()});
  auto in_img_buffer = in_img_buffer_list->At(0);

  in_img_buffer->Set("pix_fmt", std::string("bgr"));
  in_img_buffer->Set("key", std::string("image_base64"));

  e_ret = memcpy_s(in_img_buffer->MutableData(), in_img_buffer->GetBytes(),
                   base64_image_json_str.c_str(), base64_image_json_str.size());
  EXPECT_EQ(e_ret, EOK);

  int32_t total_out_size = img.cols * img.rows * 3;

  auto status = extern_data->Send("input", in_img_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  // check output
  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);

  ASSERT_EQ(output_buffer->GetBytes(), total_out_size);

  auto *mpp_buffer = (MppBuffer)output_buffer->ConstData();

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
  ASSERT_EQ(out_width, img.cols);
  ASSERT_EQ(out_height, img.rows);
  ASSERT_EQ(out_pix_fmt, std::string("bgr"));
  ASSERT_EQ(out_width_stride, img.cols * 3);
  ASSERT_EQ(out_height_stride, img.rows);

  std::shared_ptr<unsigned char> out_img_buf(
      new (std::nothrow) unsigned char[total_out_size],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_img_buf.get(), total_out_size, 0, total_out_size);
  EXPECT_EQ(e_ret, EOK);

  auto *rgbsrc = (uint8_t *)mpp_buffer_get_ptr(mpp_buffer);
  auto *rgbdst = (uint8_t *)out_img_buf.get();

  // copy to memory
  for (int i = 0; i < out_height; i++) {
    e_ret = memcpy_s(rgbdst, out_width * 3, rgbsrc, out_width * 3);
    EXPECT_EQ(e_ret, 0);
    rgbsrc += out_width * 3;
    rgbdst += out_width * 3;
  }

  auto out_file_name = std::string(TEST_ASSETS) + "/rockchip_decoder_test_bgr";
  FILE *fp_out_image = fopen(out_file_name.c_str(), "rb");
  ASSERT_NE(fp_out_image, nullptr);

  struct stat out_image_statbuf = {0};
  stat(out_file_name.c_str(), &out_image_statbuf);
  EXPECT_EQ(out_image_statbuf.st_size, total_out_size);

  std::shared_ptr<unsigned char> out_img_file_buf(
      new (std::nothrow) unsigned char[total_out_size],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_img_file_buf.get(), total_out_size, 0, total_out_size);
  EXPECT_EQ(e_ret, 0);

  auto out_image_size =
      fread(out_img_file_buf.get(), 1, total_out_size, fp_out_image);
  EXPECT_EQ(out_image_size, total_out_size);
  fclose(fp_out_image);

  // cmp memory
  EXPECT_EQ(memcmp(out_img_buf.get(), out_img_file_buf.get(), total_out_size),
            0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox