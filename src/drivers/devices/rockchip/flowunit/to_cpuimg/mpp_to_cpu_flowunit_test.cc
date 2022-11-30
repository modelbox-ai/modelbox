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

#include <fstream>
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
#include "modelbox/device/rockchip/rockchip_api.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class RockchipMppToCpuFlowUnitTest : public testing::Test {
 public:
  RockchipMppToCpuFlowUnitTest()
      : driver_flow_(std::make_shared<MockFlow>()),
        jpeg_decode_(std::make_shared<modelbox::MppJpegDecode>()) {}

 protected:
  void SetUp() override {
    // Test rockchip runtime
    auto ret = jpeg_decode_->Init();
    if (ret != modelbox::STATUS_OK) {
      MBLOG_INFO << "no rockchip device, skip test suit";
      GTEST_SKIP();
    }
  }

  void TearDown() override { driver_flow_ = nullptr; };

  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

  std::shared_ptr<modelbox::MppJpegDecode> GetJpegDecode() {
    return jpeg_decode_;
  }

 private:
  std::shared_ptr<MockFlow> driver_flow_;
  std::shared_ptr<modelbox::MppJpegDecode> jpeg_decode_;
};

std::shared_ptr<MockFlow> RockchipMppToCpuFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(RockchipMppToCpuFlowUnitTest, RunUnit) {
  std::map<int, int> size_map = {{112, 110}, {160, 120}, {640, 480}};
  for (auto &it : size_map) {
    std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                               test_data_dir + "\"]\n" +
                               R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          resize[type=flowunit, flowunit=resize, device=rockchip, deviceid=0, image_width=)" +
                               std::to_string(it.first) +
                               ", image_height=" + std::to_string(it.second) +
                               R"(]
          rk_cpuimg[type=flowunit, flowunit=rk_cpuimg, device=cpu, deviceid=0]
          input -> resize:in_image
          resize:out_image -> rk_cpuimg:in_image
          rk_cpuimg:out_image -> output
        }'''
    format = "graphviz"
  )";

    MBLOG_INFO << toml_content;

    auto driver_flow = GetDriverFlow();
    driver_flow->BuildAndRun("RunUnit", toml_content, 10);

    int w = 0;
    int h = 0;

    struct stat statbuf = {0};
    stat((std::string(TEST_ASSETS) + "/test.jpg").c_str(), &statbuf);
    FILE *fp_jpg =
        fopen((std::string(TEST_ASSETS) + "/test.jpg").c_str(), "rb");
    EXPECT_EQ(fp_jpg == nullptr, false);
    std::shared_ptr<unsigned char> img_buf(
        new (std::nothrow) unsigned char[statbuf.st_size + 1],
        std::default_delete<unsigned char[]>());
    EXPECT_EQ(img_buf == nullptr, false);
    auto s_ret =
        memset_s(img_buf.get(), statbuf.st_size + 1, 0, statbuf.st_size + 1);
    EXPECT_EQ(s_ret, EOK);
    auto jpg_size = fread(img_buf.get(), 1, statbuf.st_size, fp_jpg);
    EXPECT_EQ(jpg_size, statbuf.st_size);
    fclose(fp_jpg);
    MppFrame frame = GetJpegDecode()->Decode(img_buf.get(), jpg_size, w, h);
    EXPECT_EQ(frame == nullptr, false);
    EXPECT_EQ(w, 400);
    EXPECT_EQ(h, 300);

    auto img = cv::imread(std::string(TEST_ASSETS) + "/test.jpg");
    auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
    auto in_img_buffer_list = extern_data->CreateBufferList();
    in_img_buffer_list->Build({img.total() * img.elemSize()});
    auto in_img_buffer = in_img_buffer_list->At(0);
    in_img_buffer->Set("width", img.cols);
    in_img_buffer->Set("height", img.rows);
    in_img_buffer->Set("width_stride", img.cols * 3);
    in_img_buffer->Set("height_stride", img.rows);
    in_img_buffer->Set("pix_fmt", std::string("bgr"));
    auto e_ret =
        memcpy_s(in_img_buffer->MutableData(), in_img_buffer->GetBytes(),
                 img.data, img.total() * img.elemSize());
    EXPECT_EQ(e_ret, 0);
    auto status = extern_data->Send("input", in_img_buffer_list);
    EXPECT_EQ(status, STATUS_OK);
    // check output
    OutputBufferList map_buffer_list;
    status = extern_data->Recv(map_buffer_list);
    EXPECT_EQ(status, STATUS_OK);
    auto output_buffer_list = map_buffer_list["output"];
    ASSERT_EQ(output_buffer_list->Size(), 1);
    auto output_buffer = output_buffer_list->At(0);
    ASSERT_EQ(output_buffer->GetBytes(), it.first * it.second * 3);

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
    ASSERT_EQ(out_width, it.first);
    ASSERT_EQ(out_height, it.second);
    ASSERT_EQ(out_pix_fmt, std::string("bgr"));
    ASSERT_EQ(out_width_stride, it.first * 3);
    ASSERT_EQ(out_height_stride, it.second);

    int32_t total_out_size = it.first * it.second * 3;
    std::shared_ptr<unsigned char> out_img_buf(
        new (std::nothrow) unsigned char[total_out_size],
        std::default_delete<unsigned char[]>());
    e_ret = memset_s(out_img_buf.get(), total_out_size, 0, total_out_size);
    EXPECT_EQ(e_ret, 0);

    // copy to memory
    e_ret = memcpy_s(out_img_buf.get(), output_buffer->GetBytes(),
                     output_buffer->ConstData(), output_buffer->GetBytes());
    EXPECT_EQ(e_ret, 0);

    std::string out_file_name = std::string(TEST_ASSETS) + "/rockchip_" +
                                std::to_string(it.first) + "x" +
                                std::to_string(it.second) + "_bgr";
    struct stat out_statbuf = {0};
    stat(out_file_name.c_str(), &out_statbuf);
    EXPECT_EQ(out_statbuf.st_size, total_out_size);

    // load file
    FILE *fp_out = fopen(out_file_name.c_str(), "rb");
    EXPECT_EQ(fp_out == nullptr, false);

    std::shared_ptr<unsigned char> out_file_img_buf(
        new (std::nothrow) unsigned char[out_statbuf.st_size],
        std::default_delete<unsigned char[]>());
    e_ret = memset_s(out_file_img_buf.get(), out_statbuf.st_size, 0,
                     out_statbuf.st_size);
    EXPECT_EQ(e_ret, 0);

    auto out_size =
        fread(out_file_img_buf.get(), 1, out_statbuf.st_size, fp_out);

    EXPECT_EQ(out_size, out_statbuf.st_size);
    fclose(fp_out);

    // cmp memory
    EXPECT_EQ(
        memcmp(out_img_buf.get(), out_file_img_buf.get(), out_statbuf.st_size),
        0);

    driver_flow->GetFlow()->Wait(3 * 1000);
  }
}

}  // namespace modelbox