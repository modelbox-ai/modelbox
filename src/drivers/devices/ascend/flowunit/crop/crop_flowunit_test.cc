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

#include <acl/acl_rt.h>
#include <dsmi_common_interface.h>
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
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class CropFlowUnitTest : public testing::Test {
 public:
  CropFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    // Test ascend runtime
    int32_t count = 0;
    auto dsmi_ret = dsmi_get_device_count(&count);
    if (dsmi_ret != 0) {
      MBLOG_INFO << "no ascend device, skip test suit";
      GTEST_SKIP();
    }
  }

  void TearDown() override {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    device_mgr->Clear();
    drivers->Clear();

    driver_flow_ = nullptr;
  };

  std::shared_ptr<MockFlow> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> CropFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(CropFlowUnitTest, RunUnit) {
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
          crop[type=flowunit, flowunit=crop, device=ascend, deviceid=0]

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
  auto data_ptr = (int32_t*)in_box_buffer->MutableData();
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
  cv::Mat yuv_out_img(128 * 3 / 2, 128, CV_8UC1);
  auto acl_ret = aclrtSetDevice(0);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
  acl_ret = aclrtMemcpy(yuv_out_img.data, output_buffer->GetBytes(),
                        output_buffer->ConstData(), output_buffer->GetBytes(),
                        aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);

  auto image_size = yuv_out_img.rows * yuv_out_img.cols * yuv_out_img.elemSize();
  char expected_img[image_size];
  std::ifstream infile;
  infile.open(std::string(TEST_ASSETS) + "/ascend_crop_yuv");
  infile.read((char*)expected_img, image_size);

  EXPECT_EQ(memcmp((char*)yuv_out_img.data, expected_img, image_size), 0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox