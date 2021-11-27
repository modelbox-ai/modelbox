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

#include <opencv2/opencv.hpp>

#include "image_rotate_test_base.h"

namespace modelbox {

TEST_F(ImageRotateFlowUnitTest, CudaRotateTest) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          test_0_1_rotate[type=flowunit, flowunit=test_0_1_rotate, device=cpu, deviceid=0, label="<out_1>"]
          image_rotate[type=flowunit, flowunit=image_rotate, device=cuda, deviceid=0, label="<in_encoded_image> | <out_image>", batch_size=3]
          test_1_0_rotate[type=flowunit, flowunit=test_1_0_rotate, device=cpu, deviceid=0, label="<in_1>",batch_size=3]                                
          test_0_1_rotate:out_1 -> image_rotate:in_origin_image 
          image_rotate:out_rotate_image -> test_1_0_rotate:in_1                                                                      
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("CudaRotateTest", toml_content);
  EXPECT_EQ(ret, STATUS_STOP);

  for (auto rotate_angle : test_rotate_angle_) {
    std::string expected_file_path = std::string(TEST_ASSETS) + "/rotate_" +
                                     std::to_string(rotate_angle) + ".jpg";
    cv::Mat expected_img = cv::imread(expected_file_path);

    std::string rotate_result_file_path = std::string(TEST_DATA_DIR) +
                                          "/rotate_result_" +
                                          std::to_string(rotate_angle) + ".jpg";
    cv::Mat rotate_result_img = cv::imread(rotate_result_file_path);

    int result_data_size =
        rotate_result_img.total() * rotate_result_img.elemSize();
    int expected_data_size = expected_img.total() * expected_img.elemSize();
    EXPECT_EQ(result_data_size, expected_data_size);

    auto cmp_ret =
        memcmp(rotate_result_img.data, expected_img.data, result_data_size);
    EXPECT_EQ(cmp_ret, 0);

    auto rmret = remove(rotate_result_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  }
}

}  // namespace modelbox