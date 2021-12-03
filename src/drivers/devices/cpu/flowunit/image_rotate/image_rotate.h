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

#ifndef MODELBOX_FLOWUNIT_IMAGE_ROTATE_CPU_H_
#define MODELBOX_FLOWUNIT_IMAGE_ROTATE_CPU_H_

#include "image_rotate_base.h"
#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "image_rotate";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: An OpenCV rotate flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer type is image file binary, the "
    "output port buffer type are image. \n"
    "\t  The image type buffer contains the following meta fields:\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: rotate_angle,  Type: int32_t\n"
    "\t\tField Name: channel,       Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint:";

class ImageRotateCpuFlowUnit : public ImageRotateFlowUnitBase {
 public:
  modelbox::Status RotateOneImage(
      std::shared_ptr<modelbox::Buffer> input_buffer,
      std::shared_ptr<modelbox::Buffer> output_buffer, int32_t rotate_angle,
      int32_t width, int32_t height) override;

  std::map<int32_t, cv::RotateFlags> rotate_code_{
      {90, cv::ROTATE_90_CLOCKWISE},
      {180, cv::ROTATE_180},
      {270, cv::ROTATE_90_COUNTERCLOCKWISE}};
};

#endif  // MODELBOX_FLOWUNIT_IMAGE_ROTATE_CPU_H_
