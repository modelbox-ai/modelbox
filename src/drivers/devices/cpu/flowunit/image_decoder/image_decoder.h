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

#ifndef MODELBOX_FLOWUNIT_HTTPSERVER_CPU_H_
#define MODELBOX_FLOWUNIT_HTTPSERVER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "image_decoder";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: An OpenCV crop flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer type is image file binary, the "
    "output port buffer type are image. \n"
    "\t  The image type buffer contains the following meta fields:\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: channel,       Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint:";

class ImageDecoderFlowUnit : public modelbox::FlowUnit {
 public:
  ImageDecoderFlowUnit();
  ~ImageDecoderFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  cv::Mat BGR2YUV_NV12(const cv::Mat &src_bgr);

  std::string pixel_format_{"bgr"};
};

#endif  // MODELBOX_FLOWUNIT_HTTPSERVER_CPU_H_
