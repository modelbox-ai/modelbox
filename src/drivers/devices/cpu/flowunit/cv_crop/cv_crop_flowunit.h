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

#ifndef MODELBOX_FLOWUNIT_CV_CROP_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_CV_CROP_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "crop";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: An OpenCV crop flowunit on cpu. \n"
    "\t@Port parameter: The input port 'in_image' and the output port "
    "'out_image' buffer type are image. \n"
    "\t  The image type buffer contain the following meta fields:\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: channel,       Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t  The other input port 'in_region' buffer type is rectangle, the memory "
    "arrangement is [x,y,w,h].\n"
    "\t  it contain the following meta fields: \n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint: The field value range of this flowunit support: 'pix_fmt': "
    "[rgb_packed,bgr_packed], 'layout': [hwc]. One image can only be cropped "
    "with one "
    "rectangle and output one crop image.";
const int RGB_CHANNLES = 3;

class CVCropFlowUnit : public modelbox::FlowUnit {
 public:
  CVCropFlowUnit();
  virtual ~CVCropFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);
};

#endif  // MODELBOX_FLOWUNIT_CV_CROP_FLOWUNIT_CPU_H_
