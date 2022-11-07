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

#ifndef MODELBOX_FLOWUNIT_ROCKCHIP_CROP_H_
#define MODELBOX_FLOWUNIT_ROCKCHIP_CROP_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_TYPE = "rockchip";
constexpr const char *FLOWUNIT_NAME = "crop";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A crop flowunit on rockchip device. \n"
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
    "[nv12, rgb, bgr], 'layout': [hwc]. One image can only be cropped with "
    "one ";

constexpr const char *IN_IMG = "in_image";
constexpr const char *OUT_IMG = "out_image";
constexpr const char *IN_REGION = "in_region";

class RockchipCropFlowUnit : public modelbox::FlowUnit {
 public:
  RockchipCropFlowUnit();
  ~RockchipCropFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::shared_ptr<modelbox::Buffer> ProcessOneImage(
      const std::shared_ptr<modelbox::Buffer> &in_img, const im_rect &region);
};

#endif  // MODELBOX_FLOWUNIT_ROCKCHIP_CROP_H_
