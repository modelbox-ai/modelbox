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

#ifndef MODELBOX_FLOWUNIT_ROCKCHIP_MPP_TO_CPU_H_
#define MODELBOX_FLOWUNIT_ROCKCHIP_MPP_TO_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_NAME = "rk_cpuimg";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: transfer image from rockchip mpp to cpu image\n"
    "\t@Port parameter: The input port buffer type and the output port buffer "
    "type are image. \n"
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
    "\t@Constraint: The field value range of this flowunit supports: "
    "'pix_fmt': "
    "[rgb_packed,bgr_packed], 'layout': [hwc]. ";
	
constexpr const char *IN_IMG = "in_image";
constexpr const char *OUT_IMG = "out_image";

class MppToCpuFlowUnit : public modelbox::FlowUnit {
 public:
  MppToCpuFlowUnit();
  ~MppToCpuFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::shared_ptr<modelbox::Buffer> ProcessOneImage(
      const std::shared_ptr<modelbox::Buffer> &in_img, std::string &pix_fmt,
      int32_t w, int32_t h, int32_t ws, int32_t hs);
};

#endif  // MODELBOX_FLOWUNIT_ROCKCHIP_MPP_TO_CPU_H_
