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


#ifndef MODELBOX_FLOWUNIT_ASCEND_CROP_H_
#define MODELBOX_FLOWUNIT_ASCEND_CROP_H_

#define ENABLE_DVPP_INTERFACE
#define ACL_ENABLE

#include <acl/ops/acl_dvpp.h>
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/ascend/device_ascend.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_TYPE = "ascend";
constexpr const char *FLOWUNIT_NAME = "crop";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A crop flowunit on ascend device. \n"
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
    "[nv12], 'layout': [hwc]. One image can only be cropped with one "
    "rectangle and output one crop image.";

constexpr const char *IN_IMG = "in_image";
constexpr const char *IN_BOX = "in_region";
constexpr const char *OUT_IMG = "out_image";

class CropFlowUnit : public modelbox::AscendFlowUnit {
 public:
  CropFlowUnit() = default;
  virtual ~CropFlowUnit() = default;

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status AscendProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               aclrtStream stream);

 private:
  modelbox::Status PrepareOutput(
      std::shared_ptr<modelbox::BufferList> &input_box_buffer_list,
      std::shared_ptr<modelbox::BufferList> &output_img_buffer_list);

  modelbox::Status ProcessOneImg(std::shared_ptr<modelbox::Buffer> &in_image,
                               std::shared_ptr<modelbox::Buffer> &in_box,
                               std::shared_ptr<modelbox::Buffer> &out_image,
                               aclrtStream stream);

  modelbox::Status GetInputDesc(const std::shared_ptr<modelbox::Buffer> &in_image,
                              std::shared_ptr<acldvppPicDesc> &in_img_desc);

  modelbox::Status GetOutputDesc(const std::shared_ptr<modelbox::Buffer> &in_box,
                               const std::shared_ptr<modelbox::Buffer> &out_image,
                               std::shared_ptr<acldvppPicDesc> &out_img_desc);

  modelbox::Status GetRoiCfg(const std::shared_ptr<modelbox::Buffer> &in_box,
                           std::shared_ptr<acldvppRoiConfig> &roi_cfg);

  modelbox::Status Crop(std::shared_ptr<acldvppChannelDesc> &chan_desc,
                      std::shared_ptr<acldvppPicDesc> &in_img_desc,
                      std::shared_ptr<acldvppPicDesc> &out_img_desc,
                      std::shared_ptr<acldvppRoiConfig> &roi_cfg,
                      std::shared_ptr<modelbox::Buffer> &out_image,
                      aclrtStream stream);
};

#endif  // MODELBOX_FLOWUNIT_ASCEND_CROP_H_
