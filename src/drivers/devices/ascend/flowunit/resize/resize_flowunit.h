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


#ifndef MODELBOX_FLOWUNIT_ASCEND_RESIZE_H_
#define MODELBOX_FLOWUNIT_ASCEND_RESIZE_H_

#define ENABLE_DVPP_INTERFACE
#define ACL_ENABLE

#include <acl/ops/acl_dvpp.h>
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <modelbox/device/ascend/device_ascend.h>

constexpr const char *FLOWUNIT_TYPE = "ascend";
constexpr const char *FLOWUNIT_NAME = "resize";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A resize flowunit on ascend device. \n"
    "\t@Port parameter: The input port buffer type and the output port buffer "
    "type are image. \n"
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
    "\t@Constraint: The field value range of this flowunit support: 'pix_fmt': "
    "[nv12], 'layout': [hwc]. ";
constexpr const char *IN_IMG = "in_image";
constexpr const char *OUT_IMG = "out_image";

class ResizeFlowUnit : public modelbox::AscendFlowUnit {
 public:
  ResizeFlowUnit() = default;
  virtual ~ResizeFlowUnit() = default;

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status AscendProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               aclrtStream stream);

 private:
  modelbox::Status ProcessOneImg(std::shared_ptr<modelbox::Buffer> &in_image,
                               std::shared_ptr<modelbox::Buffer> &out_image,
                               aclrtStream stream);

  modelbox::Status GetInputDesc(const std::shared_ptr<modelbox::Buffer> &in_image,
                              std::shared_ptr<acldvppPicDesc> &in_img_desc);

  modelbox::Status GetOutputDesc(const std::shared_ptr<modelbox::Buffer> &out_image,
                               std::shared_ptr<acldvppPicDesc> &out_img_desc);

  modelbox::Status Resize(std::shared_ptr<acldvppChannelDesc> &chan_desc,
                        std::shared_ptr<acldvppPicDesc> &in_img_desc,
                        std::shared_ptr<acldvppPicDesc> &out_img_desc,
                        std::shared_ptr<modelbox::Buffer> &out_image,
                        aclrtStream stream);

  uint32_t dest_width_{0};
  uint32_t dest_height_{0};
};

#endif  // MODELBOX_FLOWUNIT_ASCEND_RESIZE_H_
