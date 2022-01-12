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

#ifndef MODELBOX_FLOWUNIT_ASCEND_PADDING_H_
#define MODELBOX_FLOWUNIT_ASCEND_PADDING_H_

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
constexpr const char *FLOWUNIT_NAME = "padding";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A padding flowunit on ascend device \n"
    "\t@Port paramter: the input port buffer type and the output port buffer "
    "type are image. \n"
    "\t  The image type buffer contain the following meta fields:\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint: the field value range of this flowunit support：'pix_fmt': "
    "[nv12], 'layout': [hwc]. ";
constexpr const char *IN_IMG = "in_image";
constexpr const char *OUT_IMG = "out_image";
enum class AlignType { BEGIN, CENTER, END };
class Rect {
 public:
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
};

class ImageSize {
 public:
  int32_t width_;
  int32_t height_;
  int32_t width_stride_;
  int32_t height_stride_;
  int32_t buffer_size_;
};

class ResizeCropParam {
 public:
  std::shared_ptr<acldvppPicDesc> in_img_desc;
  std::shared_ptr<acldvppPicDesc> resize_img_desc;
  std::shared_ptr<acldvppRoiConfig> crop_area;
  std::shared_ptr<acldvppRoiConfig> paste_area;
  std::shared_ptr<acldvppPicDesc> out_img_desc;
};

class PaddingFlowUnit : public modelbox::AscendFlowUnit {
 public:
  PaddingFlowUnit() = default;
  virtual ~PaddingFlowUnit() = default;

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status AscendProcess(
      std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream);

 private:
  modelbox::Status ProcessOneImg(std::shared_ptr<modelbox::Buffer> &in_image,
                                 std::shared_ptr<modelbox::Buffer> &out_image,
                                 aclrtStream stream);

  modelbox::Status CreateDesc(const void *buffer, const int32_t &buffer_size,
                              ImageSize &image_size,
                              std::shared_ptr<acldvppPicDesc> &pic_desc,
                              const std::string &pix_fmt);

  modelbox::Status GetInputDesc(
      const std::shared_ptr<modelbox::Buffer> &in_image,
      std::shared_ptr<acldvppPicDesc> &in_img_desc);

  modelbox::Status GetOutputDesc(
      const std::shared_ptr<modelbox::Buffer> &out_image,
      std::shared_ptr<acldvppPicDesc> &out_img_desc,
      std::shared_ptr<acldvppPicDesc> &resize_img_desc);

  modelbox::Status CropResizeAndPaste(ResizeCropParam &param,
                                      aclrtStream stream);

  modelbox::Status FillDestRoi(ImageSize &in_image_size, Rect &dest_rect,
                               std::shared_ptr<acldvppRoiConfig> &crop_area,
                               std::shared_ptr<acldvppRoiConfig> &paste_area);

  uint32_t GetAlignOffset(AlignType type, uint32_t dest_range,
                          uint32_t roi_range);

  ImageSize out_image_;
  std::shared_ptr<void> buffer_{nullptr};

  AlignType vertical_align_{AlignType::BEGIN};
  AlignType horizontal_align_{AlignType::BEGIN};

  std::vector<uint8_t> padding_data_;
  bool need_scale_{true};
  int32_t interpolation_{0};
};

#endif  // MODELBOX_FLOWUNIT_ASCEND_RESIZE_H_
