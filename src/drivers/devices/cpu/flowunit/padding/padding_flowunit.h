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

#ifndef MODELBOX_FLOWUNIT_PADDINGFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_PADDINGFLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <opencv2/opencv.hpp>

#include <string>
#include <typeinfo>
#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "padding";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A padding flowunit on cpu. \n"
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
    "\t@Constraint: The field value range of this flowunit supports: 'pix_fmt': "
    "[rgb,bgr], 'layout': [hwc]. ";

enum class AlignType { BEGIN, CENTER, END };
struct dest_roi_proportions {
  int32_t dest_roi_width = 0;
  int32_t dest_roi_height = 0;
  int32_t dest_roi_x = 0;
  int32_t dest_roi_y = 0;
};

class PaddingFlowUnit : public modelbox::FlowUnit {
 public:
  PaddingFlowUnit();
  virtual ~PaddingFlowUnit();

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override { return modelbox::STATUS_OK; };

  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  cv::InterpolationFlags GetCVResizeMethod(std::string resizeType);
  modelbox::Status PaddingOneImage(
      std::shared_ptr<modelbox::Buffer> &in_image,
      std::shared_ptr<modelbox::Buffer> &out_image);

  modelbox::Status FillDestRoi(const cv::Size &src_size, cv::Mat &dest_roi,
                               struct dest_roi_proportions *p_drp);

  uint32_t GetAlignOffset(AlignType type, uint32_t dest_range,
                          uint32_t roi_range);

  modelbox::Status FillPaddingData(
      std::shared_ptr<modelbox::Buffer> &out_image);

 private:
  int32_t width_{0};
  int32_t height_{0};
  size_t output_buffer_size_{0};
  AlignType vertical_align_{AlignType::BEGIN};
  AlignType horizontal_align_{AlignType::BEGIN};
  std::vector<uint8_t> padding_data_;
  bool need_scale_{true};
  cv::InterpolationFlags interpolation_{cv::INTER_LINEAR};
};

#endif  // MODELBOX_FLOWUNIT_FLOWUNIT_CPU_H_
