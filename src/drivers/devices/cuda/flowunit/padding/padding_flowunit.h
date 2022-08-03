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


#ifndef MODELBOX_FLOWUNIT_NPPI_PADDING_GPU_H_
#define MODELBOX_FLOWUNIT_NPPI_PADDING_GPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include "modelbox/device/cuda/device_cuda.h"
#include <npp.h>

#include <string>
#include <typeinfo>

constexpr const char *FLOWUNIT_NAME = "padding";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A padding flowunit on cuda. \n"
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

class PaddingFlowUnit : public modelbox::CudaFlowUnit {
 public:
  PaddingFlowUnit();
  ~PaddingFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override { return modelbox::STATUS_OK; };

  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               cudaStream_t stream) override;

 private:
  modelbox::Status PaddingOneImage(std::shared_ptr<modelbox::Buffer> &in_image,
                                 std::shared_ptr<modelbox::Buffer> &out_image);

  modelbox::Status FillDestRoi(const NppiSize &src_size, NppiRect &dest_roi);

  uint32_t GetAlignOffset(AlignType type, uint32_t dest_range,
                          uint32_t roi_range);

  modelbox::Status FillPaddingData(std::shared_ptr<modelbox::Buffer> &out_image);

  int32_t width_{0};
  int32_t height_{0};
  size_t output_buffer_size_{0};
  AlignType vertical_align_{AlignType::BEGIN};
  AlignType horizontal_align_{AlignType::BEGIN};
  std::vector<uint8_t> padding_data_;
  bool need_scale_{true};
  NppiInterpolationMode interpolation_{NPPI_INTER_LINEAR};
};

#endif  // MODELBOX_FLOWUNIT_NPPI_PADDING_GPU_H_
