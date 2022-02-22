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


#ifndef MODELBOX_FLOWUNIT_NPPI_CROP_GPU_H_
#define MODELBOX_FLOWUNIT_NPPI_CROP_GPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <nppi_data_exchange_and_initialization.h>

#include <typeinfo>

#include "modelbox/device/cuda/device_cuda.h"

constexpr const char *FLOWUNIT_NAME = "crop";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A crop flowunit on cuda device. \n"
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
const int RGB_CHANNLES = 3;

typedef struct {
  int32_t width;
  int32_t height;
  int32_t channel;
} ImageSize;

typedef struct RoiBox {
  int32_t x, y, width, height;
} RoiBox;

class NppiCropFlowUnit : public modelbox::CudaFlowUnit {
 public:
  NppiCropFlowUnit();
  virtual ~NppiCropFlowUnit();

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override { return modelbox::STATUS_OK; };

  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                             cudaStream_t stream) override;

 private:
  modelbox::Status NppiCrop_u8_c3r(const u_char *pSrcData, ImageSize srcSize,
                                 u_char *pDstData, RoiBox dstSize);

  modelbox::Status ProcessOneImage(
      std::shared_ptr<modelbox::BufferList> &input_buffer_list,
      std::shared_ptr<modelbox::BufferList> &input_box_bufs,
      std::shared_ptr<modelbox::BufferList> &output_buffer_list, int index);

};

#endif  // MODELBOX_FLOWUNIT_NPPI_CROP_GPU_H_
