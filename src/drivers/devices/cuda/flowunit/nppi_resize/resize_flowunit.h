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


#ifndef MODELBOX_FLOWUNIT_NPPI_RESIZE_GPU_H_
#define MODELBOX_FLOWUNIT_NPPI_RESIZE_GPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <nppi_geometry_transforms.h>
#include "modelbox/device/cuda/device_cuda.h"

#include <typeinfo>

constexpr const char *FLOWUNIT_NAME = "resize";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A resize flowunit on cuda device. \n"
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
    "[rgb_packed,bgr_packed], 'layout': [hwc]. ";
const int RGB_CHANNLES = 3;

typedef struct {
  int32_t width;  /**<  Rectangle width. */
  int32_t height; /**<  Rectangle height. */
  int32_t channel;
} ImageSize;

class NppiResizeFlowUnit : public modelbox::CudaFlowUnit {
 public:
  NppiResizeFlowUnit();
  ~NppiResizeFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override { return modelbox::STATUS_OK; };

  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               cudaStream_t stream) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

 private:
  modelbox::Status NppiResize_u8_P3(const u_char *pSrcPlanarData,
                                  ImageSize srcSize, u_char *pDstPlanarData,
                                  ImageSize dstSize,
                                  NppiInterpolationMode method);

  modelbox::Status NppiResize_u8_c3r(const u_char *pSrcPlanarData,
                                   ImageSize srcSize, u_char *pDstPlanarData,
                                   ImageSize dstSize,
                                   NppiInterpolationMode method);

  NppiInterpolationMode GetNppiResizeInterpolation(std::string resizeType);

  modelbox::Status ProcessOneImage(
      std::shared_ptr<modelbox::BufferList> &input_buffer_list,
      std::shared_ptr<modelbox::BufferList> &output_buffer_list, int index);

 private:
  size_t dest_width_{0};
  size_t dest_height_{0};
  std::string interpolation_{"inter_linear"};
};

#endif  // MODELBOX_FLOWUNIT_NPPI_RESIZE_GPU_H_
