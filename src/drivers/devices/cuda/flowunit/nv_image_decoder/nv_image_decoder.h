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

#ifndef MODELBOX_FLOWUNIT_NV_IMAGE_DECODER_GPU_H_
#define MODELBOX_FLOWUNIT_NV_IMAGE_DECODER_GPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <nvjpeg.h>

#include <typeinfo>

constexpr const char *FLOWUNIT_NAME = "image_decoder";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: An OpenCV crop flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer type is image file binary, the "
    "output port buffer type are image. \n"
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
    "\t@Constraint:";

enum ImageType {
  IMAGE_TYPE_JPEG,
  IMAGE_TYPE_PNG,
  IMAGE_TYPE_BMP,
  IMAGE_TYPE_OHTER
};

std::map<ImageType, std::vector<uint8_t>> ImgStreamFormat{
    {IMAGE_TYPE_JPEG, {0xff, 0xd8}},
    {IMAGE_TYPE_PNG, {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}},
    {IMAGE_TYPE_BMP, {0x42, 0x4d}}};

class NvImageDecoderFlowUnit : public modelbox::FlowUnit {
 public:
  NvImageDecoderFlowUnit();
  virtual ~NvImageDecoderFlowUnit();

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  ImageType CheckImageType(const uint8_t *input_data);

  bool DecodeJpeg(const std::shared_ptr<modelbox::Buffer> &input_buffer,
                  std::shared_ptr<modelbox::Buffer> &output_buffer,
                  nvjpegJpegState_t &jpeg_handle);

  bool DecodeOthers(const std::shared_ptr<modelbox::Buffer> &input_buffer,
                    std::shared_ptr<modelbox::Buffer> &output_buffer);

 private:
  std::string pixel_format_{"bgr"};

  nvjpegHandle_t handle_{nullptr};
};

#endif  // MODELBOX_FLOWUNIT_NV_IMAGE_DECODER_GPU_H_