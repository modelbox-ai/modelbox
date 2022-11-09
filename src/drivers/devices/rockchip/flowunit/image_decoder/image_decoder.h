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

#ifndef MODELBOX_FLOWUNIT_IMAGE_DECODER_ROCKCHIP_H_
#define MODELBOX_FLOWUNIT_IMAGE_DECODER_ROCKCHIP_H_

#include <modelbox/base/device.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>
#include <modelbox/device/rockchip/rockchip_memory.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "image_decoder";
constexpr const char *FLOWUNIT_TYPE = "rockchip";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: jpeg decoder flowunit on rockchip. \n"
    "\t@Port parameter: The input port buffer type is image file binary, the "
    "output port buffer type are image. \n"
    "\t  The image type buffer contains the following meta fields:\n"
    "\t\tField Name: width,               Type: int32_t\n"
    "\t\tField Name: height,              Type: int32_t\n"
    "\t\tField Name: width_stride,        Type: int32_t\n"
    "\t\tField Name: height_stride,       Type: int32_t\n"
    "\t\tField Name: channel,             Type: int32_t\n"
    "\t\tField Name: pix_fmt,             Type: string\n"
    "\t\tField Name: layout,              Type: int32_t\n"
    "\t\tField Name: shape,               Type: vector<size_t>\n"
    "\t\tField Name: type,                Type: "
    "ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint:";

class ImageDecoderFlowUnit : public modelbox::FlowUnit {
 public:
  ImageDecoderFlowUnit();
  ~ImageDecoderFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ct) override;

 private:
  MppFrame JpegDec(std::shared_ptr<modelbox::Buffer> &buffer, int &w, int &h);
  std::shared_ptr<modelbox::Buffer> DecodeFromCPU(
      std::shared_ptr<modelbox::Buffer> &in_buffer);
  cv::Mat BGR2YUV_NV12(const cv::Mat &src_bgr);

 private:
  std::string pixel_format_{modelbox::IMG_DEFAULT_FMT};
  modelbox::MppJpegDecode jpeg_dec_;
  RgaSURF_FORMAT out_pix_fmt_{RK_FORMAT_YCbCr_420_SP};
};

#endif  // MODELBOX_FLOWUNIT_IMAGE_DECODER_ROCKCHIP_H_
