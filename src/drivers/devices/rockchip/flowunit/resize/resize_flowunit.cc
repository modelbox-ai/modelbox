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

#include "resize_flowunit.h"

#include "modelbox/base/status.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "securec.h"

#define MIN_SIZE 32

modelbox::Status ResizeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  dest_width_ = opts->GetUint32("image_width", 0);

  dest_height_ = opts->GetUint32("image_height", 0);

  if (dest_width_ < MIN_SIZE || dest_height_ < MIN_SIZE) {
    std::string msg =
        "Dest width or dest height must great equal than 32, dest_width: " +
        std::to_string(dest_width_) +
        " dest_height: " + std::to_string(dest_height_);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_BADCONF, msg};
  }

  align_width_ = MPP_ALIGN(dest_width_, MPP_ALIGN_WIDTH);
  align_height_ = MPP_ALIGN(dest_height_, MPP_ALIGN_HEIGHT);

  return modelbox::STATUS_SUCCESS;
}

void ResizeFlowUnit::WriteData(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::string &pix_fmt, RgaSURF_FORMAT rga_fmt,
    std::shared_ptr<modelbox::Buffer> &out_image) {
  auto output_bufs = data_ctx->Output(OUT_IMG);
  out_image->Set("pix_fmt", pix_fmt);
  out_image->Set("width", (int32_t)dest_width_);
  out_image->Set("height", (int32_t)dest_height_);
  if (RK_FORMAT_RGB_888 == rga_fmt || RK_FORMAT_BGR_888 == rga_fmt) {
    out_image->Set("width_stride", (int32_t)(align_width_ * 3));
  } else {
    out_image->Set("width_stride", (int32_t)align_width_);
  }

  out_image->Set("height_stride", (int32_t)align_height_);
  out_image->Set("layout", std::string("hwc"));
  size_t height = dest_height_;
  size_t channel = 3;
  if (rga_fmt == RK_FORMAT_YCbCr_420_SP || rga_fmt == RK_FORMAT_YCrCb_420_SP) {
    height = dest_height_ * 3 / 2;
    channel = 1;
  }

  out_image->Set("channel", (int32_t)channel);
  out_image->Set("shape",
                 std::vector<size_t>{height, (size_t)dest_width_, channel});
  out_image->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);

  output_bufs->PushBack(out_image);
}

modelbox::Status ResizeFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_img_buffer_list = data_ctx->Input(IN_IMG);
  auto img_count = input_img_buffer_list->Size();
  if (img_count == 0) {
    MBLOG_ERROR << "input img buffer list is empty";
    return {modelbox::STATUS_INVALID, "input img buffer list is empty"};
  }

  for (size_t i = 0; i < img_count; ++i) {
    auto in_image = input_img_buffer_list->At(i);

    std::string pix_fmt;
    RgaSURF_FORMAT rga_fmt = RK_FORMAT_UNKNOWN;
    in_image->Get("pix_fmt", pix_fmt);
    rga_fmt = modelbox::GetRGAFormat(pix_fmt);
    if (rga_fmt == RK_FORMAT_UNKNOWN) {
      MBLOG_ERROR << "rga fmt unknow";
      return {modelbox::STATUS_NOTSUPPORT, "rga fmt unknow"};
    }

    rga_buffer_t in_buf;
    if (modelbox::GetRGAFromImgBuffer(in_image, rga_fmt, in_buf) !=
        modelbox::STATUS_SUCCESS) {
      MBLOG_WARN << "input img can not change to rga buffer";
      return {modelbox::STATUS_NOTSUPPORT,
              "input img can not change to rga buffer"};
    }

    auto device = this->GetBindDevice();
    rga_buffer_t out_buf;
    auto out_image = modelbox::CreateEmptyMppImg(dest_width_, dest_height_,
                                                 rga_fmt, device, out_buf);
    if (out_image == nullptr) {
      MBLOG_ERROR << "failed to create mpp img";
      return {modelbox::STATUS_NOTSUPPORT, "failed to create mpp img"};
    }

    IM_STATUS status = imresize(in_buf, out_buf);
    if (status != IM_STATUS_SUCCESS) {
      MBLOG_ERROR << "rga resize failed: " << status;
      return {modelbox::STATUS_NOTSUPPORT,
              "rga resize failed: " + std::to_string(status)};
    }

    WriteData(data_ctx, pix_fmt, rga_fmt, out_image);
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status ResizeFlowUnit::Close() { return modelbox::STATUS_SUCCESS; }

MODELBOX_FLOWUNIT(ResizeFlowUnit, rk_resize_desc) {
  rk_resize_desc.SetFlowUnitName(FLOWUNIT_NAME);
  rk_resize_desc.SetFlowUnitGroupType("Image");
  rk_resize_desc.AddFlowUnitInput({IN_IMG});
  rk_resize_desc.AddFlowUnitOutput({OUT_IMG});
  rk_resize_desc.SetFlowType(modelbox::NORMAL);
  rk_resize_desc.SetDescription(FLOWUNIT_DESC);
  rk_resize_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "image_width", "int", true, "0", "the resize width"));
  rk_resize_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "image_height", "int", true, "0", "the resize height"));
}

MODELBOX_DRIVER_FLOWUNIT(rk_resize_desc) {
  rk_resize_desc.Desc.SetName(FLOWUNIT_NAME);
  rk_resize_desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  rk_resize_desc.Desc.SetType(modelbox::DEVICE_TYPE);
  rk_resize_desc.Desc.SetDescription(FLOWUNIT_DESC);
  rk_resize_desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}