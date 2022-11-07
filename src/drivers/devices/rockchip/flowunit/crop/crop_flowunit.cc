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

#include "crop_flowunit.h"

#include "image_process.h"
#include "modelbox/base/status.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "securec.h"

RockchipCropFlowUnit::RockchipCropFlowUnit() = default;
RockchipCropFlowUnit::~RockchipCropFlowUnit() = default;

modelbox::Status RockchipCropFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status RockchipCropFlowUnit::Close() {
  return modelbox::STATUS_SUCCESS;
}

std::shared_ptr<modelbox::Buffer> RockchipCropFlowUnit::ProcessOneImage(
    const std::shared_ptr<modelbox::Buffer> &in_img, const im_rect &region) {
  std::string pix_fmt;
  RgaSURF_FORMAT rga_fmt = RK_FORMAT_UNKNOWN;
  in_img->Get("pix_fmt", pix_fmt);
  rga_fmt = modelbox::GetRGAFormat(pix_fmt);
  if (rga_fmt == RK_FORMAT_UNKNOWN) {
    MBLOG_ERROR << "unsupport pix format, pix_fmt: " << pix_fmt;
    return nullptr;
  }

  rga_buffer_t in_buf;
  if (GetRGAFromImgBuffer(in_img, rga_fmt, in_buf) !=
      modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "input img can not change to rga buffer";
    return nullptr;
  }

  auto device = this->GetBindDevice();
  rga_buffer_t out_buf;
  auto out_img =
      CreateEmptyMppImg(region.width, region.height, rga_fmt, device, out_buf);
  if (out_img == nullptr) {
    MBLOG_ERROR << "failed to create mpp img";
    return nullptr;
  }

  IM_STATUS status = imcrop(in_buf, out_buf, region);
  if (status != IM_STATUS_SUCCESS) {
    MBLOG_ERROR << "rga crop failed: " << status;
    return nullptr;
  }

  out_img->CopyMeta(in_img);
  out_img->Set("width", (int32_t)region.width);
  out_img->Set("height", (int32_t)region.height);
  auto ws = (int32_t)MPP_ALIGN(region.width, MPP_ALIGN_WIDTH);
  if (RK_FORMAT_BGR_888 == rga_fmt || RK_FORMAT_RGB_888 == rga_fmt) {
    out_img->Set("width_stride", ws * 3);
  } else {
    out_img->Set("width_stride", ws);
  }

  out_img->Set("height_stride",
               (int32_t)MPP_ALIGN(region.height, MPP_ALIGN_HEIGHT));

  return out_img;
}

modelbox::Status RockchipCropFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_img_bufs = data_ctx->Input(IN_IMG);
  auto input_box_bufs = data_ctx->Input(IN_REGION);
  if (input_img_bufs->Size() != input_box_bufs->Size()) {
    auto msg = "in_img and in_region mismatch: " +
               std::to_string(input_img_bufs->Size()) + ":" +
               std::to_string(input_box_bufs->Size());
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  auto output_bufs = data_ctx->Output(OUT_IMG);

  for (size_t i = 0; i < input_img_bufs->Size(); ++i) {
    const auto *const bbox = static_cast<const imageprocess::RoiBox *>(
        input_box_bufs->At(i)->ConstData());
    if (bbox == nullptr) {
      MBLOG_ERROR << "input region is invalid.";
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageCrop.CropFailed", "input region is invalid.");
      output_bufs->PushBack(buffer);
      continue;
    }

    im_rect region;
    region.x = bbox->x;
    region.y = bbox->y;
    region.width = bbox->w;
    region.height = bbox->h;

    int32_t in_width = 0;
    int32_t in_height = 0;

    input_img_bufs->At(i)->Get("width", in_width);
    input_img_bufs->At(i)->Get("height", in_height);
    if (in_width <= 0 || in_height <= 0) {
      MBLOG_ERROR << "input size is invalid.";
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageCrop.CropFailed", "input size is invalid");
      output_bufs->PushBack(buffer);
      continue;
    }

    if (bbox->x < 0 || bbox->x > in_width || bbox->y < 0 ||
        bbox->y > in_height || bbox->w < 0 || bbox->w > in_width ||
        bbox->h < 0 || bbox->h > in_height) {
      MBLOG_ERROR << "bbox region is invalid.";
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageCrop.CropFailed", "bbox region is invalid.");
      output_bufs->PushBack(buffer);
      continue;
    }

    auto ret = ProcessOneImage(input_img_bufs->At(i), region);
    if (ret == nullptr) {
      auto msg = "crop image failed, index is " + std::to_string(i);
      MBLOG_ERROR << msg;
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageCrop.CropFailed", msg);
      output_bufs->PushBack(buffer);
      continue;
    }

    output_bufs->PushBack(ret);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(RockchipCropFlowUnit, rk_crop_desc) {
  rk_crop_desc.SetFlowUnitName(FLOWUNIT_NAME);
  rk_crop_desc.SetFlowUnitGroupType("Image");
  rk_crop_desc.AddFlowUnitInput({IN_IMG, modelbox::DEVICE_TYPE});
  rk_crop_desc.AddFlowUnitInput({IN_REGION, "cpu"});
  rk_crop_desc.AddFlowUnitOutput({OUT_IMG, modelbox::DEVICE_TYPE});
  rk_crop_desc.SetFlowType(modelbox::NORMAL);
  rk_crop_desc.SetInputContiguous(false);
  rk_crop_desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(rk_crop_desc) {
  rk_crop_desc.Desc.SetName(FLOWUNIT_NAME);
  rk_crop_desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  rk_crop_desc.Desc.SetType(modelbox::DEVICE_TYPE);
  rk_crop_desc.Desc.SetDescription(FLOWUNIT_DESC);
  rk_crop_desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}
