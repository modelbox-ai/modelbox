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

#include "crop_flowunit.h"
#include "image_process.h"
#include "modelbox/flowunit_api_helper.h"

using namespace imageprocess;
const int MIN_WIDTH_STRIDE = 32;
const std::string OUTPUT_IMG_PIX_FMT = "nv12";

modelbox::Status CropFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::AscendProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream) {
  auto input_img_buffer_list = data_ctx->Input(IN_IMG);
  auto input_box_buffer_list = data_ctx->Input(IN_BOX);
  auto box_count = input_img_buffer_list->Size();
  auto img_count = input_box_buffer_list->Size();
  if (box_count != img_count) {
    auto err_msg = "box buffer size " + std::to_string(box_count) +
                   " and img buffer size " + std::to_string(img_count) +
                   " not equal";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  auto output_img_buffer_list = data_ctx->Output(OUT_IMG);
  auto ret = PrepareOutput(input_box_buffer_list, output_img_buffer_list);
  if (!ret) {
    MBLOG_ERROR << "Preapre output failed " << ret;
    return ret;
  }

  output_img_buffer_list->CopyMeta(input_img_buffer_list);
  for (size_t i = 0; i < img_count; ++i) {
    auto in_img_buffer = input_img_buffer_list->At(i);
    auto in_box_buffer = input_box_buffer_list->At(i);
    auto out_img_buffer = output_img_buffer_list->At(i);
    auto ret =
        ProcessOneImg(in_img_buffer, in_box_buffer, out_img_buffer, stream);
    if (!ret) {
      MBLOG_ERROR << "Crop image failed, err " << ret;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::PrepareOutput(
    std::shared_ptr<modelbox::BufferList> &input_box_buffer_list,
    std::shared_ptr<modelbox::BufferList> &output_img_buffer_list) {
  auto img_count = input_box_buffer_list->Size();
  std::vector<size_t> output_shape;
  for (size_t i = 0; i < img_count; ++i) {
    auto box_buffer = input_box_buffer_list->At(i);
    auto box_ptr = (BoxInt32 *)box_buffer->ConstData();
    size_t bytes = 0;
    int32_t align_w = align_up(box_ptr->w, ASCEND_WIDTH_ALIGN);
    align_w = std::max(align_w, MIN_WIDTH_STRIDE);
    int32_t align_h = align_up(box_ptr->h, ASCEND_HEIGHT_ALIGN);
    auto ret = GetImageBytes(OUTPUT_IMG_PIX_FMT, align_w, align_h, bytes);
    if (!ret) {
      return ret;
    }

    output_shape.emplace_back(bytes);
  }

  return output_img_buffer_list->Build(output_shape, false);
}

modelbox::Status CropFlowUnit::ProcessOneImg(
    std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<modelbox::Buffer> &in_box,
    std::shared_ptr<modelbox::Buffer> &out_image, aclrtStream stream) {
  auto chan_desc = GetDvppChannel(dev_id_);
  if (chan_desc == nullptr) {
    return {modelbox::STATUS_FAULT, "Get dvpp channel failed"};
  }

  std::shared_ptr<acldvppPicDesc> in_img_desc;
  auto ret = GetInputDesc(in_image, in_img_desc);
  if (!ret) {
    return ret;
  }

  std::shared_ptr<acldvppPicDesc> out_img_desc;
  ret = GetOutputDesc(in_box, out_image, out_img_desc);
  if (!ret) {
    return ret;
  }

  std::shared_ptr<acldvppRoiConfig> roi_cfg;
  ret = GetRoiCfg(in_box, roi_cfg);
  if (!ret) {
    return ret;
  }

  ret = Crop(chan_desc, in_img_desc, out_img_desc, roi_cfg, out_image, stream);
  if (!ret) {
    return ret;
  }

  return SetOutImgMeta(out_image, OUTPUT_IMG_PIX_FMT, out_img_desc);
}

modelbox::Status CropFlowUnit::GetInputDesc(
    const std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<acldvppPicDesc> &in_img_desc) {
  std::string in_pix_fmt;
  int32_t in_img_width = 0, in_img_height = 0, in_img_width_stride = 0,
          in_img_height_stride = 0;
  auto ret = GetImgParam(in_image, in_pix_fmt, in_img_width, in_img_height,
                         in_img_width_stride, in_img_height_stride);
  if (!ret) {
    return ret;
  }

  in_img_desc = CreateImgDesc(
      in_image->GetBytes(), (void *)in_image->ConstData(), in_pix_fmt,
      ImageShape{in_img_width, in_img_height, in_img_width_stride,
                 in_img_height_stride},
      ImgDescDestroyFlag::DESC_ONLY);
  if (in_img_desc == nullptr) {
    return modelbox::StatusError;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::GetOutputDesc(
    const std::shared_ptr<modelbox::Buffer> &in_box,
    const std::shared_ptr<modelbox::Buffer> &out_image,
    std::shared_ptr<acldvppPicDesc> &out_img_desc) {
  auto box_ptr = (const BoxInt32 *)in_box->ConstData();
  if (box_ptr->x % 2 != 0 || box_ptr->y % 2 != 0 || box_ptr->w % 2 != 0 ||
      box_ptr->h % 2 != 0) {
    return {modelbox::STATUS_INVALID,
            "Input box[x:" + std::to_string(box_ptr->x) +
                ", y:" + std::to_string(box_ptr->y) +
                ", w:" + std::to_string(box_ptr->w) +
                ", h:" + std::to_string(box_ptr->h) +
                "] is invalid, value must be even"};
  }

  auto align_w = align_up(box_ptr->w, ASCEND_WIDTH_ALIGN);
  align_w = std::max(align_w, MIN_WIDTH_STRIDE);
  auto align_h = align_up(box_ptr->h, ASCEND_HEIGHT_ALIGN);
  out_img_desc = CreateImgDesc(
      out_image->GetBytes(), (void *)out_image->MutableData(),
      OUTPUT_IMG_PIX_FMT, ImageShape{box_ptr->w, box_ptr->h, align_w, align_h},
      ImgDescDestroyFlag::DESC_ONLY);
  if (out_img_desc == nullptr) {
    return modelbox::StatusError;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::GetRoiCfg(
    const std::shared_ptr<modelbox::Buffer> &in_box,
    std::shared_ptr<acldvppRoiConfig> &roi_cfg) {
  auto box_ptr = (const BoxInt32 *)in_box->ConstData();
  uint32_t left = box_ptr->x;
  uint32_t right = box_ptr->x + box_ptr->w - 1;
  uint32_t top = box_ptr->y;
  uint32_t bottom = box_ptr->y + box_ptr->h - 1;
  auto roi_cfg_ptr = acldvppCreateRoiConfig(left, right, top, bottom);
  if (roi_cfg_ptr == nullptr) {
    return {modelbox::STATUS_FAULT, "acldvppCreateRoiConfig return null"};
  }

  roi_cfg.reset(roi_cfg_ptr,
                [](acldvppRoiConfig *ptr) { acldvppDestroyRoiConfig(ptr); });
  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::Crop(
    std::shared_ptr<acldvppChannelDesc> &chan_desc,
    std::shared_ptr<acldvppPicDesc> &in_img_desc,
    std::shared_ptr<acldvppPicDesc> &out_img_desc,
    std::shared_ptr<acldvppRoiConfig> &roi_cfg,
    std::shared_ptr<modelbox::Buffer> &out_image, aclrtStream stream) {
  auto acl_ret = acldvppVpcCropAsync(chan_desc.get(), in_img_desc.get(),
                                     out_img_desc.get(), roi_cfg.get(), stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "acldvppVpcCropAsync failed, err " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  acl_ret = aclrtSynchronizeStream(stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSynchronizeStream failed, err " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CropFlowUnit::Close() { return modelbox::STATUS_OK; }

MODELBOX_FLOWUNIT(CropFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({IN_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.AddFlowUnitInput({IN_BOX, "cpu"});
  desc.AddFlowUnitOutput({OUT_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}