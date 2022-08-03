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

#include "image_process.h"
#include "modelbox/flowunit_api_helper.h"

const std::string output_img_pix_fmt = "nv12";

modelbox::Status ResizeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  dest_width_ = opts->GetUint32("width", 0);
  if (dest_width_ == 0) {
    dest_width_ = opts->GetUint32("image_width", 0);
  }

  dest_height_ = opts->GetUint32("height", 0);
  if (dest_height_ == 0) {
    dest_height_ = opts->GetUint32("image_height", 0);
  }

  if (dest_width_ == 0 || dest_height_ == 0) {
    MBLOG_ERROR << "Dest width or dest height not valid";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ResizeFlowUnit::AscendProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream) {
  auto input_img_buffer_list = data_ctx->Input(IN_IMG);
  auto img_count = input_img_buffer_list->Size();
  if (img_count == 0) {
    MBLOG_ERROR << "input img buffer list is empty";
    return modelbox::STATUS_INVALID;
  }

  auto output_img_buffer_list = data_ctx->Output(OUT_IMG);
  size_t buffer_size = 0;
  auto align_w =
      imageprocess::align_up(dest_width_, imageprocess::ASCEND_WIDTH_ALIGN);
  auto align_h =
      imageprocess::align_up(dest_height_, imageprocess::ASCEND_HEIGHT_ALIGN);
  auto ret = imageprocess::GetImageBytes(output_img_pix_fmt, align_w, align_h,
                                         buffer_size);
  if (!ret) {
    MBLOG_ERROR << "get image bytes failed, err " << ret;
    return ret;
  }

  std::vector<size_t> output_shape(img_count, buffer_size);
  ret = output_img_buffer_list->Build(output_shape, false);
  if (!ret) {
    MBLOG_ERROR << "Build output failed, err " << ret;
    return ret;
  }

  output_img_buffer_list->CopyMeta(input_img_buffer_list);
  for (size_t i = 0; i < img_count; ++i) {
    auto in_img_buffer = input_img_buffer_list->At(i);
    auto out_img_buffer = output_img_buffer_list->At(i);
    auto ret = ProcessOneImg(in_img_buffer, out_img_buffer, stream);
    if (!ret) {
      MBLOG_ERROR << "Resize image failed, err " << ret;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ResizeFlowUnit::ProcessOneImg(
    std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<modelbox::Buffer> &out_image, aclrtStream stream) {
  auto chan_desc = imageprocess::GetDvppChannel(dev_id_);
  if (chan_desc == nullptr) {
    return {modelbox::STATUS_FAULT, "Get dvpp channel failed"};
  }

  std::shared_ptr<acldvppPicDesc> in_img_desc;
  auto ret = GetInputDesc(in_image, in_img_desc);
  if (!ret) {
    return ret;
  }

  std::shared_ptr<acldvppPicDesc> out_img_desc;
  ret = GetOutputDesc(out_image, out_img_desc);
  if (!ret) {
    return ret;
  }

  ret = Resize(chan_desc, in_img_desc, out_img_desc, out_image, stream);
  if (!ret) {
    return ret;
  }

  return imageprocess::SetOutImgMeta(out_image, output_img_pix_fmt,
                                     out_img_desc);
}

modelbox::Status ResizeFlowUnit::GetInputDesc(
    const std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<acldvppPicDesc> &in_img_desc) {
  std::string in_pix_fmt;
  int32_t in_img_width = 0;
  int32_t in_img_height = 0;
  int32_t in_img_width_stride = 0;
  int32_t in_img_height_stride = 0;
  auto ret = imageprocess::GetImgParam(in_image, in_pix_fmt, in_img_width,
                                       in_img_height, in_img_width_stride,
                                       in_img_height_stride);
  if (!ret) {
    return ret;
  }

  if (!modelbox::IsMemAligned((uintptr_t)in_image->ConstData(),
                              modelbox::ASCEND_ASYNC_ALIGN)) {
    return {modelbox::STATUS_FAULT,
            "Input mem not aligned, ptr " +
                std::to_string((uintptr_t)in_image->ConstData())};
  }

  in_img_desc = CreateImgDesc(
      in_image->GetBytes(), (void *)in_image->ConstData(), in_pix_fmt,
      imageprocess::ImageShape{in_img_width, in_img_height, in_img_width_stride,
                               in_img_height_stride},
      imageprocess::ImgDescDestroyFlag::DESC_ONLY);
  if (in_img_desc == nullptr) {
    return modelbox::StatusError;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ResizeFlowUnit::GetOutputDesc(
    const std::shared_ptr<modelbox::Buffer> &out_image,
    std::shared_ptr<acldvppPicDesc> &out_img_desc) {
  if (!modelbox::IsMemAligned((uintptr_t)out_image->MutableData(),
                              modelbox::ASCEND_ASYNC_ALIGN)) {
    return {modelbox::STATUS_FAULT,
            "Output mem not aligned, ptr " +
                std::to_string((uintptr_t)out_image->MutableData())};
  }

  auto align_w = imageprocess::align_up((int32_t)dest_width_,
                                        imageprocess::ASCEND_WIDTH_ALIGN);
  auto align_h = imageprocess::align_up((int32_t)dest_height_,
                                        imageprocess::ASCEND_HEIGHT_ALIGN);
  out_img_desc = CreateImgDesc(
      out_image->GetBytes(), (void *)out_image->MutableData(),
      output_img_pix_fmt,
      imageprocess::ImageShape{(int32_t)dest_width_, (int32_t)dest_height_,
                               align_w, align_h},
      imageprocess::ImgDescDestroyFlag::DESC_ONLY);
  if (out_img_desc == nullptr) {
    return modelbox::StatusError;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ResizeFlowUnit::Resize(
    std::shared_ptr<acldvppChannelDesc> &chan_desc,
    std::shared_ptr<acldvppPicDesc> &in_img_desc,
    std::shared_ptr<acldvppPicDesc> &out_img_desc,
    std::shared_ptr<modelbox::Buffer> &out_image, aclrtStream stream) {
  auto *resize_cfg = acldvppCreateResizeConfig();
  if (resize_cfg == nullptr) {
    return {modelbox::STATUS_FAULT, "acldvppCreateResizeConfig return null"};
  }

  Defer { acldvppDestroyResizeConfig(resize_cfg); };
  auto acl_ret = acldvppVpcResizeAsync(chan_desc.get(), in_img_desc.get(),
                                       out_img_desc.get(), resize_cfg, stream);
  if (acl_ret != ACL_SUCCESS) {
    std::string err_msg =
        "acldvppVpcCropAsync failed, err " + std::to_string(acl_ret);
    return {modelbox::STATUS_FAULT, err_msg};
  }

  acl_ret = aclrtSynchronizeStream(stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSynchronizeStream failed, err " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ResizeFlowUnit::Close() { return modelbox::STATUS_OK; }

MODELBOX_FLOWUNIT(ResizeFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({IN_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.AddFlowUnitOutput({OUT_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_width", "int", true,
                                                  "0", "the resize width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_height", "int", true,
                                                  "0", "the resize height"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}