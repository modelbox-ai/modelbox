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

#include "padding_flowunit.h"
#include "image_process.h"
#include "modelbox/flowunit_api_helper.h"

using namespace imageprocess;

#define YUV420SP_SIZE(width, height) ((width) * (height)*3 / 2)
#define ALIGNMENT_DOWN(size) (size & 0xfffffffe)
#define MINI_WIDTH_STRIDE 32
#define MINI_WIDTHE_OFFSET 15

const std::string output_img_pix_fmt = "nv12";

const std::map<std::string, AlignType> kVerticalAlignType = {
    {"top", AlignType::BEGIN},
    {"center", AlignType::CENTER},
    {"bottom", AlignType::END}};

const std::map<std::string, AlignType> kHorizontalAlignType = {
    {"left", AlignType::BEGIN},
    {"center", AlignType::CENTER},
    {"right", AlignType::END}};

const std::map<std::string, int32_t> AsendResizeInterpolation = {
    {"default", 0},
    {"bilinear_opencv", 1},
    {"nearest_neighbor_opencv", 2},
    {"bilinear_tensorflow", 3},
    {"nearest_neighbor_tensorflow", 4}};

modelbox::Status PaddingFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  out_image_.width_ = opts->GetUint32("width", 0);
  if (out_image_.width_ == 0) {
    out_image_.width_ = opts->GetUint32("image_width", 0);
  }

  out_image_.height_ = opts->GetUint32("height", 0);
  if (out_image_.height_ == 0) {
    out_image_.height_ = opts->GetUint32("image_height", 0);
  }

  if (out_image_.width_ == 0 || out_image_.height_ == 0) {
    MBLOG_ERROR << "Dest width or dest height not valid";
    return modelbox::STATUS_BADCONF;
  }

  auto vertical_align_str = opts->GetString("vertical_align", "top");
  auto item = kVerticalAlignType.find(vertical_align_str);
  if (item == kVerticalAlignType.end()) {
    MBLOG_ERROR << "vertical align must be one of [top|center|bottom]";
    return modelbox::STATUS_BADCONF;
  }
  vertical_align_ = item->second;

  auto horizontal_align_str = opts->GetString("horizontal_align", "left");
  item = kHorizontalAlignType.find(horizontal_align_str);
  if (item == kHorizontalAlignType.end()) {
    MBLOG_ERROR << "horizontal align must be one of [left|center|right]";
    return modelbox::STATUS_BADCONF;
  }
  horizontal_align_ = item->second;

  padding_data_ = opts->GetUint8s("padding_data", {0, 0, 0});
  if (padding_data_.size() != 3) {
    MBLOG_ERROR << "padding data size must be 3";
    return modelbox::STATUS_BADCONF;
  }

  need_scale_ = opts->GetBool("need_scale", true);
  auto interpolation_str = opts->GetString("interpolation", "default");
  auto interpolation_item = AsendResizeInterpolation.find(interpolation_str);
  if (interpolation_item == AsendResizeInterpolation.end()) {
    MBLOG_ERROR << "not support interpolation " << interpolation_str;
    return modelbox::STATUS_BADCONF;
  }
  interpolation_ = interpolation_item->second;

  out_image_.width_stride_ = align_up(out_image_.width_, ASCEND_WIDTH_ALIGN);
  out_image_.height_stride_ = align_up(out_image_.height_, ASCEND_HEIGHT_ALIGN);
  size_t buffer_size = 0;
  auto ret = GetImageBytes(output_img_pix_fmt, out_image_.width_stride_,
                           out_image_.height_stride_, buffer_size);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "get image bytes failed, err " << ret;
    return ret;
  }
  out_image_.buffer_size_ = buffer_size;

  auto acl_ret = aclrtSetDevice(dev_id_);
  if (acl_ret != 0) {
    MBLOG_ERROR << "failed set device " << dev_id_;
    return modelbox::STATUS_FAULT;
  }

  auto alloc_buffer = [&](uint32_t size,
                          std::string device) -> std::shared_ptr<void> {
    void *buffer = nullptr;
    if (device == "cpu") {
      if (0 == aclrtMallocHost(&buffer, size)) {
        std::shared_ptr<void> shared_buffer_cpu(
            buffer, [this](void *ptr) { aclrtFreeHost(ptr); });
        return shared_buffer_cpu;
      }

    } else {
      if (0 == acldvppMalloc((void **)&buffer, size)) {
        std::shared_ptr<void> shared_buffer_device(
            buffer, [this](void *ptr) { acldvppFree(ptr); });
        return shared_buffer_device;
      }
    }

    return nullptr;
  };
  auto host_buffer = alloc_buffer(buffer_size, "cpu");
  if (host_buffer == nullptr) {
    MBLOG_ERROR << "malloc host buffer failed, buffer size: " << buffer_size;
    return modelbox::STATUS_FAULT;
  }

  size_t y_size = out_image_.width_stride_ * out_image_.height_stride_;
  size_t uv_size = y_size / 2;
  aclrtMemset(host_buffer.get(), y_size, padding_data_[0], y_size);

  u_int8_t *uv_buffer = (u_int8_t *)host_buffer.get() + y_size;
  for (size_t i = 0; i < uv_size; i = i + 2) {
    uv_buffer[i] = padding_data_[1];
    uv_buffer[i + 1] = padding_data_[2];
  }

  buffer_ = alloc_buffer(buffer_size, "ascend");
  if (buffer_ == nullptr) {
    MBLOG_ERROR << "malloc device buffer failed, buffer size: " << buffer_size;
    return modelbox::STATUS_FAULT;
  }

  acl_ret = aclrtMemcpy(buffer_.get(), buffer_size, host_buffer.get(),
                        buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_ret) {
    return {modelbox::STATUS_FAULT, "failed copy host to device"};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::AscendProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, aclrtStream stream) {
  auto input_img_buffer_list = data_ctx->Input(IN_IMG);
  auto img_count = input_img_buffer_list->Size();
  if (img_count == 0) {
    MBLOG_ERROR << "input img buffer list is empty";
    return modelbox::STATUS_INVALID;
  }

  auto output_img_buffer_list = data_ctx->Output(OUT_IMG);
  out_image_.width_stride_ = align_up(out_image_.width_, ASCEND_WIDTH_ALIGN);
  out_image_.height_stride_ = align_up(out_image_.height_, ASCEND_HEIGHT_ALIGN);
  size_t buffer_size = 0;
  auto ret = GetImageBytes(output_img_pix_fmt, out_image_.width_stride_,
                           out_image_.height_stride_, buffer_size);
  if (!ret) {
    MBLOG_ERROR << "get image bytes failed, err " << ret;
    return ret;
  }
  out_image_.buffer_size_ = buffer_size;

  std::vector<size_t> output_shape(img_count, out_image_.buffer_size_);
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
      MBLOG_ERROR << "Padding image failed, err " << ret;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status SetInImageSize(std::shared_ptr<modelbox::Buffer> &in_image,
                                ImageSize &ori_image, std::string &pix_fmt) {
  auto ret = in_image->Get("width", ori_image.width_);
  if (!ret) {
    return {modelbox::STATUS_FAULT, "get in_image width failed"};
  }
  ret = in_image->Get("height", ori_image.height_);
  if (!ret) {
    return {modelbox::STATUS_FAULT, "get in_image height failed"};
  }

  ret = in_image->Get("pix_fmt", pix_fmt);
  if (!ret) {
    return {modelbox::STATUS_FAULT, "get in_image pix_fmt failed"};
  }

  ori_image.width_stride_ = align_up(ori_image.width_, ASCEND_WIDTH_ALIGN);
  ori_image.width_stride_ = ori_image.width_stride_ < MINI_WIDTH_STRIDE
                                ? MINI_WIDTH_STRIDE
                                : ori_image.width_stride_;

  ori_image.height_stride_ = align_up(ori_image.height_, ASCEND_HEIGHT_ALIGN);
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status PaddingFlowUnit::ProcessOneImg(
    std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<modelbox::Buffer> &out_image, aclrtStream stream) {
  ResizeCropParam param;
  ImageSize ori_image;
  std::string pix_fmt = "nv12";
  if (modelbox::STATUS_SUCCESS !=
      SetInImageSize(in_image, ori_image, pix_fmt)) {
    MBLOG_ERROR << "get in image property failed";
    return {modelbox::STATUS_FAULT, "get in image property failed"};
  }

  Rect dest_rect;
  if (modelbox::STATUS_SUCCESS !=
      FillDestRoi(ori_image, dest_rect, param.crop_area, param.paste_area)) {
    MBLOG_ERROR << "FillDestRoi failed";
    return {modelbox::STATUS_FAULT, "FillDestRoi failed"};
  }

  auto status_ret = CreateDesc(in_image->ConstData(), in_image->GetBytes(),
                               ori_image, param.in_img_desc, pix_fmt);
  if (status_ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "create desc for in_image failed";
    return status_ret;
  }

  if (aclrtMemcpy((void *)out_image->ConstData(), out_image_.buffer_size_,
                  buffer_.get(), out_image_.buffer_size_,
                  ACL_MEMCPY_DEVICE_TO_DEVICE) != 0) {
    MBLOG_ERROR << "failed copy padding data to out image";
    return modelbox::STATUS_FAULT;
  }
  status_ret = CreateDesc(out_image->ConstData(), out_image->GetBytes(),
                          out_image_, param.out_img_desc, output_img_pix_fmt);
  if (status_ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "create out image descprition failed";
    return status_ret;
  }

  status_ret = CropResizeAndPaste(param, stream);
  if (!status_ret) {
    MBLOG_ERROR << "CropResizeAndPaste failed";
    return status_ret;
  }

  return SetOutImgMeta(out_image, output_img_pix_fmt, param.out_img_desc);
}

modelbox::Status PaddingFlowUnit::CreateDesc(
    const void *buffer, const int32_t &buffer_size, ImageSize &image_size,
    std::shared_ptr<acldvppPicDesc> &pic_desc, const std::string &pix_fmt) {
  if (!modelbox::IsMemAligned((uintptr_t)buffer,
                              modelbox::ASCEND_ASYNC_ALIGN)) {
    return {modelbox::STATUS_FAULT,
            "Input mem not aligned, ptr " + std::to_string((uintptr_t)buffer)};
  }

  pic_desc = CreateImgDesc(
      buffer_size, (void *)buffer, pix_fmt,
      ImageShape{image_size.width_, image_size.height_,
                 image_size.width_stride_, image_size.height_stride_},
      ImgDescDestroyFlag::DESC_ONLY);
  if (pic_desc == nullptr) {
    MBLOG_ERROR << "CreateImgDesc failed";
    return modelbox::StatusError;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::FillDestRoi(
    ImageSize &in_image_size, Rect &dest_roi,
    std::shared_ptr<acldvppRoiConfig> &crop_area,
    std::shared_ptr<acldvppRoiConfig> &paste_area) {
  if (need_scale_) {
    MBLOG_DEBUG << "in image width:" << in_image_size.width_
                << ",height:" << in_image_size.height_;
    auto w_scale = (float)in_image_size.width_ / out_image_.width_;
    auto h_scale = (float)in_image_size.height_ / out_image_.height_;
    auto scale = std::max(w_scale, h_scale);
    dest_roi.width = in_image_size.width_ / scale;
    dest_roi.height = in_image_size.height_ / scale;

    auto min_x = (out_image_.width_ - dest_roi.width) >> 1;
    min_x = (min_x + MINI_WIDTHE_OFFSET) >> 4 << 4;
    dest_roi.width = out_image_.width_ - 2 * min_x;
    if (dest_roi.width < MINI_WIDTH_STRIDE) {
      MBLOG_ERROR << "input w/h is too small than output";
      return modelbox::STATUS_INVALID;
    }
    scale = (float)in_image_size.width_ / dest_roi.width;
    dest_roi.height = in_image_size.height_ / scale;
    dest_roi.height = dest_roi.height >> 1 << 1;
    dest_roi.width =
        dest_roi.width > out_image_.width_ ? out_image_.width_ : dest_roi.width;
    dest_roi.height = dest_roi.height > out_image_.height_ ? out_image_.height_
                                                           : dest_roi.height;
    MBLOG_DEBUG << "dest_roi width:" << dest_roi.width
                << ",height:" << dest_roi.height;
  } else {
    if (in_image_size.width_ > out_image_.width_ ||
        in_image_size.height_ > out_image_.height_) {
      MBLOG_ERROR << "src image[w:" << in_image_size.width_
                  << ",h:" << in_image_size.height_
                  << "] is great than dest size[w:" << out_image_.width_
                  << ",h:" << out_image_.height_
                  << "]. But need_scale is false";
      return modelbox::STATUS_INVALID;
    }

    dest_roi.width = in_image_size.width_;
    dest_roi.height = in_image_size.height_;
  }

  auto crop_area_local = acldvppCreateRoiConfig(0, in_image_size.width_ - 1, 0,
                                                in_image_size.height_ - 1);
  if (crop_area_local == nullptr) {
    MBLOG_ERROR << "failed create roi config for crop area";
    return modelbox::STATUS_FAULT;
  }

  crop_area = std::shared_ptr<acldvppRoiConfig>(
      crop_area_local, [](acldvppRoiConfig *config) {
        if (config != nullptr) {
          acldvppDestroyRoiConfig(config);
        }
      });
  dest_roi.x =
      GetAlignOffset(horizontal_align_, out_image_.width_, dest_roi.width) >>
      4 << 4;
  dest_roi.width = ALIGNMENT_DOWN(dest_roi.width);
  dest_roi.y = ALIGNMENT_DOWN(
      GetAlignOffset(vertical_align_, out_image_.height_, dest_roi.height));
  dest_roi.height = ALIGNMENT_DOWN(dest_roi.height);
  auto paste_area_local =
      acldvppCreateRoiConfig(dest_roi.x, dest_roi.x + dest_roi.width - 1,
                             dest_roi.y, dest_roi.y + dest_roi.height - 1);
  if (paste_area_local == nullptr) {
    MBLOG_ERROR << "failed create roi config for paste area";
    return modelbox::STATUS_FAULT;
  }
  paste_area = std::shared_ptr<acldvppRoiConfig>(
      paste_area_local, [](acldvppRoiConfig *config) {
        if (config != nullptr) {
          acldvppDestroyRoiConfig(config);
        }
      });

  return modelbox::STATUS_OK;
}

uint32_t PaddingFlowUnit::GetAlignOffset(AlignType type, uint32_t dest_range,
                                         uint32_t roi_range) {
  if (roi_range >= dest_range) {
    return 0;
  }

  uint32_t offset = 0;
  switch (type) {
    case AlignType::BEGIN:
      break;

    case AlignType::CENTER:
      offset = (dest_range - roi_range) / 2;
      break;

    case AlignType::END:
      offset = dest_range - roi_range;
      break;

    default:
      break;
  }

  return offset;
}

modelbox::Status PaddingFlowUnit::CropResizeAndPaste(ResizeCropParam &param,
                                                     aclrtStream stream) {
  auto chan_desc = GetDvppChannel(dev_id_);
  if (chan_desc == nullptr) {
    MBLOG_ERROR << "Get dvpp channel failed";
    return {modelbox::STATUS_FAULT, "Get dvpp channel failed"};
  }
  auto resize_cfg = acldvppCreateResizeConfig();
  if (resize_cfg == nullptr) {
    MBLOG_ERROR << "acldvppCreateResizeConfig return null";
    return {modelbox::STATUS_FAULT, "acldvppCreateResizeConfig return null"};
  }

  Defer { acldvppDestroyResizeConfig(resize_cfg); };
  auto acl_ret =
      acldvppSetResizeConfigInterpolation(resize_cfg, interpolation_);
  if (acl_ret != 0) {
    MBLOG_ERROR << "failed set interpolation for resize config";
    return {modelbox::STATUS_FAULT,
            "failed set interpolation for resize config"};
  }
  acl_ret = acldvppVpcCropResizePasteAsync(
      chan_desc.get(), param.in_img_desc.get(), param.out_img_desc.get(),
      param.crop_area.get(), param.paste_area.get(), resize_cfg, stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "acldvppVpcCropResizePasteAsync failed, err " +
                       std::to_string(acl_ret);
    return modelbox::STATUS_FAULT;
  }

  acl_ret = aclrtSynchronizeStream(stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSynchronizeStream failed, err " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::Close() { return modelbox::STATUS_OK; }

MODELBOX_FLOWUNIT(PaddingFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({IN_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.AddFlowUnitOutput({OUT_IMG, modelbox::ASCEND_MEM_DVPP});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_width", "int", true,
                                                  "0", "the padding width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_height", "int", true,
                                                  "0", "the padding height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "vertical_align", "string", false, "top", "vertical align type"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "horizontal_align", "string", false, "left", "horizontal align type"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "padding_data", "string", false, "0,0,0", "the padding data"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}