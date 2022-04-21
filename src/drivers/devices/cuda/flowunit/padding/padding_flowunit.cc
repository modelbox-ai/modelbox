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

#include "modelbox/flowunit_api_helper.h"

const std::map<std::string, NppiInterpolationMode> kNppiResizeInterpolation = {
    {"inter_nn", NPPI_INTER_NN},           {"inter_linear", NPPI_INTER_LINEAR},
    {"inter_cubic", NPPI_INTER_CUBIC},     {"inter_super", NPPI_INTER_SUPER},
    {"inter_lanczos", NPPI_INTER_LANCZOS},
};

const std::map<std::string, AlignType> kVerticalAlignType = {
    {"top", AlignType::BEGIN},
    {"center", AlignType::CENTER},
    {"bottom", AlignType::END}};

const std::map<std::string, AlignType> kHorizontalAlignType = {
    {"left", AlignType::BEGIN},
    {"center", AlignType::CENTER},
    {"right", AlignType::END}};

PaddingFlowUnit::PaddingFlowUnit(){};
PaddingFlowUnit::~PaddingFlowUnit(){};

modelbox::Status PaddingFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  width_ = opts->GetUint32("width", 0);
  if (width_ == 0) {
    width_ = opts->GetUint32("image_width", 0);
  }

  height_ = opts->GetUint32("height", 0);
  if (height_ == 0) {
    height_ = opts->GetUint32("image_height", 0);
  }

  if (width_ == 0 || height_ == 0) {
    MBLOG_ERROR << "width and height must set in config";
    return modelbox::STATUS_BADCONF;
  }

  output_buffer_size_ = width_ * height_ * 3;
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
  auto interpolation_str = opts->GetString("interpolation", "inter_linear");
  auto interpolation_item = kNppiResizeInterpolation.find(interpolation_str);
  if (interpolation_item == kNppiResizeInterpolation.end()) {
    MBLOG_ERROR << "not support interpolation " << interpolation_str;
    return modelbox::STATUS_BADCONF;
  }
  interpolation_ = interpolation_item->second;

  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> ctx, cudaStream_t stream) {
  auto input_buffer_list = ctx->Input("in_image");
  auto output_buffer_list = ctx->Output("out_image");
  auto image_count = input_buffer_list->Size();
  if (image_count == 0) {
    MBLOG_ERROR << "input buffer count is zero";
    return modelbox::STATUS_FAULT;
  }

  std::vector<size_t> output_shape(image_count, output_buffer_size_);
  auto ret = output_buffer_list->Build(output_shape);
  if (!ret) {
    MBLOG_ERROR << "build output buffer failed, count " << image_count
                << ",size " << output_buffer_size_;
    return modelbox::STATUS_FAULT;
  }

  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "sync stream  " << stream << " failed, err " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < image_count; ++i) {
    auto in_image = input_buffer_list->At(i);
    auto out_image = output_buffer_list->At(i);
    auto ret = PaddingOneImage(in_image, out_image);
    if (!ret) {
      MBLOG_ERROR << "padding image failed, err " << ret;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::PaddingOneImage(
    std::shared_ptr<modelbox::Buffer> &in_image,
    std::shared_ptr<modelbox::Buffer> &out_image) {
  int32_t ori_width = 0;
  int32_t ori_height = 0;
  std::string pix_fmt;
  auto ret = in_image->Get("width", ori_width);
  ret = ret && in_image->Get("height", ori_height);
  if (!ret) {
    MBLOG_ERROR << "input image must has width and height in meta";
    return modelbox::STATUS_INVALID;
  }

  in_image->Get("pix_fmt", pix_fmt);
  if (pix_fmt != "rgb" && pix_fmt != "bgr") {
    MBLOG_ERROR << "unsupport pix format " << pix_fmt;
    return modelbox::STATUS_NOTSUPPORT;
  }

  NppiSize src_size{.width = ori_width, .height = ori_height};
  NppiRect src_roi{.x = 0, .y = 0, .width = ori_width, .height = ori_height};
  NppiSize dest_size{.width = width_, .height = height_};
  NppiRect dest_roi;
  auto status = FillDestRoi(src_size, dest_roi);
  if (!status) {
    MBLOG_ERROR << "fill dest roi failed";
    return status;
  }

  status = FillPaddingData(out_image);
  if (!status) {
    MBLOG_ERROR << "fill padding data failed";
    return status;
  }

  auto nppi_ret =
      nppiResize_8u_C3R((const Npp8u *)in_image->ConstData(), ori_width * 3,
                        src_size, src_roi, (Npp8u *)out_image->MutableData(),
                        width_ * 3, dest_size, dest_roi, interpolation_);
  if (nppi_ret != NPP_SUCCESS) {
    MBLOG_ERROR << "nppiResize_8u_C3R failed, src_w:" << ori_width
                << ", src_h:" << ori_height << ", dest_w:" << width_
                << ", dest_h:" << height_ << "dest_roi[x:" << dest_roi.x
                << ",y:" << dest_roi.y << ",w:" << dest_roi.width
                << ",h:" << dest_roi.height << "], ret " << nppi_ret;
    return modelbox::STATUS_FAULT;
  }

  out_image->Set("width", width_);
  out_image->Set("height", height_);
  out_image->Set("width_stride", width_ * 3);
  out_image->Set("height_stride", height_);
  out_image->Set("channel", (int32_t)3);
  out_image->Set("pix_fmt", pix_fmt);
  std::string data_format = "hwc";
  auto data_type = modelbox::MODELBOX_UINT8;
  std::vector<size_t> data_shape = {(size_t)height_, (size_t)width_, 3};
  out_image->Set("layout", data_format);
  out_image->Set("type", data_type);
  out_image->Set("shape", data_shape);
  return modelbox::STATUS_OK;
}

modelbox::Status PaddingFlowUnit::FillDestRoi(const NppiSize &src_size,
                                              NppiRect &dest_roi) {
  if (need_scale_) {
    auto w_scale = (float)src_size.width / width_;
    auto h_scale = (float)src_size.height / height_;
    auto scale = std::max(w_scale, h_scale);
    dest_roi.width = src_size.width / scale;
    dest_roi.height = src_size.height / scale;
  } else {
    if (src_size.width > width_ || src_size.height > height_) {
      MBLOG_ERROR << "src image[w:" << src_size.width
                  << ",h:" << src_size.height
                  << "] is great than dest size[w:" << width_
                  << ",h:" << height_ << "]. But need_scale is false";
      return modelbox::STATUS_INVALID;
    }

    dest_roi.width = src_size.width;
    dest_roi.height = src_size.height;
  }

  dest_roi.x = GetAlignOffset(horizontal_align_, width_, dest_roi.width);
  dest_roi.y = GetAlignOffset(vertical_align_, height_, dest_roi.height);
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

modelbox::Status PaddingFlowUnit::FillPaddingData(
    std::shared_ptr<modelbox::Buffer> &out_image) {
  NppiSize size{.width = width_, .height = height_};
  auto ret =
      nppiSet_8u_C3R(padding_data_.data(), (Npp8u *)out_image->MutableData(),
                     width_ * 3, size);
  if (ret != NPP_SUCCESS) {
    MBLOG_ERROR << "nppiSet_8u_C3R failed, size[w:" << width_
                << ",h:" << height_ << "], ret " << ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddingFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_image", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);

  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("width", "int", true, "0", "Output img width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("height", "int", true, "0",
                                                  "Output img height"));
  std::map<std::string, std::string> vertical_align_list{
      {"top", "top"}, {"center", "center"}, {"bottom", "bottom"}};
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "vertical_align", "list", false, "top", "Output roi vertical align type",
      vertical_align_list));
  std::map<std::string, std::string> horizontal_align_list{
      {"left", "left"}, {"center", "center"}, {"right", "right"}};
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "horizontal_align", "list", false, "left",
      "Output roi horizontal align type", horizontal_align_list));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "padding_data", "string", false, "0,0,0", "Data for padding"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("need_scale", "bool", false, "true",
                               "Will scale roi to fit output image"));
  std::map<std::string, std::string> interpolation_list{
      {"inter_nn", "inter_nn"},
      {"inter_linear", "inter_linear"},
      {"inter_cubic", "inter_cubic"},
      {"inter_super", "inter_super"},
      {"inter_lanczos", "inter_lanczos"}};
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "interpolation", "list", false, "inter_linear",
      "Interpolation method to scale roi", interpolation_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
