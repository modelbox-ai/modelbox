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

#include "nppi_crop_flowunit.h"

#include <npp.h>

#include "modelbox/flowunit_api_helper.h"

NppiCropFlowUnit::NppiCropFlowUnit(){};
NppiCropFlowUnit::~NppiCropFlowUnit(){};

std::vector<std::string> kNppiCropMethod = {"u8c3r"};

modelbox::Status NppiCropFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status NppiCropFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> ctx, cudaStream_t stream) {
  auto input_img_bufs = ctx->Input("in_image");
  if (input_img_bufs->Size() <= 0) {
    auto errMsg =
        "input images size is " + std::to_string(input_img_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto input_box_bufs = ctx->Input("in_region");
  if (input_box_bufs->Size() <= 0) {
    auto errMsg =
        "in_region roi box batch is " + std::to_string(input_box_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  if (input_img_bufs->Size() != input_box_bufs->Size()) {
    auto errMsg = "in_image batch is not match in_region batch. in_image is " +
                  std::to_string(input_img_bufs->Size()) + ",in_region is " +
                  std::to_string(input_box_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto output_bufs = ctx->Output("out_image");

  std::vector<size_t> shape_vector;
  int32_t channel = RGB_CHANNLES;
  for (size_t i = 0; i < input_img_bufs->Size(); ++i) {
    auto buff = input_box_bufs->ConstBufferData(i);
    if (buff == nullptr) {
      MBLOG_WARN << "input buffer " << i << " is invalid.";
      continue;
    }
    auto bbox = static_cast<const RoiBox *>(buff);
    if (bbox == nullptr) {
      MBLOG_WARN << "buffer is not box, buffer index: " << i;
      continue;
    }

    MBLOG_DEBUG << "crop bbox : " << bbox->x << " " << bbox->y << " "
                << bbox->width << " " << bbox->height;
    shape_vector.push_back((bbox->width) * (bbox->height) * channel *
                           sizeof(u_char));
  }

  output_bufs->Build(shape_vector);

  output_bufs->CopyMeta(input_img_bufs);

  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "sync stream  " << stream << " failed, err " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < input_img_bufs->Size(); ++i) {
    auto ret = ProcessOneImage(input_img_bufs, input_box_bufs, output_bufs, i);
    if (ret != modelbox::STATUS_OK) {
      MBLOG_ERROR << "nppi crop image failed, index is " << i;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status NppiCropFlowUnit::ProcessOneImage(
    std::shared_ptr<modelbox::BufferList> &input_img_buffer_list,
    std::shared_ptr<modelbox::BufferList> &input_box_buffer_list,
    std::shared_ptr<modelbox::BufferList> &output_buffer_list, int index) {
  ImageSize src_size;
  std::string pix_fmt;

  bool exists = false;

  exists = input_img_buffer_list->At(index)->Get("height", src_size.height);
  if (!exists) {
    MBLOG_ERROR << "meta don't have key height";
    return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
  }

  exists = input_img_buffer_list->At(index)->Get("width", src_size.width);
  if (!exists) {
    MBLOG_ERROR << "meta don't have key width";
    return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
  }

  exists = input_img_buffer_list->At(index)->Get("pix_fmt", pix_fmt);
  if (!exists &&
      !input_img_buffer_list->At(index)->Get("channel", src_size.channel)) {
    MBLOG_ERROR << "meta don't have key pix_fmt or channel";
    return {modelbox::STATUS_NOTSUPPORT,
            "meta don't have key pix_fmt or channel"};
  }

  if (exists && pix_fmt != "rgb" && pix_fmt != "bgr") {
    MBLOG_ERROR << "unsupport pix format.";
    return {modelbox::STATUS_NOTSUPPORT, "unsupport pix format."};
  }

  src_size.channel = RGB_CHANNLES;

  MBLOG_DEBUG << "Input image width " << src_size.width << " height "
              << src_size.height << " channel " << src_size.channel;

  auto bbox = static_cast<const RoiBox *>(
      input_box_buffer_list->ConstBufferData(index));
  if (bbox == nullptr) {
    MBLOG_ERROR << "input data at " << index << " is invalid.";
    return modelbox::STATUS_NODATA;
  }

  RoiBox dst_size;
  dst_size.width = bbox->width;
  dst_size.height = bbox->height;
  dst_size.x = bbox->x;
  dst_size.y = bbox->y;

  auto input_data = static_cast<const u_char *>(
      input_img_buffer_list->ConstBufferData(index));

  auto output_data =
      static_cast<u_char *>(output_buffer_list->MutableBufferData(index));

  modelbox::Status ret = modelbox::STATUS_OK;
  ret = NppiCrop_u8_c3r(input_data, src_size, output_data, dst_size);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  auto output_buffer = output_buffer_list->At(index);
  output_buffer->Set("width", dst_size.width);
  output_buffer->Set("height", dst_size.height);
  output_buffer->Set("width_stride", dst_size.width * 3);
  output_buffer->Set("height_stride", dst_size.height);
  output_buffer->Set("channel", src_size.channel);
  output_buffer->Set("pix_fmt", pix_fmt);
  output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  output_buffer->Set("shape", std::vector<size_t>{(size_t)dst_size.height,
                                                  (size_t)dst_size.width, 3});
  output_buffer->Set("layout", std::string("hwc"));

  return modelbox::STATUS_OK;
}

modelbox::Status NppiCropFlowUnit::NppiCrop_u8_c3r(const u_char *p_src_data,
                                                   ImageSize src_size,
                                                   u_char *p_dst_data,
                                                   RoiBox dst_size) {
  const Npp8u *p_src = p_src_data;

  p_src =
      p_src + (dst_size.y * src_size.width + dst_size.x) * sizeof(u_char) * 3;

  Npp8u *p_dst = p_dst_data;

  NppiSize dst_npp_size;
  dst_npp_size.width = dst_size.width;
  dst_npp_size.height = dst_size.height;

  NppStatus status =
      nppiCopy_8u_C3R(p_src, src_size.width * sizeof(u_char) * 3, p_dst,
                      dst_size.width * sizeof(u_char) * 3, dst_npp_size);
  if (NPP_SUCCESS != status) {
    MBLOG_ERROR << "nppi error code " << status;
    std::string errMsg = "cuda Crop failed, error code " +
                         std::to_string(status) +
                         ", src image size: " + std::to_string(src_size.width) +
                         " x " + std::to_string(src_size.height);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_NODATA, errMsg};
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(NppiCropFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_image", "cuda"));
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_region", "cpu"));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image", "cuda"));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
