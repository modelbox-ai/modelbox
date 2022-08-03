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

#include <npp.h>

#include "modelbox/flowunit_api_helper.h"

NppiResizeFlowUnit::NppiResizeFlowUnit() = default;
NppiResizeFlowUnit::~NppiResizeFlowUnit() = default;

std::map<std::string, NppiInterpolationMode> kNppiResizeInterpolation = {
    {"inter_nn", NPPI_INTER_NN},           {"inter_linear", NPPI_INTER_LINEAR},
    {"inter_cubic", NPPI_INTER_CUBIC},     {"inter_super", NPPI_INTER_SUPER},
    {"inter_lanczos", NPPI_INTER_LANCZOS},
};

std::vector<std::string> kNppiResizeMethod = {"u8c3r", "u8p3"};

modelbox::Status NppiResizeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  dest_width_ = opts->GetUint32("width", 0);
  if (dest_width_ == 0) {
    dest_width_ = opts->GetUint32("image_width", 0);
  }

  dest_height_ = opts->GetUint32("height", 0);
  if (dest_height_ == 0) {
    dest_height_ = opts->GetUint32("image_height", 0);
  }
  if (dest_width_ <= 0 || dest_height_ <= 0) {
    const auto *errMsg = "resize width or height is not configured or invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  interpolation_ = opts->GetString("interpolation", "inter_linear");
  if (kNppiResizeInterpolation.find(interpolation_) ==
      kNppiResizeInterpolation.end()) {
    auto errMsg =
        "resize interpolation is invalid, configure is :" + interpolation_;
    MBLOG_ERROR << errMsg;
    std::string valid_interpolation;
    for (const auto &iter : kNppiResizeInterpolation) {
      if (valid_interpolation.length() > 0) {
        valid_interpolation += ", ";
      }
      valid_interpolation += iter.first;
    }
    MBLOG_ERROR << "Valid interpolation is: " << valid_interpolation;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  MBLOG_DEBUG << "resize dest width " << dest_width_ << ", resize dest height "
              << dest_height_ << ", resize interpolation " << interpolation_;
  return modelbox::STATUS_OK;
}

modelbox::Status NppiResizeFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  auto input_bufs = data_ctx->Input("in_image");
  auto output_bufs = data_ctx->Output("out_image");

  if (input_bufs->Size() <= 0) {
    auto errMsg = "input images size is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  size_t channel = RGB_CHANNLES;
  std::vector<size_t> sub_shape{dest_width_, dest_height_, channel};
  std::vector<size_t> tensor_shape(
      input_bufs->Size(), modelbox::Volume(sub_shape) * sizeof(u_char));
  output_bufs->Build(tensor_shape);

  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "sync stream  " << stream << " failed, err " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  output_bufs->CopyMeta(input_bufs);
  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    auto ret = ProcessOneImage(input_bufs, output_bufs, i);
    if (ret != modelbox::STATUS_OK) {
      MBLOG_ERROR << "nppi resize image failed, index is " << i;
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status NppiResizeFlowUnit::ProcessOneImage(
    std::shared_ptr<modelbox::BufferList> &input_buffer_list,
    std::shared_ptr<modelbox::BufferList> &output_buffer_list, int index) {
  ImageSize srcResize;
  std::string pix_fmt;
  bool exists = false;
  exists = input_buffer_list->At(index)->Get("height", srcResize.height);
  if (!exists) {
    MBLOG_ERROR << "meta don't have key width";
    return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
  }

  exists = input_buffer_list->At(index)->Get("width", srcResize.width);
  if (!exists) {
    MBLOG_ERROR << "meta don't have key height";
    return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
  }

  exists = input_buffer_list->At(index)->Get("pix_fmt", pix_fmt);
  if (!exists &&
      !input_buffer_list->At(index)->Get("channel", srcResize.channel)) {
    MBLOG_ERROR << "meta don't have key pix_fmt or channel";
    return {modelbox::STATUS_NOTSUPPORT,
            "meta don't have key pix_fmt or channel"};
  }

  if (exists && pix_fmt != "rgb" && pix_fmt != "bgr") {
    MBLOG_ERROR << "unsupport pix format.";
    return {modelbox::STATUS_NOTSUPPORT, "unsupport pix format."};
  }

  srcResize.channel = RGB_CHANNLES;
  MBLOG_DEBUG << "get " << srcResize.width << " rows " << srcResize.height
              << " channel " << srcResize.channel;

  ImageSize dstResize;
  dstResize.height = dest_height_;
  dstResize.width = dest_width_;
  dstResize.channel = srcResize.channel;

  const auto *input_data =
      static_cast<const u_char *>(input_buffer_list->ConstBufferData(index));
  auto *output_data =
      static_cast<u_char *>(output_buffer_list->MutableBufferData(index));

  // resize image
  auto nppiMethod = GetNppiResizeInterpolation(interpolation_);

  modelbox::Status ret = modelbox::STATUS_OK;
  ret = NppiResize_u8_c3r(input_data, srcResize, output_data, dstResize,
                          nppiMethod);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  // output resize image
  auto output_buffer = output_buffer_list->At(index);
  output_buffer->Set("width", dstResize.width);
  output_buffer->Set("height", dstResize.height);
  output_buffer->Set("width_stride", dstResize.width * 3);
  output_buffer->Set("height_stride", dstResize.height);
  output_buffer->Set("channel", srcResize.channel);
  output_buffer->Set("pix_fmt", pix_fmt);
  output_buffer->Set("layout", "hwc");
  output_buffer->Set(
      "shape", std::vector<size_t>({static_cast<size_t>(dstResize.height),
                                    static_cast<size_t>(dstResize.width),
                                    static_cast<size_t>(srcResize.channel)}));
  output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  return modelbox::STATUS_OK;
}

modelbox::Status NppiResizeFlowUnit::NppiResize_u8_P3(
    const u_char *pSrcPlanarData, ImageSize srcSize, u_char *pDstPlanarData,
    ImageSize dstSize, NppiInterpolationMode method) {
  const Npp8u *pSrc[3];
  pSrc[0] = pSrcPlanarData;
  pSrc[1] = pSrcPlanarData + srcSize.width * srcSize.height;
  pSrc[2] = pSrcPlanarData + srcSize.width * srcSize.height * 2;

  NppiRect oSrcRectROI;
  oSrcRectROI.x = 0;
  oSrcRectROI.y = 0;
  oSrcRectROI.width = srcSize.width;
  oSrcRectROI.height = srcSize.height;

  Npp8u *pDst[3];
  pDst[0] = pDstPlanarData;
  pDst[1] = pDstPlanarData + dstSize.width * dstSize.height;
  pDst[2] = pDstPlanarData + dstSize.width * dstSize.height * 2;

  NppiRect oDstRectROI;
  oDstRectROI.x = 0;
  oDstRectROI.y = 0;
  oDstRectROI.width = dstSize.width;
  oDstRectROI.height = dstSize.height;

  NppiSize srcNppSize;
  srcNppSize.width = srcSize.width;
  srcNppSize.height = srcSize.height;

  NppiSize dstNppSize;
  dstNppSize.width = dstSize.width;
  dstNppSize.height = dstSize.height;

  NppStatus status = nppiResize_8u_P3R(
      pSrc, srcSize.width * sizeof(u_char), srcNppSize, oSrcRectROI, pDst,
      dest_width_ * sizeof(u_char), dstNppSize, oDstRectROI, method);
  if (NPP_SUCCESS != status) {
    MBLOG_ERROR << "npp error code " << status;
    std::string errMsg = "Nppi resize failed, error code " +
                         std::to_string(status) +
                         ", src image size: " + std::to_string(srcSize.width) +
                         " x " + std::to_string(srcSize.height);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_NODATA, errMsg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status NppiResizeFlowUnit::NppiResize_u8_c3r(
    const u_char *pSrcPlanarData, ImageSize srcSize, u_char *pDstPlanarData,
    ImageSize dstSize, NppiInterpolationMode method) {
  const Npp8u *pSrc = pSrcPlanarData;

  NppiRect oSrcRectROI;
  oSrcRectROI.x = 0;
  oSrcRectROI.y = 0;
  oSrcRectROI.width = srcSize.width;
  oSrcRectROI.height = srcSize.height;

  Npp8u *pDst = pDstPlanarData;

  NppiRect oDstRectROI;
  oDstRectROI.x = 0;
  oDstRectROI.y = 0;
  oDstRectROI.width = dstSize.width;
  oDstRectROI.height = dstSize.height;

  NppiSize srcNppSize;
  srcNppSize.width = srcSize.width;
  srcNppSize.height = srcSize.height;

  NppiSize dstNppSize;
  dstNppSize.width = dstSize.width;
  dstNppSize.height = dstSize.height;

  NppStatus status = nppiResize_8u_C3R(
      pSrc, srcSize.width * sizeof(u_char) * 3, srcNppSize, oSrcRectROI, pDst,
      dest_width_ * sizeof(u_char) * 3, dstNppSize, oDstRectROI, method);
  if (NPP_SUCCESS != status) {
    MBLOG_ERROR << "nppi error code " << status;
    std::string errMsg = "cuda resize failed, error code " +
                         std::to_string(status) +
                         ", src image size: " + std::to_string(srcSize.width) +
                         " x " + std::to_string(srcSize.height);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_NODATA, errMsg};
  }
  return modelbox::STATUS_OK;
}

NppiInterpolationMode NppiResizeFlowUnit::GetNppiResizeInterpolation(
    std::string resizeType) {
  transform(resizeType.begin(), resizeType.end(), resizeType.begin(),
            ::tolower);

  if (kNppiResizeInterpolation.find(resizeType) ==
      kNppiResizeInterpolation.end()) {
    MBLOG_WARN << "cuda resize not support method \"" << resizeType << "\"";
    MBLOG_WARN << "using defalt method \"inter_linear\"";
    return NPPI_INTER_LINEAR;
  }

  return kNppiResizeInterpolation[resizeType];
}

MODELBOX_FLOWUNIT(NppiResizeFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_image", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_width", "int", true,
                                                  "640", "the resize width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_height", "int", true,
                                                  "480", "the resize height"));
  std::map<std::string, std::string> interpolation_list;

  for (auto &item : kNppiResizeInterpolation) {
    interpolation_list[item.first] = item.first;
  }

  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("interpolation", "list", true, "inter_linear",
                               "the resize interpolation", interpolation_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
