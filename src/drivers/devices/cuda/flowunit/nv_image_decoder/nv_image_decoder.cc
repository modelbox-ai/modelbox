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

#include "nv_image_decoder.h"
#include <cuda_runtime_api.h>
#include <npp.h>
#include <nppi_data_exchange_and_initialization.h>
#include <opencv2/opencv.hpp>
#include "modelbox/flowunit_api_helper.h"
#include "modelbox/device/cuda/device_cuda.h"

std::map<std::string, nvjpegOutputFormat_t> NvImgPixelFormat{
    {"bgr", NVJPEG_OUTPUT_BGR}, {"rgb", NVJPEG_OUTPUT_RGB}};

NvImageDecoderFlowUnit::NvImageDecoderFlowUnit(){};
NvImageDecoderFlowUnit::~NvImageDecoderFlowUnit(){};

modelbox::Status NvImageDecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  pixel_format_ = opts->GetString("pix_fmt", "bgr");
  if (NvImgPixelFormat.find(pixel_format_) == NvImgPixelFormat.end()) {
    auto errMsg = "pixel_format is invalid, configure is :" + pixel_format_;
    MBLOG_ERROR << errMsg;
    std::string valid_format;
    for (const auto &iter : NvImgPixelFormat) {
      if (valid_format.length() > 0) {
        valid_format += ", ";
      }
      valid_format += iter.first;
    }
    MBLOG_ERROR << "Valid pixel_format is: " << valid_format;
    return {modelbox::STATUS_BADCONF, errMsg};
  }
  MBLOG_DEBUG << "pixel_format " << pixel_format_;

  nvjpegStatus_t ret = NVJPEG_STATUS_SUCCESS;
  ret = nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &handle_);
  if (ret != NVJPEG_STATUS_SUCCESS) {
    MBLOG_ERROR << "nvjpegCreateSimple failed, ret " << ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status NvImageDecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image decode";

  // get input
  auto input_bufs = data_ctx->Input("in_encoded_image");
  auto output_bufs = data_ctx->Output("out_image");

  if (input_bufs->Size() <= 0) {
    auto err_msg =
        "input images batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  // each thread instantiated nvjpegJpegState_t
  nvjpegJpegState_t jpeg_handle{nullptr};
  auto jpeg_ret = nvjpegJpegStateCreate(handle_, &jpeg_handle);
  if (jpeg_ret != NVJPEG_STATUS_SUCCESS) {
    MBLOG_ERROR << "nvjpegJpegStateCreate failed, ret " << jpeg_ret;
    return modelbox::STATUS_FAULT;
  }
  Defer {
    jpeg_ret = nvjpegJpegStateDestroy(jpeg_handle);
    if (jpeg_ret != NVJPEG_STATUS_SUCCESS) {
      MBLOG_ERROR << "nvjpegJpegStateDestroy failed, ret " << jpeg_ret;
    }
  };

  // image decode
  for (auto &buffer : *input_bufs) {
    auto output_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
    auto input_data = static_cast<const uint8_t *>(buffer->ConstData());
    bool decode_ret = false;
    if (CheckImageType(input_data) == IMAGE_TYPE_JPEG) {
      decode_ret = DecodeJpeg(buffer, output_buffer, jpeg_handle);
    } 
    
    if (!decode_ret) {
      decode_ret = DecodeOthers(buffer, output_buffer);
    }

    if (!decode_ret) {
      output_buffer->SetError("ImageDecoder.DecodeFailed", "NvImageDecoder decode failed.");
    }
    output_bufs->PushBack(output_buffer);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status NvImageDecoderFlowUnit::Close() {
  if (handle_ != nullptr) {
    auto des_ret = nvjpegDestroy(handle_);
    if (des_ret != NVJPEG_STATUS_SUCCESS) {
      MBLOG_ERROR << "nvjpegDestroy failed, ret " << des_ret;
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_OK;
};

ImageType NvImageDecoderFlowUnit::CheckImageType(const uint8_t *input_data) {
  for (auto &format_value : ImgStreamFormat) {
    int ret = memcmp(input_data, format_value.second.data(),
                     format_value.second.size());
    if (ret == 0) {
      return format_value.first;
    }
  }

  return IMAGE_TYPE_OHTER;
}

bool NvImageDecoderFlowUnit::DecodeJpeg(
    const std::shared_ptr<modelbox::Buffer> &input_buffer,
    std::shared_ptr<modelbox::Buffer> &output_buffer,
    nvjpegJpegState_t &jpeg_handle) {
  int n_component = 0;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  auto input_data = static_cast<const uint8_t *>(input_buffer->ConstData());
  auto ret = nvjpegGetImageInfo(handle_, input_data, input_buffer->GetBytes(),
                                &n_component, &subsampling, widths, heights);
  if (ret != NVJPEG_STATUS_SUCCESS) {
    MBLOG_ERROR << "get input encode image info failed, ret " << ret;
    return false;
  }
  MBLOG_DEBUG << "widths: " << widths[0] << " " << widths[1] << " " << widths[2]
              << " " << widths[3];
  MBLOG_DEBUG << "heights " << heights[0] << " " << heights[1] << " "
              << heights[2] << " " << heights[3];

  // build planner buffer
  auto planner_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
  auto modelbox_ret =
      planner_buffer->Build((size_t)(widths[0] * heights[0] * n_component));
  if (modelbox_ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "build planner buffer failed, ret " << modelbox_ret;
    return false;
  }
  auto planner_data = static_cast<uint8_t *>(planner_buffer->MutableData());
  auto cuda_mem = std::dynamic_pointer_cast<modelbox::CudaMemory>(
      planner_buffer->GetDeviceMemory());
  cuda_mem->BindStream();
  auto stream = cuda_mem->GetBindStream();

  nvjpegImage_t imgdesc = {{planner_data, planner_data + widths[0] * heights[0],
                            planner_data + widths[0] * heights[0] * 2,
                            planner_data + widths[0] * heights[0] * 3},
                           {(uint32_t)widths[0], (uint32_t)widths[0],
                            (uint32_t)widths[0], (uint32_t)widths[0]}};

  ret = nvjpegDecode(handle_, jpeg_handle, input_data, input_buffer->GetBytes(),
                     NvImgPixelFormat[pixel_format_], &imgdesc, stream->Get());
  cudaStreamSynchronize(stream->Get());
  if (ret != NVJPEG_STATUS_SUCCESS) {
    MBLOG_ERROR << "nvjpegDecode failed, ret " << ret;
    return false;
  }

  // build output buffer
  output_buffer->Build((size_t)(widths[0] * heights[0] * n_component));
  auto output_data = static_cast<uint8_t *>(output_buffer->MutableData());

  // planner to packed image copy
  Npp8u *dst_planer[3] = {(Npp8u *)(planner_data),
                          (Npp8u *)(planner_data + widths[0] * heights[0]),
                          (Npp8u *)(planner_data + widths[0] * heights[0] * 2)};
  NppiSize dst_size = {widths[0], heights[0]};
  auto nppi_ret = nppiCopy_8u_P3C3R(dst_planer, widths[0], (Npp8u *)output_data,
                                    widths[0] * 3, dst_size);
  if (nppi_ret != NPP_SUCCESS) {
    MBLOG_ERROR << "nppiCopy_8u_P3C3R failed. ret is " << nppi_ret;
    return false;
  }

  output_buffer->Set("width", (int32_t)widths[0]);
  output_buffer->Set("height", (int32_t)heights[0]);
  output_buffer->Set("width_stride", (int32_t)widths[0]);
  output_buffer->Set("height_stride", (int32_t)heights[0]);
  output_buffer->Set("channel", (int32_t)n_component);
  output_buffer->Set("pix_fmt", pixel_format_);
  output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  output_buffer->Set("shape",
                     std::vector<size_t>{(size_t)heights[0], (size_t)widths[0],
                                         (size_t)n_component});
  output_buffer->Set("layout", std::string("hwc"));
  return true;
}

bool NvImageDecoderFlowUnit::DecodeOthers(
    const std::shared_ptr<modelbox::Buffer> &input_buffer,
    std::shared_ptr<modelbox::Buffer> &output_buffer) {
  auto input_data = static_cast<const uint8_t *>(input_buffer->ConstData());
  std::vector<uint8_t> input_data2(
      input_data, input_data + input_buffer->GetBytes() / sizeof(uint8_t));

  cv::Mat img_bgr = cv::imdecode(input_data2, cv::IMREAD_COLOR);
  if (img_bgr.data == nullptr || img_bgr.size == 0) {
    MBLOG_ERROR << "input image buffer is invalid, imdecode failed.";
    return false;
  }

  cv::Mat img_dest;
  if (pixel_format_ == "bgr") {
    img_dest = img_bgr;
  } else if (pixel_format_ == "rgb") {
    cv::cvtColor(img_bgr, img_dest, cv::COLOR_BGR2RGB);
  }

  output_buffer->Set("height", (int32_t)img_dest.rows);
  output_buffer->Set("width", (int32_t)img_dest.cols);
  output_buffer->Set("height_stride", (int32_t)img_dest.rows);
  output_buffer->Set("width_stride", (int32_t)img_dest.cols * 3);
  output_buffer->Set("channel", (int32_t)img_dest.channels());
  output_buffer->Set("pix_fmt", pixel_format_);
  output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  output_buffer->Set(
      "shape", std::vector<size_t>{(size_t)img_dest.rows, (size_t)img_dest.cols,
                                   (size_t)img_dest.channels()});
  output_buffer->Set("layout", std::string("hwc"));
  output_buffer->BuildFromHost(img_dest.data,
                               img_dest.total() * img_dest.elemSize(), nullptr);

  return true;
}

MODELBOX_FLOWUNIT(NvImageDecoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_encoded_image", "cpu"));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetResourceNice(false);
  desc.SetDescription(FLOWUNIT_DESC);
  std::map<std::string, std::string> format_list;

  for (auto &item : NvImgPixelFormat) {
    format_list[item.first] = item.first;
  }

  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "list", true, "bgr", "the imdecode output pixel format",
      format_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
