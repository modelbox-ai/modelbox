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

#include "image_decoder.h"

#include "modelbox/flowunit_api_helper.h"

#include <securec.h>

ImageDecoderFlowUnit::ImageDecoderFlowUnit(){};
ImageDecoderFlowUnit::~ImageDecoderFlowUnit(){};

std::vector<std::string> CvImgPixelFormat{"bgr", "rgb", "nv12"};

modelbox::Status ImageDecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  pixel_format_ = opts->GetString("pix_fmt", "bgr");
  if (find(CvImgPixelFormat.begin(), CvImgPixelFormat.end(), pixel_format_) ==
      CvImgPixelFormat.end()) {
    auto errMsg = "pixel_format is invalid, configure is :" + pixel_format_;
    MBLOG_ERROR << errMsg;
    std::string valid_format;
    for (const auto &iter : CvImgPixelFormat) {
      if (valid_format.length() > 0) {
        valid_format += ", ";
      }
      valid_format += iter;
    }
    MBLOG_ERROR << "Valid pixel_format is: " << valid_format;
    return {modelbox::STATUS_BADCONF, errMsg};
  }
  MBLOG_DEBUG << "pixel_format " << pixel_format_;

  return modelbox::STATUS_OK;
}

modelbox::Status ImageDecoderFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status ImageDecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image decode";

  // get input
  auto input_bufs = data_ctx->Input("in_encoded_image");
  auto output_bufs = data_ctx->Output("out_image");
  if (input_bufs->Size() <= 0) {
    auto errMsg = "input images batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  // decode
  std::vector<cv::Mat> output_img_list;
  std::vector<size_t> output_shape;
  for (auto &buffer : *input_bufs) {
    const auto *input_data = static_cast<const u_char *>(buffer->ConstData());
    std::vector<u_char> input_data2(
        input_data, input_data + buffer->GetBytes() / sizeof(u_char));

    cv::Mat img_bgr = cv::imdecode(input_data2, cv::IMREAD_COLOR);
    if (img_bgr.data == nullptr || img_bgr.size == nullptr) {
      std::string error_msg = "input image buffer is invalid, imdecode failed.";
      MBLOG_ERROR << error_msg;
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
      output_bufs->PushBack(buffer);
      continue;
    }
    cv::Mat img_dest;
    if (pixel_format_ == "bgr") {
      img_dest = img_bgr;
    } else if (pixel_format_ == "rgb") {
      cv::cvtColor(img_bgr, img_dest, cv::COLOR_BGR2RGB);
    } else if (pixel_format_ == "nv12") {
      img_dest = BGR2YUV_NV12(img_bgr);
    }

    if (!modelbox::StatusError) {
      std::string error_msg = "input image decode success, but transform nv12 format failed.";
      MBLOG_ERROR << error_msg;
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
      output_bufs->PushBack(buffer);
      continue;
    }

    MBLOG_DEBUG << "decode image clos : " << img_bgr.cols
                << ", rows : " << img_bgr.rows
                << "channles : " << img_bgr.channels();

    // build output_buffer
    output_bufs->EmplaceBack(
        img_dest.data, img_dest.total() * img_dest.elemSize(),
        [img_dest](void *unused) { /* hold img dest*/ });
    auto output_buffer = output_bufs->Back();
    output_buffer->Set("width", (int32_t)img_bgr.cols);
    output_buffer->Set("height", (int32_t)img_bgr.rows);
    auto width_stride = (int32_t)img_bgr.cols;
    if (pixel_format_ == "rgb" || pixel_format_ == "bgr") {
      width_stride *= 3;
    }
    
    output_buffer->Set("width_stride", width_stride);
    output_buffer->Set("height_stride", (int32_t)img_bgr.rows);
    output_buffer->Set("channel", (int32_t)img_dest.channels());
    output_buffer->Set("pix_fmt", pixel_format_);
    output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    output_buffer->Set(
        "shape",
        std::vector<size_t>{(size_t)img_dest.rows, (size_t)img_dest.cols,
                            (size_t)img_dest.channels()});
    output_buffer->Set("layout", std::string("hwc"));
  }

  return modelbox::STATUS_OK;
}

cv::Mat ImageDecoderFlowUnit::BGR2YUV_NV12(const cv::Mat &src_bgr) {
  modelbox::StatusError = modelbox::STATUS_OK;
  cv::Mat dst_nv12(src_bgr.rows * 1.5, src_bgr.cols, CV_8UC1, cv::Scalar(0));
  cv::Mat src_yuv_i420;
  cv::cvtColor(src_bgr, src_yuv_i420, cv::COLOR_BGR2YUV_I420);

  size_t len_y = src_bgr.rows * src_bgr.cols;
  size_t len_u = len_y / 4;
  auto ret = memcpy_s(dst_nv12.data, len_y, src_yuv_i420.data, len_y);
  if (ret != EOK) {
    MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", size " << len_y;
    dst_nv12.release();
    modelbox::StatusError = {modelbox::STATUS_FAULT};
    return dst_nv12;
  }
  for (size_t i = 0; i < len_u; ++i) {
    dst_nv12.data[len_y + 2 * i] = src_yuv_i420.data[len_y + i];
    dst_nv12.data[len_y + 2 * i + 1] = src_yuv_i420.data[len_y + len_u + i];
  }

  return dst_nv12;
}

MODELBOX_FLOWUNIT(ImageDecoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_encoded_image"});
  desc.AddFlowUnitOutput({"out_image"});

  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "string", true, "bgr", "the output pixel format"));

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
