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

#include "image_decoder.h"

#include <modelbox/base/crypto.h>
#include <securec.h>

#include "modelbox/flowunit_api_helper.h"

ImageDecoderFlowUnit::ImageDecoderFlowUnit() = default;
ImageDecoderFlowUnit::~ImageDecoderFlowUnit() = default;

std::vector<std::string> CvImgPixelFormat{"bgr", "rgb", "nv12"};

modelbox::Status ImageDecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  pixel_format_ = opts->GetString("pix_fmt", modelbox::IMG_DEFAULT_FMT);
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
  out_pix_fmt_ = modelbox::GetRGAFormat(pixel_format_);

  return jpeg_dec_.Init();
}

MppFrame ImageDecoderFlowUnit::JpegDec(
    std::shared_ptr<modelbox::Buffer> &buffer, int &w, int &h) {
  auto *input_data = (u_char *)(buffer->ConstData());
  auto data_len = buffer->GetBytes();
  if (data_len <= 0) {
    MBLOG_ERROR << "buffer size is invalid";
    return nullptr;
  }

  return jpeg_dec_.Decode(input_data, data_len, w, h);
}

std::shared_ptr<modelbox::Buffer> ImageDecoderFlowUnit::DecodeFromCPU(
    std::shared_ptr<modelbox::Buffer> &in_buffer) {
  const auto *input_data_ptr =
      static_cast<const u_char *>(in_buffer->ConstData());
  std::vector<u_char> input_data(
      input_data_ptr, input_data_ptr + in_buffer->GetBytes() / sizeof(u_char));

  cv::Mat img_bgr = cv::imdecode(input_data, cv::IMREAD_COLOR);
  if (img_bgr.data == nullptr) {
    std::string error_msg = "input image buffer is invalid, imdecode failed.";
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  cv::Mat img_dest;
  if (pixel_format_ == "bgr") {
    img_dest = img_bgr;
  } else if (pixel_format_ == "rgb") {
    cv::cvtColor(img_bgr, img_dest, cv::COLOR_BGR2RGB);
  } else if (pixel_format_ == "nv12") {
    img_dest = BGR2YUV_NV12(img_bgr);
  } else {
    std::string error_msg = "no support pixel format:" + pixel_format_;
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  if (!modelbox::StatusError) {
    std::string error_msg =
        "input image decode success, but transform nv12 format failed.";
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  auto output_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
  if (output_buffer == nullptr) {
    std::string error_msg = "create output buffer fail.";
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  auto img_size = img_dest.total() * img_dest.elemSize();
  MBLOG_ERROR << img_size;

  auto ret = output_buffer->Build(img_size);
  if (ret != modelbox::STATUS_OK) {
    auto error_msg = "Create buffer fail, size=" + std::to_string(img_size);
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  auto *mpp_buf = (MppBuffer)(output_buffer->MutableData());
  auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);

  auto e_ret =
      memcpy_s(cpu_buf, output_buffer->GetBytes(), img_dest.data, img_size);
  if (e_ret != EOK) {
    auto error_msg = "memcpy_s fail, e_ret=" + std::to_string(e_ret);
    MBLOG_ERROR << error_msg;
    auto buffer = std::make_shared<modelbox::Buffer>();
    buffer->SetError("ImageDecoder.DecodeFailed", error_msg);
    return buffer;
  }

  output_buffer->Set("width", (int32_t)img_bgr.cols);
  output_buffer->Set("height", (int32_t)img_bgr.rows);
  auto width_stride = (int32_t)img_bgr.cols;
  if (pixel_format_ == "rgb" || pixel_format_ == "bgr") {
    width_stride *= 3;
  }

  output_buffer->Set("width_stride", width_stride);
  output_buffer->Set("height_stride", (int32_t)img_bgr.rows);
  output_buffer->Set("channel", (int32_t)img_dest.channels());
  output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  output_buffer->Set(
      "shape", std::vector<size_t>{(size_t)img_dest.rows, (size_t)img_dest.cols,
                                   (size_t)img_dest.channels()});
  output_buffer->Set("layout", std::string("hwc"));
  return output_buffer;
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

modelbox::Status ImageDecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  // get input
  auto input_bufs = ctx->Input("in_encoded_image");
  auto output_bufs = ctx->Output("out_image");
  if (input_bufs->Size() <= 0) {
    auto msg = "input images batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  std::vector<size_t> output_shape;
  for (auto &buffer : *input_bufs) {
    int w = 0;
    int h = 0;
    std::shared_ptr<modelbox::Buffer> out_buf = nullptr;
    MppFrame frame = JpegDec(buffer, w, h);
    if (frame == nullptr) {
      const auto *msg = "failed to MppJpegDec";
      MBLOG_ERROR << msg;
      out_buf = DecodeFromCPU(buffer);
    } else {
      out_buf = modelbox::ColorChange(frame, out_pix_fmt_, GetBindDevice());
      if (out_buf == nullptr) {
        const auto *msg = "failed to color change";
        MBLOG_ERROR << msg;
        out_buf = DecodeFromCPU(buffer);
      }
    }

    // build out_buf
    out_buf->Set("pix_fmt", pixel_format_);
    output_bufs->PushBack(out_buf);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ImageDecoderFlowUnit, rk_imgdec_desc) {
  rk_imgdec_desc.SetFlowUnitName(FLOWUNIT_NAME);
  rk_imgdec_desc.SetFlowUnitGroupType("Image");
  rk_imgdec_desc.AddFlowUnitInput({"in_encoded_image", "cpu"});
  rk_imgdec_desc.AddFlowUnitOutput({"out_image"});

  rk_imgdec_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "string", true, modelbox::IMG_DEFAULT_FMT,
      "the output pixel format"));

  rk_imgdec_desc.SetFlowType(modelbox::NORMAL);
  rk_imgdec_desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(rk_imgdec_desc) {
  rk_imgdec_desc.Desc.SetName(FLOWUNIT_NAME);
  rk_imgdec_desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  rk_imgdec_desc.Desc.SetType(modelbox::DEVICE_TYPE);
  rk_imgdec_desc.Desc.SetDescription(FLOWUNIT_DESC);
  rk_imgdec_desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}
