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


#include "face_preprocess_flowunit.h"

#include <securec.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

PreProcessFlowUnit::PreProcessFlowUnit(){};
PreProcessFlowUnit::~PreProcessFlowUnit(){};

std::map<std::string, cv::InterpolationFlags> kCVResizeMethod = {
    {"inter_nearest", cv::INTER_NEAREST},
    {"inter_linear", cv::INTER_LINEAR},
    {"inter_cubic", cv::INTER_CUBIC},
    {"inter_area", cv::INTER_AREA},
    {"inter_lanczos4", cv::INTER_LANCZOS4},
    {"inter_max", cv::INTER_MAX},
    {"warp_fill_outliers", cv::WARP_FILL_OUTLIERS},
    {"warp_inverse_map", cv::WARP_INVERSE_MAP},
};

modelbox::Status PreProcessFlowUnit::Open(
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
    auto errMsg = "resize width or height is not configured or invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  method_ = opts->GetString("method", "inter_linear");
  if (kCVResizeMethod.find(method_) == kCVResizeMethod.end()) {
    auto errMsg = "resize method is invalid, configure is :" + method_;
    MBLOG_ERROR << errMsg;
    std::string validmethod;
    for (const auto &iter : kCVResizeMethod) {
      if (validmethod.length() > 0) {
        validmethod += ", ";
      }
      validmethod += iter.first;
    }
    MBLOG_ERROR << "Valid method is: " << validmethod;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  MBLOG_DEBUG << "resize dest width " << dest_width_ << ", resize dest height "
              << dest_height_ << ", resize method " << method_;
  return modelbox::STATUS_OK;
}
modelbox::Status PreProcessFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status PreProcessFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "process image cvresize";

  auto input_bufs = ctx->Input(PREPROCESS_UNIT_IN_NAME[0]);
  auto output_bufs = ctx->Output(PREPROCESS_UNIT_OUT_NAME[0]);

  if (input_bufs->Size() <= 0) {
    auto errMsg = "input images batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  size_t channel = RGB_CHANNELS;
  std::vector<size_t> sub_shape{dest_width_, dest_height_, channel};
  std::vector<size_t> tensor_shape(input_bufs->Size(),
                                   modelbox::Volume(sub_shape) * sizeof(float));
  auto ret = output_bufs->Build(tensor_shape);
  if (!ret) {
    auto errMsg = "build empty output failed in face preprocess unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  errno_t err;

  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    int32_t width, height, channel;
    std::string pix_fmt;
    bool exists = false;
    exists = input_bufs->At(i)->Get("height", height);
    if (!exists) {
      MBLOG_ERROR << "meta don't have key height";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
    }

    exists = input_bufs->At(i)->Get("width", width);
    if (!exists) {
      MBLOG_ERROR << "meta don't have key width";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
    }

    exists = input_bufs->At(i)->Get("pix_fmt", pix_fmt);
    if (!exists && !input_bufs->At(i)->Get("channel", channel)) {
      MBLOG_ERROR << "meta don't have key pix_fmt or channel";
      return {modelbox::STATUS_NOTSUPPORT,
              "meta don't have key pix_fmt or channel"};
    }

    if (exists && pix_fmt != "rgb" && pix_fmt != "bgr") {
      MBLOG_ERROR << "unsupport pix format.";
      return {modelbox::STATUS_NOTSUPPORT, "unsupport pix format."};
    }

    channel = RGB_CHANNELS;
    MBLOG_DEBUG << "get " << width << " rows " << height << " channel "
                << channel;

    auto input_data =
        static_cast<const u_char *>(input_bufs->ConstBufferData(i));
    cv::Mat img_data(cv::Size(width, height), CV_8UC3);
    err = memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                   input_data, input_bufs->At(i)->GetBytes());
    if (err != 0) {
      auto errMsg =
          "Copying origin image from input port in face preprocess unit "
          "failed.";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }

    float scale =
        cv::min(float(dest_width_) / width, float(dest_height_) / height);
    auto scaleSize = cv::Size(width * scale, height * scale);
    cv::Mat resized;
    cv::resize(img_data, resized, scaleSize, 0, 0);

    cv::Mat cropped(dest_height_, dest_width_, CV_8UC3);
    cv::Rect rect((dest_width_ - scaleSize.width) / 2,
                  (dest_height_ - scaleSize.height) / 2, scaleSize.width,
                  scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1.);

    auto output = static_cast<float *>(output_bufs->MutableBufferData(i));
    err = memcpy_s(output, output_bufs->At(i)->GetBytes(), img_float.data,
                   img_float.total() * img_float.elemSize());
    if (err != 0) {
      auto errMsg =
          "Copying image to output port in face preprocess unit failed.";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    const uint8_t CHANNEL_NUM = 3;
    size_t size =
        (output_bufs->At(i)->GetBytes() / sizeof(float)) / CHANNEL_NUM;

    for (size_t j = 0; j < size; j++) {
      for (size_t c = 0; c < CHANNEL_NUM; c++) {
        output[j * CHANNEL_NUM + c] =
            ((output[j * CHANNEL_NUM + c] * normalizes_[c]) - localmeans_[c]) /
            variances_[c];
      }
    }

    output_bufs->At(i)->Set("width", (int32_t)dest_width_);
    output_bufs->At(i)->Set("height", (int32_t)dest_height_);
    output_bufs->At(i)->Set("channel", channel);
    output_bufs->At(i)->Set("pix_fmt", std::string("bgr"));
    output_bufs->At(i)->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
  }

  return modelbox::STATUS_OK;
}

cv::InterpolationFlags PreProcessFlowUnit::GetCVResizeMethod(
    std::string resizeType) {
  transform(resizeType.begin(), resizeType.end(), resizeType.begin(),
            ::tolower);

  if (kCVResizeMethod.find(resizeType) == kCVResizeMethod.end()) {
    MBLOG_WARN << "opencv resize not support method \"" << resizeType << "\"";
    MBLOG_WARN << "using defalt method \"inter_linear\"";
    return cv::INTER_LINEAR;
  }

  return kCVResizeMethod[resizeType];
}

MODELBOX_FLOWUNIT(PreProcessFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  for (auto port : PREPROCESS_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }

  for (auto port : PREPROCESS_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
  desc.SetFlowType(modelbox::NORMAL);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_width", "int", true,
                                                "640", "the resize width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("image_height", "int", true,
                                                "480", "the resize height"));

  std::map<std::string, std::string> method_list;
  for (auto &item : kCVResizeMethod) {
    method_list[item.first] = item.first;
  }

  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("method", "list", true, "inter_linear",
                             "the resize method", method_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
