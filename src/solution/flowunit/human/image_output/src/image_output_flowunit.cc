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


#include "image_output_flowunit.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

ImageOutputFlowUnit::ImageOutputFlowUnit(){};
ImageOutputFlowUnit::~ImageOutputFlowUnit(){};

modelbox::Status ImageOutputFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  default_dest_ = opts->GetString("default_dest", "");
  if (default_dest_.length() <= 0) {
    MBLOG_ERROR
        << "image_output flowunit : can not get default_dest from config file";
    return modelbox::STATUS_BADCONF;
  }

  MBLOG_DEBUG << "image output dest: " << default_dest_;
  return modelbox::STATUS_OK;
}

modelbox::Status ImageOutputFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto input_buf = ctx->Input("frame_info");
  auto saveframeID =
      std::static_pointer_cast<int>(ctx->GetPrivate(SAVEFRAMEID));

  int32_t cols = 0;
  int32_t rows = 0;
  int32_t channels = 0;

  for (size_t i = 0; i < input_buf->Size(); i++) {
    if (!input_buf->At(i)->Get("width", cols)) {
      MBLOG_ERROR
          << "image_output flowunit can not get input 'width' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
    }
    if (!input_buf->At(i)->Get("height", rows)) {
      MBLOG_ERROR
          << "image_output flowunit can not get input 'height' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
    }
    if (!input_buf->At(i)->Get("channel", channels)) {
      MBLOG_ERROR
          << "image_output flowunit can not get input 'channel' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key channel"};
    }
    auto input_data = static_cast<const uchar *>(input_buf->ConstBufferData(i));

    cv::Mat img_data(cv::Size(cols, rows), CV_8UC3);
    auto ret = memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                        input_data, input_buf->At(i)->GetBytes());
    if (EOK != ret) {
      MBLOG_ERROR << "cpu image_output failed, ret " << ret;
      return modelbox::STATUS_FAULT;
    }

    // RGB2BGR
    cv::Mat bgr_img;
    cv::cvtColor(img_data, bgr_img, cv::COLOR_RGB2BGR);
    std::string img_name =
        default_dest_ + std::to_string(*saveframeID) + std::string(".jpg");
    cv::imwrite(img_name, bgr_img);
    (*saveframeID)++;
  }

  ctx->SetPrivate(SAVEFRAMEID, saveframeID);

  return modelbox::STATUS_OK;
}

modelbox::Status ImageOutputFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto saveframeID = std::make_shared<int>();
  *saveframeID = 0;
  data_ctx->SetPrivate(SAVEFRAMEID, saveframeID);

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ImageOutputFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(
      modelbox::FlowUnitInput("frame_info", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::STREAM);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("method", "string", true, "",
                                                "path to save output image"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}