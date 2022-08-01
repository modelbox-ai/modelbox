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

#include "cv_crop_flowunit.h"

#include <securec.h>

#include "image_process.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
using namespace imageprocess;
CVCropFlowUnit::CVCropFlowUnit() = default;
CVCropFlowUnit::~CVCropFlowUnit() = default;

modelbox::Status CVCropFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status CVCropFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status CVCropFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image cv_crop";

  auto input_img_bufs = data_ctx->Input("in_image");
  if (input_img_bufs->Size() <= 0) {
    auto errMsg =
        "in_image image batch is " + std::to_string(input_img_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto input_box_bufs = data_ctx->Input("in_region");
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

  auto output_bufs = data_ctx->Output("out_image");
  output_bufs->CopyMeta(input_img_bufs);

  for (size_t i = 0; i < input_img_bufs->Size(); ++i) {
    int32_t width;
    int32_t height;
    int32_t width_dest;
    int32_t height_dest;
    int32_t channel;
    std::string pix_fmt;

    bool exists = false;
    auto img_buffer = input_img_bufs->At(i);
    exists = img_buffer->Get("height", height);
    if (!exists) {
      MBLOG_ERROR << "meta don't have key height";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
    }

    exists = img_buffer->Get("width", width);
    if (!exists) {
      MBLOG_ERROR << "meta don't have key width";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
    }

    exists = img_buffer->Get("pix_fmt", pix_fmt);
    if (!exists && !img_buffer->Get("channel", channel)) {
      MBLOG_ERROR << "meta don't have key pix_fmt or channel";
      return {modelbox::STATUS_NOTSUPPORT,
              "meta don't have key pix_fmt or channel"};
    }

    if (exists && pix_fmt != "rgb" && pix_fmt != "bgr") {
      MBLOG_ERROR << "unsupport pix format.";
      return {modelbox::STATUS_NOTSUPPORT, "unsupport pix format."};
    }

    channel = RGB_CHANNLES;

    const auto *bbox =
        static_cast<const RoiBox *>(input_box_bufs->ConstBufferData(i));
    if (!CheckRoiBoxVaild(bbox, width, height)) {
      return {modelbox::STATUS_FAULT, "roi box param is invaild !"};
    }

    MBLOG_DEBUG << "crop bbox :  " << bbox->x << " " << bbox->y << " "
                << bbox->w << " " << bbox->h;
    auto *input_data = const_cast<void *>(input_img_bufs->ConstBufferData(i));
    cv::Mat img_data(cv::Size(width, height), CV_8UC3, input_data);
    MBLOG_DEBUG << "ori image : cols " << img_data.cols << " rows "
                << img_data.rows << " channel " << img_data.channels();

    cv::Rect my_roi(bbox->x, bbox->y, bbox->w, bbox->h);
    cv::Mat cropped;
    cropped = img_data(my_roi);
    auto img_dest = std::make_shared<cv::Mat>();
    cropped.copyTo(*img_dest);
    size_t size_bytes = img_dest->total() * img_dest->elemSize();
    output_bufs->EmplaceBack(img_dest->data, size_bytes, [img_dest](void *ptr) {
      /* Only capture pkt */
    });
    auto output_buffer = output_bufs->Back();
    width_dest = img_dest->cols;
    height_dest = img_dest->rows;
    output_buffer->Set("width", width_dest);
    output_buffer->Set("height", height_dest);
    output_buffer->Set("width_stride", width_dest * 3);
    output_buffer->Set("height_stride", height_dest);
    output_buffer->Set("channel", channel);
    output_buffer->Set("pix_fmt", pix_fmt);
    output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    output_buffer->Set("shape", std::vector<size_t>{(size_t)height_dest,
                                                    (size_t)width_dest, 3});
    output_buffer->Set("layout", std::string("hwc"));
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(CVCropFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_region"});
  desc.AddFlowUnitOutput({"out_image"});
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
