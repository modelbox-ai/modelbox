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

#include "draw_bbox_flowunit.h"
#include <securec.h>
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

DrawBBoxFlowUnit::DrawBBoxFlowUnit() = default;
DrawBBoxFlowUnit::~DrawBBoxFlowUnit() = default;

modelbox::Status DrawBBoxFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status DrawBBoxFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status DrawBBoxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_INFO << "process image draw bbox on cpu";

  auto input1_bufs = data_ctx->Input("in_region");
  if (input1_bufs->Size() <= 0) {
    auto errMsg = "in_region batch is " + std::to_string(input1_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto input2_bufs = data_ctx->Input("in_image");
  if (input2_bufs->Size() <= 0) {
    auto errMsg = "in_image batch is " + std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  if (input1_bufs->Size() != input2_bufs->Size()) {
    auto errMsg = "in_image batch is not match in_region batch. in_image is " +
                  std::to_string(input1_bufs->Size()) + ",in_region is " +
                  std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto output_bufs = data_ctx->Output("out_image");

  std::vector<size_t> shape;
  for (size_t i = 0; i < input2_bufs->Size(); ++i) {
    shape.emplace_back(input2_bufs->At(i)->GetBytes());
  }

  output_bufs->Build(shape);
  MBLOG_INFO << "begin process batch";
  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    // get bboxes
    size_t num_bboxes = input1_bufs->At(i)->GetBytes() / sizeof(BBox);

    MBLOG_INFO << "num_bboxes: " << num_bboxes;

    std::vector<std::shared_ptr<BBox>> bboxs;
    for (size_t j = 0; j < num_bboxes; ++j) {
      std::shared_ptr<BBox> b = std::make_shared<BBox>();
      memcpy_s(
          b.get(), sizeof(BBox),
          (const char *)(input1_bufs->ConstBufferData(i)) + (sizeof(BBox) * j),
          sizeof(BBox));
      bboxs.push_back(b);
    }

    // get images
    int32_t width = 0;
    int32_t height = 0;
    int32_t channel = 0;
    int32_t rate_den = 0;
    int32_t rate_num = 0;
    input2_bufs->At(i)->Get("width", width);
    input2_bufs->At(i)->Get("height", height);
    input2_bufs->At(i)->Get("channel", channel);
    input2_bufs->At(i)->Get("rate_den", rate_den);
    input2_bufs->At(i)->Get("rate_num", rate_num);
    std::string pix_fmt = "rgb";
    input2_bufs->At(i)->Get("pix_fmt", pix_fmt);

    MBLOG_INFO << "w:" << width << ",h:" << height << ",c:" << channel;

    cv::Mat image(height, width, CV_8UC3);
    memcpy_s(image.data, image.total() * image.elemSize(),
             input2_bufs->ConstBufferData(i), input2_bufs->At(i)->GetBytes());
    MBLOG_INFO << "end get images";

    // draw bboxes
    for (auto &b : bboxs) {
      MBLOG_DEBUG << "draw bbox : has box " << b->x << " " << b->y << " "
                  << b->w << " " << b->h << " " << b->score << " "
                  << b->category;
      cv::rectangle(image, cv::Point(b->x, b->y),
                    cv::Point(b->x + b->w, b->y + b->h), cv::Scalar(255, 0, 0),
                    5, 8, 0);
    }

    // output data
    auto output_buffer = output_bufs->At(i);
    auto *output_data = output_buffer->MutableData();
    memcpy_s(output_data, output_buffer->GetBytes(), image.data,
             image.total() * image.elemSize());
    output_buffer->Set("width", width);
    output_buffer->Set("height", height);
    output_buffer->Set("width_stride", width);
    output_buffer->Set("height_stride", height);
    output_buffer->Set("channel", channel);
    output_buffer->Set("pix_fmt", pix_fmt);
    output_buffer->Set("layout", std::string("hwc"));
    output_buffer->Set(
        "shape",
        std::vector<size_t>{(size_t)height, (size_t)width, (size_t)channel});
    output_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    output_buffer->Set("rate_den", rate_den);
    output_buffer->Set("rate_num", rate_num);
  }

  MBLOG_INFO << "draw bbox finish";
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(DrawBBoxFlowUnit, desc) {
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
