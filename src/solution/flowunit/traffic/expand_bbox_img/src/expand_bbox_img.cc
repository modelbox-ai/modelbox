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


#include "expand_bbox_img.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

ExpandBBoxImgFlowUnit::ExpandBBoxImgFlowUnit(){};
ExpandBBoxImgFlowUnit::~ExpandBBoxImgFlowUnit(){};

modelbox::Status ExpandBBoxImgFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExpandBBoxImgFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status ExpandBBoxImgFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_INFO << "process image crop and resize on cpu";
  // In_img : image  {[image]}
  auto input_img = ctx->Input("In_img");

  // In_bbox : boxes  {[bbox, ..., bbox]}
  auto input_bbox = ctx->Input("In_bbox");

  // Out_img : images {[image], ... ,[image]}
  auto output_img = ctx->Output("Out_img");

  // Out_bbox: car_bboxes {[bbox], ... ,[bbox]}
  auto output_bbox = ctx->Output("Out_bbox");

  MBLOG_INFO << "begin process batch";

  /* get bboxes */
  size_t num_bboxes = input_bbox->At(0)->GetBytes() / sizeof(BBox);

  if (num_bboxes > 0) {
    // Out_bbox
    std::vector<size_t> bbox_shape(num_bboxes, sizeof(BBox));
    output_bbox->Build(bbox_shape);

    std::vector<std::shared_ptr<BBox>> bboxs;
    for (size_t i = 0; i < num_bboxes; ++i) {
      std::shared_ptr<BBox> b = std::make_shared<BBox>();
      memcpy_s(
          b.get(), sizeof(BBox),
          (const char *)(input_bbox->ConstBufferData(0)) + (sizeof(BBox) * i),
          sizeof(BBox));
      bboxs.push_back(b);

      auto bbox_data = (BBox *)output_bbox->MutableBufferData(i);
      memcpy_s(bbox_data, sizeof(BBox), b.get(), sizeof(BBox));
    }

    /* get images */
    int32_t width, height, channel, rate_den, rate_num;
    input_img->At(0)->Get("width", width);
    input_img->At(0)->Get("height", height);
    input_img->At(0)->Get("channel", channel);
    input_img->At(0)->Get("rate_den", rate_den);
    input_img->At(0)->Get("rate_num", rate_num);

    // RGB
    cv::Mat image(height, width, CV_8UC3);
    memcpy_s(image.data, image.total() * image.elemSize(),
             input_img->ConstBufferData(0), input_img->At(0)->GetBytes());
    MBLOG_INFO << "end get images";

    auto device = GetBindDevice();
    for (auto &b : bboxs) {
      std::shared_ptr<modelbox::Buffer> buffer =
          std::make_shared<modelbox::Buffer>(device);

      // ROI crop
      cv::Rect crop_roi = cv::Rect(b->x, b->y, b->w, b->h);
      cv::Mat crop_img;

      crop_img = image(crop_roi);
      size_t data_size = crop_img.total() * crop_img.elemSize();
      buffer->Build(data_size);

      int32_t crop_width = crop_img.cols;
      int32_t crop_height = crop_img.rows;
      cv::Mat img_dest(crop_height, crop_width, CV_8UC3);
      crop_img.copyTo(img_dest);
      memcpy_s(buffer->MutableData(), data_size, img_dest.data, data_size);

      buffer->Set("width", crop_width);
      buffer->Set("height", crop_height);
      buffer->Set("channel", RGB_CHANNELS);
      buffer->Set("pix_fmt", std::string("rgb"));
      buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);

      output_img->PushBack(buffer);
    }
  } else {
    // Out_bbox bbox {[]}
    std::vector<size_t> bbox_shape{0};
    output_bbox->Build(bbox_shape);

    // Out_img {[]}
    std::vector<size_t> img_shape{0};
    output_img->Build(img_shape);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ExpandBBoxImgFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_img", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_bbox", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(
      modelbox::FlowUnitOutput("Out_img", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(
      modelbox::FlowUnitOutput("Out_bbox", modelbox::DEVICE_TYPE));
  desc.SetOutputType(modelbox::EXPAND);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}