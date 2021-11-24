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


#include "split_img_bbox.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

SplitImgBBoxFlowUnit::SplitImgBBoxFlowUnit(){};
SplitImgBBoxFlowUnit::~SplitImgBBoxFlowUnit(){};

modelbox::Status SplitImgBBoxFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status SplitImgBBoxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  // {[bbox, img], ... , [bbox, img]}
  auto img_bbox_input = ctx->Input("In_true");
  // {[img], ... ,[img]}
  auto img_bufs = ctx->Output("Out_img");
  // {[bbox], ... ,[bbox]}
  auto bbox_bufs = ctx->Output("Out_bbox");

  for (size_t i = 0; i < img_bbox_input->Size(); ++i) {
    auto input_data = (const BBox*)(img_bbox_input->ConstBufferData(i));
    auto device = GetBindDevice();

    std::shared_ptr<modelbox::Buffer> bbox_buffer =
        std::make_shared<modelbox::Buffer>(device);

    std::shared_ptr<modelbox::Buffer> img_buffer =
        std::make_shared<modelbox::Buffer>(device);

    size_t bbox_buffer_data_size = sizeof(BBox);
    size_t data_size = img_bbox_input->At(i)->GetBytes();
    size_t img_buffer_data_size = (data_size - bbox_buffer_data_size);

    bbox_buffer->Build(bbox_buffer_data_size);
    img_buffer->Build(img_buffer_data_size);

    img_buffer->CopyMeta(img_bbox_input->At(i));

    memcpy_s(bbox_buffer->MutableData(), bbox_buffer_data_size, input_data,
             bbox_buffer_data_size);
    memcpy_s(img_buffer->MutableData(), img_buffer_data_size, ++input_data,
             img_buffer_data_size);

    img_bufs->PushBack(img_buffer);
    bbox_bufs->PushBack(bbox_buffer);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(SplitImgBBoxFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_true", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_img", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_bbox", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}