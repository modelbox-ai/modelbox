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


#include "car_condition.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

CarConditionFlowUnit::CarConditionFlowUnit(){};
CarConditionFlowUnit::~CarConditionFlowUnit(){};

modelbox::Status CarConditionFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status CarConditionFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  // {[img], ... , [img]}
  auto img_input = ctx->Input("In_img");
  // {[box], ... ,[img]}
  auto bbox_input = ctx->Input("In_bbox");
  // {[box, img], ....,[box, img]}
  auto true_output = ctx->Output("Out_true");
  // {[], ... ,[]}
  auto false_output = ctx->Output("Out_false");

  auto device = GetBindDevice();
  for (size_t i = 0; i < bbox_input->Size(); ++i) {
    size_t flag = bbox_input->At(i)->GetBytes();

    std::shared_ptr<modelbox::Buffer> buffer =
        std::make_shared<modelbox::Buffer>(device);

    if (flag > 0) {
      size_t img_data_size = img_input->At(i)->GetBytes();
      size_t bbox_data_size = bbox_input->At(i)->GetBytes();
      buffer->Build(img_data_size + bbox_data_size);
      buffer->CopyMeta(img_input->At(i));

      auto img_data = (img_input->ConstBufferData(i));
      auto bbox_data = (bbox_input->ConstBufferData(i));

      memcpy_s(buffer->MutableData(), bbox_data_size, bbox_data,
               bbox_data_size);

      memcpy_s((char*)buffer->MutableData() + bbox_data_size, img_data_size,
               img_data, img_data_size);

      true_output->PushBack(buffer);
    } else {
      buffer->Build(0);
      false_output->PushBack(buffer);
    }
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(CarConditionFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_img", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_bbox", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(
      modelbox::FlowUnitOutput("Out_true", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(
      modelbox::FlowUnitOutput("Out_false", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetConditionType(modelbox::IF_ELSE);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
