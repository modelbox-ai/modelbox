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


#include "face_condition.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

FaceConditionFlowUnit::FaceConditionFlowUnit(){};
FaceConditionFlowUnit::~FaceConditionFlowUnit(){};

modelbox::Status FaceConditionFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status FaceConditionFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceConditionFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_INFO << "process face condition";
  // {[img], ... , [img]}
  auto input_bufs = ctx->Input("In_img");
  if (input_bufs->Size() <= 0) {
    auto errMsg = "In_img batch in face condition unit is " +
                  std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // {[img], ....,[img]}
  auto true_output = ctx->Output("Out_true");
  // {[], ... ,[]}
  auto false_output = ctx->Output("Out_false");

  if (input_bufs->At(0)->GetBytes() == 0) {
    auto ret = false_output->Build({0});
    if (!ret) {
      auto errMsg = "build empty output failed in face condition unit";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    return modelbox::STATUS_OK;
  }
  errno_t err;
  std::vector<size_t> tensor_shape;
  for (size_t batch_idx = 0; batch_idx < input_bufs->Size(); ++batch_idx) {
    tensor_shape.emplace_back(input_bufs->At(batch_idx)->GetBytes());
  }
  auto ret = true_output->Build(tensor_shape);
  if (!ret) {
    auto errMsg = "build unempty output failed in face condition unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  for (size_t batch_idx = 0; batch_idx < input_bufs->Size(); ++batch_idx) {
    int32_t width = 0, height = 0, channel = 0, rate_den = 0, rate_num = 0;
    input_bufs->At(batch_idx)->Get("width", width);
    input_bufs->At(batch_idx)->Get("height", height);
    input_bufs->At(batch_idx)->Get("channel", channel);
    input_bufs->At(batch_idx)->Get("rate_den", rate_den);
    input_bufs->At(batch_idx)->Get("rate_num", rate_num);

    auto input_data = (float *)(input_bufs->At(batch_idx)->ConstData());
    auto output_data =
        static_cast<float *>((*true_output)[batch_idx]->MutableData());
    err = memcpy_s(output_data, input_bufs->At(batch_idx)->GetBytes(),
                   input_data, input_bufs->At(batch_idx)->GetBytes());
    if (err != 0) {
      auto errMsg = "Copying face aligned image in face condition unit failed";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    true_output->At(batch_idx)->Set("width", width);
    true_output->At(batch_idx)->Set("height", height);
    true_output->At(batch_idx)->Set("channel", channel);
    true_output->At(batch_idx)->Set("pix_fmt", std::string("rgb"));
    true_output->At(batch_idx)->Set("rate_den", rate_den);
    true_output->At(batch_idx)->Set("rate_num", rate_num);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceConditionFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : CONDITION_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }

  for (auto port : CONDITION_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
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
