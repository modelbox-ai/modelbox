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


#include "face_expand.h"

#include "securec.h"

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

FaceExpandFlowUnit::FaceExpandFlowUnit(){};
FaceExpandFlowUnit::~FaceExpandFlowUnit(){};

modelbox::Status FaceExpandFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status FaceExpandFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceExpandFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "process expand faces";
  // In_img : faces {[face, ... ,face]}
  auto input_bufs = ctx->Input(EXPAND_UNNIT_IN_NAME[0]);
  if (input_bufs->Size() <= 0) {
    auto errMsg = "In_img batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // Out_img : faces {[face], ... ,[face]}
  auto output_bufs = ctx->Output(EXPAND_UNNIT_OUT_NAME[0]);

  std::vector<size_t> tensor_shape;
  errno_t err;
  if (input_bufs->At(0)->GetBytes() == 0) {
    tensor_shape.emplace_back(0);
    auto ret = output_bufs->Build(tensor_shape);
    if (!ret) {
      auto errMsg = "build empty output failed in face expand unit";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    return modelbox::STATUS_OK;
  }

  int32_t width = 0, height = 0, channel = 0, rateDen = 0, rate_num = 0;
  input_bufs->At(0)->Get("width", width);
  input_bufs->At(0)->Get("height", height);
  input_bufs->At(0)->Get("channel", channel);
  input_bufs->At(0)->Get("rate_den", rateDen);
  input_bufs->At(0)->Get("rate_num", rate_num);

  int imgSize = width * height * channel;
  size_t faceNum = input_bufs->At(0)->GetBytes() / (imgSize * sizeof(float));
  for (size_t i = 0; i < faceNum; ++i) {
    // get face number in each img
    tensor_shape.emplace_back(imgSize * sizeof(float));
  }
  auto ret = output_bufs->Build(tensor_shape);
  if (!ret) {
    auto errMsg = "build output failed in face expand unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  for (size_t batch_idx = 0; batch_idx < input_bufs->Size(); ++batch_idx) {
    auto input_data = (float *)(input_bufs->At(batch_idx)->ConstData());
    for (size_t face_idx = 0; face_idx < faceNum; ++face_idx) {
      auto output_data =
          static_cast<float *>((*output_bufs)[face_idx]->MutableData());
      err =
          memcpy_s(output_data, imgSize * sizeof(float),
                   (input_data + face_idx * imgSize), imgSize * sizeof(float));
      if (err != 0) {
        auto errMsg =
            "Copying expression score to output port in face expand unit "
            "failed.";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
    }

    output_bufs->At(batch_idx)->Set("width", width);
    output_bufs->At(batch_idx)->Set("height", height);
    output_bufs->At(batch_idx)->Set("channel", channel);
    output_bufs->At(batch_idx)->Set("pix_fmt", std::string("rgb"));
    output_bufs->At(batch_idx)->Set("rate_den", rateDen);
    output_bufs->At(batch_idx)->Set("rate_num", rate_num);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceExpandFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : EXPAND_UNNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }
  for (auto port : EXPAND_UNNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
  desc.SetOutputType(modelbox::EXPAND);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}