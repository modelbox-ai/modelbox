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


#include "face_color_transpose_flowunit.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

FaceColorTransposeFlowUnit::FaceColorTransposeFlowUnit(){};
FaceColorTransposeFlowUnit::~FaceColorTransposeFlowUnit(){};

modelbox::Status FaceColorTransposeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status FaceColorTransposeFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceColorTransposeFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "color_transpose process begin";
  auto input_buf = ctx->Input("in_image");
  if (input_buf->Size() <= 0) {
    auto errMsg = "In_img batch is " + std::to_string(input_buf->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  auto output_buf = ctx->Output("out_image");

  std::vector<size_t> shape_vector;
  for (size_t i = 0; i < input_buf->Size(); ++i) {
    shape_vector.push_back(input_buf->At(i)->GetBytes());
  }
  auto ret = output_buf->Build(shape_vector);
  if (!ret) {
    auto errMsg = "build output failed in face color transpose unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  for (size_t i = 0; i < input_buf->Size(); ++i) {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channel = 0;
    std::string pix_fmt;
    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    bool metaresult = true;
    metaresult = input_buf->At(i)->Get("width", width) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("height", height) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("channel", channel) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("pix_fmt", pix_fmt) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("type", type) ? metaresult : false;
    if (metaresult == false) {
      auto msg = "buffer meta is invalid.";
      MBLOG_INFO << msg;
      return {modelbox::STATUS_BADCONF, msg};
    }

    size_t elem_size = width * height;
    auto input_data = static_cast<const float *>(input_buf->ConstBufferData(i));
    auto output_data = static_cast<float *>(output_buf->MutableBufferData(i));
    if (input_data == nullptr || output_data == nullptr) {
      return {modelbox::STATUS_NOMEM};
    }

    for (size_t i = 0; i < (size_t)channel; ++i) {
      for (size_t j = 0; j < elem_size; ++j) {
        output_data[i * elem_size + j] = input_data[j * channel + i];
      }
    }

    output_buf->At(i)->CopyMeta(input_buf->At(i));
    output_buf->At(i)->Set("width", width);
    output_buf->At(i)->Set("height", height);
    output_buf->At(i)->Set("channel", channel);
    output_buf->At(i)->Set("pix_fmt", pix_fmt);
    output_buf->At(i)->Set("type", type);
  }

  MBLOG_DEBUG << "color_transpose process data finish";
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceColorTransposeFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  for (auto port : TRANSPOSE_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }
  for (auto port : TRANSPOSE_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
  desc.SetFlowType(modelbox::NORMAL);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
