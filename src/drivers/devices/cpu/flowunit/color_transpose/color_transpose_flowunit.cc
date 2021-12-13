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

#include "color_transpose_flowunit.h"

#include <securec.h>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

ColorTransposeFlowUnit::ColorTransposeFlowUnit(){};
ColorTransposeFlowUnit::~ColorTransposeFlowUnit(){};

modelbox::Status ColorTransposeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status ColorTransposeFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status ColorTransposeFlowUnit::CheckParam(
    modelbox::ModelBoxDataType type, const std::string &pix_fmt,
    const std::string &layout) {
  if (type != modelbox::ModelBoxDataType::MODELBOX_UINT8) {
    return {modelbox::STATUS_INVALID, "type must be uint8"};
  }

  if (pix_fmt != "rgb" && pix_fmt != "bgr") {
    return {modelbox::STATUS_INVALID, "pix_fmt should be [rgb, bgr]"};
  }

  if (layout != "hwc") {
    return {modelbox::STATUS_INVALID, "layout must be hwc"};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ColorTransposeFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "color_transpose process begin";
  auto input_buf = ctx->Input("in_image");
  auto output_buf = ctx->Output("out_image");

  std::vector<size_t> shape_vector;
  for (size_t i = 0; i < input_buf->Size(); ++i) {
    shape_vector.push_back(input_buf->At(i)->GetBytes());
  }
  output_buf->Build(shape_vector);
  output_buf->CopyMeta(input_buf);

  for (size_t i = 0; i < input_buf->Size(); ++i) {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channel = 0;
    std::string pix_fmt;
    std::string layout;
    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    bool metaresult = true;
    metaresult = input_buf->At(i)->Get("width", width) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("height", height) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("channel", channel) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("pix_fmt", pix_fmt) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("type", type) ? metaresult : false;
    metaresult = input_buf->At(i)->Get("layout", layout) ? metaresult : false;

    if (metaresult == false) {
      auto msg = "buffer meta is invalid.";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_BADCONF, msg};
    }

    auto ret = CheckParam(type, pix_fmt, layout);
    if (!ret) {
      MBLOG_ERROR << "input buffer meta invalid, detail: " << ret;
      return ret;
    }

    size_t elem_size = width * height;

    auto input_data =
        static_cast<const u_char *>(input_buf->ConstBufferData(i));
    auto output_data = static_cast<u_char *>(output_buf->MutableBufferData(i));
    if (input_data == nullptr || output_data == nullptr) {
      return {modelbox::STATUS_NOMEM};
    }

    for (size_t i = 0; i < (size_t)channel; ++i) {
      for (size_t j = 0; j < elem_size; ++j) {
        output_data[i * elem_size + j] = input_data[j * channel + i];
      }
    }
    auto buffer = output_buf->At(i);
    buffer->CopyMeta(input_buf->At(i));
    buffer->Set("width", width);
    buffer->Set("height", height);
    buffer->Set("width_stride", width);
    buffer->Set("height_stride", height);
    buffer->Set("channel", channel);
    buffer->Set("pix_fmt", pix_fmt);
    buffer->Set("layout", std::string("chw"));
    buffer->Set("shape", std::vector<size_t>{(size_t)channel, (size_t)height,
                                             (size_t)width});
    buffer->Set("type", type);
  }

  MBLOG_DEBUG << "color_transpose process data finish";
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ColorTransposeFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitOutput({"out_image"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
