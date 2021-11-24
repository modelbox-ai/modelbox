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


#include "mean_flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "modelbox/type.h"

MeanFlowUnit::MeanFlowUnit(){};
MeanFlowUnit::~MeanFlowUnit(){};

modelbox::Status MeanFlowUnit::Process(std::shared_ptr<modelbox::DataContext> ctx) {
  const auto input_bufs = ctx->Input("in_data");
  if (!CheckBufferListValid(input_bufs)) {
    MBLOG_ERROR << "mean flowunit in_image invalied";
    return modelbox::STATUS_FAULT;
  }

  auto output_bufs = ctx->Output("out_data");
  if (!BuildOutputBufferList(input_bufs, output_bufs)) {
    MBLOG_ERROR << "build out_image BufferList failed";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    auto input_buf = input_bufs->At(i);
    std::vector<size_t> shape;
    if (!input_buf->Get("shape", shape)) {
      MBLOG_ERROR << "mean flowunit can not get shape from meta";
      continue;
    }

    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    if (!input_bufs->At(i)->Get("type", type)) {
      MBLOG_ERROR << "mean flowunit can not get input type from meta";
      continue;
    }

    float *in_data_f32 = nullptr;
    uint8_t *in_data_uint8 = nullptr;
    if (type == modelbox::ModelBoxDataType::MODELBOX_FLOAT) {
      Process(in_data_f32, input_buf, output_bufs->At(i));
    } else {
      Process(in_data_uint8, input_buf, output_bufs->At(i));
    }
  }

  return modelbox::STATUS_OK;
}

template <typename T>
void MeanFlowUnit::Process(const T *input_data,
                           std::shared_ptr<modelbox::Buffer> input_buf,
                           std::shared_ptr<modelbox::Buffer> out_buff) {
  input_data = static_cast<T *>(const_cast<void *>(input_buf->ConstData()));
  if (input_data == nullptr) {
    MBLOG_ERROR << "mean flowunit data is nullptr";
    return;
  }

  size_t size = (input_buf->GetBytes() / sizeof(T)) / CHANNEL_NUM;
  out_buff->CopyMeta(input_buf);
  out_buff->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
  auto out_data = static_cast<float *>(out_buff->MutableData());
  if (out_data == nullptr) {
    MBLOG_ERROR << "output is null";
    return;
  }

  for (size_t c = 0; c < CHANNEL_NUM; c++) {
    for (size_t j = size * c; j < size * (c + 1); j++) {
      out_data[j] = input_data[j] - params_.means_[c];
    }
  }
}

MODELBOX_FLOWUNIT(MeanFlowUnit, desc) {
  desc.SetFlowUnitName("mean");
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_data", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_data", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("mean", "string", true, "", "the mean param"));
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