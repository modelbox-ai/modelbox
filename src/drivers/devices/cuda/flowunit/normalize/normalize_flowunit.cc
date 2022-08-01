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

#include "normalize_flowunit.h"
#include "modelbox/flowunit_api_helper.h"

NormalizeFlowUnit::NormalizeFlowUnit() = default;
NormalizeFlowUnit::~NormalizeFlowUnit() = default;

modelbox::Status NormalizeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  if (!opts->Contain("standard_deviation_inverse")) {
    MBLOG_ERROR << "normalize flow unit does not contain normalize param";
    return modelbox::STATUS_BADCONF;
  }

  auto input_params = opts->GetDoubles("standard_deviation_inverse");
  if (input_params.size() != CHANNEL_NUM) {
    MBLOG_ERROR << "normalize param error";
    return modelbox::STATUS_BADCONF;
  }

  params_.normalizes_.assign(input_params.begin(), input_params.end());
  return modelbox::STATUS_OK;
}

modelbox::Status NormalizeFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status NormalizeFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  const auto input_bufs = data_ctx->Input("in_data");
  if (input_bufs->Size() == 0) {
    MBLOG_ERROR << "normalize flowunit input invalied";
    return modelbox::STATUS_FAULT;
  }

  auto output_bufs = data_ctx->Output("out_data");
  if (!BuildOutputBufferList(input_bufs, output_bufs)) {
    MBLOG_ERROR << "build output BufferList failed";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    auto input_buf = input_bufs->At(i);
    int32_t width, height;
    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    if (!CheckBufferValid(input_buf, width, height, type)) {
      MBLOG_FATAL << "normalize flowunit input_buf invalied";
      continue;
    }

    auto out_buff = output_bufs->At(i);
    out_buff->CopyMeta(input_buf);
    out_buff->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
    auto out_data = static_cast<float *>(out_buff->MutableData());

    if (type == modelbox::ModelBoxDataType::MODELBOX_FLOAT) {
      auto *in_data_f32 =
          static_cast<float *>(const_cast<void *>(input_buf->ConstData()));
      if (in_data_f32 == nullptr) {
        MBLOG_ERROR << "normalize flowunit data is nullptr";
        continue;
      }

      cudaMemcpy(out_data, in_data_f32, input_buf->GetBytes(),
                 cudaMemcpyDeviceToDevice);
    } else {
      auto *in_data_uint8 =
          static_cast<uint8_t *>(const_cast<void *>(input_buf->ConstData()));
      if (in_data_uint8 == nullptr) {
        MBLOG_ERROR << "normalize flowunit data is nullptr";
        continue;
      }

      std::vector<uint8_t> host_data_uint8(input_buf->GetBytes());
      cudaMemcpy(host_data_uint8.data(), in_data_uint8, input_buf->GetBytes(),
                 cudaMemcpyDeviceToHost);
      std::vector<float> host_data_f32(input_buf->GetBytes());
      for (size_t i = 0; i < input_buf->GetBytes(); i++) {
        host_data_f32[i] = host_data_uint8[i];
      }

      cudaMemcpy(out_data, host_data_f32.data(),
                 input_buf->GetBytes() * sizeof(float), cudaMemcpyHostToDevice);
    }

    int32_t ret = NormalizeOperator(out_data, width, height);
    if (ret < 0) {
      MBLOG_ERROR << "normalize FlowUnit process failed";
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_OK;
}

bool NormalizeFlowUnit::CheckBufferValid(
    std::shared_ptr<modelbox::Buffer> buffer, int32_t &width, int32_t &height,
    modelbox::ModelBoxDataType &type) {
  std::vector<size_t> shape;
  if (!buffer->Get("shape", shape)) {
    MBLOG_ERROR << "mean flowunit can not get shape from meta";
    return false;
  }

  if (shape.size() != SHAPE_SIZE) {
    MBLOG_ERROR << "mean flowunit only support hwc data";
    return false;
  }

  if (shape[2] != CHANNEL_NUM) {
    MBLOG_ERROR << "mean flowunit only support hwc and C is " << CHANNEL_NUM;
    return false;
  }

  height = shape[0];
  width = shape[1];

  if (!buffer->Get("type", type)) {
    MBLOG_ERROR << "mean flowunit can not get input type from meta";
    return false;
  }

  return true;
}

bool NormalizeFlowUnit::NormalizeOperator(float *data, int32_t width,
                                          int32_t height) {
  ImageRect roi;
  roi.x = roi.y = 0;
  roi.width = width;
  roi.height = height;

  int32_t ret = Scale_32f_C1IR(data, width, roi, params_.normalizes_[0]);
  if (ret < 0) {
    MBLOG_ERROR << "normalize FlowUnit process channel_0 failed";
    return ret;
  }

  ret =
      Scale_32f_C1IR(data + width * height, width, roi, params_.normalizes_[1]);
  if (ret < 0) {
    MBLOG_ERROR << "normalize FlowUnit process channel_1 failed";
    return ret;
  }

  ret = Scale_32f_C1IR(data + width * height * 2, width, roi,
                       params_.normalizes_[2]);
  if (ret < 0) {
    MBLOG_ERROR << "normalize FlowUnit process channel_2 failed";
  }

  return ret;
}

MODELBOX_FLOWUNIT(NormalizeFlowUnit, desc) {
  desc.SetFlowUnitName("normalize");
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_data", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_data", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "standard_deviation_inverse", "string", true, "", "the normalize param"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
