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

MeanFlowUnit::MeanFlowUnit(){};
MeanFlowUnit::~MeanFlowUnit(){};

modelbox::Status MeanFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  if (!opts->Contain("mean")) {
    MBLOG_ERROR << "mean flow unit does not contain mean param";
    return modelbox::STATUS_BADCONF;
  }

  auto input_params = opts->GetDoubles("mean");
  if (input_params.size() != CHANNEL_NUM) {
    MBLOG_ERROR << "mean param error";
    return modelbox::STATUS_BADCONF;
  }

  params_.means_.assign(input_params.begin(), input_params.end());
  return modelbox::STATUS_OK;
}

modelbox::Status MeanFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status MeanFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> ctx, cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  const auto input_bufs = ctx->Input("in_data");
  if (input_bufs->Size() == 0) {
    MBLOG_ERROR << "mean flowunit in_data invalied";
    return modelbox::STATUS_FAULT;
  }

  auto output_bufs = ctx->Output("out_data");
  if (!BuildOutputBufferList(input_bufs, output_bufs)) {
    MBLOG_ERROR << "build output BufferList failed";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    auto input_buf = input_bufs->At(i);
    int32_t width, height;
    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    if (!CheckBufferValid(input_buf, width, height, type)) {
      MBLOG_FATAL << "mean flowunit input_buf invalied";
      continue;
    }

    auto out_buff = output_bufs->At(i);
    out_buff->CopyMeta(input_buf);
    out_buff->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
    auto out_data = static_cast<float *>(out_buff->MutableData());

    if (type == modelbox::ModelBoxDataType::MODELBOX_FLOAT) {
      float *in_data_f32 =
          static_cast<float *>(const_cast<void *>(input_buf->ConstData()));
      if (in_data_f32 == nullptr) {
        MBLOG_ERROR << "mean flowunit data is nullptr";
        continue;
      }

      cudaMemcpy(out_data, in_data_f32, input_buf->GetBytes(),
                 cudaMemcpyDeviceToDevice);
    } else {
      uint8_t *in_data_uint8 =
          static_cast<uint8_t *>(const_cast<void *>(input_buf->ConstData()));
      if (in_data_uint8 == nullptr) {
        MBLOG_ERROR << "mean flowunit data is nullptr";
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

    int32_t ret = MeanOperator(out_data, width, height);
    if (ret < 0) {
      MBLOG_ERROR << "mean flowunit process failed";
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_OK;
}

bool MeanFlowUnit::CheckBufferValid(std::shared_ptr<modelbox::Buffer> buffer,
                                    int32_t &width, int32_t &height,
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

bool MeanFlowUnit::MeanOperator(const float *data, int32_t width,
                                int32_t height) {
  /* sub the mean value of BGR channels */
  ImageRect roi;
  roi.x = roi.y = 0;
  roi.width = width;
  roi.height = height;

  ImageMean_32f mean;
  mean.channel_0 = params_.means_[0];
  mean.channel_1 = params_.means_[1];
  mean.channel_2 = params_.means_[2];

  return Mean_PLANAR_32f_P3R(data, width, height, roi, mean, nullptr);
}

MODELBOX_FLOWUNIT(MeanFlowUnit, desc) {
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetFlowUnitName("mean");
  desc.SetFlowUnitGroupType("Image");
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_data", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_data", FLOWUNIT_TYPE));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("mean", "string", true, "", "the mean param"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
