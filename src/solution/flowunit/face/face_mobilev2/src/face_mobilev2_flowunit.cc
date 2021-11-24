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


#include "face_mobilev2_flowunit.h"

#include <securec.h>

#include <cmath>
#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using std::map;
using std::vector;
using namespace modelbox;

FaceMobilev2FlowUnit::FaceMobilev2FlowUnit(){};
FaceMobilev2FlowUnit::~FaceMobilev2FlowUnit(){};

modelbox::Status FaceMobilev2FlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status FaceMobilev2FlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceMobilev2FlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image mobilev2";
  auto layer = data_ctx->Input(MOBILE_UNIT_IN_NAME[0]);
  auto output_bufs = data_ctx->Output(MOBILE_UNIT_OUT_NAME[0]);
  // checkout input batch_size and shape
  std::vector<std::shared_ptr<BufferList>> infer_layers;
  auto status = GetInferLayers(data_ctx, infer_layers);
  if (!status) {
    return status;
  }

  std::vector<size_t> tensor_shape;
  for (size_t batch_idx = 0; batch_idx < layer->Size(); ++batch_idx) {
    // get face number in each img
    size_t face_num =
        infer_layers[0]->At(batch_idx)->GetBytes() / (7 * sizeof(float));
    tensor_shape.emplace_back(sizeof(FaceExpressionScore) * face_num);
  }
  auto ret = output_bufs->Build(tensor_shape);
  if (!ret) {
    auto errMsg = "build output failed in face mobilev2 unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  for (size_t batch_idx = 0; batch_idx < layer->Size(); ++batch_idx) {
    float *center_data = (float *)(infer_layers[0]->At(batch_idx)->ConstData());
    auto tmp_buffer_size = infer_layers[0]->At(batch_idx)->GetBytes();
    int face_num = tmp_buffer_size / (7 * sizeof(float));
    auto output_data =
        (FaceExpressionScore *)((*output_bufs)[batch_idx]->MutableData());
    for (int face_idx = 0; face_idx < face_num; ++face_idx) {
      float tempData[7] = {0};
      float sum = 0.0f;
      for (int idx = 0; idx < 7; ++idx) {
        tempData[idx] = *(center_data + idx);
      }
      for (int i = 0; i < 7; i++) {
        sum += exp(tempData[i]);
      }
      for (int i = 0; i < 7; i++) {
        tempData[i] = exp(tempData[i]) / sum;
      }

      output_data->surprise_score = tempData[0];
      output_data->fear_score = tempData[1];
      output_data->disgust_score = tempData[2];
      output_data->happy_score = tempData[3];
      output_data->sad_score = tempData[4];
      output_data->angry_score = tempData[5];
      output_data->neutral_score = tempData[6];
      output_data++;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status FaceMobilev2FlowUnit::GetInferLayers(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers) {
  size_t batch_size = 0;

  for (size_t i = 0; i < MOBILE_UNIT_IN_NAME.size(); ++i) {
    // check batch size, all input batch size is same
    std::shared_ptr<BufferList> layer = data_ctx->Input(MOBILE_UNIT_IN_NAME[i]);
    auto cur_batch_size = layer->Size();
    if (cur_batch_size <= 0 ||
        (batch_size != 0 && cur_batch_size != batch_size)) {
      auto errMsg =
          "infer layer is invalid. batch_size:" + std::to_string(batch_size) +
          " layer_name:" + MOBILE_UNIT_IN_NAME[i] +
          " cur_batch_size:" + std::to_string(cur_batch_size);
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    batch_size = cur_batch_size;

    infer_layers.push_back(layer);
  }

  if (infer_layers.empty()) {
    auto errMsg = "infer layer is empty.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceMobilev2FlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : MOBILE_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }
  for (auto port : MOBILE_UNIT_OUT_NAME) {
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