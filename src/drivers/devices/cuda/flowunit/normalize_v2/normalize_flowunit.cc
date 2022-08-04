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
#include "normalize_flowunit_cu.h"

NormalizeFlowUnitV2::NormalizeFlowUnitV2() = default;
NormalizeFlowUnitV2::~NormalizeFlowUnitV2() = default;

constexpr int COLOR_CHANNEL_COUNT = 3;
constexpr int GRAY_CHANNEL_COUNT = 1;

modelbox::Status NormalizeFlowUnitV2::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  if (!opts->Contain("output_layout")) {
    MBLOG_ERROR << "config must has output_layout";
    return modelbox::STATUS_BADCONF;
  }

  output_layout_ = opts->GetString("output_layout", "");
  if (output_layout_ != "chw" && output_layout_ != "hwc") {
    MBLOG_ERROR << "Invalid config output_layout = " << output_layout_;
    return modelbox::STATUS_BADCONF;
  }

  std::vector<float> default_mean{0, 0, 0};
  mean_ = opts->GetFloats("mean", default_mean);

  std::vector<float> default_std{1, 1, 1};
  std_ = opts->GetFloats("standard_deviation_inverse", default_std);

  auto device = GetBindDevice();
  mean_buffer_ = std::make_shared<modelbox::Buffer>(device);
  mean_buffer_->BuildFromHost(mean_.data(), mean_.size() * sizeof(float));

  std_buffer_ = std::make_shared<modelbox::Buffer>(device);
  std_buffer_->BuildFromHost(std_.data(), std_.size() * sizeof(float));

  return modelbox::STATUS_OK;
}

modelbox::Status GetParm(const std::shared_ptr<modelbox::Buffer> &buffer,
                         std::vector<size_t> &shape, std::string &input_layout,
                         modelbox::ModelBoxDataType &type) {
  if (!buffer->Get("shape", shape)) {
    MBLOG_ERROR << "can not get shape from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (shape.size() != 1 && shape.size() != 3) {
    MBLOG_ERROR << "unsupport image shape: " << shape.size();
    return modelbox::STATUS_INVALID;
  }

  if (!buffer->Get("layout", input_layout)) {
    MBLOG_ERROR << "can not get layout from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (input_layout != "chw" && input_layout != "hwc") {
    MBLOG_ERROR << "unsupport layout: " << input_layout
                << " support chw or hwc";
    return modelbox::STATUS_INVALID;
  }

  if (!buffer->Get("type", type)) {
    MBLOG_ERROR << "can not get type from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (type != modelbox::ModelBoxDataType::MODELBOX_UINT8) {
    MBLOG_ERROR << "unsupport type: " << type
                << " support modelbox::ModelBoxDataType::MODELBOX_UINT8";
    return modelbox::STATUS_INVALID;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status GetAndCheckParm(
    const std::shared_ptr<modelbox::BufferList> &input,
    std::vector<size_t> &shape, std::string &input_layout,
    modelbox::ModelBoxDataType &type) {
  std::vector<size_t> tmp_shape;
  std::string tmp_input_layout;
  modelbox::ModelBoxDataType tmp_type = modelbox::MODELBOX_TYPE_INVALID;

  for (auto &buffer : *input) {
    if (buffer == *input->begin()) {
      if (!GetParm(buffer, shape, input_layout, type)) {
        return modelbox::STATUS_INVALID;
      }
    }

    if (!GetParm(buffer, tmp_shape, tmp_input_layout, tmp_type)) {
      return modelbox::STATUS_INVALID;
    }

    if (tmp_shape != shape) {
      MBLOG_ERROR << "all image must has same shape.";
      return modelbox::STATUS_INVALID;
    }

    if (tmp_input_layout != input_layout) {
      MBLOG_ERROR << "all image must has same layout.";
      return modelbox::STATUS_INVALID;
    }

    if (tmp_type != type) {
      MBLOG_ERROR << "all image must has same type.";
      return modelbox::STATUS_INVALID;
    }
  }

  return modelbox::STATUS_OK;
}

/* run when processing data */
modelbox::Status NormalizeFlowUnitV2::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  auto input = data_ctx->Input("in_image");
  auto output = data_ctx->Output("out_data");

  std::vector<size_t> shape;
  std::string input_layout;
  modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;

  auto status = GetAndCheckParm(input, shape, input_layout, type);
  if (!status) {
    return status;
  }

  int H = 0;
  int W = 0;
  int C = 0;
  if (input_layout == "hwc") {
    H = shape[0];
    W = shape[1];
    C = shape[2];
  } else {
    MBLOG_ERROR << "only support hwc, but input layout is " << input_layout;
    return modelbox::STATUS_INVALID;
  }

  if (C != GRAY_CHANNEL_COUNT && C != COLOR_CHANNEL_COUNT) {
    MBLOG_ERROR << "invalid image channels: " << C << " support 1 or 3";
    return modelbox::STATUS_FAULT;
  }

  const auto *data = input->ConstData();
  // TODO sizeof arg used config
  output->Build(std::vector<size_t>(input->Size(), H * W * C * sizeof(float)));
  std::vector<size_t> output_shape;
  if (output_layout_ == "hwc") {
    Normalize((uint8_t *)data, input->Size(), H, W, C,
              (const float *)mean_buffer_->ConstData(),
              (const float *)std_buffer_->ConstData(),
              (float *)output->MutableData(), stream);
    output_shape = {(size_t)H, (size_t)W, (size_t)C};
  } else {
    NormalizeAndCHW((uint8_t *)data, input->Size(), H, W, C,
                    (const float *)mean_buffer_->ConstData(),
                    (const float *)std_buffer_->ConstData(),
                    (float *)output->MutableData(), stream);
    output_shape = {(size_t)C, (size_t)H, (size_t)W};
  }

  output->CopyMeta(input);
  output->Set("layout", output_layout_);
  output->Set("shape", output_shape);
  output->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(NormalizeFlowUnitV2, desc) {
  desc.SetFlowUnitName("image_preprocess");
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(
      modelbox::FlowUnitInput("in_image", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(
      modelbox::FlowUnitOutput("out_data", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "output_layout", "string", true, "", "the normalize output layout"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("mean", "string", false, "",
                                                  "the normalize mean"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "standard_deviation_inverse", "string", false, "", "the normalize std"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
