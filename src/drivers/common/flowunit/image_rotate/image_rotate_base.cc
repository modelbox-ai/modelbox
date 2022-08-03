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

#include "image_rotate_base.h"

#include <securec.h>

#include <sstream>
#include <string>

ImageRotateFlowUnitBase::ImageRotateFlowUnitBase() = default;
ImageRotateFlowUnitBase::~ImageRotateFlowUnitBase() = default;

modelbox::Status ImageRotateFlowUnitBase::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  has_rotate_angle_ = opts->Contain("rotate_angle");
  if (has_rotate_angle_) {
    rotate_angle_ = opts->GetInt32("rotate_angle", 0);
    auto ret = CheckRotateAngle(rotate_angle_);
    if (ret != modelbox::STATUS_OK) {
      return ret;
    }
  }
  MBLOG_DEBUG << "has  rotate_angle" << has_rotate_angle_ << ", rotate_angle"
              << rotate_angle_;

  return modelbox::STATUS_OK;
}

modelbox::Status ImageRotateFlowUnitBase::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status ImageRotateFlowUnitBase::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_bufs = data_ctx->Input("in_image");
  auto output_bufs = data_ctx->Output("out_image");
  if (input_bufs->Size() <= 0) {
    auto errMsg = "input images batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  for (auto &buffer : *input_bufs) {
    if (CheckImageType(buffer) != modelbox::STATUS_OK) {
      return modelbox::STATUS_FAULT;
    }

    int32_t rotate_angle(0);
    if (has_rotate_angle_) {
      rotate_angle = rotate_angle_;
    } else if (!buffer->Get("rotate_angle", rotate_angle)) {
      MBLOG_ERROR << "get buffer meta rotate_angle failed.";
      return modelbox::STATUS_FAULT;
    }

    if (rotate_angle == 0) {
      output_bufs->PushBack(buffer);
      continue;
    }
    auto check_ret = CheckRotateAngle(rotate_angle);
    if (check_ret != modelbox::STATUS_OK) {
      return check_ret;
    }

    int32_t width;
    int32_t height;
    buffer->Get("width", width);
    buffer->Get("height", height);

    int32_t output_width(width);
    int32_t output_height(height);
    if (rotate_angle == 90 || rotate_angle == 270) {
      output_width = height;
      output_height = width;
    }

    // rotate
    auto output_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
    RotateOneImage(buffer, output_buffer, rotate_angle, width, height);

    output_buffer->CopyMeta(buffer);
    output_buffer->Set("width", output_width);
    output_buffer->Set("height", output_height);
    output_buffer->Set("width_stride", output_width);
    output_buffer->Set("height_stride", output_height);
    output_buffer->Set("rotate_angle", 0);
    output_bufs->PushBack(output_buffer);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ImageRotateFlowUnitBase::CheckImageType(
    const std::shared_ptr<modelbox::Buffer> &input_buffer) {
  auto input_type = modelbox::ModelBoxDataType::MODELBOX_TYPE_INVALID;
  if (!input_buffer->Get("type", input_type) ||
      input_type != modelbox::ModelBoxDataType::MODELBOX_UINT8) {
    MBLOG_ERROR << "input image buffer type must be MODELBOX_UINT8";
    return modelbox::STATUS_FAULT;
  }

  std::string input_layout;
  if (!input_buffer->Get("layout", input_layout) || input_layout != "hwc") {
    MBLOG_ERROR << "input image buffer layout must be hwc";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ImageRotateFlowUnitBase::CheckRotateAngle(
    const int32_t &rotate_angle) {
  if (rotate_value_.find(rotate_angle) == rotate_value_.end()) {
    MBLOG_ERROR << "rotate_angle is invalid, configure is :" +
                       std::to_string(rotate_angle);
    std::stringstream err_msg;
    err_msg << "Valid rotate_angle is: ";
    for (auto value : rotate_value_) {
      err_msg << std::to_string(value) << " ";
    }
    MBLOG_ERROR << err_msg.str();
    return {modelbox::STATUS_BADCONF, err_msg.str()};
  }
  return modelbox::STATUS_OK;
}
