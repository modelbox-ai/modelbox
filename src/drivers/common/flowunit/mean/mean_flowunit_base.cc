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


#include "mean_flowunit_base.h"

MeanFlowUnitBase::MeanFlowUnitBase() = default;
MeanFlowUnitBase::~MeanFlowUnitBase() = default;

modelbox::Status MeanFlowUnitBase::Open(
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

bool MeanFlowUnitBase::CheckBufferListValid(
    const std::shared_ptr<modelbox::BufferList> &buffer_list) {
  if (buffer_list == nullptr) {
    MBLOG_ERROR << "mean flowunit input is null";
    return false;
  }

  if (buffer_list->Size() == 0) {
    MBLOG_ERROR << "mean flowunit input size is 0";
    return false;
  }

  return true;
}

bool BuildOutputBufferList(
    const std::shared_ptr<modelbox::BufferList> &input_bufs,
    std::shared_ptr<modelbox::BufferList> &output_bufs) {
  std::vector<size_t> shape;
  for (size_t i = 0; i < input_bufs->Size(); ++i) {
    modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
    if (!input_bufs->At(i)->Get("type", type)) {
      MBLOG_FATAL << "mean flowunit can not get input type from meta";
      return false;
    }

    if ((type != modelbox::ModelBoxDataType::MODELBOX_FLOAT) &&
        (type != modelbox::ModelBoxDataType::MODELBOX_UINT8)) {
      MBLOG_FATAL << "mean flowunit input type error, type is " << type;
      return false;
    }

    size_t size = 0;
    if (type == modelbox::ModelBoxDataType::MODELBOX_FLOAT) {
      size = input_bufs->At(i)->GetBytes();
    } else {
      size = (input_bufs->At(i)->GetBytes() / sizeof(uint8_t)) * sizeof(float);
    }

    shape.emplace_back(size);
  }

  output_bufs->Build(shape);
  return true;
}

modelbox::Status MeanFlowUnitBase::Close() { return modelbox::STATUS_OK; }

modelbox::Status MeanFlowUnitBase::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status MeanFlowUnitBase::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}
