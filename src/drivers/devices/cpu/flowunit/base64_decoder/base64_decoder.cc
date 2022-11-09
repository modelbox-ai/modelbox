/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#include "base64_decoder.h"

#include <modelbox/base/crypto.h>
#include <securec.h>

#include <nlohmann/json.hpp>

#include "modelbox/flowunit_api_helper.h"

using nlohmann::json;

Base64DecoderFlowUnit::Base64DecoderFlowUnit() = default;
Base64DecoderFlowUnit::~Base64DecoderFlowUnit() = default;

modelbox::Status Base64DecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  // data_format only surport [raw,json]
  data_format_ = opts->GetString("data_format", "raw");
  if (data_format_ != "raw" && data_format_ != "json") {
    MBLOG_ERROR << "Valid data_format is: " << data_format_;
    return {modelbox::STATUS_BADCONF, "Valid data_format is: " + data_format_};
  }

  decoder_key_ = opts->GetString("key", "");
  return modelbox::STATUS_OK;
}

std::string Base64DecoderFlowUnit::JsonDecode(const std::string &buffer) {
  try {
    auto data_body = json::parse(buffer);

    if (!data_body.contains(decoder_key_)) {
      MBLOG_ERROR << decoder_key_ << " isn't exist";
      return "";
    }

    if (!data_body[decoder_key_].is_string()) {
      MBLOG_ERROR << "data isn't string, key:" << decoder_key_;
      return "";
    }

    return std::move(data_body[decoder_key_].get<std::string>());
  } catch (std::exception const &e) {
    MBLOG_ERROR << "failed to json decode exception: " << e.what();
    return "";
  }
}

std::shared_ptr<modelbox::Buffer> Base64DecoderFlowUnit::Base64Decoder(
    const std::string &buffer) {
  if (buffer.empty()) {
    MBLOG_ERROR << "input is empty";
    return nullptr;
  }

  std::vector<u_char> input_data;
  auto ret = modelbox::Base64Decode(buffer, &input_data);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "base64 decode fail reason: " << ret.Errormsg();
    return nullptr;
  }

  auto out_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
  if (out_buffer == nullptr) {
    MBLOG_ERROR << "failed to make out buffer";
    return nullptr;
  }

  ret = out_buffer->Build(input_data.size());
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "build buffer fail size: " << input_data.size()
                << " reason: " << ret.Errormsg();
    return nullptr;
  }

  auto e_ret = memcpy_s(out_buffer->MutableData(), out_buffer->GetBytes(),
                        input_data.data(), input_data.size());
  if (e_ret != EOK) {
    MBLOG_ERROR << "failed to memcpy ret: " << e_ret;
    return nullptr;
  }

  return out_buffer;
}

modelbox::Status Base64DecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  // get input
  auto input_bufs = ctx->Input("in_data");
  auto output_bufs = ctx->Output("out_data");
  if (input_bufs->Size() <= 0) {
    auto msg = "input data batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  std::vector<size_t> output_shape;
  for (auto &buffer : *input_bufs) {
    int data_len = buffer->GetBytes();
    if (data_len <= 0) {
      const auto *msg = "in data size is invalied";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    std::string in_data_str((char *)buffer->ConstData(), data_len);

    if (data_format_ == "json") {
      in_data_str = JsonDecode(in_data_str);
    }

    auto out_buf = Base64Decoder(in_data_str);
    if (out_buf == nullptr) {
      const auto *msg = "out buf is nullptr";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    output_bufs->PushBack(out_buf);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(Base64DecoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_data", "cpu"});
  desc.AddFlowUnitOutput({"out_data"});

  desc.SetFlowType(modelbox::NORMAL);
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
