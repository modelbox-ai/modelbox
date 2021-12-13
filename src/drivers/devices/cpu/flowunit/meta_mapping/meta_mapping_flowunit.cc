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

#include "meta_mapping_flowunit.h"

#include <memory>

#include "modelbox/base/config.h"
#include "modelbox/base/utils.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

#define CASTER_IMPL(code) \
  [](std::stringstream &ss, modelbox::Any *any) { code; }

#define SETTER_IMPL(code)                                                     \
  [this](std::shared_ptr<modelbox::Buffer> &buffer, const std::string &str) { \
    code;                                                                     \
  }

MetaMappingFlowUnit::MetaMappingFlowUnit(){};
MetaMappingFlowUnit::~MetaMappingFlowUnit(){};

modelbox::Status MetaMappingFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  src_meta_name_ = opts->GetString("src_meta");
  if (src_meta_name_.empty()) {
    MBLOG_ERROR << "Missing src_meta in flowunit config";
    return modelbox::STATUS_BADCONF;
  }

  dest_meta_name_ = opts->GetString("dest_meta");
  if (dest_meta_name_.empty()) {
    MBLOG_ERROR << "Missing dest_meta in flowunit config";
    return modelbox::STATUS_BADCONF;
  }

  auto rules = opts->GetStrings("rules");
  auto ret = ParseRules(rules);
  if (!ret) {
    MBLOG_ERROR << "parser rules failed";
    return ret;
  }

  InitToStringCasters();
  InitBufferMetaSetters();

  return modelbox::STATUS_SUCCESS;
}

void MetaMappingFlowUnit::InitToStringCasters() {
  to_string_casters_ = {
      {typeid(int8_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<int8_t>(*any);)},
      {typeid(uint8_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<uint8_t>(*any);)},
      {typeid(int16_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<int16_t>(*any);)},
      {typeid(uint16_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<uint16_t>(*any);)},
      {typeid(int32_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<int32_t>(*any);)},
      {typeid(uint32_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<uint32_t>(*any);)},
      {typeid(int64_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<int64_t>(*any);)},
      {typeid(uint64_t).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<uint64_t>(*any);)},
      {typeid(float).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<float>(*any);)},
      {typeid(double).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<double>(*any);)},
      {typeid(bool).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<bool>(*any);)},
      {typeid(std::string).hash_code(),
       CASTER_IMPL(ss << modelbox::any_cast<std::string>(*any);)}};
}

void MetaMappingFlowUnit::InitBufferMetaSetters() {
  buffer_meta_setters_ = {
      {typeid(int8_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (int8_t)std::stoi(str));)},
      {typeid(uint8_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (uint8_t)std::stoi(str));)},
      {typeid(int16_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (int16_t)std::stoi(str));)},
      {typeid(uint16_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (uint16_t)std::stoi(str));)},
      {typeid(int32_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (int32_t)std::stoi(str));)},
      {typeid(uint32_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (uint32_t)std::stol(str));)},
      {typeid(int64_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (int64_t)std::stol(str));)},
      {typeid(uint64_t).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (uint64_t)std::stoul(str));)},
      {typeid(float).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (float)std::stof(str));)},
      {typeid(double).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, (double)std::stod(str));)},
      {typeid(bool).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, str == "true");)},
      {typeid(std::string).hash_code(),
       SETTER_IMPL(buffer->Set(dest_meta_name_, str);)}};
}

modelbox::Status MetaMappingFlowUnit::ParseRules(
    const std::vector<std::string> &rules) {
  for (auto &rule : rules) {
    auto rule_v = modelbox::StringSplit(rule, '=');
    if (rule_v.size() != 2) {
      return modelbox::STATUS_BADCONF;
    }

    MBLOG_INFO << "Add map rule " << rule_v[0] << "=" << rule_v[1];
    mapping_rules_[rule_v[0]] = rule_v[1];
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status MetaMappingFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status MetaMappingFlowUnit::ToString(modelbox::Any *any,
                                               std::string &val) {
  auto &type = any->type();
  auto caster_item = to_string_casters_.find(type.hash_code());
  if (caster_item == to_string_casters_.end()) {
    MBLOG_ERROR << "Not support meta type " << type.name();
    return modelbox::STATUS_NOTSUPPORT;
  }

  std::stringstream ss;
  caster_item->second(ss, any);
  val = ss.str();
  return modelbox::STATUS_OK;
}

modelbox::Status MetaMappingFlowUnit::SetValue(
    std::shared_ptr<modelbox::Buffer> &buffer, std::string &str,
    const std::type_info &type) {
  try {
    auto setter_item = buffer_meta_setters_.find(type.hash_code());
    if (setter_item == buffer_meta_setters_.end()) {
      MBLOG_ERROR << "Not support meta type " << type.name();
      return modelbox::STATUS_NOTSUPPORT;
    }

    setter_item->second(buffer, str);
  } catch (std::invalid_argument &e) {
    MBLOG_ERROR << "Can not convert " << str << " to target type "
                << type.name();
    return modelbox::STATUS_FAULT;
  } catch (std::out_of_range &e) {
    MBLOG_ERROR << "Value " << str << " is out of range for type "
                << type.name();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MetaMappingFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto input_buffer_list = ctx->Input(INPUT_DATA);
  auto output_buffer_list = ctx->Output(OUTPUT_DATA);
  for (auto &buffer : *input_buffer_list) {
    output_buffer_list->PushBack(buffer);
    modelbox::Any *src_val = nullptr;
    bool exist = false;
    std::tie(src_val, exist) = buffer->Get(src_meta_name_);
    if (!exist) {
      continue;
    }

    modelbox::Any src_val_cpy = *src_val;
    // Only copy src meta to dest meta
    if (mapping_rules_.empty()) {
      buffer->Set(dest_meta_name_, src_val_cpy);
      continue;
    }

    // Try map src meta value to dest meta value
    std::string src_val_str;
    auto ret = ToString(src_val, src_val_str);
    if (!ret) {
      buffer->Set(dest_meta_name_, src_val_cpy);
      continue;
    }

    auto item = mapping_rules_.find(src_val_str);
    if (item == mapping_rules_.end()) {
      buffer->Set(dest_meta_name_, src_val_cpy);
      continue;
    }

    SetValue(buffer, item->second, src_val->type());
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MetaMappingFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
};

modelbox::Status MetaMappingFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
};

MODELBOX_FLOWUNIT(MetaMappingFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput({INPUT_DATA});
  desc.AddFlowUnitOutput({OUTPUT_DATA});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetFlowUnitGroupType("Image");
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("src_meta", "string", true,
                                                  "", "the source meta"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("dest_meta", "string", true,
                                                  "", "the dest meta"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("rules", "string", false, "",
                                                  "the meta mapping rules"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
