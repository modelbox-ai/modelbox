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


#include "generate_plugin.h"

std::shared_ptr<TensorRTInferencePlugin> CreatePlugin() {
  return std::make_shared<OriginInferencePlugin>();
}

modelbox::Status OriginInferencePlugin::PluginInit(
    std::shared_ptr<modelbox::Configuration> config) {
  modelbox::Status status = modelbox::STATUS_OK;
  std::vector<std::string> names, types;
  status = SetUpInputOutput(config, "input", names, types);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "set up input failed, error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  input_name_list_.swap(names);
  input_type_list_.swap(types);

  status = SetUpInputOutput(config, "output", names, types);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "set up output failed, error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  output_name_list_.swap(names);
  output_type_list_.swap(types);

  return status;
}

modelbox::Status OriginInferencePlugin::SetUpInputOutput(
    std::shared_ptr<modelbox::Configuration> config, const std::string &type,
    std::vector<std::string> &names, std::vector<std::string> &types) {
  auto keys = config->GetSubKeys(type);
  for (unsigned int i = 1; i <= keys.size(); ++i) {
    std::string inner_name, inner_type;
    auto key = type + "." + type + std::to_string(i);
    auto item_table = config->GetSubKeys(key);
    if (item_table.empty()) {
      auto err_msg = "the key " + key + " is not found in config file.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto name_index = key + ".name";
    inner_name = config->GetString(name_index);
    if (inner_name.empty()) {
      auto err_msg = "the key " + key + " should have key name.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto type_index = key + ".type";
    inner_type = config->GetString(type_index);
    if (inner_type.empty()) {
      auto err_msg = "the key " + key + " should have key type.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    names.push_back(inner_name);
    types.push_back(inner_type);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::PreProcess(
    std::shared_ptr<modelbox::DataContext> ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::PostProcess(
    std::shared_ptr<modelbox::DataContext> ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::DataPre(
    std::shared_ptr<modelbox::DataContext> ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::DataPost(
    std::shared_ptr<modelbox::DataContext> ctx) {
  return modelbox::STATUS_OK;
}