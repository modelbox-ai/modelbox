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

#include "obs_output_broker.h"

#include <modelbox/base/uuid.h>
#include <modelbox/device/cpu/device_cpu.h>
#include <modelbox/iam_auth.h>
#include <securec.h>

#include <nlohmann/json.hpp>

ObsOutputBroker::ObsOutputBroker() = default;
ObsOutputBroker::~ObsOutputBroker() = default;

modelbox::Status ObsOutputBroker::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  obs_status ret_status = OBS_STATUS_BUTT;
  ret_status = obs_initialize(OBS_INIT_ALL);
  if (OBS_STATUS_OK != ret_status) {
    const auto *obs_status_name = obs_get_status_name(ret_status);
    if (obs_status_name == nullptr) {
      obs_status_name = "null";
    }
    MBLOG_ERROR << "failed to initialize OBS SDK: " << obs_status_name;
    return {modelbox::STATUS_FAULT};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ObsOutputBroker::Deinit() { return modelbox::STATUS_OK; }

std::shared_ptr<modelbox::OutputBrokerHandle> ObsOutputBroker::Open(
    const std::shared_ptr<modelbox::Configuration> &session_config,
    const std::string &config) {
  nlohmann::json config_json;
  std::shared_ptr<OBSOutputInfo> output_info =
      std::make_shared<OBSOutputInfo>();

  try {
    config_json = nlohmann::json::parse(config);

    auto value = config_json["obsEndPoint"];
    if (value.empty()) {
      MBLOG_ERROR << "obsEndPoint is empty!";
      return nullptr;
    }
    std::string http_header = "http://";
    std::string https_header = "https://";
    std::string end_point = value;

    if (end_point.find(http_header) == 0) {
      end_point = end_point.substr(http_header.length());
    } else if (end_point.find(https_header) == 0) {
      end_point = end_point.substr(https_header.length());
    }
    output_info->end_point = end_point;

    value = config_json["bucket"];
    if (value.empty()) {
      MBLOG_ERROR << "bucket is empty!";
      return nullptr;
    }
    output_info->bucket = value;

    value = config_json["path"];
    if (value.empty()) {
      MBLOG_ERROR << "path is empty!";
      return nullptr;
    }
    output_info->path = value;

    if (config_json.contains("domainName")) {
      value =
          config_json["domainName"];  // domainName maybe empty in edge scene.
      output_info->domain_name = value;
    }
    if (config_json.contains("xroleName")) {
      value = config_json["xroleName"];  // xroleName maybe empty in edge scene.
      output_info->xrole_name = value;
    }

    if (config_json.contains("userId")) {
      output_info->user_id = config_json["userId"];
    }
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Failed to parse json config, detail: " << e.what();
    return nullptr;
  }
  output_info->file_key_index = 0;

  auto handle = std::make_shared<modelbox::OutputBrokerHandle>();
  std::string uuid;
  if (modelbox::STATUS_OK != modelbox::GetUUID(&uuid)) {
    MBLOG_ERROR << "Failed to generate a uuid for the OBS output broker!";
    return nullptr;
  }
  handle->broker_id_ = uuid;
  std::lock_guard<std::mutex> lock(output_cfgs_mutex_);
  output_configs_[handle->broker_id_] = output_info;

  return handle;
}

modelbox::Status ObsOutputBroker::Write(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  size_t data_size = buffer->GetBytes();
  auto *data = const_cast<char *>((const char *)buffer->ConstData());
  if (buffer == nullptr || data == nullptr || data_size == 0) {
    MBLOG_WARN << "Invalid buffer: buffer is nullptr!";
    return modelbox::STATUS_NODATA;
  }

  // get the broker configuration
  std::unique_lock<std::mutex> lock(output_cfgs_mutex_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker configuration, type: "
        << handle->output_broker_type_ << ", id: " << handle->broker_id_;
    return modelbox::STATUS_FAULT;
  }
  std::shared_ptr<OBSOutputInfo> output_info = iter->second;
  lock.unlock();

  // set OBS file key
  std::string file_key;
  buffer->Get(META_OUTPUT_FILE_NAME, file_key);
  if (file_key.empty()) {
    file_key =
        handle->broker_id_ + "_" + std::to_string(output_info->file_key_index);
  }

  std::string path;
  if (!output_info->path.empty()) {
    if ('/' != output_info->path.at(output_info->path.length() - 1)) {
      path = output_info->path + "/";
    } else {
      path = output_info->path;
    }
  }

  file_key = path + file_key;  // File Path: [bucket]:[path]/[output_file_name]

  auto obs_client = modelbox::ObsClient::GetInstance();
  modelbox::ObsOptions obs_opt;
  obs_opt.end_point = output_info->end_point;
  obs_opt.bucket = output_info->bucket;
  obs_opt.path = file_key;
  obs_opt.domain_name = output_info->domain_name;
  obs_opt.xrole_name = output_info->xrole_name;
  obs_opt.user_id = output_info->user_id;
  auto ret = obs_client->PutObject(obs_opt, data, data_size);
  if (modelbox::STATUS_AGAIN == ret) {
    MBLOG_WARN << ret.Errormsg();
    PrintObsConfig(obs_opt);
    return modelbox::STATUS_AGAIN;
  }

  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << ret.Errormsg();
    return ret;
  }

  ++output_info->file_key_index;
  return modelbox::STATUS_OK;
}

modelbox::Status ObsOutputBroker::Sync(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  return modelbox::STATUS_OK;
}

modelbox::Status ObsOutputBroker::Close(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  std::unique_lock<std::mutex> lock(output_cfgs_mutex_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR << "broker handle not found, type: "
                << handle->output_broker_type_
                << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  output_configs_.erase(handle->broker_id_);
  return modelbox::STATUS_OK;
}

void ObsOutputBroker::PrintObsConfig(const modelbox::ObsOptions &opt) {
  MBLOG_INFO << "obs option - endpoint: " << opt.end_point;
}
