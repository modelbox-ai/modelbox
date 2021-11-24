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


#include "vcn_source_parser.h"
#include <dirent.h>
#include <securec.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <string>
#include "modelbox/base/utils.h"
#include "modelbox/device/cpu/device_cpu.h"

void RemoveFileCallback(std::string uri);

VcnSourceParser::VcnSourceParser() {}
VcnSourceParser::~VcnSourceParser() {}

modelbox::Status VcnSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  ReadConf(opts);
  retry_enabled_ = opts->GetBool("vcn_retry_enable", retry_enabled_);
  retry_interval_ = opts->GetInt32("vcn_retry_interval_ms", retry_interval_);
  retry_max_times_ = opts->GetInt32("vcn_retry_count_limit", retry_max_times_);
  return modelbox::STATUS_OK;
}

modelbox::Status VcnSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status VcnSourceParser::Parse(const std::string &config,
                                      std::string &uri,
                                      DestroyUriFunc &destroy_uri_func) {
  modelbox::VcnInfo vcn_info;
  uri = "";

  // read info from cfg
  auto ret = GetVcnInfo(vcn_info, config);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << "failed to get vcn info.";
    return ret;
  }

  auto vcn_client = modelbox::VcnClient::GetInstance();
  if (nullptr == vcn_client) {
    MBLOG_ERROR << "failed to get vcn client instance.";
    return modelbox::STATUS_FAULT;
  }

  std::shared_ptr<modelbox::VcnStream> stream;
  ret = vcn_client->AddVcnStream(vcn_info, stream);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << ret.Errormsg();
    return ret;
  }

  uri = stream->GetUrl();
  destroy_uri_func = [stream](const std::string &uri) {};

  return modelbox::STATUS_OK;
}

modelbox::Status VcnSourceParser::GetVcnInfo(modelbox::VcnInfo &vcn_info,
                                           const std::string &config) {
  nlohmann::json config_json;
  try {
    config_json = nlohmann::json::parse(config);
    auto value = config_json["userName"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn userName is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.user_name = value;

    value = config_json["password"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn password is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.password = value;

    value = config_json["ip"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn ip is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.ip = value;

    value = config_json["port"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn port is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.port = value;

    value = config_json["cameraCode"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn camera code is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.camera_code = value;

    value = config_json["streamType"];
    if (value.empty()) {
      MBLOG_ERROR << "vcn stream type is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    vcn_info.stream_type = value;
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }

  return modelbox::STATUS_OK;
}
