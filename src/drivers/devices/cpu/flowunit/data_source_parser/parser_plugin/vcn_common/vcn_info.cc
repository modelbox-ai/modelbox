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

#include "vcn_info.h"

#include <modelbox/base/log.h>

#include <nlohmann/json.hpp>

namespace modelbox {
bool IsVcnInfoValid(const VcnInfo &info) {
  return (!(info.user_name.empty() || info.password.empty() ||
            info.ip.empty() || info.port.empty()));
}

modelbox::Status GetVcnInfo(modelbox::VcnInfo &vcn_info,
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

void ReadConfVcnCommon(const std::shared_ptr<modelbox::Configuration> &opts,
                       int32_t &retry_enabled, int32_t &retry_interval,
                       int32_t &retry_max_times) {
  retry_enabled = opts->GetBool("vcn_retry_enable", retry_enabled);
  retry_interval = opts->GetInt32("vcn_retry_interval_ms", retry_interval);
  retry_max_times = opts->GetInt32("vcn_retry_count_limit", retry_max_times);
}
}  // namespace modelbox