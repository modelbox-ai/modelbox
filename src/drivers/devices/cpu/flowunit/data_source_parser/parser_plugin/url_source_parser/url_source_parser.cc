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

#include "url_source_parser.h"

#include <securec.h>

#include <nlohmann/json.hpp>

#include "modelbox/device/cpu/device_cpu.h"
#define RETRY_PARAMS_NOT_SET (-2)

UrlSourceParser::UrlSourceParser() = default;
UrlSourceParser::~UrlSourceParser() = default;

modelbox::Status UrlSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  retry_enabled_ = opts->GetBool("retry_enable", DATASOURCE_PARSER_RETRY_ON);
  retry_interval_ = opts->GetInt32("retry_interval_ms",
                                   DATASOURCE_PARSER_DEFAULT_RETRY_INTERVAL);
  retry_max_times_ = opts->GetInt32("retry_count_limit", RETRY_PARAMS_NOT_SET);

  retry_enabled_ = opts->GetBool("url_retry_enable", retry_enabled_);
  retry_interval_ = opts->GetInt32("url_retry_interval_ms", retry_interval_);
  file_retry_interval_ =
      opts->GetInt32("url_file_retry_interval_ms", retry_interval_);
  stream_retry_interval_ =
      opts->GetInt32("url_stream_retry_interval_ms", retry_interval_);
  retry_max_times_ = opts->GetInt32("url_retry_count_limit", retry_max_times_);
  file_retry_times_ =
      opts->GetInt32("url_file_retry_count_limit",
                     retry_max_times_ == RETRY_PARAMS_NOT_SET
                         ? DATASOURCE_PARSER_FILE_DEFAULT_RETRY_TIMES
                         : retry_max_times_);
  stream_retry_times_ =
      opts->GetInt32("url_stream_retry_count_limit",
                     retry_max_times_ == RETRY_PARAMS_NOT_SET
                         ? DATASOURCE_PARSER_STREAM_DEFAULT_RETRY_TIMES
                         : retry_max_times_);

  MBLOG_INFO << "url source parser config retry_enabled:" << retry_enabled_
             << " stream_retry_interval:" << stream_retry_interval_
             << " file_retry_interval:" << file_retry_interval_
             << " keep_alive_interval:" << file_retry_times_
             << " keep_alive_interval:" << stream_retry_times_;

  return modelbox::STATUS_OK;
}

modelbox::Status UrlSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status UrlSourceParser::GetStreamType(const std::string &config,
                                                std::string &stream_type) {
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(config);

    std::string url_type = json["url_type"].get<std::string>();
    if (url_type.empty()) {
      return {modelbox::STATUS_BADCONF, "url_type is empty"};
    }

    if (url_type == "file") {
      stream_type = "file";
    } else {
      stream_type = "stream";
    }
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }
  return modelbox::STATUS_OK;
}

modelbox::Status UrlSourceParser::Parse(
    const std::shared_ptr<modelbox::SessionContext> &session_context,
    const std::string &config, std::string &uri,
    modelbox::DestroyUriFunc &destroy_uri_func) {
  nlohmann::json json;
  std::string url_type;
  if (GetStreamType(config, url_type) != modelbox::STATUS_OK) {
    return {modelbox::STATUS_BADCONF, "url_type is empty"};
  }
  if (url_type == "file") {
    retry_interval_ = file_retry_interval_;
    retry_max_times_ = file_retry_times_;
  } else if (url_type == "stream") {
    retry_interval_ = stream_retry_interval_;
    retry_max_times_ = stream_retry_times_;
  } else {
    MBLOG_ERROR << "url input type: " << url_type << " is not supported";
    return modelbox::STATUS_BADCONF;
  }

  try {
    json = nlohmann::json::parse(config);
    uri = json["url"].get<std::string>();
    if (uri.empty()) {
      return {modelbox::STATUS_BADCONF, "uri is empty"};
    }
    MBLOG_DEBUG << "Get url address success.";
    return modelbox::STATUS_OK;
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }
}
