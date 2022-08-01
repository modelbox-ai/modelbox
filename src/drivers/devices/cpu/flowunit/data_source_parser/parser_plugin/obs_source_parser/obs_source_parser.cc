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

#include "obs_source_parser.h"

#include <dirent.h>
#include <securec.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <ctime>
#include <nlohmann/json.hpp>
#include <string>

#include "iam_auth.h"
#include "modelbox/base/utils.h"
#include "modelbox/base/uuid.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "obs_client.h"
#include "obs_file_handler.h"

#define OBS_RETRY_INTERVAL_DEFALUT 1000
#define OBS_RETRY_TIMES_DEFALUT 5

#define OBS_STREAM_READ_SIZE_LOW 1
#define OBS_STREAM_READ_SIZE_NORMAL 5
#define OBS_STREAM_READ_SIZE_HIGH 20

void RemoveFileCallback(std::string uri);

ObsSourceParser::ObsSourceParser() = default;
ObsSourceParser::~ObsSourceParser() = default;

modelbox::Status ObsSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  ReadConf(opts);
  retry_enabled_ = opts->GetBool("obs_retry_enable", retry_enabled_);
  retry_interval_ =
      opts->GetInt32("obs_retry_interval_ms", OBS_RETRY_INTERVAL_DEFALUT);
  retry_max_times_ =
      opts->GetInt32("obs_retry_count_limit", OBS_RETRY_TIMES_DEFALUT);
  read_type_ = opts->GetString("obs_download_method", "file");
  if (read_type_ != "stream") {
    return modelbox::STATUS_OK;
  }
  stream_memory_mode_ = opts->GetString("obs_stream_memory_mode", "low");
  max_read_size_ = OBS_STREAM_READ_SIZE_LOW;
  if (stream_memory_mode_ == "normal") {
    max_read_size_ = OBS_STREAM_READ_SIZE_NORMAL;
  } 
  if (stream_memory_mode_ == "high") {
    max_read_size_ = OBS_STREAM_READ_SIZE_HIGH;
  }
  return modelbox::STATUS_OK;
}

modelbox::Status ObsSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status ObsSourceParser::Parse(
    std::shared_ptr<modelbox::SessionContext> session_context,
    const std::string &config, std::string &uri,
    DestroyUriFunc &destroy_uri_func) {
  OBSDownloadInfo download_info;
  uri = "";

  // read info from cfg
  auto ret = GetObsInfo(download_info, config);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << "failed to get obs info";
    return ret;
  }

  modelbox::ObsOptions obs_opt;
  obs_opt.end_point = download_info.end_point;
  obs_opt.bucket = download_info.bucket;
  obs_opt.path = download_info.file_key;
  obs_opt.domain_name = download_info.domain_name;
  obs_opt.xrole_name = download_info.xrole_name;
  obs_opt.ak = download_info.ak;
  obs_opt.sk = download_info.sk;
  obs_opt.token = download_info.token;
  obs_opt.user_id = download_info.user_id;

  std::string uuid;
  if (modelbox::STATUS_OK != GetUUID(&uuid)) {
    MBLOG_WARN << "Failed to generate a uuid for the OBS output broker! Use "
                  "default id: yyyymmddhhmmss";
    time_t now = time(nullptr);
    uuid = GetTimeString(&now);
  }

  if ("stream" == read_type_) {
    std::shared_ptr<modelbox::OBSFileHandler> obs_handler =
        std::make_shared<modelbox::OBSFileHandler>();
    obs_handler->SetOBSOption(obs_opt);
    std::string obs_uri =
        std::string("/obs/") + uuid + std::string("/") + download_info.file_key;
    uri = DEFAULT_FILE_REQUEST_URI + obs_uri;
    modelbox::FileRequester::GetInstance()->RegisterUrlHandler(obs_uri,
                                                               obs_handler);
    modelbox::FileRequester::GetInstance()->SetMaxFileReadSize(max_read_size_);
    destroy_uri_func = [obs_uri](std::string uri) {
      modelbox::FileRequester::GetInstance()->DeregisterUrl(obs_uri);
    };
    return modelbox::STATUS_OK;
  }

  download_info.file_local_path =
      OBS_TEMP_PATH + uuid + "_" +
      download_info.file_key.substr(download_info.file_key.rfind('/') + 1);

  auto obs_client = modelbox::ObsClient::GetInstance();
  ret = obs_client->GetObject(obs_opt, download_info.file_local_path);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << ret.Errormsg();
    return ret;
  }

  uri = download_info.file_local_path;
  destroy_uri_func = RemoveFileCallback;

  return modelbox::STATUS_OK;
}

modelbox::Status ObsSourceParser::GetStreamType(const std::string &config,
                                                std::string &stream_type) {
  stream_type = "file";
  return modelbox::STATUS_OK;
}

modelbox::Status ObsSourceParser::GetObsInfo(OBSDownloadInfo &download_info,
                                             const std::string &config) {
  nlohmann::json config_json;
  try {
    config_json = nlohmann::json::parse(config);

    auto value = config_json["obsEndPoint"];
    if (value.empty()) {
      MBLOG_ERROR << "obsEndPoint is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    std::string http_header = "http://";
    std::string https_header = "https://";
    std::string end_point = value;

    if (end_point.find(http_header) == 0) {
      end_point = end_point.substr(http_header.length());
    } else if (end_point.find(https_header) == 0) {
      end_point = end_point.substr(https_header.length());
    }
    download_info.end_point = end_point;

    value = config_json["bucket"];
    if (value.empty()) {
      MBLOG_ERROR << "bucket is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    download_info.bucket = value;

    value = config_json["path"];
    if (value.empty()) {
      MBLOG_ERROR << "path is empty!";
      return {modelbox::STATUS_BADCONF};
    }
    download_info.file_key = value;

    if (config_json.contains("userId")) {
      value = config_json["userId"];
      download_info.user_id = value;
    }
    if (download_info.user_id.empty()) {
      MBLOG_DEBUG << "userId is empty!";
    }
    auto domainName = config_json["domainName"];
    auto xroleName = config_json["xroleName"];
    auto ak = config_json["ak"];
    auto sk = config_json["sk"];
    auto token = config_json["token"];
    if (!domainName.empty() && !xroleName.empty()) {
      download_info.domain_name = domainName;
      download_info.xrole_name = xroleName;
    }
    if (!ak.empty() && !sk.empty()) {
      download_info.ak = ak;
      download_info.sk = sk;
      if (!token.empty()) {
        download_info.token = token;
      }
    }
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }

  return modelbox::STATUS_OK;
}

std::string ObsSourceParser::GetTimeString(time_t *time) {
  if (nullptr == time) {
    return "";
  }

  tm gmtm;
  gmtime_r(time, &gmtm);
  return std::to_string(gmtm.tm_year + 1900) + std::to_string(gmtm.tm_mon + 1) +
         std::to_string(gmtm.tm_mday) + std::to_string(gmtm.tm_hour) +
         std::to_string(gmtm.tm_min) + std::to_string(gmtm.tm_sec);
}

// TODO: 多路流使用同一个输入文件时，必须在最后一路退出后再删除文件
void RemoveFileCallback(std::string uri) {
  if (uri.empty()) {
    MBLOG_WARN << "Empty uri to be removed." << uri;
    return;
  }

  struct stat stat_buffer;
  stat(uri.c_str(), &stat_buffer);
  if (stat_buffer.st_mode & S_IFREG) {
    if (0 == std::remove(uri.c_str())) {
    } else {
      MBLOG_WARN << "Failed to remove obs downloaded file: " << uri;
    }
  }
}
