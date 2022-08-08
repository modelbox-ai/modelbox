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

#include "dis_output_broker.h"

#include <securec.h>

#include <nlohmann/json.hpp>

#include "iam_auth.h"
#include "modelbox/base/uuid.h"
#include "modelbox/device/cpu/device_cpu.h"

#define MAX_AK_SIZE 64
#define MAX_SK_SIZE 128
#define MAX_TOKEN_SIZE 16 * 1024
#define MAX_PROJECT_ID_SIZE 64
#define SERIALIZE_MODE "base64"

DisOutputConfigurations DisOutputBroker::output_configs_;
std::mutex DisOutputBroker::output_configs_lock_;
thread_local modelbox::Status callback_status;

DisOutputBroker::DisOutputBroker() {}
DisOutputBroker::~DisOutputBroker() {}

std::atomic_bool DisOutputBroker::init_flag_{false};

modelbox::Status DisOutputBroker::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  bool expect = false;
  if (!init_flag_.compare_exchange_strong(expect, true,
                                          std::memory_order_acq_rel)) {
    return modelbox::STATUS_OK;
  }

  FILE *logFile = NULL;
  logFile = stdout;
  if (NULL == logFile) {
    MBLOG_ERROR << "Output dis sdk log failed";
    return modelbox::STATUS_FAULT;
  }
  int ret = 0;
  ret = DisInit(logFile, GetUserAuthInfo);
  if (0 != ret) {
    MBLOG_ERROR << "Init dis sdk failed: " << ret;
    return modelbox::STATUS_FAULT;
  }
  MBLOG_INFO << "Init dis sdk success";

  return modelbox::STATUS_OK;
}

modelbox::Status DisOutputBroker::Deinit() {
  bool expect = true;
  if (!init_flag_.compare_exchange_strong(expect, false,
                                          std::memory_order_acq_rel)) {
    return modelbox::STATUS_OK;
  }

  DisDeinit();
  MBLOG_INFO << "Deinit dis sdk success";
  return modelbox::STATUS_OK;
}

std::shared_ptr<modelbox::OutputBrokerHandle> DisOutputBroker::Open(
    const std::string &config) {
  std::string uuid;
  if (modelbox::STATUS_OK != modelbox::GetUUID(&uuid)) {
    MBLOG_ERROR << "Failed to generate a uuid for the dis output broker!";
    return nullptr;
  }

  auto handle = std::make_shared<modelbox::OutputBrokerHandle>();
  handle->broker_id_ = uuid;

  if (modelbox::STATUS_OK != ParseConfig(handle, config)) {
    MBLOG_ERROR << "Parse config to json failed";
    return nullptr;
  }

  return handle;
}

modelbox::Status DisOutputBroker::Write(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  if (buffer == nullptr) {
    MBLOG_ERROR << "Invalid buffer: buffer is nullptr!";
    return modelbox::STATUS_NODATA;
  }

  size_t data_size = buffer->GetBytes();
  char *data = const_cast<char *>((const char *)buffer->ConstData());
  if (data == nullptr || data_size == 0) {
    MBLOG_WARN << "Invalid data! Nothing to be upload!";
  }

  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker configuration, type: "
        << handle->output_broker_type_ << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }
  guard.unlock();

  std::shared_ptr<DisOutputInfo> output_info = iter->second;

  char *method = const_cast<char *>(SERIALIZE_MODE);
  DISSetSerializedMode(method);

  char *region = const_cast<char *>(output_info->region.c_str());
  char *host = const_cast<char *>(output_info->end_point.c_str());
  char *stream_name = const_cast<char *>(output_info->stream_name.c_str());

  int ret = 0;
  char project_id_mask[MAX_PROJECT_ID_SIZE];
  ret = snprintf_s(project_id_mask, MAX_PROJECT_ID_SIZE,
                   handle->broker_id_.size(), handle->broker_id_.c_str());
  if (ret == -1) {
    MBLOG_ERROR << "Failed to copy broker_id to project_id."
                << " ret: " << ret
                << ", broker_id size: " << handle->broker_id_.size()
                << ", MAX_PROJECT_ID_SIZE: " << MAX_PROJECT_ID_SIZE;
    if (handle->broker_id_.size() >= MAX_PROJECT_ID_SIZE) {
      MBLOG_ERROR << "MAX_PROJECT_ID_SIZE must be larger than broker_id size.";
    }
    return modelbox::STATUS_FAULT;
  }

  callback_status = modelbox::STATUS_OK;

  DISResponseInfo rsp_info = {0};
  DISPutRecord record = {0};
  record.recordData.stringLen = data_size;
  record.recordData.data = data;
  record.partitionKey = const_cast<char *>(handle->broker_id_.c_str());

  ret = PutRecords(host, project_id_mask, region, stream_name, 1, &record,
                   PutRecordCallBack, &rsp_info);
  if (ret == 0 && rsp_info.HttpResponseCode < 300 &&
      callback_status == modelbox::STATUS_OK) {
    MBLOG_DEBUG << "Send record success, ret: " << ret
                << ". Http response code: " << rsp_info.HttpResponseCode;
    return modelbox::STATUS_OK;
  } else if (JudgeTryAgain(rsp_info.HttpResponseCode) ==
                 modelbox::STATUS_FAULT ||
             callback_status == modelbox::STATUS_FAULT) {
    MBLOG_ERROR << "Send record failed, the httprspcode is: "
                << rsp_info.HttpResponseCode << ", ret: " << ret
                << ". Error code: " << rsp_info.ErrorCode
                << ". Error detail: " << rsp_info.ErrorDetail;
    return modelbox::STATUS_FAULT;
  } else {
    if (JudgeUpdateCert(rsp_info.HttpResponseCode)) {
      MBLOG_WARN << "Send dis failed, try to update cert info.";
      modelbox::AgencyInfo agency_info;
      agency_info.user_domain_name = output_info->domain_name;
      agency_info.xrole_name = output_info->xrole_name;
      auto hw_cert = modelbox::IAMAuth::GetInstance();
      if (hw_cert == nullptr) {
        MBLOG_ERROR << "Failed to get hw_cert instance!";
        return {modelbox::STATUS_FAULT};
      }
      hw_cert->ExpireUserAgencyProjectCredential(agency_info);
    }

    MBLOG_WARN << "Try to send dis again.";

    return modelbox::STATUS_AGAIN;
  }
}

modelbox::Status DisOutputBroker::Sync(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  return modelbox::STATUS_OK;
}

modelbox::Status DisOutputBroker::Close(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR << "Broker handle not found, type: "
                << handle->output_broker_type_
                << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  output_configs_.erase(handle->broker_id_);
  guard.unlock();

  return modelbox::STATUS_OK;
}

DISStatus DisOutputBroker::GetUserAuthInfo(char *project_id, char *ak_array,
                                           char *sk_array,
                                           char *x_security_token) {
  std::string broker_id;
  broker_id = project_id;
  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(broker_id);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker configuration, id: "
        << broker_id;
    return DISStatusGetUserAuthInfoErr;
  }
  guard.unlock();

  std::shared_ptr<DisOutputInfo> output_info = iter->second;

  if (GetCertInfo(output_info) != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to send data! Invalid authorization.";
    return DISStatusGetUserAuthInfoErr;
  }
  int ret = 0;
  ret = strncpy_s(ak_array, MAX_AK_SIZE, output_info->ak.c_str(),
                  output_info->ak.size());
  if (ret != 0) {
    MBLOG_ERROR << "Failed to copy output_info->ak to ak_array."
                << " ret: " << ret
                << ". output_info->ak size: " << output_info->ak.size()
                << ", MAX_AK_SIZE: " << MAX_AK_SIZE;
    if (output_info->ak.size() >= MAX_AK_SIZE) {
      MBLOG_ERROR << "MAX_AK_SIZE must be larger than output_info->ak size.";
    }
    return DISStatusGetUserAuthInfoErr;
  }
  ret = strncpy_s(sk_array, MAX_SK_SIZE, output_info->sk.c_str(),
                  output_info->sk.size());
  if (ret != 0) {
    MBLOG_ERROR << "Failed to copy output_info->sk to sk_array."
                << " ret: " << ret
                << ". output_info->sk size: " << output_info->sk.size()
                << ", MAX_SK_SIZE: " << MAX_SK_SIZE;
    if (output_info->sk.size() >= MAX_SK_SIZE) {
      MBLOG_ERROR << "MAX_SK_SIZE must be larger than output_info->ak size.";
    }
    return DISStatusGetUserAuthInfoErr;
  }
  ret = strncpy_s(project_id, MAX_PROJECT_ID_SIZE,
                  output_info->project_id.c_str(),
                  output_info->project_id.size());
  if (ret != 0) {
    MBLOG_ERROR << "Failed to copy output_info->project_id to project_id."
                << " ret: " << ret << ". output_info->project_id size: "
                << output_info->project_id.size()
                << ", MAX_PROJECT_ID_SIZE: " << MAX_PROJECT_ID_SIZE;
    if (output_info->project_id.size() >= MAX_PROJECT_ID_SIZE) {
      MBLOG_ERROR << "MAX_PROJECT_ID_SIZE must be larger than "
                     "output_info->project_id size.";
    }
    return DISStatusGetUserAuthInfoErr;
  }
  ret = strncpy_s(x_security_token, MAX_TOKEN_SIZE, output_info->token.c_str(),
                  output_info->token.size());
  if (ret != 0) {
    MBLOG_ERROR << "Failed to copy output_info->token to x_security_token."
                << " ret: " << ret
                << ". output_info->token size: " << output_info->token.size()
                << ", MAX_TOKEN_SIZE: " << MAX_TOKEN_SIZE;
    if (output_info->token.size() >= MAX_TOKEN_SIZE) {
      MBLOG_ERROR
          << "MAX_TOKEN_SIZE must be larger than output_info->token size.";
    }
    return DISStatusGetUserAuthInfoErr;
  }

  MBLOG_DEBUG << "Get user auth info success";

  return DISStatusOK;
}

modelbox::Status DisOutputBroker::GetCertInfo(
    std::shared_ptr<DisOutputInfo> &output_info) {
  MBLOG_DEBUG << "Try to get cert info.";

  modelbox::UserAgencyCredential credential;
  modelbox::AgencyInfo agent_info;

  agent_info.user_domain_name = output_info->domain_name;
  agent_info.xrole_name = output_info->xrole_name;

  auto hw_cert = modelbox::IAMAuth::GetInstance();
  if (hw_cert == nullptr) {
    MBLOG_ERROR << "Failed to get hw_cert instance!";
    return {modelbox::STATUS_FAULT};
  }

  auto ret = hw_cert->GetUserAgencyProjectCredential(credential, agent_info,
                                                     output_info->user_id);

  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to get credential info!";
    return {modelbox::STATUS_FAULT};
  }

  if (credential.user_ak.empty()) {
    MBLOG_ERROR << "Failed to get credential ak info! String is empty.";
    return {modelbox::STATUS_FAULT};
  }
  if (credential.user_sk.empty()) {
    MBLOG_ERROR << "Failed to get credential sk info! String is empty.";

    return {modelbox::STATUS_FAULT};
  }
  if (credential.user_secure_token.empty()) {
    MBLOG_ERROR << "Failed to get credential token info! String is empty.";
    return {modelbox::STATUS_FAULT};
  }

  output_info->ak = credential.user_ak;
  output_info->sk = credential.user_sk;
  output_info->token = "X-Security-Token:" + credential.user_secure_token;

  MBLOG_DEBUG << "Get cert info success.";

  return modelbox::STATUS_OK;
}

modelbox::Status DisOutputBroker::ParseConfig(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::string &config) {
  nlohmann::json json;
  std::shared_ptr<DisOutputInfo> output_info =
      std::make_shared<DisOutputInfo>();
  try {
    json = nlohmann::json::parse(config);

    std::string end_point;
    end_point = json["disEndPoint"].get<std::string>();
    std::string::size_type idx;
    std::string https_endpoint = "https://";
    idx = end_point.find(https_endpoint);
    if (idx == 0) {
      end_point = end_point.erase(idx, https_endpoint.size());
    }
    output_info->end_point = end_point;
    if (end_point.empty()) {
      MBLOG_ERROR
          << "Invalid disEndPoint, value of key <disEndPoint> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    output_info->region = json["region"].get<std::string>();
    if (output_info->region.empty()) {
      MBLOG_ERROR << "Invalid region, value of key <region> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    output_info->stream_name = json["streamName"].get<std::string>();
    if (output_info->stream_name.empty()) {
      MBLOG_ERROR << "Invalid streamName, value of key <streamName> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    output_info->project_id = json["projectId"].get<std::string>();
    if (output_info->project_id.empty()) {
      MBLOG_ERROR << "Invalid projectId, value of key <projectId> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    if (json.contains("domainName")) {
      output_info->domain_name = json["domainName"].get<std::string>();
      if (output_info->domain_name.empty()) {
        MBLOG_DEBUG << "Value of key <domainName> is empty!";
      }
    }

    if (json.contains("xroleName")) {
      output_info->xrole_name = json["xroleName"].get<std::string>();
      if (output_info->xrole_name.empty()) {
        MBLOG_DEBUG << "Value of key <xroleName> is empty!";
      }
    }

    if (json.contains("userId")) {
      output_info->user_id = json["userId"].get<std::string>();
      MBLOG_DEBUG << "Value of key <userId> is " << output_info->user_id;
    }

    std::unique_lock<std::mutex> guard(output_configs_lock_);
    output_configs_[handle->broker_id_] = output_info;
    guard.unlock();

    MBLOG_DEBUG << "Parse cfg json success.";

    return modelbox::STATUS_OK;
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse output config to json failed, detail: " << e.what();

    return modelbox::STATUS_BADCONF;
  }
}

DISStatus DisOutputBroker::PutRecordCallBack(
    char *error_code, char *error_details, char *stream_name,
    DISPutRecord *put_record, char *seq_number, char *partitiod_id) {
  if (NULL == seq_number) {
    MBLOG_WARN << "Send record failed, key: " << put_record->partitionKey
               << ", error code: " << error_code
               << ", message: " << error_details;
    if (strncmp(error_code, "DIS.4219", 8) == 0 ||
        strncmp(error_code, "DIS.4223", 8) == 0) {
      callback_status = modelbox::STATUS_FAULT;
    } else {
      MBLOG_WARN << "Set try again flag.";
      callback_status = modelbox::STATUS_AGAIN;
    }
    return DISStatusError;
  } else {
    MBLOG_DEBUG << "Send record success, key: " << put_record->partitionKey
                << ", seqnum: " << seq_number << ", pid: " << partitiod_id;
    callback_status == modelbox::STATUS_OK;
    return DISStatusOK;
  }
}

modelbox::Status DisOutputBroker::JudgeTryAgain(long http_response_code) {
  switch (http_response_code) {
    case 400:
    case 403:
    case 404:
    case 405:
    case 503:
      return modelbox::STATUS_FAULT;
    default:
      return modelbox::STATUS_AGAIN;
  }
}

bool DisOutputBroker::JudgeUpdateCert(long http_response_code) {
  switch (http_response_code) {
    case 401:
    case 407:
    case 441:
      return true;
    default:
      return false;
  }
}
