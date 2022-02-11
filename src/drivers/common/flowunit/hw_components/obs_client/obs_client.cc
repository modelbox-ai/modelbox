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

#include "obs_client.h"

#include <securec.h>

#include "iam_auth/iam_auth.h"
#include "modelbox/base/utils.h"

#define OBS_SDK_MAX_KEYS 1000
#define MAX_RETRY_COUNTS 3
#define OBJECTS_LIST_MARKER_SIZE 4096

namespace modelbox {

std::mutex ObsClient::obs_client_lock_;

// callbacks for OBS SDK
obs_status ResponsePropertiesCallback(const obs_response_properties *properties,
                                      void *callback_data);
void ListObjectCompleteCallback(obs_status status,
                                const obs_error_details *error,
                                void *callback_data);
obs_status ListObjectsCallback(int is_truncated, const char *next_marker,
                               int contents_count,
                               const obs_list_objects_content *contents,
                               int common_prefixes_count,
                               const char **common_prefixes,
                               void *callback_data);
obs_status GetPropertiesCallback(const obs_response_properties *properties,
                                 void *callback_data);
void GetObjectCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data);
obs_status GetObjectDataCallback(int buffer_size, const char *buffer,
                                 void *callback_data);
void PutBufferCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data);
int PutBufferDataCallback(int buffer_size, char *buffer, void *callback_data);
obs_status GetObjectSizeCallback(const obs_response_properties *properties,
                                 void *callback_data);

void GetObjectSizeCompleteCallback(obs_status status,
                                   const obs_error_details *error,
                                   void *callback_data);

obs_status GetBufferCallback(int buffer_size, const char *buffer,
                             void *callback_data);

void GetBufferCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data);

// data struct for OBS SDK callbacks
using GetObjectListData = struct {
  int is_truncated;
  char next_marker[OBJECTS_LIST_MARKER_SIZE];
  std::vector<std::string> object_keys_list;
  obs_status ret_status;
};

using GetObjectCallbackData = struct {
  std::shared_ptr<FILE> out_file;
  obs_status ret_status;
};

using GetBufferCallbackData = struct {
  unsigned char *get_buffer;
  uint64_t buffer_size;
  obs_status ret_status;
};

using GetObejectSizeCallbackData = struct {
  uint64_t content_length;
  obs_status ret_status;
};

using PutBufferObjectCallbackData = struct {
  char *put_buffer;
  uint64_t buffer_size;
  uint64_t cur_offset;
  obs_status ret_status;
};

std::shared_ptr<ObsClient> ObsClient::GetInstance() {
  std::lock_guard<std::mutex> lock(obs_client_lock_);
  static std::shared_ptr<ObsClient> obs_client = nullptr;
  if (nullptr == obs_client) {
    obs_client = std::shared_ptr<ObsClient>(new ObsClient());
    if (nullptr == obs_client) {
      MBLOG_ERROR << "Failed to construct obs client!";
      return nullptr;
    }
  }

  static bool is_initialized = false;
  if (true == is_initialized) {
    return obs_client;
  }
  auto ret = obs_client->InitObsSdk();
  if (modelbox::STATUS_OK != ret.Code()) {
    MBLOG_ERROR << ret.Errormsg();
    return nullptr;
  }
  is_initialized = true;

  return obs_client;
}

ObsClient::~ObsClient() { DeInitObsSdk(); };

modelbox::Status ObsClient::InitObsSdk() {
  obs_status ret_status = OBS_STATUS_BUTT;
  ret_status = obs_initialize(OBS_INIT_ALL);
  if (OBS_STATUS_OK != ret_status) {
    auto obs_err = obs_get_status_name(ret_status);
    std::string err_msg = "Failed to initialize OBS SDK";
    if (obs_err) {
      err_msg += ", ";
      err_msg += obs_err;
    }
    return {modelbox::STATUS_FAULT, err_msg};
  }
  return modelbox::STATUS_OK;
}

void ObsClient::DeInitObsSdk() { obs_deinitialize(); }

modelbox::Status ObsClient::GetAuthInfo(const std::string &domain_name,
                                        const std::string &xrole_name,
                                        const std::string &user_id,
                                        std::string &access_key,
                                        std::string &secret_key,
                                        std::string &security_token) {
  std::string err_msg = "";

  modelbox::AgencyInfo agent_info;
  agent_info.user_domain_name = domain_name;
  agent_info.xrole_name = xrole_name;

  auto hw_auth = modelbox::IAMAuth::GetInstance();
  if (hw_auth == nullptr) {
    err_msg = "Failed to get hw_auth instance!";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  modelbox::UserAgencyCredential credential;
  auto ret =
      hw_auth->GetUserAgencyProjectCredential(credential, agent_info, user_id);
  if (ret != modelbox::STATUS_OK) {
    err_msg = "Failed to get credential info!";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  access_key = credential.user_ak;
  secret_key = credential.user_sk;
  security_token = credential.user_secure_token;

  return modelbox::STATUS_OK;
}

modelbox::Status ObsClient::GetUpdatedAuthInfo(const std::string &domain_name,
                                               const std::string &xrole_name,
                                               const std::string &user_id,
                                               std::string &access_key,
                                               std::string &secret_key,
                                               std::string &security_token) {
  modelbox::AgencyInfo agency_info;
  agency_info.user_domain_name = domain_name;
  agency_info.xrole_name = xrole_name;

  std::string err_msg = "";
  auto hw_auth = modelbox::IAMAuth::GetInstance();
  if (hw_auth == nullptr) {
    err_msg = "Failed to get hw_auth instance!";
    return {modelbox::STATUS_FAULT, err_msg};
  }
  hw_auth->ExpireUserAgencyProjectCredential(agency_info);

  modelbox::UserAgencyCredential user_credential;
  auto ret = hw_auth->GetUserAgencyProjectCredential(user_credential,
                                                     agency_info, user_id);
  if (ret != modelbox::STATUS_OK) {
    err_msg = "Failed to get the renewed credential info!";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  access_key = user_credential.user_ak;
  secret_key = user_credential.user_sk;
  security_token = user_credential.user_secure_token;

  return modelbox::STATUS_OK;
}

modelbox::Status ObsClient::NotifyToUpdateAuthInfo(
    const std::string &domain_name, const std::string &xrole_name) {
  modelbox::AgencyInfo agency_info;
  agency_info.user_domain_name = domain_name;
  agency_info.xrole_name = xrole_name;

  std::string err_msg = "";
  auto hw_auth = modelbox::IAMAuth::GetInstance();
  if (hw_auth == nullptr) {
    err_msg = "Failed to update Auth info";
    return {modelbox::STATUS_FAULT, err_msg};
  }
  hw_auth->ExpireUserAgencyProjectCredential(agency_info);

  return modelbox::STATUS_OK;
}

bool ObsClient::IsValidOptionExceptPath(const ObsOptions &opt) {
  if (opt.end_point.empty() || opt.bucket.empty()) {
    return false;
  }
  return true;
}

bool ObsClient::IsValidOptionIncludingPath(const ObsOptions &opt) {
  if (!IsValidOptionExceptPath(opt) || opt.path.empty()) {
    return false;
  }
  return true;
}

void ObsClient::SetObsOption(const ObsOptions &src, const std::string &ak,
                             const std::string &sk,
                             const std::string &security_token,
                             obs_options &dst) {
  dst.bucket_options.host_name = const_cast<char *>(src.end_point.c_str());
  dst.bucket_options.bucket_name = const_cast<char *>(src.bucket.c_str());
  dst.bucket_options.access_key = const_cast<char *>(ak.c_str());
  dst.bucket_options.secret_access_key = const_cast<char *>(sk.c_str());
  dst.bucket_options.token = const_cast<char *>(security_token.c_str());
}

std::shared_ptr<FILE> ObsClient::OpenLocalFile(
    const std::string &full_file_path) {
  if (full_file_path.empty()) {
    return nullptr;
  }

  std::string path = full_file_path.substr(0, full_file_path.rfind("/"));

  if (modelbox::CreateDirectory(path) != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to create folder for obs object (" << full_file_path
                << "), error info: " << modelbox::StrError(errno);
    return nullptr;
  }

  auto path_name = modelbox::PathCanonicalize(full_file_path);
  FILE *out_file = fopen(path_name.c_str(), "wb");
  if (!out_file) {
    MBLOG_ERROR << "Failed to create file " << full_file_path
                << ", because: " << modelbox::StrError(errno);
    return nullptr;
  }
  auto file_ptr = std::shared_ptr<FILE>(out_file, [](FILE *file) {
    fflush(file);
    fclose(file);
  });
  return file_ptr;
}

bool ObsClient::NeedUpdateAuthInfo(obs_status status) {
  switch (status) {
    case OBS_STATUS_InvalidAccessKeyId:
    case OBS_STATUS_NoToken:
    case OBS_STATUS_ExpiredToken:
    case OBS_STATUS_InvalidToken:
    case OBS_STATUS_TokenRefreshRequired:
      return true;
    default:
      return false;
  }
}

bool ObsClient::NeedTryAgain(obs_status status) {
  switch (status) {
    case OBS_STATUS_EntityTooSmall:
    case OBS_STATUS_EntityTooLarge:
    case OBS_STATUS_InlineDataTooLarge:
    case OBS_STATUS_NoSuchBucket:
    case OBS_STATUS_NoSuchKey:
    case OBS_STATUS_OK:
      return false;
    default:
      return true;
  }
}

modelbox::Status ObsClient::GetObjectsList(
    const ObsOptions &opt, std::vector<std::string> &object_list) {
  if (!IsValidOptionExceptPath(opt)) {
    std::string err_msg = "Invalid parameters!";
    return {modelbox::STATUS_INVALID, err_msg};
  }

  // get Authorization info
  std::string ak;
  std::string sk;
  std::string security_token;
  auto ret = GetAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, ak, sk,
                         security_token);
  if (modelbox::STATUS_OK != ret) {
    return ret;
  }

  // create and initialize the obs option
  obs_options option;
  init_obs_options(&option);
  SetObsOption(opt, ak, sk, security_token, option);

  // set callbacks
  obs_list_objects_handler list_bucket_objects_handler = {
      {&ResponsePropertiesCallback, &ListObjectCompleteCallback},
      &ListObjectsCallback};

  // user-defined callback data
  GetObjectListData data;
  memset_s(&data, sizeof(GetObjectListData), 0, sizeof(GetObjectListData));

  int retry_count = 0;
  char next_marker[OBJECTS_LIST_MARKER_SIZE] = {0};

  // list objects
  while (retry_count < MAX_RETRY_COUNTS) {
    if (opt.path.empty()) {
      list_bucket_objects(&option, nullptr, next_marker, nullptr,
                          OBS_SDK_MAX_KEYS, &list_bucket_objects_handler,
                          &data);
    } else {
      list_bucket_objects(&option, opt.path.c_str(), next_marker, nullptr,
                          OBS_SDK_MAX_KEYS, &list_bucket_objects_handler,
                          &data);
    }

    if (OBS_STATUS_OK == data.ret_status) {
      retry_count = 0;  // reset
      // successfully get complete list
      if (!data.is_truncated) {
        break;
      }
      auto len =
          snprintf_s(next_marker, OBJECTS_LIST_MARKER_SIZE,
                     OBJECTS_LIST_MARKER_SIZE - 1, "%s", data.next_marker);
      if (len < 0 || len >= OBJECTS_LIST_MARKER_SIZE - 1) {
        MBLOG_WARN << "marker is too long: " << next_marker;
        return {modelbox::STATUS_INVALID, "marker too long"};
      }

      continue;
    }

    ++retry_count;
    if (NeedUpdateAuthInfo(data.ret_status)) {
      MBLOG_WARN << "Auth info expire and need to be updated.";
      auto ret = GetUpdatedAuthInfo(opt.domain_name, opt.xrole_name,
                                    opt.user_id, ak, sk, security_token);
      if (modelbox::STATUS_OK != ret) {
        MBLOG_WARN << "Failed to update auth info!! obs_ret_status: "
                   << obs_get_status_name(data.ret_status)
                   << ", try count: " << retry_count;
        continue;
      }
      MBLOG_WARN << "Auth info is updated successfully. obs_ret_status: "
                 << obs_get_status_name(data.ret_status)
                 << ", try count: " << retry_count;
      SetObsOption(opt, ak, sk, security_token, option);
      continue;
    }
    MBLOG_ERROR << "Failed to list objects! obs_ret_status: "
                << obs_get_status_name(data.ret_status)
                << ", try count: " << retry_count;
  }

  std::string err_msg = "";
  if (retry_count >= MAX_RETRY_COUNTS) {
    err_msg = "Failed to list objects after " + std::to_string(retry_count) +
              " tries!";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  object_list = std::move(data.object_keys_list);
  return modelbox::STATUS_OK;
}

modelbox::Status ObsClient::GetObject(const ObsOptions &opt,
                                      const std::string &file_local_path) {
  std::string err_msg = "";
  if (!IsValidOptionIncludingPath(opt) || file_local_path.empty()) {
    err_msg = "Failed to download obs object: Invalid parameters! file key: " +
              file_local_path;
    return {modelbox::STATUS_INVALID, err_msg};
  }

  // get Authorization info
  std::string ak;
  std::string sk;
  std::string security_token;
  auto ret = GetAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, ak, sk,
                         security_token);
  if (modelbox::STATUS_OK != ret) {
    return ret;
  }

  // Initialize the download option
  obs_options option;
  init_obs_options(&option);
  SetObsOption(opt, ak, sk, security_token, option);

  obs_object_info object_info = {0};
  object_info.key = const_cast<char *>(opt.path.c_str());

  GetObjectCallbackData data;
  data.ret_status = OBS_STATUS_BUTT;
  data.out_file = OpenLocalFile(file_local_path);
  if (nullptr == data.out_file) {
    err_msg =
        "Failed to download obs object: can't open local file to accept "
        "data: " +
        file_local_path;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  // define the download range; 0 indicates to download the whole object.
  obs_get_conditions get_condition = {0};
  init_get_properties(&get_condition);
  get_condition.start_byte = 0;  // the start position of the object
  get_condition.byte_count =
      0;  // download length, default is 0, up to the end of the object
  obs_get_object_handler get_object_handler = {
      {&GetPropertiesCallback, &GetObjectCompleteCallback},
      &GetObjectDataCallback};

  // download
  get_object(&option, &object_info, &get_condition, 0, &get_object_handler,
             &data);

  if (NeedUpdateAuthInfo(data.ret_status)) {
    MBLOG_WARN
        << "Denied to access OBS. Maybe Auth info expired. Try to update.";
    ret = GetUpdatedAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, ak,
                             sk, security_token);
    if (modelbox::STATUS_OK != ret) {
      MBLOG_WARN << "Failed to update hw_auth info.";
    } else {
      // try to get object again.
      SetObsOption(opt, ak, sk, security_token, option);
      get_object(&option, &object_info, &get_condition, 0, &get_object_handler,
                 &data);
    }
  }

  if (OBS_STATUS_OK != data.ret_status) {
    auto obs_status_name = obs_get_status_name(data.ret_status);
    if (obs_status_name == nullptr) {
      obs_status_name = "null";
    }
    err_msg = "Failed to download obs object: [" + opt.bucket + "] - " +
              opt.path + " err_msg: (" + std::to_string(data.ret_status) +
              ": " + obs_status_name + ").";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ObsClient::GetBuffer(ObsOptions &opt, unsigned char *buf,
                                      uint64_t size, uint64_t offset) {
  std::string err_msg = "";

  obs_object_info object_info = {0};
  object_info.key = const_cast<char *>(opt.path.c_str());

  GetBufferCallbackData data;
  data.ret_status = OBS_STATUS_BUTT;
  data.get_buffer = buf;
  data.buffer_size = 0;

  // get Authorization info
  if (opt.ak.empty() || opt.sk.empty()) {
    auto ret = GetAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, opt.ak,
                           opt.sk, opt.token);
    if (modelbox::STATUS_OK != ret) {
      return ret;
    }
  }

  // Initialize the download option
  obs_options option;
  init_obs_options(&option);
  SetObsOption(opt, opt.ak, opt.sk, opt.token, option);

  // define the download range; 0 indicates to download the whole object.
  obs_get_conditions get_condition = {0};
  init_get_properties(&get_condition);
  get_condition.start_byte = offset;  // the start position of the object
  get_condition.byte_count =
      size;  // download length, default is 0, up to the end of the object
  obs_get_object_handler get_object_handler = {
      {&GetPropertiesCallback, &GetBufferCompleteCallback}, &GetBufferCallback};

  // download
  get_object(&option, &object_info, &get_condition, 0, &get_object_handler,
             &data);

  if (NeedUpdateAuthInfo(data.ret_status)) {
    MBLOG_WARN
        << "Denied to access OBS. Maybe Auth info expired. Try to update.";
    auto ret = GetUpdatedAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id,
                                  opt.ak, opt.sk, opt.token);
    if (modelbox::STATUS_OK != ret) {
      MBLOG_WARN << "Failed to update hw_auth info.";
    } else {
      // try to get object again.
      SetObsOption(opt, opt.ak, opt.sk, opt.token, option);
      get_object(&option, &object_info, &get_condition, 0, &get_object_handler,
                 &data);
    }
  }

  if (OBS_STATUS_OK != data.ret_status) {
    err_msg = "Failed to get buffer from obs data, bucket: " + opt.bucket +
              ", file key: " + opt.path +
              ", buffer size: " + std::to_string(size) +
              ", obs status: " + std::to_string(data.ret_status) + " (" +
              obs_get_status_name(data.ret_status) + ").";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

uint64_t ObsClient::GetObjectSize(ObsOptions &opt) {
  std::string err_msg = "";

  obs_object_info object_info = {0};
  object_info.key = const_cast<char *>(opt.path.c_str());

  GetObejectSizeCallbackData data;
  data.ret_status = OBS_STATUS_BUTT;
  data.content_length = 0;

  // get Authorization info
  if (opt.ak.empty() || opt.sk.empty()) {
    auto ret = GetAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, opt.ak,
                           opt.sk, opt.token);
    if (modelbox::STATUS_OK != ret) {
      return 0;
    }
  }

  // Initialize the download option
  obs_options option;
  init_obs_options(&option);
  SetObsOption(opt, opt.ak, opt.sk, opt.token, option);

  obs_response_handler response_handler = {&GetObjectSizeCallback,
                                           &GetObjectSizeCompleteCallback};

  get_object_metadata(&option, &object_info, 0, &response_handler, &data);

  if (NeedUpdateAuthInfo(data.ret_status)) {
    MBLOG_WARN
        << "Denied to access OBS. Maybe Auth info expired. Try to update.";
    auto ret = GetUpdatedAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id,
                                  opt.ak, opt.sk, opt.token);
    if (modelbox::STATUS_OK != ret) {
      MBLOG_WARN << "Failed to update hw_auth info.";
    } else {
      // try to get object again.
      SetObsOption(opt, opt.ak, opt.sk, opt.token, option);
      get_object_metadata(&option, &object_info, 0, &response_handler, &data);
    }
  }

  if (OBS_STATUS_OK != data.ret_status) {
    err_msg = "Failed to get obs object size, bucket: " + opt.bucket +
              ", file key: " + opt.path +
              ", obs status: " + std::to_string(data.ret_status) + " (" +
              obs_get_status_name(data.ret_status) + ").";
    MBLOG_ERROR << err_msg;
  }

  return data.content_length;
}

modelbox::Status ObsClient::PutObject(const ObsOptions &opt, const char *data,
                                      size_t data_size) {
  std::string err_msg = "";
  if (!IsValidOptionIncludingPath(opt)) {
    err_msg = "Failed to output obs data: Invalid obs options!";
    return {modelbox::STATUS_INVALID, err_msg};
  }

  if (data == nullptr || data_size == 0) {
    err_msg = "Failed to output obs data: Invalid data!";
    return {modelbox::STATUS_INVALID, err_msg};
  }

  // get Authorization info
  std::string ak;
  std::string sk;
  std::string security_token;
  auto ret = GetAuthInfo(opt.domain_name, opt.xrole_name, opt.user_id, ak, sk,
                         security_token);
  if (modelbox::STATUS_OK != ret) {
    return ret;
  }

  // initialize obs option
  obs_options option;
  init_obs_options(&option);
  SetObsOption(opt, ak, sk, security_token, option);

  // initialize put properties
  obs_put_properties put_properties;
  init_put_properties(&put_properties);

  // initialize put
  PutBufferObjectCallbackData data_to_put = {0};
  data_to_put.put_buffer = const_cast<char *>(data);
  data_to_put.buffer_size = data_size;

  // set callback functions
  obs_put_object_handler putobjectHandler = {
      {&ResponsePropertiesCallback, &PutBufferCompleteCallback},
      &PutBufferDataCallback};

  put_object(&option, const_cast<char *>(opt.path.c_str()), data_size,
             &put_properties, 0, &putobjectHandler, &data_to_put);

  if (NeedUpdateAuthInfo(data_to_put.ret_status)) {
    MBLOG_WARN
        << "Access obs denied. Maybe Auth info is expired. Try to update.";
    ret = NotifyToUpdateAuthInfo(opt.domain_name, opt.xrole_name);
    if (modelbox::STATUS_OK != ret) {
      MBLOG_ERROR << ret.Errormsg();
    }
    return {modelbox::STATUS_AGAIN,
            "Failed to output obs data. Try again please."};
  }

  if (NeedTryAgain(data_to_put.ret_status)) {
    err_msg = "Failed to output obs data, bucket: " + opt.bucket +
              ", file key: " + opt.path +
              ", data size: " + std::to_string(data_size) +
              ", obs status: " + std::to_string(data_to_put.ret_status) + " (" +
              obs_get_status_name(data_to_put.ret_status) + ").";
    return {modelbox::STATUS_AGAIN, err_msg};
  }

  if (OBS_STATUS_OK != data_to_put.ret_status) {
    err_msg = "Failed to output obs data, bucket: " + opt.bucket +
              ", file key: " + opt.path +
              ", data size: " + std::to_string(data_size) +
              ", obs status: " + std::to_string(data_to_put.ret_status) + " (" +
              obs_get_status_name(data_to_put.ret_status) + ").";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

obs_status ResponsePropertiesCallback(const obs_response_properties *properties,
                                      void *callback_data) {
  return OBS_STATUS_OK;
}

void ListObjectCompleteCallback(obs_status status,
                                const obs_error_details *error,
                                void *callback_data) {
  auto data = (GetObjectListData *)callback_data;
  data->ret_status = status;

  if (data->ret_status == OBS_STATUS_OK && error && error->message) {
    // sometimes there would be an error, eventhough the ret_status is OK.
    data->ret_status = OBS_STATUS_InternalError;
  }
}

obs_status ListObjectsCallback(int is_truncated, const char *next_marker,
                               int contents_count,
                               const obs_list_objects_content *contents,
                               int common_prefixes_count,
                               const char **common_prefixes,
                               void *callback_data) {
  if (contents_count < 0) {
    MBLOG_WARN << "Illegal contents count: " << contents_count;
    return OBS_STATUS_AbortedByCallback;
  }

  auto data = (GetObjectListData *)callback_data;

  data->is_truncated = is_truncated;
  // This is tricky.  S3 doesn't return the NextMarker if there is no
  // delimiter.  Why, I don't know, since it's still useful for paging
  // through results.  We want NextMarker to be the last content in the
  // list, so set it to that if necessary.
  if ((!next_marker || !next_marker[0]) && contents_count) {
    next_marker = contents[contents_count - 1].key;
  }
  if (next_marker) {
    auto ret = snprintf_s(data->next_marker, OBJECTS_LIST_MARKER_SIZE,
                          OBJECTS_LIST_MARKER_SIZE - 1, "%s", next_marker);
    if (ret < 0 || ret >= OBJECTS_LIST_MARKER_SIZE - 1) {
      MBLOG_WARN << "marker is too long: " << next_marker;
      return OBS_STATUS_AbortedByCallback;
    }
  } else {
    data->next_marker[0] = 0;
  }

  for (uint32_t i = 0; i < (uint32_t)contents_count; i++) {
    std::string strFilePath = contents[i].key;
    std::string lastchar = strFilePath.substr(strFilePath.length() - 1);

    // skip empty directories.
    if ("/" == lastchar) {
      continue;
    }

    data->object_keys_list.push_back(strFilePath);
  }

  return OBS_STATUS_OK;
}

obs_status GetPropertiesCallback(const obs_response_properties *properties,
                                 void *callback_data) {
  return OBS_STATUS_OK;
}

void GetObjectCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data) {
  if (nullptr != error->message) {
    MBLOG_WARN << "OBS error message: " << error->message;
  }
  if (nullptr != error->resource) {
    MBLOG_WARN << "OBS error resource: " << error->resource;
  }
  if (nullptr != error->further_details) {
    MBLOG_WARN << "OBS error further detail: " << error->further_details;
  }
  if (OBS_STATUS_OK != status) {
    MBLOG_WARN << "OBS status: " << status;
  }
  auto data = (GetObjectCallbackData *)callback_data;
  if (nullptr == data) {
    return;
  }
  data->ret_status = status;
}

obs_status GetObjectDataCallback(int buffer_size, const char *buffer,
                                 void *callback_data) {
  auto data = (GetObjectCallbackData *)callback_data;
  if (nullptr == data || nullptr == data->out_file) {
    return OBS_STATUS_AbortedByCallback;
  }
  size_t wrote = fwrite(buffer, buffer_size, 1, data->out_file.get());
  return ((wrote < (size_t)1) ? OBS_STATUS_AbortedByCallback : OBS_STATUS_OK);
}

void PutBufferCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data) {
  auto data = (PutBufferObjectCallbackData *)callback_data;
  data->ret_status = status;
}

int PutBufferDataCallback(int buffer_size, char *buffer, void *callback_data) {
  auto data = (PutBufferObjectCallbackData *)callback_data;

  int toRead = 0;
  if (data->buffer_size) {
    toRead =
        ((data->buffer_size > (unsigned)buffer_size) ? (unsigned)buffer_size
                                                     : data->buffer_size);
    auto ret = memcpy_s(buffer, buffer_size,
                        data->put_buffer + data->cur_offset, toRead);
    if (EOK != ret) {
      MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size " << toRead
                  << ", dest size " << buffer_size;
      return 0;
    }
  }

  data->buffer_size -= toRead;
  data->cur_offset += toRead;

  return toRead;
}

obs_status GetObjectSizeCallback(const obs_response_properties *properties,
                                 void *callback_data) {
  auto data = (GetObejectSizeCallbackData *)callback_data;
  data->content_length = properties->content_length;
  return OBS_STATUS_OK;
}

void GetObjectSizeCompleteCallback(obs_status status,
                                   const obs_error_details *error,
                                   void *callback_data) {
  auto data = (GetObejectSizeCallbackData *)callback_data;
  data->ret_status = status;
}

obs_status GetBufferCallback(int buffer_size, const char *buffer,
                             void *callback_data) {
  auto data = (GetBufferCallbackData *)callback_data;
  if (nullptr == data || nullptr == data->get_buffer) {
    return OBS_STATUS_AbortedByCallback;
  }
  auto ret = memcpy_s(data->get_buffer + data->buffer_size, buffer_size, buffer,
                      buffer_size);
  if (EOK != ret) {
    MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size "
                << buffer_size << ", dest size " << buffer_size;
    return OBS_STATUS_InternalError;
  }
  data->buffer_size += buffer_size;
  return OBS_STATUS_OK;
}

void GetBufferCompleteCallback(obs_status status,
                               const obs_error_details *error,
                               void *callback_data) {
  auto data = (GetBufferCallbackData *)callback_data;
  data->ret_status = status;
}

}  // namespace modelbox
