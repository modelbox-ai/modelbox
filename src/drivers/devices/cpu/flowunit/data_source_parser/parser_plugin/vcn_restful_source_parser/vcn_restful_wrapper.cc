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

#include "vcn_restful_wrapper.h"

#include <functional>

#include "modelbox/base/log.h"
#include "nlohmann/json.hpp"

constexpr const char *RESTFUL_LOGIN_URL = "/loginInfo/login/v1.0";
constexpr const char *RESTFUL_LOGOUT_URL = "/users/logout";
constexpr const char *RESTFUL_KEEP_ALIVE_URL = "/common/keepAlive";
constexpr const char *RESTFUL_GET_RTSP_URL = "/video/rtspurl/v1.0";

constexpr const uint64_t IVS_RESULT_SUCCESS = 0;
constexpr const uint64_t IVS_RESULT_PWD_EXPIRED = 119101305;
constexpr const uint64_t IVS_RESULT_FIRST_LOGIN = 119101308;

constexpr const uint32_t RESTFUL_STREAM_TYPE_MAX = 3;

constexpr const int HTTP_STATUS_CODE_OK = 200;

namespace modelbox {

VcnRestfulWrapper::VcnRestfulWrapper() {
  httpcli_func_map_[REQ_GET] = [](httplib::Client &cli, const std::string &path,
                                  const httplib::Headers &headers,
                                  const std::string &body) {
    return cli.Get(path.c_str(), headers);
  };

  httpcli_func_map_[REQ_POST] =
      [](httplib::Client &cli, const std::string &path,
         const httplib::Headers &headers, const std::string &body) {
        return cli.Post(path.c_str(), headers, body, nullptr);
      };
}

modelbox::Status VcnRestfulWrapper::SendRequest(const std::string &uri,
                                                const std::string &path,
                                                const std::string &body,
                                                const httplib::Headers &headers,
                                                REQ_METHOD method,
                                                httplib::Response &resp) {
  httplib::Client cli(uri);
  cli.enable_server_certificate_verification(false);
  cli.set_write_timeout(std::chrono::seconds(30));

  auto func_item = httpcli_func_map_.find(method);
  if (func_item == httpcli_func_map_.end()) {
    return {modelbox::STATUS_NOTSUPPORT, "Not support http method"};
  }

  try {
    auto result = func_item->second(cli, path, headers, body);
    if (result == nullptr) {
      return {modelbox::STATUS_FAULT,
              "Failed to send request " + httplib::to_string(result.error())};
    }

    resp = result.value();
  } catch (const std::exception &e) {
    auto msg =
        std::string("Failed to send request, exception reason: ") + e.what();
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::Login(VcnRestfulInfo &restful_info) {
  if (!IsRestfulInfoValid(restful_info, true)) {
    std::string msg = "Failed to restful login, restful info is invalid";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_INVALID, msg};
  }

  std::string uri = "https://" + restful_info.ip + ":" + restful_info.port;

  httplib::Headers headers;
  headers.insert({"Content-Type", "application/json"});
  headers.insert({"Cache-Control", "no-cache"});

  nlohmann::json body;
  body["userName"] = restful_info.user_name;
  body["password"] = restful_info.password;

  httplib::Response resp;
  auto ret =
      SendRequest(uri, RESTFUL_LOGIN_URL, body.dump(), headers, REQ_POST, resp);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful login, send request fail reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (resp.status != HTTP_STATUS_CODE_OK) {
    std::string msg =
        "Failed to restful login, result code:" + std::to_string(resp.status);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  ret = ParseRestfulLoginResult(resp, restful_info);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful login, parse result reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::ParseRestfulLoginResult(
    const httplib::Response &resp, VcnRestfulInfo &restful_info) {
  try {
    auto result_json = nlohmann::json::parse(resp.body);
    if (!result_json.contains("resultCode") ||
        !result_json["resultCode"].is_number_unsigned()) {
      std::string msg =
          "Failed to parse login result, result code isn't exist or invalid";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    auto result_code = result_json["resultCode"].get<uint64_t>();
    if (result_code != IVS_RESULT_SUCCESS &&
        result_code != IVS_RESULT_PWD_EXPIRED &&
        result_code != IVS_RESULT_FIRST_LOGIN) {
      std::string msg = "Failed to parse login result, result code:" +
                        std::to_string(result_code);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    if (result_code == IVS_RESULT_PWD_EXPIRED) {
      MBLOG_WARN << "restful login password has expired";
    }

    if (result_code == IVS_RESULT_FIRST_LOGIN) {
      MBLOG_WARN << "restful is first login";
    }

    if (!resp.has_header("Set-Cookie")) {
      std::string msg = "Failed to parse login result, cookie isn't exist";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    auto cookie = resp.get_header_value("Set-Cookie");
    auto cookies = modelbox::StringSplit(cookie, ';');
    const std::string COOKIE_PREFIX = "JSESSIONID=";
    for (auto &cookie_value : cookies) {
      auto pos = cookie_value.find(COOKIE_PREFIX);
      if (pos != std::string::npos) {
        restful_info.jsession_id = cookie_value.substr(COOKIE_PREFIX.size());
        break;
      }
    }

    if (restful_info.jsession_id.empty()) {
      std::string msg = "Failed to parse login result, jsession id is empty";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  } catch (std::exception const &e) {
    std::string msg = std::string("catch exception:") + e.what();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::Logout(const VcnRestfulInfo &restful_info) {
  if (!IsRestfulInfoValid(restful_info)) {
    std::string msg = "Failed to restful logout, restful info is invalid";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_INVALID, msg};
  }

  std::string uri = "https://" + restful_info.ip + ":" + restful_info.port;
  const std::string COOKIE_PREFIX = "JSESSIONID=";

  httplib::Headers headers;
  headers.insert({"Content-Type", "application/json"});
  headers.insert({"Cache-Control", "no-cache"});
  headers.insert({"Cookie", COOKIE_PREFIX + restful_info.jsession_id});

  httplib::Response resp;
  auto ret = SendRequest(uri, RESTFUL_LOGOUT_URL, "", headers, REQ_GET, resp);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful logout, send request fail reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (resp.status != HTTP_STATUS_CODE_OK) {
    std::string msg =
        "Failed to restful logout, result code:" + std::to_string(resp.status);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  try {
    auto result_json = nlohmann::json::parse(resp.body);
    if (!result_json.contains("resultCode") ||
        !result_json["resultCode"].is_number_unsigned()) {
      std::string msg =
          "Failed to parse logout, result code isn't exist or invalid";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    auto result_code = result_json["resultCode"].get<uint64_t>();
    if (result_code != IVS_RESULT_SUCCESS) {
      std::string msg =
          "Failed to parse logout, result code:" + std::to_string(result_code);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::GetUrl(const VcnRestfulInfo &restful_info,
                                           std::string &rtsp_url) {
  if (!IsRestfulInfoValid(restful_info)) {
    std::string msg = "Failed to restful get url, restful info is invalid";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_INVALID, msg};
  }

  if (restful_info.camera_code.empty() ||
      restful_info.stream_type > RESTFUL_STREAM_TYPE_MAX ||
      restful_info.stream_type < 0) {
    std::string msg = "Failed to restful get url, parameters is invalid";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_INVALID, msg};
  }

  std::string uri = "https://" + restful_info.ip + ":" + restful_info.port;
  const std::string COOKIE_PREFIX = "JSESSIONID=";

  httplib::Headers headers;
  headers.insert({"Content-Type", "application/json"});
  headers.insert({"Cache-Control", "no-cache"});
  headers.insert({"Cookie", COOKIE_PREFIX + restful_info.jsession_id});

  auto body = nlohmann::json::parse(R"({
    "mediaURLParam":{
        "broadCastType":0,
        "packProtocolType":1,
        "protocolType":2,
        "serviceType":1,
        "streamType":1,
        "transMode":0,
        "clientType":1
    }
})");
  body["cameraCode"] = restful_info.camera_code;
  body["mediaURLParam"]["streamType"] = restful_info.stream_type;

  httplib::Response resp;
  auto ret = SendRequest(uri, RESTFUL_GET_RTSP_URL, body.dump(), headers,
                         REQ_POST, resp);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful get url, send request fail reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (resp.status != HTTP_STATUS_CODE_OK) {
    std::string msg =
        "Failed to restful get url, result code:" + std::to_string(resp.status);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  ret = ParseRestfulGetUrlResult(resp, rtsp_url);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful get url, parse result fail reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::ParseRestfulGetUrlResult(
    const httplib::Response &resp, std::string &rtsp_url) {
  try {
    auto result_json = nlohmann::json::parse(resp.body);
    if (!result_json.contains("resultCode") ||
        !result_json["resultCode"].is_number_unsigned()) {
      std::string msg =
          "Failed to parse get url result, result code isn't exist or invalid";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    auto result_code = result_json["resultCode"].get<uint64_t>();
    if (result_code != IVS_RESULT_SUCCESS) {
      std::string msg = "Failed to parse get url result, result code:" +
                        std::to_string(result_code);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    if (!result_json.contains("rtspURL") ||
        !result_json["rtspURL"].is_string()) {
      std::string msg =
          "Failed to get url result, rtspURL isn't exist or invalid";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    rtsp_url = result_json["rtspURL"].get<std::string>();

    if (rtsp_url.empty()) {
      std::string msg = "Failed to parse get url result, rtsp_url is empty";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapper::KeepAlive(
    const VcnRestfulInfo &restful_info) {
  if (!IsRestfulInfoValid(restful_info)) {
    std::string msg = "Failed to restful keep alive, restful info is invalid";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_INVALID, msg};
  }

  std::string uri = "https://" + restful_info.ip + ":" + restful_info.port;
  const std::string COOKIE_PREFIX = "JSESSIONID=";

  httplib::Headers headers;
  headers.insert({"Content-Type", "application/json"});
  headers.insert({"Cache-Control", "no-cache"});
  headers.insert({"Cookie", COOKIE_PREFIX + restful_info.jsession_id});

  httplib::Response resp;
  auto ret =
      SendRequest(uri, RESTFUL_KEEP_ALIVE_URL, "", headers, REQ_GET, resp);
  if (modelbox::STATUS_OK != ret) {
    std::string msg =
        "Failed to restful keep alive, send request fail reason:" +
        ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (resp.status != HTTP_STATUS_CODE_OK) {
    std::string msg = "Failed to restful keep alive, result code:" +
                      std::to_string(resp.status);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  try {
    auto result_json = nlohmann::json::parse(resp.body);
    if (!result_json.contains("resultCode") ||
        !result_json["resultCode"].is_number_unsigned()) {
      std::string msg =
          "Failed to restful keep alive, result code isn't exist or invalid";
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    auto result_code = result_json["resultCode"].get<uint64_t>();
    if (result_code != IVS_RESULT_SUCCESS) {
      std::string msg = "Failed to restful keep alive, result code:" +
                        std::to_string(result_code);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

bool VcnRestfulWrapper::IsRestfulInfoValid(const VcnRestfulInfo &restful_info,
                                           bool is_login) {
  if (restful_info.user_name.empty()) {
    MBLOG_ERROR << "user name is empty";
    return false;
  }

  if (restful_info.password.empty()) {
    MBLOG_ERROR << "user password is empty";
    return false;
  }

  if (restful_info.ip.empty()) {
    MBLOG_ERROR << "ip is empty";
    return false;
  }

  if (restful_info.port.empty()) {
    MBLOG_ERROR << "port is empty";
    return false;
  }

  if (is_login) {
    return true;
  }

  if (restful_info.jsession_id.empty()) {
    MBLOG_ERROR << "jsession id is empty";
    return false;
  }

  return true;
}

}  // namespace modelbox
