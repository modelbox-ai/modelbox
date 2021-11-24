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


#include "vcn_sdk_wrapper_mock_test.h"
#include <modelbox/base/log.h>

#define SESSION_ID_FOR_TEST 127

namespace modelbox {

IVS_INT32 VcnSdkWrapperMock::VcnSdkInit() { return IVS_SUCCEED; }

IVS_INT32 VcnSdkWrapperMock::VcnSdkLogin(IVS_LOGIN_INFO *login_req_info,
                                         int32_t *session_id) {
  if (nullptr == login_req_info || nullptr == session_id) {
    return IVS_PARA_INVALID;
  }
  std::string user_name = login_req_info->cUserName;
  std::string ip = login_req_info->stIP.cIP;
  int port = login_req_info->uiPort;
  int ip_type = login_req_info->stIP.uiIPType;

  std::string user_name_should_be = "user";
  char pwd_should_be[] = "password";
  std::string ip_should_be = "192.168.1.1";
  int port_should_be = 666;
  int ip_type_should_be = 0;

  do {
    if (user_name != user_name_should_be) {
      MBLOG_ERROR << "Failed to login, user name is not correct.";
      break;
    }
    if (strcmp(login_req_info->pPWD, pwd_should_be) != 0) {
      MBLOG_ERROR << "Failed to login, pwd is not correct.";
      break;
    }
    if (ip != ip_should_be) {
      MBLOG_ERROR << "Failed to login, ip is not correct.";
      break;
    }
    if (port != port_should_be) {
      MBLOG_ERROR << "Failed to login, port is not correct.";
      break;
    }
    if (ip_type != ip_type_should_be) {
      MBLOG_ERROR << "Failed to login, ip type is not correct.";
      break;
    }
    *session_id = SESSION_ID_FOR_TEST;
    return IVS_SUCCEED;
  } while (false);
  
  return IVS_PARA_INVALID;
}

IVS_INT32 VcnSdkWrapperMock::VcnSdkLogout(IVS_INT32 session_id) {
  if (session_id < SESSION_ID_MIN || session_id > SESSION_ID_MAX) {
    MBLOG_ERROR << "Invalid session id: " << session_id;
    return IVS_PARA_INVALID;
  }

  return IVS_SUCCEED;
}

IVS_INT32 VcnSdkWrapperMock::VcnSdkGetUrl(
    IVS_INT32 session_id, const IVS_CHAR *camera_code,
    const IVS_URL_MEDIA_PARAM *url_media_param, std::string &rtsp_url) {
  if (nullptr == camera_code || nullptr == url_media_param) {
    MBLOG_ERROR
        << "Invalid parameters: camera_code or url_media_param is nullptr";
    rtsp_url = "";
    return IVS_PARA_INVALID;
  }

  std::string camera_code_str = camera_code;
  std::string camera_code_should_be =
      "01234567890123456789#01234567890123456789012345678901";
  if (url_media_param->StreamType != STREAM_TYPE_MAIN ||
      camera_code_str != camera_code_should_be) {
    MBLOG_ERROR << "Invalid parameters: StreamType: "
                << url_media_param->StreamType
                << ", camera code: " << camera_code_str;
    MBLOG_ERROR << "Parameters should be: StreamType: "
                << STREAM_TYPE_MAIN
                << ", camera code: " << camera_code_should_be;
    rtsp_url = "";
    return IVS_PARA_INVALID;
  }

  rtsp_url = "https://www.Hello_World.com";
  return IVS_SUCCEED;
}

IVS_INT32 VcnSdkWrapperMock::VcnSdkCleanup() { return IVS_SUCCEED; }

}  // namespace modelbox