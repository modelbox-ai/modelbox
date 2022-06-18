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

#include "vcn_restful_wrapper_mock_test.h"

#include <modelbox/base/log.h>

const std::string J_SESSION_ID_SHOULD_BE =
    "C3AECD84E65268D2731DA3146FDEF9B0C4B14B5C2A37EDA381D47927E993B";

namespace modelbox {
modelbox::Status VcnRestfulWrapperMock::Login(VcnRestfulInfo &restful_info) {
  std::string user_name_should_be = "user";
  std::string pwd_should_be = "password";
  std::string ip_should_be = "192.168.1.1";
  std::string port_should_be = "666";
  std::string camera_code_should_be =
      "01234567890123456789#01234567890123456789012345678901";
  uint32_t ip_type_should_be = 1;

  if (user_name_should_be != restful_info.user_name) {
    std::string msg = "Failed to login, user name is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (pwd_should_be != restful_info.password) {
    std::string msg = "Failed to login, pwd is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (ip_should_be != restful_info.ip) {
    std::string msg = "Failed to login, ip is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (port_should_be != restful_info.port) {
    std::string msg = "Failed to login, port is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (camera_code_should_be != restful_info.camera_code) {
    std::string msg = "Failed to login, camera code is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  if (ip_type_should_be != restful_info.stream_type) {
    std::string msg = "Failed to login, stream type is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  restful_info.jsession_id = J_SESSION_ID_SHOULD_BE;

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapperMock::Logout(
    const VcnRestfulInfo &restful_info) {
  if (J_SESSION_ID_SHOULD_BE != restful_info.jsession_id) {
    std::string msg = "Failed to logout, session id is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapperMock::GetUrl(
    const VcnRestfulInfo &restful_info, std::string &rtsp_url) {
  if (J_SESSION_ID_SHOULD_BE != restful_info.jsession_id) {
    std::string msg = "Failed to get url, session id is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  rtsp_url = "https://www.Hello_World.com";

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulWrapperMock::KeepAlive(
    const VcnRestfulInfo &restful_info) {
  if (J_SESSION_ID_SHOULD_BE != restful_info.jsession_id) {
    std::string msg = "Failed to keep alive, session id is not correct.";
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return modelbox::STATUS_OK;
}
}  // namespace modelbox