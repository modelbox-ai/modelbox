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

#ifndef MODELBOX_FLOWUNIT_VCN_INFO_H_
#define MODELBOX_FLOWUNIT_VCN_INFO_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>

#include <string>

namespace modelbox {

typedef struct tag_VcnInfo {
  std::string ip;
  std::string port;
  std::string user_name;
  std::string password;
  std::string camera_code;
  uint32_t stream_type;
} VcnInfo;

/**
 * @brief   Check whether the vcn info contains valid user name/password/ip
 * and port.
 * @param   info - in, vcn info.
 * @return  true for valid, vice versa.
 */
bool IsVcnInfoValid(const VcnInfo &info);

class VcnAccountBase {
 public:
  VcnAccountBase(const VcnInfo &info)
      : ip_(info.ip),
        port_(info.port),
        user_name_(info.user_name),
        password_(info.password) {
    streams_count_ = 0;
  };

  virtual ~VcnAccountBase(){};

  /**
   * @brief   get vcn user name
   * @return  user name
   */
  std::string GetUserName() const { return user_name_; };

  /**
   * @brief   get vcn user password
   * @return  user password
   */
  std::string GetPassword() const { return password_; };

  /**
   * @brief   get vcn ip
   * @return  vcn ip
   */
  std::string GetIp() const { return ip_; };

  /**
   * @brief   get vcn port
   * @return  vcn port
   */
  std::string GetPort() const { return port_; };

  /**
   * @brief   get vcn stream count
   * @return  stream count
   */
  uint32_t GetStreamsCount() const { return streams_count_; };

 public:
  void AddStream() { ++streams_count_; };
  void RemoveStream() {
    if (streams_count_ > 0) {
      --streams_count_;
    }
  };

  std::string ip_;
  std::string port_;
  std::string user_name_;
  std::string password_;
  uint32_t streams_count_;
};

modelbox::Status GetVcnInfo(modelbox::VcnInfo &vcn_info,
                            const std::string &config);

class VcnStreamBase {
 public:
  VcnStreamBase(const std::string &url, const std::string &camera_code)
      : url_(url), camera_code_(camera_code){};

  virtual ~VcnStreamBase(){};

  /**
   * @brief   get stream url
   * @return  stream url
   */
  std::string GetUrl() { return url_; };

 protected:
  std::string url_;
  std::string camera_code_;
};

void ReadConfVcnCommon(const std::shared_ptr<modelbox::Configuration> &opts,
                       int32_t &retry_enabled, int32_t &retry_interval,
                       int32_t &retry_max_times);

}  // namespace modelbox
#endif  // MODELBOX_FLOWUNIT_VCN_INFO_H_