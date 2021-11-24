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


#ifndef MODELBOX_FLOWUNIT_VCN_CLIENT_H_
#define MODELBOX_FLOWUNIT_VCN_CLIENT_H_

#include <modelbox/base/log.h>
#include <modelbox/base/status.h>

#include <mutex>
#include <vector>

#include "vcn_sdk_wrapper.h"

namespace modelbox {

typedef struct tag_VcnInfo {
  std::string ip = "";
  std::string port = "";
  std::string user_name = "";
  std::string password = "";
  std::string camera_code = "";
  uint32_t stream_type;
} VcnInfo;

class VcnAccount;
class VcnStream;

/**
 * @brief This is a singleton class, in charge of all about the VCN SDK.
 */
class VcnClient {
 public:
  /**
   * @brief   Get an VcnClient object.
   * @return  Pointer to an VcnClient object.
   *          Notes: return nullptr if it's failed to initialize VCN SDK.
   */
  static std::shared_ptr<VcnClient> GetInstance();

  /**
   * @brief   Add a vcn stream and get its url.
   * @param   info - in, Vcn info.
   * @param   stream - out, pointer to a VcnStream object. This object hold a
   * vcn url;
   * @return  Successful or not
   */
  modelbox::Status AddVcnStream(VcnInfo &info,
                              std::shared_ptr<VcnStream> &stream);

  /**
   * @brief   Remove the vcn stream from VcnClient, and logout the responding
   * account if necessary.
   * @param   stream - in, pointer to a VcnStream object to be remove;
   * @return  Successful or not
   */
  modelbox::Status RemoveVcnStream(VcnStream *stream);

  /**
   * @brief   Set a mock VCN SDK wrapper for the Unit Test.
   * @param   _sdk_wrapper - pointer to an object which is derived from class
   * 'VcnSdkWrapper';
   * @return  Successful or not
   */
  modelbox::Status SetSDKWrapper(std::shared_ptr<VcnSdkWrapper> _sdk_wrapper);

  virtual ~VcnClient();

 private:
  VcnClient() = default;

  /**
   * @brief   Initialize the VCN SDK.
   * @return  Successful or not
   */
  modelbox::Status Init();

  /**
   * @brief   Deinitialize the VCN SDK.
   * @return  Successful or not
   */
  modelbox::Status DeInit();

  /**
   * @brief   loglogin a vcn account
   * @param   account - in, pointer to a VcnAccount object, containing
   * information to login.
   * @param   session_id - out, An id assigned to every vcn account that
   * successfully login.
   * @return  Successful or not
   */
  modelbox::Status LoginVcnAccount(std::shared_ptr<const VcnAccount> account,
                                 int32_t &session_id);

  /**
   * @brief   logout a vcn account
   * @param   account - in, pointer to a VcnAccount object, containing
   * information to logout.
   * @return  Successful or not
   */
  modelbox::Status LogoutVcnAccount(std::shared_ptr<VcnAccount> account);

  /**
   * @brief   get the vcn stream url
   * @param   session_id - in, pointer to a VcnAccount object, containing
   * information to logout.
   * @param   camera_code - in, camera code from vcn config.
   * @param   stream_type - in, stream type from vcn config.
   * @param   url - out, stream url
   * @return  Successful or not
   */
  modelbox::Status GetUrl(int32_t session_id, const std::string &camera_code,
                        uint32_t stream_type, std::string &url);

  /**
   * @brief   Check whether the vcn info contains valid user name/password/ip
   * and port.
   * @param   info - in, vcn info.
   * @return  true for valid, vice versa.
   */
  bool IsVcnInfoValid(const VcnInfo &info);

  /**
   * @brief   check whether the error can be solved by a RETRY.
   * @param   error_code - in, vcn sdk error code.
   * @return  true or false, need to retry or not.
   */
  bool NeedToRetry(const IVS_INT32 error_code);

  static std::mutex vcn_client_lock_;
  std::mutex vcn_accounts_lock_;  // lock before any operations applied to the
                                  // vcn_accounts_
  std::vector<std::shared_ptr<VcnAccount>> vcn_accounts_;
  std::shared_ptr<VcnSdkWrapper> sdk_wrapper_;
};

class VcnAccount {
  friend class VcnClient;

 public:
  VcnAccount(const VcnInfo &info)
      : ip_(info.ip),
        port_(info.port),
        user_name_(info.user_name),
        password_(info.password) {
    streams_count_ = 0;
    session_id_ = -1;
  };

  virtual ~VcnAccount(){};
  bool GetLoginState() {
    return session_id_ >= SESSION_ID_MIN && session_id_ <= SESSION_ID_MAX;
  };

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
   * @brief   get vcn session id
   * @return  session id
   */
  int32_t GetSessionId() const { return session_id_; };

  /**
   * @brief   get vcn stream count
   * @return  stream count
   */
  uint32_t GetStreamsCount() const { return streams_count_; };

 private:
  void SetSessionId(int32_t session_id) { session_id_ = session_id; };
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
  int32_t session_id_;
  uint32_t streams_count_;
};

class VcnStream {
  friend class modelbox::VcnClient;

 public:
  VcnStream(const std::string &url, const std::string &camera_code,
            const int32_t session_id, std::shared_ptr<VcnAccount> account)
      : url_(url),
        camera_code_(camera_code),
        session_id_(session_id),
        account_(account){};

  virtual ~VcnStream(){};

  /**
   * @brief   get stream url
   * @return  stream url
   */
  std::string GetUrl() { return url_; };

 private:
  std::shared_ptr<VcnAccount> GetAccount() { return account_; };
  std::string url_;
  std::string camera_code_;
  int32_t session_id_{0};
  std::shared_ptr<VcnAccount> account_;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_CLIENT_H_