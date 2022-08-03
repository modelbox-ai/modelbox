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

#include "vcn_info.h"
#include "vcn_sdk_wrapper.h"

namespace modelbox {

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

class VcnAccount : public VcnAccountBase {
  friend class VcnClient;

 public:
  VcnAccount(const VcnInfo &info) : VcnAccountBase(info) { session_id_ = -1; };

  virtual ~VcnAccount(){};
  bool GetLoginState() {
    return session_id_ >= SESSION_ID_MIN && session_id_ <= SESSION_ID_MAX;
  };

  /**
   * @brief   get vcn session id
   * @return  session id
   */
  int32_t GetSessionId() const { return session_id_; };

 private:
  void SetSessionId(int32_t session_id) { session_id_ = session_id; };

  int32_t session_id_;
};

class VcnStream : public VcnStreamBase {
  friend class modelbox::VcnClient;

 public:
  VcnStream(std::string url, std::string camera_code,
            int32_t session_id, std::shared_ptr<VcnAccount> account)
      : VcnStreamBase(url, camera_code),
        account_(account),
        session_id_(session_id){};

  virtual ~VcnStream(){};

 private:
  std::shared_ptr<VcnAccount> GetAccount() { return account_; };
  std::shared_ptr<VcnAccount> account_;
  int32_t session_id_{0};
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_CLIENT_H_