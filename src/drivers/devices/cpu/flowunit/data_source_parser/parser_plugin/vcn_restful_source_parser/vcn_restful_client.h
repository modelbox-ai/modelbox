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

#ifndef MODELBOX_FLOWUNIT_VCN_RESTFUL_CLIENT_H_
#define MODELBOX_FLOWUNIT_VCN_RESTFUL_CLIENT_H_

#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <mutex>
#include <utility>
#include <vector>

#include "vcn_info.h"
#include "vcn_restful_wrapper.h"

#define KEEP_ALIVE_INTERVAL_DEFAULT_SEC 600

namespace modelbox {

class VcnStreamRestful;
class VcnAccountRestful;

/**
 * @brief This is a singleton class, in charge of all about the VCN restful.
 */
class VcnRestfulClient {
 public:
  /**
   * @brief   Get an VcnRestfulClient object.
   * @return  Pointer to an VcnRestfulClient object.
   *          Notes: return nullptr if it's failed to initialize VCN restful.
   */
  static std::shared_ptr<VcnRestfulClient> GetInstance(
      int32_t keep_alive_interval);

  virtual ~VcnRestfulClient() = default;

  modelbox::Status Init();

  /**
   * @brief   Add a vcn stream and get its url.
   * @param   info - in, Vcn info.
   * @param   stream - out, pointer to a VcnStreamRestful object. This object
   * hold a vcn url;
   * @return  Successful or not
   */
  modelbox::Status AddVcnStream(VcnInfo &info,
                                std::shared_ptr<VcnStreamRestful> &stream);

  /**
   * @brief   Remove the vcn stream from VcnRestfulClient, and logout the
   * responding account if necessary.
   * @param   stream - in, pointer to a VcnStreamRestful object to be remove;
   * @return  Successful or not
   */
  modelbox::Status RemoveVcnStream(VcnStreamRestful *stream);

  /**
   * @brief   Set a mock VCN restful wrapper for the Unit Test.
   * @param   _restful_wrapper - pointer to an object which is derived from
   * class 'VcnRestfulWrapper';
   * @return  Successful or not
   */
  modelbox::Status SetRestfulWrapper(
      const std::shared_ptr<VcnRestfulWrapper> &_restful_wrapper);

  void SetKeepAliveInterval(int32_t keep_alive_interval) {
    keep_alive_interval_ = keep_alive_interval;
  }

 private:
  VcnRestfulClient(int32_t keep_alive_interval)
      : restful_wrapper_(nullptr),
        keep_alive_interval_(keep_alive_interval),
        keep_alive_timer_task_(nullptr) {}

  modelbox::Status GetVcnAccount(const VcnInfo &info,
                                 std::shared_ptr<VcnAccountRestful> &account);
  modelbox::Status CreateVcnAccount(
      const VcnInfo &info, std::shared_ptr<VcnAccountRestful> &account);

  modelbox::Status GetVcnUrl(const VcnInfo &info,
                             const std::shared_ptr<VcnAccountRestful> &account,
                             std::string &url);

  modelbox::Status RemoveVcnAccount(
      const std::shared_ptr<VcnAccountRestful> &account);

  modelbox::Status KeepAliveProcess();
  void GetRestfulInfoFromAccount(
      const std::shared_ptr<const VcnAccountRestful> &account,
      VcnRestfulInfo &info);
  void PullKeepAliveThread();

  static std::mutex vcn_client_lock_;
  std::mutex vcn_account_lock_;
  std::vector<std::shared_ptr<VcnAccountRestful>> vcn_accounts_;
  std::shared_ptr<VcnRestfulWrapper> restful_wrapper_;
  int32_t keep_alive_interval_;
  std::shared_ptr<modelbox::TimerTask> keep_alive_timer_task_;
  modelbox::Timer timer_;
};

class VcnAccountRestful : public VcnAccountBase {
 public:
  VcnAccountRestful(const VcnInfo &info)
      : VcnAccountBase(info), keep_alive_time_(0) {}

  std::string GetSessionId() const { return session_id_; }
  void SetSessionId(const std::string &session_id) { session_id_ = session_id; }

  /**
   * @brief   get keep alive use restful
   * @return  keep alive
   */
  time_t GetKeepAliveTime() const { return keep_alive_time_; }
  void SetKeepAliveTime(time_t keep_alive_time) {
    keep_alive_time_ = keep_alive_time;
  }

 private:
  std::string session_id_;
  time_t keep_alive_time_;
};

class VcnStreamRestful : public VcnStreamBase {
 public:
  VcnStreamRestful(std::string url, std::string camera_code,
                   std::shared_ptr<VcnAccountRestful> account)
      : VcnStreamBase(std::move(url), std::move(camera_code)),
        account_(std::move(account)) {}

  std::shared_ptr<VcnAccountRestful> GetAccount() { return account_; }

 private:
  std::shared_ptr<VcnAccountRestful> account_;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_RESTFUL_CLIENT_H_