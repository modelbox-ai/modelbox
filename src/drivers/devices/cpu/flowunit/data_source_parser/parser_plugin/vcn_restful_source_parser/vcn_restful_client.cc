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

#include "vcn_restful_client.h"

#include <modelbox/base/log.h>

namespace modelbox {
std::mutex VcnRestfulClient::vcn_client_lock_;

std::shared_ptr<VcnRestfulClient> VcnRestfulClient::GetInstance(
    int32_t keep_alive_interval) {
  static std::shared_ptr<VcnRestfulClient> vcn_client(
      new VcnRestfulClient(keep_alive_interval));
  std::lock_guard<std::mutex> lock(vcn_client_lock_);
  static bool is_initialized = false;
  if (is_initialized) {
    return vcn_client;
  }

  auto ret = vcn_client->Init();
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "failed to init vcn restful client reason: "
                << ret.Errormsg();
    return nullptr;
  }

  is_initialized = true;
  return vcn_client;
}

VcnRestfulClient::~VcnRestfulClient() {
  is_keep_alive_ = false;
  if (keep_alive_thread_ != nullptr && keep_alive_thread_->joinable()) {
    keep_alive_thread_->join();
    keep_alive_thread_ = nullptr;
  }
}

modelbox::Status VcnRestfulClient::Init() {
  if (keep_alive_thread_ != nullptr) {
    MBLOG_INFO << "restful client repeat init";
    return modelbox::STATUS_OK;
  }

  restful_wrapper_ = std::make_shared<VcnRestfulWrapper>();
  if (restful_wrapper_ == nullptr) {
    return {modelbox::STATUS_INVALID, "failed to create vcn wrapper"};
  }

  keep_alive_thread_ = std::make_shared<std::thread>(
      &VcnRestfulClient::KeepAliveThreadProc, this);
  if (keep_alive_thread_ == nullptr) {
    return {modelbox::STATUS_INVALID,
            "failed to create vcn restful keep alive thread"};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::AddVcnStream(
    VcnInfo &info, std::shared_ptr<VcnStreamRestful> &stream) {
  std::string errmsg = "";
  const std::string errmsg_prefix = "Failed to add vcn stream: ";
  if (!IsVcnInfoValid(info)) {
    errmsg = errmsg_prefix + "invalid info.";
    return {modelbox::STATUS_INVALID, errmsg};
  }

  // check accounts record
  std::shared_ptr<VcnAccountRestful> account = nullptr;
  std::lock_guard<std::mutex> lock(vcn_account_lock_);
  auto ret = GetVcnAccount(info, account);
  if (ret != modelbox::STATUS_OK) {
    errmsg = errmsg_prefix + ret.Errormsg();
    return {modelbox::STATUS_INVALID, errmsg};
  }

  std::string url;
  ret = GetVcnUrl(info, account, url);
  if (ret != modelbox::STATUS_OK) {
    errmsg = errmsg_prefix + ret.Errormsg();
    return {modelbox::STATUS_INVALID, errmsg};
  }

  MBLOG_INFO << "User name: " << info.user_name
             << ", successfully get url: " << url;

  stream = std::shared_ptr<VcnStreamRestful>(
      new VcnStreamRestful(url, info.camera_code, account),
      [this](VcnStreamRestful *stream) {
        this->RemoveVcnStream(stream);
        delete stream;
      });

  account->AddStream();

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::RemoveVcnStream(VcnStreamRestful *stream) {
  std::string errmsg = "";
  const std::string errmsg_prefix = "Failed to remove vcn restful stream: ";
  if (nullptr == stream) {
    MBLOG_ERROR << errmsg_prefix + "stream ptr is nullptr.";
    return modelbox::STATUS_INVALID;
  }

  std::lock_guard<std::mutex> lock(vcn_account_lock_);
  auto account = stream->GetAccount();
  account->RemoveStream();

  if (account->GetStreamsCount() > 0) {
    return modelbox::STATUS_OK;
  }

  auto ret = RemoveVcnAccount(account);
  if (ret != modelbox::STATUS_OK) {
    std::string errmsg = errmsg_prefix + ret.Errormsg();
    return {modelbox::STATUS_INVALID, errmsg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::GetVcnAccount(
    const VcnInfo &info, std::shared_ptr<VcnAccountRestful> &account) {
  auto iter =
      std::find_if(vcn_accounts_.begin(), vcn_accounts_.end(),
                   [&info](std::shared_ptr<const VcnAccountRestful> account_) {
                     return (account_->GetUserName() == info.user_name &&
                             account_->GetIp() == info.ip &&
                             account_->GetPort() == info.port &&
                             account_->GetPassword() == info.password);
                   });
  if (iter != vcn_accounts_.end()) {
    account = *iter;
    return modelbox::STATUS_OK;
  }

  auto ret = CreateVcnAccount(info, account);
  if (ret != modelbox::STATUS_OK) {
    std::string errmsg =
        std::string("failed to get vcn restful account reason: ") +
        ret.Errormsg();
    return {modelbox::STATUS_INVALID, errmsg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::CreateVcnAccount(
    const VcnInfo &info, std::shared_ptr<VcnAccountRestful> &account) {
  account = std::make_shared<VcnAccountRestful>(info);
  if (account == nullptr) {
    return {modelbox::STATUS_INVALID, "failed to create vcn account"};
  }

  VcnRestfulInfo restful_info(info);
  auto ret = restful_wrapper_->Login(restful_info);
  if (ret != modelbox::STATUS_OK) {
    std::string errreason =
        std::string("failed to login vcn restful client reason: ") +
        ret.Errormsg();
    return {modelbox::STATUS_INVALID, errreason};
  }

  account->SetSessionId(restful_info.jsession_id);

  vcn_accounts_.emplace_back(account);

  MBLOG_INFO << "Successfully login vcn restful, User name: "
             << restful_info.user_name;

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::GetVcnUrl(
    const VcnInfo &info, const std::shared_ptr<VcnAccountRestful> &account,
    std::string &url) {
  VcnRestfulInfo restful_info(info);
  restful_info.jsession_id = account->GetSessionId();
  auto ret = restful_wrapper_->GetUrl(restful_info, url);
  if (ret != modelbox::STATUS_OK) {
    std::string errreason =
        std::string("failed to get vcn restful url reason: ") + ret.Errormsg();
    return {modelbox::STATUS_INVALID, errreason};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulClient::RemoveVcnAccount(
    const std::shared_ptr<VcnAccountRestful> &account) {
  VcnRestfulInfo restful_info;
  GetRestfulInfoFromAccount(account, restful_info);

  auto ret = restful_wrapper_->Logout(restful_info);
  if (ret != modelbox::STATUS_OK) {
    std::string errreason =
        std::string("failed to logout vcn restful client reason: ") +
        ret.Errormsg();
    return {modelbox::STATUS_INVALID, errreason};
  }

  auto iter = std::find_if(
      vcn_accounts_.begin(), vcn_accounts_.end(),
      [&account](std::shared_ptr<const VcnAccountRestful> account_) {
        return (account_->GetUserName() == account->GetUserName() &&
                account_->GetIp() == account->GetIp() &&
                account_->GetPort() == account->GetPort() &&
                account_->GetPassword() == account->GetPassword());
      });
  if (iter == vcn_accounts_.end()) {
    std::string errreason =
        "failed to logout vcn restful client reason: The account to be deleted "
        "is NOT FOUND";
    return {modelbox::STATUS_NOTFOUND, errreason};
  }

  vcn_accounts_.erase(iter);

  MBLOG_INFO << "remove vcn restful success ip:" << restful_info.ip
             << " user name:" << restful_info.user_name;

  return modelbox::STATUS_OK;
}

void VcnRestfulClient::KeepAliveThreadProc() {
  while (is_keep_alive_) {
    KeepAliveProcess();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  MBLOG_INFO << "keep alive thread exit";
}

void VcnRestfulClient::KeepAliveProcess() {
  std::lock_guard<std::mutex> lock(vcn_account_lock_);
  if (restful_wrapper_ == nullptr) {
    MBLOG_ERROR << "wrapper is nullptr";
    return;
  }

  for (auto &account : vcn_accounts_) {
    time_t now = time(NULL);
    if (now - account->GetKeepAliveTime() < keep_alive_interval_) {
      continue;
    }

    VcnRestfulInfo info;
    GetRestfulInfoFromAccount(account, info);
    auto ret = restful_wrapper_->KeepAlive(info);
    if (modelbox::STATUS_OK != ret) {
      std::string msg = "Failed to vcn restful keep alive " + ret.Errormsg();
      MBLOG_ERROR << msg;
      continue;
    }

    account->SetKeepAliveTime(now);
  }
}

void VcnRestfulClient::GetRestfulInfoFromAccount(
    const std::shared_ptr<const VcnAccountRestful> &account,
    VcnRestfulInfo &info) {
  info.ip = account->GetIp();
  info.port = account->GetPort();
  info.user_name = account->GetUserName();
  info.password = account->GetPassword();
  info.jsession_id = account->GetSessionId();
}

modelbox::Status VcnRestfulClient::SetRestfulWrapper(
    std::shared_ptr<VcnRestfulWrapper> _restful_wrapper) {
  if (nullptr == _restful_wrapper) {
    return {modelbox::STATUS_INVALID, "wrapper pointer is nullptr."};
  }

  restful_wrapper_ = _restful_wrapper;
  return modelbox::STATUS_OK;
}

}  // namespace modelbox