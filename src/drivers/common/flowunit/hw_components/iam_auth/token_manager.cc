#include "token_manager.h"

#include <cpprest/http_client.h>
#include <cpprest/json.h>

#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "iam_api.h"
#include "modelbox/base/log.h"
#include "signer.h"

using namespace web;                   // NOLINT
using namespace web::http;             // NOLINT
using namespace web::http::client;     // NOLINT
using namespace utility;               // NOLINT
using namespace concurrency::streams;  // NOLINT

#define INTERVAL_TIME (3600 * 1000)

namespace modelbox {
TokenManager::TokenManager()
    : request_credential_uri_("/v3.0/OS-CREDENTIAL/securitytokens"),
      request_token_uri_("/v3/auth/tokens") {}

TokenManager::~TokenManager() = default;

modelbox::Status TokenManager::Init() {
  if (!init_flag_) {
    timer_ = std::make_shared<modelbox::TimerTask>();
    timer_->Callback(&TokenManager::OnTimer, this);

    TimerGlobal::Schedule(timer_, 1000, INTERVAL_TIME, false);
    init_flag_ = true;
  }
  return modelbox::STATUS_OK;
}

void TokenManager::OnTimer() {
  if (++async_count > 1) {
    async_count--;
    return;
  }
  Defer { async_count--; };
  RequestAgencyToken();
}

void TokenManager::RequestAgencyToken() {
  std::unique_lock<std::mutex> lock_credential(credential_lock_);
  auto origin_user_credential_map = user_credential_map_;
  lock_credential.unlock();
  for (auto &credential_item : origin_user_credential_map) {
    auto status = RequestAgencyProjectCredential(credential_item.first, true);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "failed to get project credential, user name : "
                  << credential_item.first.user_domain_name;
    }
  }

  std::unique_lock<std::mutex> lock_token(token_lock_);
  auto origin_user_token_map = user_token_map_;
  lock_token.unlock();
  for (auto user_item : origin_user_token_map) {
    auto &project_map = user_item.second;
    for (auto &project_item : project_map) {
      auto status =
          RequestAgencyProjectToken(user_item.first, project_item.first, true);
      if (status != modelbox::STATUS_OK) {
        MBLOG_ERROR << "failed to get project credential, user name : "
                    << user_item.first.user_domain_name;
      }
    }
  }
}

modelbox::Status TokenManager::SetConsigneeInfo(const std::string &ak,
                                                const std::string &sk,
                                                const std::string &domain_id,
                                                const std::string &project_id) {
  consignee_info_.ak = ak;
  consignee_info_.sk = sk;
  consignee_info_.domain_id = domain_id;
  consignee_info_.project_id = project_id;
  return modelbox::STATUS_OK;
}
modelbox::Status TokenManager::SetHostAddress(const std::string &host) {
  request_host_ = host;
  IAMApi::SetRequestHost(host);
  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::RequestAgencyProjectToken(
    const AgencyInfo &agency_info, const ProjectInfo project_info, bool force) {
  if (!force) {
    if (modelbox::STATUS_EXIST ==
        FindUserAgencyToken(agency_info, project_info)) {
      return modelbox::STATUS_OK;
    }
  }

  IAMApi::SetRequestTokenUri(request_token_uri_);
  IAMApi::SetRequestCredentialUri(request_credential_uri_);
  UserAgencyToken token;
  if (IsExpire(agent_token_.expires_time_) && token_update_callback_) {
    token_update_callback_();
  }
  if (modelbox::STATUS_OK !=
      IAMApi::GetAgencyProjectTokenWithToken(agent_token_, agency_info,
                                             project_info, token)) {
    if (modelbox::STATUS_OK !=
        IAMApi::GetAgencyProjectTokenWithAK(consignee_info_, agency_info,
                                            project_info, token)) {
      return modelbox::STATUS_FAULT;
    }
  }
  if (modelbox::STATUS_OK !=
      SaveUserAgencyToken(agency_info, project_info, token)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::GetAgencyProjectToken(
    const AgencyInfo &agency_info, const ProjectInfo &project_info,
    UserAgencyToken &agency_token) {
  std::lock_guard<std::mutex> lock(token_lock_);
  auto item = user_token_map_.find(agency_info);
  if (item == user_token_map_.end()) {
    return modelbox::STATUS_FAULT;
  }
  auto token_item = item->second.find(project_info);
  if (token_item == item->second.end()) {
    return modelbox::STATUS_FAULT;
  }
  agency_token = token_item->second;
  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::RequestAgencyProjectCredential(
    const AgencyInfo &agency_info, bool force) {
  if (!force) {
    if (modelbox::STATUS_EXIST == FindUserAgencyCredential(agency_info)) {
      return modelbox::STATUS_OK;
    }
  }

  IAMApi::SetRequestCredentialUri(request_credential_uri_);
  ProjectInfo project_info;
  UserAgencyCredential credential;

  if (modelbox::STATUS_OK != IAMApi::GetAgencyProjectCredentialWithAK(
                                 consignee_info_, agency_info, credential)) {
    if (IsExpire(agent_token_.expires_time_) && token_update_callback_) {
      token_update_callback_();
      return modelbox::STATUS_FAULT;
    }
    if (modelbox::STATUS_OK != IAMApi::GetAgencyProjectCredentialWithToken(
                                   agent_token_, agency_info, credential)) {
      return modelbox::STATUS_FAULT;
    }
  }
  if (modelbox::STATUS_OK !=
      SaveUserAgencyCredential(agency_info, credential)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::GetAgencyProjectCredential(
    const AgencyInfo &agency_info, UserAgencyCredential &agency_credential) {
  std::lock_guard<std::mutex> lock(credential_lock_);
  auto iter = user_credential_map_.find(agency_info);
  if (iter == user_credential_map_.end()) {
    return modelbox::STATUS_FAULT;
  }
  agency_credential = iter->second;
  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::SaveUserAgencyCredential(
    const AgencyInfo &agency_info, UserAgencyCredential &credential) {
  std::lock_guard<std::mutex> lock(credential_lock_);
  user_credential_map_[agency_info] = credential;
  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::SaveUserAgencyToken(
    const AgencyInfo &agency_info, const ProjectInfo &project_info,
    UserAgencyToken &agency_token) {
  std::lock_guard<std::mutex> lock(token_lock_);
  user_token_map_[agency_info][project_info] = agency_token;
  return modelbox::STATUS_OK;
}

void TokenManager::DeleteUserAgencyCredential(const AgencyInfo &agency_info) {
  std::lock_guard<std::mutex> lock(credential_lock_);
  auto iter = user_credential_map_.find(agency_info);
  if (iter != user_credential_map_.end()) {
    user_credential_map_.erase(iter);
  }
}

void TokenManager::DeleteUserAgencyToken(const AgencyInfo &agency_info) {
  std::lock_guard<std::mutex> lock(token_lock_);
  auto iter = user_token_map_.find(agency_info);
  if (iter != user_token_map_.end()) {
    user_token_map_.erase(iter);
  }
}

modelbox::Status TokenManager::FindUserAgencyCredential(
    const AgencyInfo &agency_info) {
  std::lock_guard<std::mutex> lock(credential_lock_);
  auto iter = user_credential_map_.find(agency_info);
  if (iter == user_credential_map_.end()) {
    return modelbox::STATUS_NOTFOUND;
  }
  return modelbox::STATUS_EXIST;
}

modelbox::Status TokenManager::FindUserAgencyToken(
    const AgencyInfo &agency_info, const ProjectInfo &project_info) {
  std::lock_guard<std::mutex> lock(token_lock_);
  auto iter = user_token_map_.find(agency_info);
  if (iter == user_token_map_.end()) {
    return modelbox::STATUS_NOTFOUND;
  }

  auto token_iter = iter->second.find(project_info);
  if (token_iter == iter->second.end()) {
    return modelbox::STATUS_NOTFOUND;
  }
  return modelbox::STATUS_EXIST;
}

void TokenManager::SetPersistUserAgencyCredential(
    const UserAgencyCredential &credential) {
  std::lock_guard<std::mutex> lock(persist_credential_lock_);
  persist_credential_[credential.user_id] = credential;
}

void TokenManager::RemovePersistUserAgencyCredential(
    const std::string &userId) {
  std::lock_guard<std::mutex> lock(persist_credential_lock_);
  if (persist_credential_.find(userId) == persist_credential_.end()) {
    MBLOG_WARN << "RemovePersistUserAgencyCredential: " << userId
               << " presist credential info isn't exist";
    return;
  }
  persist_credential_.erase(userId);
}

modelbox::Status TokenManager::GetPersistUserAgencyCredential(
    UserAgencyCredential &credential, const std::string &userId) {
  std::lock_guard<std::mutex> lock(persist_credential_lock_);
  if (persist_credential_.find(userId) == persist_credential_.end()) {
    MBLOG_WARN << userId << " presist credential info isn't exist";
    return modelbox::STATUS_FAULT;
  }
  UserAgencyCredential userCredential = persist_credential_[userId];
  if (userCredential.user_ak.empty()) {
    MBLOG_ERROR
        << "presist credential is empty, please set presist credential first.";
    return modelbox::STATUS_FAULT;
  }
  credential.user_id = userCredential.user_id;
  credential.user_ak = userCredential.user_ak;
  credential.user_sk = userCredential.user_sk;
  credential.user_secure_token = userCredential.user_secure_token;
  return modelbox::STATUS_OK;
}

modelbox::Status TokenManager::SetAgentToken(const AgentToken &token) {
  agent_token_ = token;
  return modelbox::STATUS_SUCCESS;
}

void TokenManager::SetUpdateAgentTokenRequestCallback(
    std::function<void()> &callback) {
  token_update_callback_ = callback;
}

bool TokenManager::IsExpire(const std::string &expire) const {
  auto expire_tp = datetime::from_string(expire, datetime::ISO_8601);
  if (!expire_tp.is_initialized()) {
    auto msg = std::string("expire not ISO_8601 format.") +
               std::string(" expire:") + expire;
    return true;
  }

  auto now_tp = datetime::utc_now() + datetime::from_minutes(30);
  if (!now_tp.is_initialized()) {
    auto msg = std::string("expire not ISO_8601 format.") +
               std::string(" expire:") + expire;
    return true;
  }

  if (expire_tp.to_interval() < now_tp.to_interval()) {
    auto msg = std::string("expire timeout.") + std::string(" expire:") +
               expire + std::string(" expire_tp:") +
               now_tp.to_string(datetime::ISO_8601);
    MBLOG_WARN << msg;
    return true;
  }

  return false;
}

}  // namespace modelbox
