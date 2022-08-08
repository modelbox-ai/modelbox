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

#include <modelbox/base/log.h>
#include <modelbox/iam_auth.h>

#include "token_manager.h"
namespace modelbox {

static std::shared_ptr<TokenManager> token_manager_ =
    std::make_shared<TokenManager>();

std::shared_ptr<IAMAuth> IAMAuth::GetInstance() {
  static std::shared_ptr<IAMAuth> instance(new IAMAuth());
  return instance;
}

modelbox::Status IAMAuth::SetConsigneeInfo(const std::string &service_ak,
                                           const std::string &service_sk,
                                           const std::string &domain_id,
                                           const std::string &project_id) {
  return token_manager_->SetConsigneeInfo(service_ak, service_sk, domain_id,
                                          project_id);
}

modelbox::Status IAMAuth::GetUserAgencyProjectCredential(
    UserAgencyCredential &agency_user_credential, const AgencyInfo &agency_info,
    const std::string &user_id) {
  if (agency_info.user_domain_name.empty()) {
    if (modelbox::STATUS_OK != token_manager_->GetPersistUserAgencyCredential(
                                   agency_user_credential, user_id)) {
      MBLOG_ERROR << "failed to get user credential info, user_id: " << user_id;
      return modelbox::STATUS_FAULT;
    }
    return modelbox::STATUS_OK;
  }

  modelbox::Status code =
      token_manager_->RequestAgencyProjectCredential(agency_info, false);
  if (code != modelbox::STATUS_OK) {
    MBLOG_ERROR << "failed request agency project credential";
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK != token_manager_->GetAgencyProjectCredential(
                                 agency_info, agency_user_credential)) {
    MBLOG_ERROR << "failed get agency project credential";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status IAMAuth::GetUserAgencyProjectToken(
    UserAgencyToken &agency_user_token, const AgencyInfo &agency_info,
    const ProjectInfo &project_info) {
  modelbox::Status code = token_manager_->RequestAgencyProjectToken(
      agency_info, project_info, false);
  if (code != modelbox::STATUS_OK) {
    MBLOG_ERROR << "failed request agency project token";
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK !=
      token_manager_->GetAgencyProjectToken(agency_info, project_info,
                                            agency_user_token)) {
    MBLOG_ERROR << "failed get agency project token";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

void IAMAuth::ExpireUserAgencyProjectCredential(const AgencyInfo &agency_info) {
  token_manager_->DeleteUserAgencyCredential(agency_info);
}

void IAMAuth::ExpireUserAgencyProjectToken(const AgencyInfo &agency_info) {
  token_manager_->DeleteUserAgencyToken(agency_info);
}

void IAMAuth::SetIAMHostAddress(const std::string &host) {
  token_manager_->SetHostAddress(host);
}

modelbox::Status IAMAuth::Init() { return token_manager_->Init(); }

void IAMAuth::SetUserAgencyCredential(const UserAgencyCredential &credential) {
  token_manager_->SetPersistUserAgencyCredential(credential);
}

void IAMAuth::RemoveUserAgencyCredential(const std::string &userId) {
  token_manager_->RemovePersistUserAgencyCredential(userId);
}

void IAMAuth::SetAgentToken(const AgentToken &token) {
  token_manager_->SetAgentToken(token);
}

void IAMAuth::SetUpdateAgentTokenCallBack(std::function<void()> &callback) {
  token_manager_->SetUpdateAgentTokenRequestCallback(callback);
}

}  // namespace modelbox