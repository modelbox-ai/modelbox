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

#ifndef GET_TOKEN_H_
#define GET_TOKEN_H_

#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <map>

#include <modelbox/token_header.h>

namespace modelbox {
class TokenManager {
 public:
  TokenManager();

  virtual ~TokenManager();

  /**
   * @brief   Initializing a Scheduled Task
   * @return  Successful or not
   */
  modelbox::Status Init();

  /**
   * @brief   Save Consiginee information and AK/SK
   * @param   ak - in, access_key
   * @param   sk - in, secret_key
   * @param   domian_id - in, consignee's domain id
   * @param   project_id - in, current project id
   * @return  Successful or not
   */
  modelbox::Status SetConsigneeInfo(const std::string &ak,
                                    const std::string &sk,
                                    const std::string &domain_id,
                                    const std::string &project_id);

  /**
   * @brief   Save IAM host address
   * @param   host - in, iam host address
   * @return  Successful or not
   */
  modelbox::Status SetHostAddress(const std::string &host);

  /**
   * @brief   request agency credential
   * @param   agency_info - in, agency info
   * @param   force - in,
   *                  true: request credential from host,
   *                  false: check credeential is exist in cache before request
   * @return  Successful or not
   */
  modelbox::Status RequestAgencyProjectCredential(const AgencyInfo &agency_info,
                                                  bool force);

  /**
   * @brief   get agency credential
   * @param   agency_info - in, agency information
   * @param   agency_credential - out, credential inforamtion
   * @return  Successful or not
   */
  modelbox::Status GetAgencyProjectCredential(
      const AgencyInfo &agency_info, UserAgencyCredential &agency_credential);

  /**
   * @brief   request agency token
   * @param   agency_info - in, agency info
   * @param   force - in,
   *                  true: request token from host,
   *                  false: check token is exist in cache before request
   * @return  Successful or not
   */
  modelbox::Status RequestAgencyProjectToken(const AgencyInfo &agency_info,
                                             const ProjectInfo &project_info,
                                             bool force);

  /**
   * @brief   get agency token
   * @param   agency_info - in, agency information
   * @param   agency_token - out, token inforamtion
   * @return  Successful or not
   */
  modelbox::Status GetAgencyProjectToken(const AgencyInfo &agency_info,
                                         const ProjectInfo &project_info,
                                         UserAgencyToken &agency_token);

  /**
   * @brief   delete agency credential
   * in cache
   * @param   agency_info - in, agency
   * information
   * @return  Successful or not
   */
  void DeleteUserAgencyCredential(const AgencyInfo &agency_info);

  /**
   * @brief   delete agency token in
   * cache
   * @param   agency_info - in, agency
   * information
   * @return  Successful or not
   */
  void DeleteUserAgencyToken(const AgencyInfo &agency_info);

  /**
   * @brief   save agency
   * credential in cache
   * @param   credential - in,
   * credential information
   */
  void SetPersistUserAgencyCredential(const UserAgencyCredential &credential);

  /**
   * @brief   remove agency
   * remove credential in cache
   * @param   userId - in,
   * remove credential information
   */
  void RemovePersistUserAgencyCredential(const std::string &userId);

  /**
   * @brief   get agency credential in
   * cache
   * @param   credential - out,
   * credential information
   */
  modelbox::Status GetPersistUserAgencyCredential(
      UserAgencyCredential &credential, const std::string &userId = "");

  modelbox::Status SetAgentToken(const AgentToken &token);
  /*

  */
  void SetUpdateAgentTokenRequestCallback(std::function<void()> &callback);

 private:
  void OnTimer();

  modelbox::Status SaveUserAgencyToken(const AgencyInfo &agency_info,
                                       const ProjectInfo &project_info,
                                       UserAgencyToken &agency_token);

  modelbox::Status SaveUserAgencyCredential(const AgencyInfo &agency_info,
                                            UserAgencyCredential &credential);

  modelbox::Status FindUserAgencyCredential(const AgencyInfo &agency_info);

  modelbox::Status FindUserAgencyToken(const AgencyInfo &agency_info,
                                       const ProjectInfo &project_info);

  void RequestAgencyToken();

  bool IsExpire(const std::string &expire) const;

  bool init_flag_{false};
  std::string request_credential_uri_;
  std::string request_token_uri_;
  std::string request_host_;
  std::string request_method_;
  ConsigneeInfo consignee_info_;
  std::map<AgencyInfo, UserAgencyCredential> user_credential_map_;
  std::map<AgencyInfo, std::map<ProjectInfo, UserAgencyToken> > user_token_map_;
  std::mutex credential_lock_;
  std::mutex token_lock_;
  std::mutex persist_credential_lock_;
  std::map<std::string, UserAgencyCredential> persist_credential_;
  AgentToken agent_token_;
  std::function<void(void)> token_update_callback_;
  std::atomic<int32_t> async_count{0};
  std::shared_ptr<modelbox::TimerTask> timer_;
};
}  // namespace modelbox

#endif
