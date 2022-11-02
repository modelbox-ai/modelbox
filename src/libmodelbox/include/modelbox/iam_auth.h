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

#ifndef MODELBOX_FLOWUNIT_IAM_AUTH_H_
#define MODELBOX_FLOWUNIT_IAM_AUTH_H_

#include <modelbox/base/status.h>
#include <modelbox/token_header.h>

#include <functional>
#include <memory>

namespace modelbox {
/**
 * @brief
 * User: means algorithm user
 * Service: means algorithm developer or platform developer
 */
class IAMAuth {
 public:
  IAMAuth();
  virtual ~IAMAuth();

  /*
   * @brief get iamauth instance
   * @return iamauth instance
   */
  static std::shared_ptr<IAMAuth> GetInstance();

  /**
   * @brief initilize timer
   * @return successful or fault
   */
  modelbox::Status Init();

  /**
   * @brief set iam host address
   * @param host - in, iam host address
   */
  void SetIAMHostAddress(const std::string &host);

  /**
   * @brief Set Consignee info: ak, sk, domain_id and project_id
   * @param service_ak - in, access key for vas
   * @param service_sk - in, secret key for vas
   * @param domain_id - in, domain id
   * @param project_id - in, project id
   * @return successful or fault
   */
  modelbox::Status SetConsigneeInfo(const std::string &service_ak,
                                    const std::string &service_sk,
                                    const std::string &domain_id,
                                    const std::string &project_id);
  /**
   * @brief If service cert has been set, then you can get
   * user agency Project credential to access user cloud resource
   * @param agency_user_credential - out, agency credential
   * @param agency_info - in, agency info
   * @param user_id - in, user id
   * @return successful or fault
   */
  modelbox::Status GetUserAgencyProjectCredential(
      UserAgencyCredential &agency_user_credential,
      const AgencyInfo &agency_info, const std::string &user_id = "");

  /**
   * @brief If service cert has been set, then you can get
   * user agency Project token to access user cloud resource
   * @param agency_user_token - out, agency user token
   * @param agency_info - in, agency info
   * @param project_info - in, project info
   * @return successful or fault
   */
  modelbox::Status GetUserAgencyProjectToken(UserAgencyToken &agency_user_token,
                                             const AgencyInfo &agency_info,
                                             const ProjectInfo &project_info);

  /**
   * @brief If user agency Project credential expires,notice me
   * @param agency_info - in, agency info
   */
  void ExpireUserAgencyProjectCredential(const AgencyInfo &agency_info);

  /**
   * @brief If user agency Project token expires,notice me
   * @param agency_info - in, agency info
   */
  void ExpireUserAgencyProjectToken(const AgencyInfo &agency_info);

  /**
   * @brief Save agency project credential
   * @param credential - in, credential token
   */
  void SetUserAgencyCredential(const UserAgencyCredential &credential);

  /**
   * @brief Remove agency project credential
   * @param userId - in, user id
   */
  void RemoveUserAgencyCredential(const std::string &userId);

  /**
   * @brief Save vas token
   * @param token - in, vas token from iva
   */
  void SetAgentToken(const AgentToken &token);

  /**
   * @brief set update token callback function
   * @param callback -in
   */
  void SetUpdateAgentTokenCallBack(std::function<void()> &callback);
};
}  // namespace modelbox
#endif  // MODELBOX_FLOWUNIT_IAM_AUTH_H_