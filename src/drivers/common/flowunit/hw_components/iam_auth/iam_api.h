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


#ifndef MODELBOX_FLOWUNIT_IAM_API_H_
#define MODELBOX_FLOWUNIT_IAM_API_H_

#include <modelbox/base/status.h>
#include <cpprest/http_client.h>
#include <cpprest/json.h>

#include <string>

#include "token_header.h"
#define TOKEN_REQUEST 0
#define CREDENTIAL_REQUEST 1
#define ONE_DAY_SECONDS 86400

namespace modelbox {
class IAMApi {
 public:
  /*
   * @brief  Get agency credential with ak/sk
   * @param  consignee_info - in,  consignee information
   * @param  AgencyInfo - in, agency information
   * @param  user_agency_credential - out, user credential
   * @return successful or fault
   */
  static modelbox::Status GetAgencyProjectCredentialWithAK(
      const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
      UserAgencyCredential &user_agency_credential);
  /*
   * @brief  Get agency credential with token
   * @param  agent_token - in,  agent token
   * @param  AgencyInfo - in, agency information
   * @param  user_agency_credential - out, user credential
   * @return successful or fault
   */
  static modelbox::Status GetAgencyProjectCredentialWithToken(
      const AgentToken &agent_token, const AgencyInfo &agency_info,
      UserAgencyCredential &user_agency_credential);

  /*
   * @brief  Get agency credential
   * @param  consignee_info - in,  consignee information
   * @param  AgencyInfo - in, agency information
   * @param  project_info - in, project information: project_name and project_id
   * @param  user_agency_token - out, user token
   * @return successful or fault
   */
  static modelbox::Status GetAgencyProjectTokenWithAK(
      const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
      const ProjectInfo &project_info, UserAgencyToken &project_agency_token);

  /*
   * @brief  Get agency token
   * @param  token - in,  vas token
   * @param  agency_info - in, agency information
   * @param  project_info - in, project information: project_name and project_id
   * @param  user_agency_token - out, user token
   * @return successful or fault
   */
  static modelbox::Status GetAgencyProjectTokenWithToken(
      const AgentToken &agent_token, const AgencyInfo &agency_info,
      const ProjectInfo &project_info, UserAgencyToken &user_agency_token);

  /*
   * @brief  Set iam host address
   * @param  request_host - in,  iam host address
   */
  static void SetRequestHost(std::string request_host);

  /*
   * @brief  Set token request uri
   * @param  request_token_uri - in,   token request uri
   */
  static void SetRequestTokenUri(std::string request_token_uri);

  /*
   * @brief  Set credential request uri
   * @param  request_credential_uri - in,   credential request uri
   */
  static void SetRequestCredentialUri(std::string request_credential_uri);

  /*
   * @brief  Set validate certificates flag
   * @param  validate_certificates_ - in,   Indicates whether to validate the
   * certificate.
   */
  static void SetValidateCertificates(bool validate_certificates);
  /*
   * @brief  Set certificate file and path
   * @param  cert_file - in,   certificate file name
   * @param  cert_file_path - in, certificate file
   */
  static void SetCertFilePath(std::string cert_file,
                              std::string cert_file_path);
  /*
   * @brief  Create a signer request
   * @param  consignee_info - in,  consignee info
   * @param  agency_info - in, agency info
   * @param  project_info - project info(id and name)
   * @param  token_type - token or credential
   * @return return request
   */
  static std::shared_ptr<void> CreateSignerRequest(
      const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
      const ProjectInfo &project_info, int32_t token_flag);

  static std::string request_host_;
  static std::string request_credential_uri_;
  static std::string request_token_uri_;
  static std::string cert_file_path_;
  static std::string cert_file_;
  static bool validate_certificates_;
};
}  // namespace modelbox
#endif  // MODELBOX_FLOWUNIT_IAM_API_H_