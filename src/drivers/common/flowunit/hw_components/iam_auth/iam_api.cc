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


#include "iam_api.h"

#include <iostream>
#include <nlohmann/json.hpp>

#include "modelbox/base/log.h"
#include "signer.h"

using namespace std;
using namespace utility;
using namespace web;
using namespace web::http;
using namespace web::http::client;

namespace modelbox {
bool IAMApi::validate_certificates_ = false;
std::string IAMApi::request_host_ = "";
std::string IAMApi::request_token_uri_ = "";
std::string IAMApi::request_credential_uri_ = "";
std::string IAMApi::cert_file_ = "/etc/ssl/certs/ca-certificates.crt";
std::string IAMApi::cert_file_path_ = "/etc/ssl/certs";

void SetHttpConfig(http_client_config &config, bool validate_certificates) {
  config.set_validate_certificates(validate_certificates);
  config.set_timeout(utility::seconds(60));
  if (validate_certificates) {
    config.set_ssl_context_callback([&](boost::asio::ssl::context &ctx) {
      ctx.load_verify_file(IAMApi::cert_file_path_);
      ctx.add_verify_path(IAMApi::cert_file_path_);
    });
  }
}

modelbox::Status CreateRequestBody(const AgencyInfo &agency_info,
                                 const ProjectInfo &project_info,
                                 const int32_t &token_flag,
                                 web::json::value &request_body) {
  try {
    request_body["auth"]["identity"]["methods"][0] =
        web::json::value::string(U("assume_role"));
    request_body["auth"]["identity"]["assume_role"]["domain_name"] =
        web::json::value::string(U(agency_info.user_domain_name));
    request_body["auth"]["identity"]["assume_role"]["agency_name"] =
        web::json::value::string(U(agency_info.xrole_name));
    request_body["auth"]["identity"]["assume_role"]["duration_seconds"] = 
        ONE_DAY_SECONDS;
    if (token_flag == TOKEN_REQUEST) {
      if (!project_info.project_name.empty()) {
        request_body["auth"]["scope"]["project"]["name"] =
            web::json::value::string(U(project_info.project_name));
      } else if (!project_info.project_id.empty()) {
        request_body["auth"]["scope"]["project"]["id"] =
            web::json::value::string(U(project_info.project_id));
      } else {
        MBLOG_ERROR << "cannot find any project info";
      }
    }
  } catch (const exception &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status GetSubjectTokenFromResponse(
    UserAgencyToken &token, web::http::http_response &response_data) {
  try {
    auto response_headers = response_data.headers();
    auto token_iter = response_headers.find("X-Subject-Token");
    if (token_iter != response_headers.end()) {
      token.user_token = token_iter->second;
    } else {
      throw "failed get user agency token ";
    }
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status GetAgencyCredentialFromResponse(
    UserAgencyCredential &user_agency_credential,
    web::http::http_response &response_data) {
  try {
    auto reponse_body = response_data.extract_json().get();
    auto nlohmann_body = nlohmann::json::parse(reponse_body.serialize());
    user_agency_credential.user_secure_token =
        nlohmann_body["credential"]["securitytoken"];
    user_agency_credential.user_ak = nlohmann_body["credential"]["access"];
    user_agency_credential.user_sk = nlohmann_body["credential"]["secret"];
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

std::shared_ptr<void> IAMApi::CreateSignerRequest(
    const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
    const ProjectInfo &project_info, int32_t token_flag) {
  if (request_host_.empty()) {
    MBLOG_ERROR << "host or uri is empty, please set host and uri value";
    return nullptr;
  }

  if (token_flag == TOKEN_REQUEST) {
    if (request_token_uri_.empty()) {
      MBLOG_ERROR << "token uri is empty, please set  uri value";
      return nullptr;
    }
  }

  if (token_flag == CREDENTIAL_REQUEST) {
    if (request_credential_uri_.empty()) {
      MBLOG_ERROR << "credential uri is empty, please set  uri value";
      return nullptr;
    }
  }

  size_t pos = request_host_.find("://", 0);
  size_t offset = string("://").length();
  std::string endpoint = request_host_.substr(pos + offset);

  // construct json data
  web::json::value request_body;
  if (modelbox::STATUS_OK !=
      CreateRequestBody(agency_info, project_info, token_flag, request_body)) {
    return nullptr;
  }

  std::string request_uri = request_token_uri_;
  if (token_flag == CREDENTIAL_REQUEST) {
    request_uri = request_credential_uri_;
  }

  std::shared_ptr<RequestParams> request_self = std::make_shared<RequestParams>(
      "POST", endpoint, request_uri, "", U(request_body.serialize()));
  request_self->addHeader("content-type", "application/json");
  request_self->addHeader("X-Domain-Id", consignee_info.domain_id);
  request_self->addHeader("X-Project-Id", consignee_info.project_id);

  Signer signer(consignee_info.ak, consignee_info.sk);
  signer.createSignature(request_self.get());
  return request_self;
}

modelbox::Status SendHttpRequest(const std::string request_host,
                               const std::string &request_uri,
                               const web::http::http_request &token_request,
                               web::http::http_response &response_data) {
  http_client_config config;
  SetHttpConfig(config, IAMApi::validate_certificates_);
  http_client token_client(U(request_host + request_uri), config);
  try {
    response_data = token_client.request(token_request).get();
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }

  if (response_data.status_code() != status_codes::Created) {
    MBLOG_ERROR << "failed to get project token, status code :"
                << response_data.status_code();
    MBLOG_ERROR << "return body: " << response_data.extract_utf8string().get();
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status IAMApi::GetAgencyProjectCredentialWithAK(
    const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
    UserAgencyCredential &user_agency_credential) {
  ProjectInfo project_info;
  std::shared_ptr<RequestParams> request_self =
      std::static_pointer_cast<RequestParams>(CreateSignerRequest(
          consignee_info, agency_info, project_info, CREDENTIAL_REQUEST));
  if (request_self == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  http_request token_request;
  token_request.set_method(methods::POST);
  token_request.set_body(U(request_self->getPayload()));
  for (auto header : *request_self->getHeaders()) {
    token_request.headers()[U(header.getKey().c_str())] =
        U(header.getValue().c_str());
  }

  web::http::http_response response_data;
  if (modelbox::STATUS_OK != SendHttpRequest(request_host_,
                                           request_credential_uri_,
                                           token_request, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK !=
      GetAgencyCredentialFromResponse(user_agency_credential, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status IAMApi::GetAgencyProjectCredentialWithToken(
    const AgentToken &agent_token, const AgencyInfo &agency_info,
    UserAgencyCredential &user_agency_credential) {
  web::json::value request_body;
  ProjectInfo project_info;
  if (modelbox::STATUS_OK != CreateRequestBody(agency_info, project_info,
                                             CREDENTIAL_REQUEST,
                                             request_body)) {
    return modelbox::STATUS_FAULT;
  }

  http_request token_request;
  token_request.headers()["Content-Type"] = "application/json;charset=utf8";
  token_request.headers()["X-Auth-Token"] = agent_token.x_subject_token_;
  token_request.set_method(methods::POST);
  token_request.set_body(request_body);
  web::http::http_response response_data;

  if (modelbox::STATUS_OK != SendHttpRequest(request_host_, request_token_uri_,
                                           token_request, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK !=
      GetAgencyCredentialFromResponse(user_agency_credential, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status IAMApi::GetAgencyProjectTokenWithAK(
    const ConsigneeInfo &consignee_info, const AgencyInfo &agency_info,
    const ProjectInfo &project_info, UserAgencyToken &project_agency_token) {
  std::shared_ptr<RequestParams> request_self =
      std::static_pointer_cast<RequestParams>(CreateSignerRequest(
          consignee_info, agency_info, project_info, TOKEN_REQUEST));
  if (request_self == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  http_request token_request;
  token_request.set_method(methods::POST);
  token_request.set_body(U(request_self->getPayload()));
  for (auto header : *request_self->getHeaders()) {
    token_request.headers()[U(header.getKey().c_str())] =
        U(header.getValue().c_str());
  }

  web::http::http_response response_data;
  if (modelbox::STATUS_OK != SendHttpRequest(request_host_, request_token_uri_,
                                           token_request, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK !=
      GetSubjectTokenFromResponse(project_agency_token, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status IAMApi::GetAgencyProjectTokenWithToken(
    const AgentToken &agent_token, const AgencyInfo &agency_info,
    const ProjectInfo &project_info, UserAgencyToken &user_agency_token) {
  web::json::value request_body;
  if (modelbox::STATUS_OK != CreateRequestBody(agency_info, project_info,
                                             TOKEN_REQUEST, request_body)) {
    return modelbox::STATUS_FAULT;
  }

  http_request token_request;
  token_request.headers()["Content-Type"] = "application/json;charset=utf8";
  token_request.headers()["X-Auth-Token"] = agent_token.x_subject_token_;
  token_request.set_method(methods::POST);
  token_request.set_body(request_body);
  web::http::http_response response_data;

  if (modelbox::STATUS_OK != SendHttpRequest(request_host_, request_token_uri_,
                                           token_request, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  if (modelbox::STATUS_OK !=
      GetSubjectTokenFromResponse(user_agency_token, response_data)) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

void IAMApi::SetRequestHost(std::string request_host) {
  request_host_ = request_host;
}

void IAMApi::SetRequestTokenUri(std::string request_token_uri) {
  request_token_uri_ = request_token_uri;
}

void IAMApi::SetRequestCredentialUri(std::string request_credential_uri) {
  request_credential_uri_ = request_credential_uri;
}

void IAMApi::SetValidateCertificates(bool validate_certificates) {
  validate_certificates_ = validate_certificates;
}

void IAMApi::SetCertFilePath(std::string cert_file,
                             std::string cert_file_path) {
  IAMApi::cert_file_ = cert_file;
  IAMApi::cert_file_path_ = cert_file_path;
}

}  // namespace modelbox