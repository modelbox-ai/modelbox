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

#include "vis_source_parser.h"

#include <dirent.h>
#include <securec.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <ctime>
#include <nlohmann/json.hpp>
#include <string>

#include "cpprest/http_client.h"
#include "iam_auth.h"
#include "modelbox/base/utils.h"
#include "modelbox/base/uuid.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "signer.h"

VisSourceParser::VisSourceParser() = default;
VisSourceParser::~VisSourceParser() = default;

modelbox::Status VisSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  ReadConf(opts);
  retry_enabled_ = opts->GetBool("vis_retry_enable", RETRY_ON);
  retry_interval_ = opts->GetInt32("vis_retry_interval_ms", retry_interval_);
  retry_max_times_ = opts->GetInt32("vis_retry_count_limit", retry_max_times_);
  return modelbox::STATUS_OK;
}

modelbox::Status VisSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status VisSourceParser::Parse(
    std::shared_ptr<modelbox::SessionContext> session_context,
    const std::string &config, std::string &uri,
    DestroyUriFunc &destroy_uri_func) {
  VisInputInfo input_info;

  if (GetVisInfo(input_info, config) != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to get vis input info";
    return modelbox::STATUS_FAULT;
  }

  if (GetTempAKSKInfo(input_info) != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to get cert info! Invalid authorization.";
    return modelbox::STATUS_FAULT;
  }

  std::string struri = "/v1/" + input_info.project_id + "/streams/" +
                       input_info.stream_name + "/endpoint";
  utility::string_t address = U(input_info.end_point + struri);

  MBLOG_DEBUG << "Vis request address: " << address;
  web::http::uri request_uri = web::http::uri(address);

  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(30));

  if (input_info.cert_flag == false) {
    client_config.set_validate_certificates(false);
  } else {
    client_config.set_ssl_context_callback([&](boost::asio::ssl::context &ctx) {
      ctx.load_verify_file("/etc/pki/tls/certs/ca-bundle.crt");
    });
  }

  std::shared_ptr<web::http::client::http_client> client;
  client = std::make_shared<web::http::client::http_client>(
      web::http::uri_builder(request_uri).to_uri(), client_config);

  web::http::http_headers headers;
  size_t pos = input_info.end_point.find("://", 0);
  size_t offset = std::string("://").length();
  std::string endpoint = input_info.end_point.substr(pos + offset);
  std::shared_ptr<RequestParams> request_self =
      std::make_shared<RequestParams>("GET", endpoint, struri, "");
  request_self->addHeader("Content-Type", "application/json");
  Signer signer(input_info.ak, input_info.sk);
  signer.createSignature(request_self.get());
  for (auto header : *request_self->getHeaders()) {
    headers.add(U(header.getKey()), U(header.getValue()));
  }
  headers.add(U("X-Project-Id"), U(input_info.project_id));
  headers.add(U("X-Security-Token"), U(input_info.token));

  web::http::http_request msg;
  msg.set_method(web::http::methods::GET);
  msg.headers() = headers;

  try {
    MBLOG_INFO << "Send vis get stream request to " << address;
    web::http::http_response resp = client->request(msg).get();

    std::string resp_info = resp.extract_string().get();
    if (resp.status_code() == 200) {
      MBLOG_INFO << "Get input from vis success. Http response code: "
                 << resp.status_code() << ". Http response body: " << resp_info;

      nlohmann::json resp_json;
      try {
        resp_json = nlohmann::json::parse(resp_info);
        if (resp_json.contains("pull_flow_address")) {
          uri = resp_json["pull_flow_address"].get<std::string>();
          if (uri.empty()) {
            MBLOG_ERROR << "pull_flow_address is empty!";
            return modelbox::STATUS_BADCONF;
          }
          MBLOG_DEBUG << "pull_flow_address: " << uri;
          return modelbox::STATUS_OK;
        }

        if (resp_json.contains("hls_pull_flow_address")) {
          uri = resp_json["hls_pull_flow_address"].get<std::string>();
          if (uri.empty()) {
            MBLOG_ERROR << "hls_pull_flow_address is empty!";
            return modelbox::STATUS_BADCONF;
          }
          MBLOG_DEBUG << "hls_pull_flow_address: " << uri;
          return modelbox::STATUS_OK;
        }

        MBLOG_ERROR << "No avaliable pull flow address string in response.";
        return modelbox::STATUS_BADCONF;
      } catch (const std::exception &e) {
        MBLOG_ERROR << "Parse response body to json failed, detail: "
                    << e.what();
        return modelbox::STATUS_INVALID;
      }
    } else {
      MBLOG_ERROR << "Get input from vis failed.  Http response code: "
                  << resp.status_code()
                  << ". Http response body: " << resp_info;
      return modelbox::STATUS_FAULT;
    }
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status VisSourceParser::GetVisInfo(VisInputInfo &input_info,
                                             const std::string &config) {
  nlohmann::json config_json;
  try {
    config_json = nlohmann::json::parse(config);

    std::string end_point;
    end_point = config_json["visEndPoint"].get<std::string>();
    std::string::size_type idx;
    std::string https_endpoint = "https://";
    idx = end_point.find(https_endpoint);
    if (idx != 0) {
      end_point = "https://" + end_point;
    }
    input_info.end_point = end_point;
    if (input_info.end_point.empty()) {
      MBLOG_ERROR << "Value of <visEndPoint> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    input_info.project_id = config_json["projectId"].get<std::string>();
    if (input_info.project_id.empty()) {
      MBLOG_ERROR << "Value of key <projectId> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    input_info.stream_name = config_json["streamName"].get<std::string>();
    if (input_info.stream_name.empty()) {
      MBLOG_ERROR << "Value of key <streamName> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    if (config_json.contains("domainName")) {
      input_info.domain_name = config_json["domainName"].get<std::string>();
      if (input_info.domain_name.empty()) {
        MBLOG_DEBUG << "Value of key <domainName> is empty!";
      }
    }

    if (config_json.contains("userId")) {
      input_info.user_id = config_json["userId"].get<std::string>();
      if (input_info.user_id.empty()) {
        MBLOG_DEBUG << "Value of key <userId> is empty!";
      }
    }

    if (config_json.contains("xroleName")) {
      input_info.xrole_name = config_json["xroleName"].get<std::string>();
      if (input_info.xrole_name.empty()) {
        MBLOG_DEBUG << "Value of key <xroleName> is empty!";
      }
    }

    if (config_json.contains("certificate")) {
      input_info.cert_flag = config_json["certificate"].get<bool>();
    } else {
      input_info.cert_flag = true;
    }

    return modelbox::STATUS_OK;
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }
}

modelbox::Status VisSourceParser::GetTempAKSKInfo(VisInputInfo &input_info) {
  modelbox::UserAgencyToken agency_user_token;
  modelbox::AgencyInfo agency_info;
  modelbox::ProjectInfo project_info;

  agency_info.user_domain_name = input_info.domain_name;
  agency_info.xrole_name = input_info.xrole_name;
  project_info.project_id = input_info.project_id;

  modelbox::UserAgencyCredential credential;
  auto ret = modelbox::IAMAuth::GetInstance()->GetUserAgencyProjectCredential(
      credential, agency_info, input_info.user_id);
  if (ret != modelbox::STATUS_OK) {
    std::string err_msg = "Failed to get credential info!";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  input_info.ak = credential.user_ak;
  input_info.sk = credential.user_sk;
  input_info.token = credential.user_secure_token;

  return modelbox::STATUS_OK;
}