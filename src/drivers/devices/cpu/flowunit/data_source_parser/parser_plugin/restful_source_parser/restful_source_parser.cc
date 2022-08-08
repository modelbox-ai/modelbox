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

#include "restful_source_parser.h"

#include <iomanip>
#include <nlohmann/json.hpp>

RestfulSourceParser::RestfulSourceParser() = default;
RestfulSourceParser::~RestfulSourceParser() = default;

modelbox::Status RestfulSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  retry_enabled_ = opts->GetBool("retry_enable", DATASOURCE_PARSER_RETRY_ON);
  retry_interval_ = opts->GetInt32("retry_interval_ms",
                                   DATASOURCE_PARSER_DEFAULT_RETRY_INTERVAL);
  retry_max_times_ = opts->GetInt32(
      "retry_count_limit", DATASOURCE_PARSER_STREAM_DEFAULT_RETRY_TIMES);

  retry_enabled_ = opts->GetBool("restful_retry_enable", retry_enabled_);
  retry_interval_ =
      opts->GetInt32("restful_retry_interval_ms", retry_interval_);
  retry_max_times_ =
      opts->GetInt32("restful_retry_count_limit", retry_max_times_);
  return modelbox::STATUS_OK;
}

modelbox::Status RestfulSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status RestfulSourceParser::Parse(
    const std::shared_ptr<modelbox::SessionContext> &session_context,
    const std::string &config, std::string &uri,
    modelbox::DestroyUriFunc &destroy_uri_func) {
  RestfulInputInfo input_info;

  if (GetRestfulInfo(input_info, config) != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Failed to get Restful input info";
    return modelbox::STATUS_FAULT;
  }

  web::http::http_response resp;
  auto ret = SendRestfulRequest(input_info, resp);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Send Restful Request failed, detail: "
                << ret.WrapErrormsgs();
    return modelbox::STATUS_FAULT;
  }

  ret = ProcessRestfulResponse(input_info, resp, uri);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Process Restful Response failed.";
    return modelbox::STATUS_FAULT;
  }
  session_context->SetPrivate(
      "data_source_parser.restful_source_parser.response",
      std::make_shared<std::string>(input_info.response_body));
  return modelbox::STATUS_OK;
}

modelbox::Status RestfulSourceParser::GetRestfulInfo(
    RestfulInputInfo &input_info, const std::string &config) {
  nlohmann::json config_json;
  try {
    config_json = nlohmann::json::parse(config);
    std::string request_url = config_json["request_url"].get<std::string>();
    if (request_url.empty()) {
      MBLOG_ERROR
          << "Invalid request url, value of key <request_url> is empty!";
      return modelbox::STATUS_BADCONF;
    }

    if (config_json.contains("params")) {
      auto brokers_json = config_json["params"];
      std::string encode_params;
      for (auto &broker_json : brokers_json) {
        auto param_key = broker_json["param_key"].get<std::string>();
        auto param_value = broker_json["param_value"].get<std::string>();
        MBLOG_DEBUG << "param_key " << param_key << ", param_value "
                    << param_value;
        encode_params += web::uri::encode_data_string(param_key) + "=" +
                         web::uri::encode_data_string(param_value) + "&";
      }
      input_info.encode_full_url =
          request_url + "?" +
          encode_params.substr(0, encode_params.length() - 1);

    } else {
      input_info.encode_full_url = request_url;
    }

    input_info.response_url_position =
        config_json["response_url_position"].get<std::string>();
    if (input_info.response_url_position.empty()) {
      MBLOG_ERROR << "Invalid response url position, value of key "
                     "<response_url_position> is empty!";
      return modelbox::STATUS_BADCONF;
    }
    MBLOG_DEBUG << "response url position: "
                << input_info.response_url_position;

    if (config_json.contains("headers")) {
      auto value = config_json["headers"].get<nlohmann::json>();
      for (auto &header : value.items()) {
        if (header.key().empty()) {
          MBLOG_ERROR << "headers key is empty!";
          return modelbox::STATUS_BADCONF;
        }
        if (!header.value().is_string()) {
          MBLOG_ERROR << "Key <" << header.key() << "> must have string value.";
          return modelbox::STATUS_BADCONF;
        }
        input_info.headers.add(_XPLATSTR(header.key()),
                               _XPLATSTR(header.value()));
      }
    }

    return modelbox::STATUS_OK;
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse data source config to json failed, detail: "
                << e.what();
    return modelbox::STATUS_INVALID;
  }
}

modelbox::Status RestfulSourceParser::SendRestfulRequest(
    RestfulInputInfo &input_info, web::http::http_response &resp) {
  utility::string_t address = _XPLATSTR(input_info.encode_full_url);
  web::http::uri request_uri = web::http::uri(address);

  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(30));
  client_config.set_validate_certificates(false);

  std::shared_ptr<web::http::client::http_client> client;
  client = std::make_shared<web::http::client::http_client>(
      web::http::uri_builder(request_uri).to_uri(), client_config);

  input_info.headers.add(_XPLATSTR("Content-Type"),
                         _XPLATSTR("application/json"));
  web::http::http_request msg;
  msg.set_method(web::http::methods::GET);
  msg.headers() = input_info.headers;
  try {
    resp = client->request(msg).get();
  } catch (std::exception const &e) {
    return {modelbox::STATUS_FAULT, e.what()};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status RestfulSourceParser::ProcessRestfulResponse(
    RestfulInputInfo &input_info, web::http::http_response &resp,
    std::string &uri) {
  if (resp.status_code() == 200) {
    std::string resp_info = resp.extract_string().get();
    MBLOG_DEBUG << "Get response from restful server success.";

    nlohmann::json resp_json;
    try {
      resp_json = nlohmann::json::parse(resp_info);

      std::vector<std::string> rtsp_url_path;
      rtsp_url_path =
          modelbox::StringSplit(input_info.response_url_position, '/');
      if (rtsp_url_path.empty()) {
        MBLOG_ERROR << "rtsp_url_path is empty!";
      }
      for (const auto &url_path : rtsp_url_path) {
        resp_json = resp_json[url_path];
      }
      uri = resp_json;
      if (uri.empty()) {
        MBLOG_ERROR << "Restful rtsp address is empty!";
        return modelbox::STATUS_FAULT;
      }

      input_info.response_body = resp_info;
      MBLOG_DEBUG << "Get restful input info success.";
      return modelbox::STATUS_OK;
    } catch (const std::exception &e) {
      MBLOG_ERROR << "Parse response body failed, detail: " << e.what();
      return modelbox::STATUS_INVALID;
    }
  } else {
    MBLOG_ERROR << "Get input from Restful failed.  Http response code: "
                << resp.status_code()
                << ". Http response body: " << resp.extract_string().get();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status RestfulSourceParser::GetStreamType(const std::string &config,
                                                    std::string &stream_type) {
  stream_type = "stream";

  return modelbox::STATUS_OK;
}