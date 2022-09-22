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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_RESTFUL_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_RESTFUL_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#define _TURN_OFF_PLATFORM_STRING
#include <modelbox/data_source_parser_plugin.h>

#include "cpprest/http_client.h"

constexpr const char *DRIVER_NAME = "restful";
constexpr const char *DRIVER_DESC =
    "An restful data source parser plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

typedef struct tag_RestfulInputInfo {
  std::string encode_full_url;
  web::http::http_headers headers;
  std::string response_url_position;
  std::string response_body;
} RestfulInputInfo;

class RestfulSourceParser : public modelbox::DataSourceParserPlugin {
 public:
  RestfulSourceParser();
  ~RestfulSourceParser() override;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  modelbox::Status Parse(
      const std::shared_ptr<modelbox::SessionContext> &session_context,
      const std::shared_ptr<modelbox::Configuration> &session_config,
      const std::string &config, std::string &uri,
      modelbox::DestroyUriFunc &destroy_uri_func) override;

  modelbox::Status GetStreamType(const std::string &config,
                                 std::string &stream_type) override;

 private:
  modelbox::Status GetRestfulInfo(RestfulInputInfo &input_info,
                                  const std::string &config);
  modelbox::Status SendRestfulRequest(RestfulInputInfo &input_info,
                                      web::http::http_response &resp);
  modelbox::Status ProcessRestfulResponse(RestfulInputInfo &input_info,
                                          web::http::http_response &resp,
                                          std::string &uri);
};

class RestfulSourceParserFactory : public modelbox::DriverFactory {
 public:
  RestfulSourceParserFactory() = default;
  ~RestfulSourceParserFactory() override = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<RestfulSourceParser>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_RESTFUL_CPU_H_