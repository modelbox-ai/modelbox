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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_URL_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_URL_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/data_source_parser_plugin.h>

constexpr const char *DRIVER_NAME = "url";
constexpr const char *DRIVER_DESC = "A url data source parser plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

class UrlSourceParser : public modelbox::DataSourceParserPlugin {
 public:
  UrlSourceParser();
  ~UrlSourceParser() override;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  modelbox::Status Parse(
      const std::shared_ptr<modelbox::SessionContext> &session_context,
      const std::string &config, std::string &uri,
      modelbox::DestroyUriFunc &destroy_uri_func) override;
  modelbox::Status GetStreamType(const std::string &config,
                                 std::string &stream_type) override;

 protected:
  int32_t file_retry_interval_ = 1;
  int32_t file_retry_times_ = 0;
  int32_t stream_retry_interval_ = 1;
  int32_t stream_retry_times_ = 0;
};

class UrlSourceParserFactory : public modelbox::DriverFactory {
 public:
  UrlSourceParserFactory() = default;
  ~UrlSourceParserFactory() override = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<UrlSourceParser>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_URL_CPU_H_
