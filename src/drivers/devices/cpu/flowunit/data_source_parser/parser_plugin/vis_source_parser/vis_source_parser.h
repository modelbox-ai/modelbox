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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VIS_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VIS_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "data_source_parser_plugin.h"

constexpr const char *DRIVER_NAME = "vis";
constexpr const char *DRIVER_DESC = "An vis data source parser plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

typedef struct tag_VisInputInfo {
  std::string ak;
  std::string sk;
  std::string token;
  std::string user_id;
  std::string domain_name;
  std::string xrole_name;
  std::string end_point;
  std::string project_id;
  std::string stream_name;
  bool cert_flag;
} VisInputInfo;

class VisSourceParser : public DataSourceParserPlugin {
 public:
  VisSourceParser();
  ~VisSourceParser();

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  modelbox::Status Parse(
      std::shared_ptr<modelbox::SessionContext> session_context,
      const std::string &config, std::string &uri,
      DestroyUriFunc &destroy_uri_func) override;

 private:
  modelbox::Status GetVisInfo(VisInputInfo &input_info,
                              const std::string &config);

  modelbox::Status GetTempAKSKInfo(VisInputInfo &input_info);
};

class VisSourceParserFactory : public modelbox::DriverFactory {
 public:
  VisSourceParserFactory() = default;
  ~VisSourceParserFactory() = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<VisSourceParser>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VIS_CPU_H_