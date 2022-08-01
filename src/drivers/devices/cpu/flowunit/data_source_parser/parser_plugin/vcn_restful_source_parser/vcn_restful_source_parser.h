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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VCN_RESTFUL_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VCN_RESTFUL_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "data_source_parser_plugin.h"
#include "vcn_restful_client.h"

constexpr const char *DRIVER_NAME = "vcn_restful";
constexpr const char *DRIVER_DESC =
    "A VCN restful data source parser plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

class VcnRestfulSourceParser : public DataSourceParserPlugin {
 public:
  VcnRestfulSourceParser() = default;
  ~VcnRestfulSourceParser() override = default;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  modelbox::Status Parse(
      std::shared_ptr<modelbox::SessionContext> session_context,
      const std::string &config, std::string &uri,
      DestroyUriFunc &destroy_uri_func) override;

 private:
  int32_t keep_alive_interval_{KEEP_ALIVE_INTERVAL_DEFAULT_SEC};
};

class VcnRestfulSourceParserFactory : public modelbox::DriverFactory {
 public:
  VcnRestfulSourceParserFactory() = default;
  ~VcnRestfulSourceParserFactory() override = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<VcnRestfulSourceParser>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_VCN_RESTFUL_CPU_H_