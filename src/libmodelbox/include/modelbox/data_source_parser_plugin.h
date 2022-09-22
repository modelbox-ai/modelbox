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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_PLUGIN_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_PLUGIN_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/status.h>
#include <modelbox/session_context.h>

#include <functional>
#include <string>

#define DATASOURCE_PARSER_STREAM_DEFAULT_RETRY_TIMES (-1)
#define DATASOURCE_PARSER_FILE_DEFAULT_RETRY_TIMES (10)
#define DATASOURCE_PARSER_DEFAULT_RETRY_INTERVAL 1000
#define DATASOURCE_PARSER_RETRY_ON 1
#define DATASOURCE_PARSER_RETRY_OFF 0

constexpr const char *DRIVER_CLASS_DATA_SOURCE_PARSER_PLUGIN =
    "DRIVER-SOURCE-PARSER";

namespace modelbox {

using DestroyUriFunc = std::function<void(const std::string &uri)>;

class DataSourceParserPlugin : public Driver {
 public:
  virtual Status Init(const std::shared_ptr<Configuration> &opts) = 0;

  virtual Status Deinit() = 0;

  virtual Status Parse(const std::shared_ptr<SessionContext> &session_context,
                       const std::shared_ptr<modelbox::Configuration> &session_config, 
                       const std::string &config, std::string &uri,
                       DestroyUriFunc &destroy_uri_func) = 0;

  virtual Status GetStreamType(const std::string &config,
                               std::string &stream_type) = 0;

  int32_t GetRetryTimes() { return retry_max_times_; };

  int32_t GetRetryInterval() { return retry_interval_; };

  int32_t GetRetryEnabled() { return retry_enabled_; };

 protected:
  int32_t retry_interval_{
      DATASOURCE_PARSER_DEFAULT_RETRY_INTERVAL};  // millisecond
  int32_t retry_max_times_{
      DATASOURCE_PARSER_STREAM_DEFAULT_RETRY_TIMES};  // -1: infinite retry
  int32_t retry_enabled_{
      DATASOURCE_PARSER_RETRY_OFF};  // 0:  retry disable  1: retry enable
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_PLUGIN_H_