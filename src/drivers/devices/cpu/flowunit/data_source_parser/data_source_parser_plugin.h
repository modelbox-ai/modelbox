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
#include <functional>
#include <string>
#include "source_context.h"
#include "source_parser.h"

#define RETRY_ON 1
#define RETRY_OFF 0

constexpr const char *DRIVER_CLASS_DATA_SOURCE_PARSER_PLUGIN =
    "DRIVER-SOURCE-PARSER";

class DataSourceParserPlugin
    : public modelbox::Driver,
      public modelbox::SourceParser,
      public std::enable_shared_from_this<DataSourceParserPlugin> {
 public:
  virtual modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) = 0;

  virtual modelbox::Status Deinit() = 0;

  virtual modelbox::Status Parse(const std::string &config, std::string &uri,
                               DestroyUriFunc &destroy_uri_func) = 0;

  virtual modelbox::Status GetStreamType(const std::string &config,
                                       std::string &stream_type) {
    stream_type = "stream";
    return modelbox::STATUS_OK;
  }

  virtual int32_t GetRetryTimes() { return retry_max_times_; };
  virtual int32_t GetRetryInterval() { return retry_interval_; };
  virtual int32_t GetRetryEnabled() { return retry_enabled_; };

  virtual void ReadConf(const std::shared_ptr<modelbox::Configuration> &opts) {
    retry_enabled_ = opts->GetBool("retry_enable", retry_enabled_);
    retry_interval_ = opts->GetInt32("retry_interval_ms", retry_interval_);
    retry_max_times_ = opts->GetInt32("retry_count_limit", retry_max_times_);
  };

  virtual bool GetRetryFlag(modelbox::Status status) { return true; };

  virtual void SetStreamType(
      std::shared_ptr<modelbox::SourceContext> source_context) {
    source_context->SetStreamType("stream");
  };

  virtual std::shared_ptr<modelbox::SourceContext> GetSourceContext(
      const std::string &source_type) {
    std::shared_ptr<modelbox::SourceContext> source_context =
        std::make_shared<modelbox::SourceContext>(shared_from_this(),
                                                source_type);
    source_context->SetRetryParam(retry_enabled_, retry_interval_,
                                  retry_max_times_);
    return source_context;
  };

  virtual modelbox::RetryStatus NeedRetry(std::string &stream_type,
                                        modelbox::Status &last_status,
                                        int32_t retry_times) {
    if (last_status == modelbox::STATUS_NODATA && stream_type == "file") {
      return modelbox::RETRY_STOP;
    }

    if (retry_enabled_ &&
        ((retry_times <= retry_max_times_) || (retry_max_times_ == -1))) {
      return modelbox::RETRY_NEED;
    }
    MBLOG_INFO << "retry_enable_: " << retry_enabled_
               << " retry_times: " << retry_times
               << " retry_max_times_: " << retry_max_times_;
    return modelbox::RETRY_NONEED;
  };

 protected:
  int32_t retry_interval_{1000};  // millisecond
  int32_t retry_max_times_{-1};
  int32_t retry_enabled_{RETRY_OFF};
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_PLUGIN_H_