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

#ifndef MODELBOX_SOURCE_CONTEXT_H_
#define MODELBOX_SOURCE_CONTEXT_H_

#include <modelbox/base/config.h>
#include <modelbox/base/device.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/data_source_parser_plugin.h>

#include "modelbox/data_context.h"

namespace modelbox {

enum RetryStatus { RETRY_NONEED = 0, RETRY_NEED = 1, RETRY_STOP = 2 };

class RetryContext {
 public:
  void SetRetryEnable(int32_t retry_enabled) {
    retry_enabled_ = retry_enabled;
  };

  void SetMaxRetryTimes(int32_t max_retry_times) {
    max_retry_times_ = max_retry_times;
  };

  void SetRetryInterval(int32_t retry_interval) {
    retry_interval_ = retry_interval;
  };

  void ResetRetryTimes() { retry_times_ = 0; };

  int32_t GetMaxRetryTimes() { return max_retry_times_; };
  int32_t GetRetryTimes() { return retry_times_; };
  int32_t GetRetryInterval() { return retry_interval_; };  // millisecond
  int32_t GetRetryEnable() { return retry_enabled_; };
  void RetryTimesInc() { retry_times_++; };

 private:
  int32_t retry_enabled_;    // retry or not
  int32_t retry_interval_;   // retry interval millisecond
  int32_t max_retry_times_;  // max retry times
  int32_t retry_times_{0};   // current retry times
};

class SourceContext {
 public:
  SourceContext(std::shared_ptr<DataSourceParserPlugin> plugin,
                std::string plugin_name);

  virtual ~SourceContext();

  void SetStreamType(std::string type);
  std::shared_ptr<std::string> GetSourceURL();
  void SetLastProcessStatus(const modelbox::Status& status);
  void SetDataSourceCfg(std::string data_source_cfg);
  void SetSessionContext(
      const std::shared_ptr<modelbox::SessionContext>& session_context);
  void SetSessionConfig(
      const std::shared_ptr<modelbox::Configuration>& session_config);

  RetryStatus NeedRetry();
  int32_t GetRetryInterval();
  void SetRetryParam(int32_t retry_enable, int32_t retry_interval,
                     int32_t retry_times);

 private:
  std::shared_ptr<modelbox::SessionContext> session_context_;
  std::shared_ptr<modelbox::Configuration> session_config_;
  std::string data_source_cfg_;
  std::shared_ptr<DataSourceParserPlugin> plugin_;
  modelbox::Status last_status_{modelbox::STATUS_SUCCESS};
  std::string stream_type_;
  RetryContext retry_context_;
  std::string plugin_name_;
};

}  // namespace modelbox
#endif