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

#include "source_context.h"

#include <utility>

namespace modelbox {

SourceContext::SourceContext(std::shared_ptr<DataSourceParserPlugin> plugin,
                             std::string plugin_name)
    : plugin_(std::move(plugin)), plugin_name_(std::move(plugin_name)) {}

SourceContext::~SourceContext() = default;

std::shared_ptr<std::string> SourceContext::GetSourceURL() {
  std::shared_ptr<std::string> uri;
  std::string uri_str;
  DestroyUriFunc destroy_uri_func;

  auto ret = plugin_->Parse(session_context_, data_source_cfg_, uri_str,
                            destroy_uri_func);
  if (!ret) {
    MBLOG_ERROR << "Parse config failed, source uri is empty";
    return nullptr;
  }

  uri = std::shared_ptr<std::string>(new std::string(uri_str),
                                     [destroy_uri_func](std::string *ptr) {
                                       if (destroy_uri_func) {
                                         destroy_uri_func(*ptr);
                                       }
                                       delete ptr;
                                     });

  return uri;
}

RetryStatus SourceContext::NeedRetry() {
  retry_context_.RetryTimesInc();

  if (last_status_ == modelbox::STATUS_NODATA && stream_type_ == "file") {
    return modelbox::RETRY_STOP;
  }

  if (plugin_->GetRetryEnabled() &&
      ((retry_context_.GetRetryTimes() <= plugin_->GetRetryTimes()) ||
       (plugin_->GetRetryTimes() == -1))) {
    return modelbox::RETRY_NEED;
  }
  MBLOG_INFO << "retry_enable_: " << plugin_->GetRetryEnabled()
             << " retry_times: " << retry_context_.GetRetryTimes()
             << " retry_max_times_: " << plugin_->GetRetryTimes();
  return modelbox::RETRY_NONEED;
}

void SourceContext::SetLastProcessStatus(const modelbox::Status &status) {
  last_status_ = status;
  if (status == modelbox::STATUS_SUCCESS) {
    retry_context_.ResetRetryTimes();
  }
}

void SourceContext::SetStreamType(std::string type) {
  stream_type_ = std::move(type);
  MBLOG_DEBUG << "plugin_name: " << plugin_name_
              << "  stream_type: " << stream_type_;
}

void SourceContext::SetRetryParam(int32_t retry_enable, int32_t retry_interval,
                                  int32_t retry_times) {
  retry_context_.SetMaxRetryTimes(retry_times);
  retry_context_.SetRetryInterval(retry_interval);
  retry_context_.SetRetryEnable(retry_enable);
  MBLOG_DEBUG << "plugin_name: " << plugin_name_
              << "  retry_enable: " << retry_context_.GetRetryEnable()
              << "  retry_interval:" << retry_context_.GetRetryInterval()
              << "  retry_times: " << retry_context_.GetMaxRetryTimes();
}

void SourceContext::SetDataSourceCfg(std::string data_source_cfg) {
  data_source_cfg_ = std::move(data_source_cfg);
}

void SourceContext::SetSessionContext(
    const std::shared_ptr<modelbox::SessionContext> &session_context) {
  session_context_ = session_context;
}

int32_t SourceContext::GetRetryInterval() {
  return retry_context_.GetRetryInterval();
}

}  // namespace modelbox
