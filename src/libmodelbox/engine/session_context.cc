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

#include "modelbox/session_context.h"

#include "modelbox/base/uuid.h"
#include "modelbox/profiler.h"
#include "modelbox/virtual_node.h"
namespace modelbox {

SessionContext::SessionContext(std::shared_ptr<StatisticsItem> graph_stats) {
  ConfigurationBuilder config_builder;
  config_ = config_builder.Build();
  auto ret = GetUUID(&session_id_);
  if (ret != STATUS_OK) {
    MBLOG_WARN << "Get uuid failed, set session id to timestamp";
    session_id_ = std::to_string(GetCurrentTime());
  }

  if (graph_stats != nullptr) {
    graph_stats_ = graph_stats;
    graph_session_stats_ = graph_stats_->AddItem(session_id_);
  }
  MBLOG_INFO << "session context start se id:" << GetSessionId();
};

SessionContext::~SessionContext() {
  MBLOG_INFO << "session context finish se id:" << GetSessionId();
  if (graph_stats_ != nullptr) {
    graph_stats_->DelItem(session_id_);
  }
};

void SessionContext::SetPrivate(const std::string &key,
                                std::shared_ptr<void> private_content) {
  std::lock_guard<std::mutex> lock(private_map_lock_);
  private_map_[key] = private_content;
};

void SessionContext::SetSessionId(const std::string &session_id) {
  session_id_ = session_id;
}

std::string SessionContext::GetSessionId() { return session_id_; }

std::shared_ptr<Configuration> SessionContext::GetConfig() { return config_; }

void SessionContext::SetError(std::shared_ptr<FlowUnitError> error) {
  error_ = error;
}

std::shared_ptr<FlowUnitError> SessionContext::GetError() { return error_; }

std::shared_ptr<StatisticsItem> SessionContext::GetStatistics(
    SessionContexStatsType type) {
  switch (type) {
    case SessionContexStatsType::SESSION:
      return graph_session_stats_;

    case SessionContexStatsType::GRAPH:
      return graph_stats_;

    default:
      return nullptr;
  }
}

std::shared_ptr<void> SessionContext::GetPrivate(const std::string &key) {
  std::lock_guard<std::mutex> lock(private_map_lock_);
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    return nullptr;
  }

  return private_map_[key];
};

}  // namespace modelbox
