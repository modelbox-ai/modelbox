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

#include "modelbox/session.h"

namespace modelbox {

Session::Session(std::shared_ptr<StatisticsItem> graph_stats)
    : ctx_(std::make_shared<SessionContext>(graph_stats)) {}

Session::~Session() {
  auto io = io_handle_.lock();
  if (io == nullptr) {
    return;
  }

  io->SessionEnd(error_);
}

void Session::AddStateListener(std::shared_ptr<SessionStateListener> listener) {
  std::lock_guard<std::mutex> lock(state_listener_list_lock_);
  state_listener_list_.push_back(listener);
}

void Session::SetSessionIO(std::shared_ptr<SessionIO> io_handle) {
  io_handle_ = io_handle;
}

std::shared_ptr<SessionIO> Session::GetSessionIO() { return io_handle_.lock(); }

std::shared_ptr<SessionContext> Session::GetSessionCtx() { return ctx_; }

/**
 * @brief will cause session end after current data in engine processed over
 **/
void Session::Close() {
  std::lock_guard<std::mutex> state_lock(state_lock_);
  if (closed_) {
    return;
  }

  closed_ = true;
  error_ = std::make_shared<FlowUnitError>("EOF");

  std::lock_guard<std::mutex> lock(state_listener_list_lock_);
  for (auto &state_listener : state_listener_list_) {
    auto listener = state_listener.lock();
    if (listener == nullptr) {
      continue;
    }

    listener->NotifySessionClose();
  }
}

bool Session::IsClosed() { return closed_; }

/**
 * @brief abort session imediately
 **/
void Session::Abort() { abort_ = true; }

bool Session::IsAbort() { return abort_; }

void Session::SetError(std::shared_ptr<FlowUnitError> error) { error_ = error; }

std::shared_ptr<FlowUnitError> Session::GetError() { return error_; }

std::shared_ptr<Session> SessionManager::CreateSession(
    std::shared_ptr<StatisticsItem> graph_stats) {
  auto *session = new Session(graph_stats);
  auto session_id = session->GetSessionCtx()->GetSessionId();
  auto session_ptr =
      std::shared_ptr<Session>(session, [session_id, this](Session *ptr) {
        DeleteSession(session_id);
        delete ptr;
      });
  std::lock_guard<std::mutex> lock(sessions_lock_);
  sessions_[session_id] = session_ptr;
  return session_ptr;
}

void SessionManager::DeleteSession(const SessionId &id) {
  std::lock_guard<std::mutex> lock(sessions_lock_);
  sessions_.erase(id);
  MBLOG_INFO << "session " << id << " is over, running session count "
             << sessions_.size();
}

std::unordered_map<SessionId, std::weak_ptr<Session>>
SessionManager::GetSessions() {
  std::lock_guard<std::mutex> lock(sessions_lock_);
  return sessions_;
}

}  // namespace modelbox