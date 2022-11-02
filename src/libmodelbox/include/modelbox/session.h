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

#ifndef MODELBOX_SESSION_H_
#define MODELBOX_SESSION_H_

#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "modelbox/error.h"
#include "modelbox/profiler.h"
#include "modelbox/session_context.h"

namespace modelbox {

using SessionId = std::string;

class SessionIO {
 public:
  SessionIO();
  virtual Status SetOutputMeta(const std::string& port_name,
                               std::shared_ptr<DataMeta> meta) = 0;
  virtual Status Send(const std::string& port_name,
                      std::shared_ptr<BufferList> buffer_list) = 0;
  virtual Status Recv(OutputBufferList& map_buffer_list, int timeout = 0) = 0;
  virtual Status Close() = 0;
  virtual Status Shutdown() = 0;

  virtual ~SessionIO();

 protected:
  friend class Session;
  virtual void SessionEnd(std::shared_ptr<FlowUnitError> error = nullptr) = 0;
};

class SessionStateListener {
 public:
  SessionStateListener();
  virtual ~SessionStateListener();

  virtual void NotifySessionClose();
};

class Session {
 public:
  Session(const std::shared_ptr<StatisticsItem>& graph_stats);

  virtual ~Session();

  void AddStateListener(const std::shared_ptr<SessionStateListener>& listener);

  void SetSessionIO(const std::shared_ptr<SessionIO>& io_handle);

  std::shared_ptr<SessionIO> GetSessionIO();

  bool HasSessionIO();

  std::shared_ptr<SessionContext> GetSessionCtx();
  /**
   * @brief will cause session end after current data in engine processed over
   **/
  void Close();

  bool IsClosed();

  /**
   * @brief abort session imediately
   **/
  void Abort();

  bool IsAbort();

  void SetError(std::shared_ptr<FlowUnitError> error);

  std::shared_ptr<FlowUnitError> GetError();

 private:
  std::atomic_bool has_io_{false};
  std::weak_ptr<SessionIO> io_handle_;  // hold by user
  std::shared_ptr<SessionContext> ctx_;

  std::mutex state_lock_;
  std::atomic_bool closed_{false};
  std::atomic_bool abort_{false};

  std::shared_ptr<FlowUnitError> error_;

  std::mutex state_listener_list_lock_;
  std::list<std::weak_ptr<SessionStateListener>> state_listener_list_;
};

class SessionManager {
 public:
  SessionManager();

  virtual ~SessionManager();

  std::shared_ptr<Session> CreateSession(
      const std::shared_ptr<StatisticsItem>& graph_stats);

  void DeleteSession(const SessionId& id);

  std::unordered_map<SessionId, std::weak_ptr<Session>> GetSessions();

 private:
  std::mutex sessions_lock_;
  std::unordered_map<SessionId, std::weak_ptr<Session>> sessions_;
};

}  // namespace modelbox

#endif  // MODELBOX_SESSION_H_