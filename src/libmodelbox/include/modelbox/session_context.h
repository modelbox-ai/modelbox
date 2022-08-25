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

#ifndef MODELBOX_SESSION_CONTEXT_H_
#define MODELBOX_SESSION_CONTEXT_H_

#include <memory>
#include <unordered_map>

#include "modelbox/buffer_list.h"
#include "modelbox/statistics.h"

namespace modelbox {
class ExternalDataMapImpl;
using OutputBufferList =
    std::unordered_map<std::string, std::shared_ptr<BufferList>>;

enum class SessionContexStatsType { SESSION, GRAPH };

class SessionContext {
 public:
  /**
   * @brief Session context
   * @param graph_stats Statistics for graph
   */
  SessionContext(const std::shared_ptr<StatisticsItem> &graph_stats = nullptr);

  virtual ~SessionContext();

  /**
   * @brief Set private data to session context
   * @param key private data key
   * @param private_content private data
   * @param type_id private data typeid
   */
  void SetPrivate(const std::string &key, std::shared_ptr<void> private_content,
                  std::size_t type_id = 0);

  /**
   * @brief Get private data from session context
   * @param key private data key
   * @return private data
   */
  std::shared_ptr<void> GetPrivate(const std::string &key);

  /**
   * @brief Get private data from session context
   * @param key private data key
   * @return private data
   */
  template <typename T>
  inline std::shared_ptr<T> GetPrivate(const std::string &key) {
    return std::static_pointer_cast<T>(GetPrivate(key));
  }

  /**
   * @brief Get private data typeid from session context
   * @param key private data key
   * @return private data typeid
   */
  std::size_t GetPrivateType(const std::string &key);

  /**
   * @brief Set session id
   * @param session_id session id
   */
  void SetSessionId(const std::string &session_id);

  /**
   * @brief Get session id
   * @return session_id session id
   */
  std::string GetSessionId();

  /**
   * @brief Get session configuration object
   * @return configuration
   */
  std::shared_ptr<Configuration> GetConfig();

  /**
   * @brief Set error to session
   * @param error run error
   */
  void SetError(std::shared_ptr<FlowUnitError> error);

  std::shared_ptr<FlowUnitError> GetError();
  /**
   * @brief Get statistics ctx in nodes.session_id level
   * @return Statistics ctx
   */
  std::shared_ptr<StatisticsItem> GetStatistics(
      SessionContexStatsType type = SessionContexStatsType::SESSION);

 private:
  std::mutex private_map_lock_;
  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;
  std::unordered_map<std::string, std::size_t> private_map_type_;
  std::string session_id_;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<FlowUnitError> error_;
  std::shared_ptr<StatisticsItem> graph_stats_;
  std::shared_ptr<StatisticsItem> graph_session_stats_;
};

}  // namespace modelbox
#endif  // MODELBOX_SESSION_CONTEXT_H_
