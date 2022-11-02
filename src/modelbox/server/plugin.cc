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

#include <modelbox/server/plugin.h>

namespace modelbox {

Plugin::Plugin() = default;

Plugin::~Plugin() = default;

bool Plugin::Check() { return true; }

modelbox::Status PluginMsgRouter::RegisterRecvFunc(
    const std::string &topic_name, const PluginRecvMsgFunc &func) {
  std::lock_guard<std::mutex> lock(receivers_lock_);
  auto &func_list = receivers_[topic_name];
  func_list.push_back(func);
  return modelbox::STATUS_OK;
}

modelbox::Status PluginMsgRouter::RouteMsg(
    const std::string &topic_name, const std::string &msg_name,
    const std::shared_ptr<const void> &msg_data, size_t msg_len) {
  std::lock_guard<std::mutex> lock(receivers_lock_);
  if (receivers_.find(topic_name) == receivers_.end()) {
    MBLOG_ERROR << "Topic " << topic_name << " is not found, send msg "
                << msg_name << " failed";
    return modelbox::STATUS_NOTFOUND;
  }

  auto &func_list = receivers_[topic_name];
  auto receive_action = [](const std::vector<PluginRecvMsgFunc> &func_list,
                           const std::string &msg_name,
                           const std::shared_ptr<const void> &msg_data,
                           size_t msg_len) {
    for (const auto &func : func_list) {
      func(msg_name, msg_data, msg_len);
    }
  };

  thread_pool_.Submit(receive_action, func_list, msg_name, msg_data, msg_len);
  return modelbox::STATUS_OK;
}

std::shared_ptr<PluginMsgRouter> PluginMsgRouter::GetInstance() {
  static auto instance = std::make_shared<PluginMsgRouter>();
  return instance;
}

}  // namespace modelbox