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

#ifndef MODELBOX_PLUGIN_H_
#define MODELBOX_PLUGIN_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/base/thread_pool.h>

#include <atomic>
#include <mutex>

namespace modelbox {

class Plugin {
 public:
  /**
   * @brief Server plugin
   */
  Plugin();

  virtual ~Plugin();

  /**
   * @brief plugin init
   * @param config plugin configuration
   * @return init result
   */
  virtual bool Init(std::shared_ptr<modelbox::Configuration> config) = 0;

  /**
   * @brief Start plugin
   * @return start result
   */
  virtual bool Start() = 0;

  /**
   * @brief Stop plugin
   * @return stop result
   */
  virtual bool Stop() = 0;

  /**
   * @brief Check plugin  status
   * @return check result
   */
  virtual bool Check();
};

using PluginRecvMsgFunc = std::function<void(
    const std::string &msg_name, const std::shared_ptr<const void> &msg_data,
    size_t msg_len)>;

class PluginMsgRouter {
  friend class Server;

 public:
  /**
   * @brief Register a func to receive msg for the topic
   * @param topic_name Msg for the topic you want
   * @param func To process the msg, should not stuck for long time
   * @return Result of register
   */
  modelbox::Status RegisterRecvFunc(const std::string &topic_name,
                                    const PluginRecvMsgFunc &func);

  /**
   * @brief Route msg to the target async
   * @param topic_name Route msg to the topic
   * @param msg_name Indentify the msg
   * @param msg_data Data route to others
   * @param msg_len Data length
   * @return Result of submit the msg
   */
  modelbox::Status RouteMsg(const std::string &topic_name,
                            const std::string &msg_name,
                            const std::shared_ptr<const void> &msg_data,
                            size_t msg_len);

  /**
   * @brief Get msg router instance
   * @return Instance of msg router
   */
  static std::shared_ptr<PluginMsgRouter> GetInstance();

 private:
  void Clear() {
    std::lock_guard<std::mutex> lck(receivers_lock_);
    receivers_.clear();
  }

  std::mutex receivers_lock_;
  std::map<std::string, std::vector<PluginRecvMsgFunc>> receivers_;
  modelbox::ThreadPool thread_pool_{2, -1, 100};
};

}  // namespace modelbox

extern "C" {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

MODELBOX_DLL_PUBLIC std::shared_ptr<modelbox::Plugin> CreatePlugin();

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

#endif  // MODELBOX_PLUGIN_H_
