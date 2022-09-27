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

#ifndef MODELBOX_SERVER_PLUGIN_H_
#define MODELBOX_SERVER_PLUGIN_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/statistics.h>

#include <memory>
#include <string>

#include "modelbox/server/plugin.h"

namespace modelbox {

class ServerPlugin {
 public:
  ServerPlugin(std::string plugin_path);
  virtual ~ServerPlugin() = default;

  virtual modelbox::Status Init(
      std::shared_ptr<modelbox::Configuration> config) = 0;
  virtual modelbox::Status Start() = 0;
  virtual modelbox::Status Stop() = 0;
  virtual modelbox::Status Check();

  static std::shared_ptr<ServerPlugin> MakePlugin(
      const std::string &plugin_path);

  std::string PluginFile();

 protected:
  std::string plugin_path_;

 private:
  friend class Server;
  bool IsInit();
  void SetInit(bool init);
  bool is_init_{false};
};

/**
 * @brief Dynamic library plugin
 */
class DlPlugin : public ServerPlugin {
 public:
  DlPlugin(const std::string &plugin_path);
  ~DlPlugin() override;

  modelbox::Status Init(
      std::shared_ptr<modelbox::Configuration> config) override;
  modelbox::Status Start() override;
  modelbox::Status Stop() override;
  modelbox::Status Check() override;

 private:
  std::shared_ptr<Plugin> plugin_;
  void *plugin_handler_{nullptr};
};

class JSCtx;

/**
 * @brief Javascript plugin
 */
class JsPlugin : public ServerPlugin {
 public:
  JsPlugin(const std::string &plugin_path);
  ~JsPlugin() override;

  modelbox::Status Init(
      std::shared_ptr<modelbox::Configuration> config) override;
  modelbox::Status Start() override;
  modelbox::Status Stop() override;

  void RegisterStatsNotify(
      const std::string &path_pattern,
      const std::set<modelbox::StatisticsNotifyType> &type_list,
      const std::string &func_name, void *priv_data, size_t delay = 0,
      size_t interval = 0);

  static void AddMap(void *runtime, JsPlugin *plugin);

  static void DelMap(void *runtime);

  static JsPlugin *GetPlugin(void *runtime);

 private:
  modelbox::Status RegisterCFunction();
  modelbox::Status LoadInitCode();

  std::shared_ptr<JSCtx> js_ctx_;
  std::vector<std::shared_ptr<modelbox::StatisticsNotifyCfg>> notify_cfg_list_;

  static std::mutex runtime_to_plugin_lock;
  static std::map<void *, JsPlugin *> runtime_to_plugin;
};

}  // namespace modelbox

#endif  // MODELBOX_SERVER_PLUGIN_H_