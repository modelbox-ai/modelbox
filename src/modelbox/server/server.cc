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

#include "server.h"

#include <functional>

#include "config.h"
#include "modelbox/base/configuration.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

namespace modelbox {

Server::~Server() { plugins_.clear(); }

modelbox::Status Server::Init() {
  if (config_ == nullptr) {
    return modelbox::STATUS_BADCONF;
  }

  auto ret = control_.Init(config_);
  if (!ret) {
    MBLOG_ERROR << "Init control failed.";
    return modelbox::STATUS_FAULT;
  }

  ret = GetPluginList();
  if (!ret) {
    MBLOG_ERROR << "server parse config failed";
    return modelbox::STATUS_FAULT;
  }

  for (auto &plugin : plugins_) {
    auto ret = plugin->Init(config_);
    if (!ret) {
      MBLOG_ERROR << "init plugin " << plugin->PluginFile() << " failed";
      return ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status Server::Start() {
  MBLOG_INFO << "app server start";
  if (!control_.Start()) {
    MBLOG_ERROR << "start control failed.";
    return modelbox::STATUS_FAULT;
  }

  std::shared_ptr<ServerPlugin> last_plugin;
  try {
    for (auto &plugin : plugins_) {
      auto ret = plugin->Start();
      if (!ret) {
        MBLOG_ERROR << "Plugin, start failed, " << plugin->PluginFile();
        return modelbox::STATUS_FAULT;
      }
      plugin->SetInit(true);
      last_plugin = plugin;
    }
  } catch (const std::exception &e) {
    if (last_plugin) {
      MBLOG_ERROR << "Plugin, start failed, " << last_plugin->PluginFile()
                  << " reason: " << e.what();
    } else {
      MBLOG_ERROR << "Plugin, start failed, "
                  << " reason: " << e.what();
    }
    return {modelbox::STATUS_FAULT,
            std::string("start plugin failed, ") + e.what()};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status Server::Stop() {
  MBLOG_INFO << "app server stop";
  for (auto &plugin : plugins_) {
    if (plugin->IsInit() == false) {
      continue;
    }
    plugin->Stop();
  }

  auto router = PluginMsgRouter::GetInstance();
  router->Clear();
  control_.Stop();

  return modelbox::STATUS_OK;
}

modelbox::Status Server::Check() {
  for (auto &plugin : plugins_) {
    if (plugin->Check() != modelbox::STATUS_OK) {
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_OK;
}


modelbox::Status Server::GetPluginList() {
  auto plugin_path_list = config_->GetStrings("plugin.files");
  if (plugin_path_list.size() <= 0) {
    MBLOG_ERROR << "can not find plugin path from config file";
    return modelbox::STATUS_FAULT;
  }

  MBLOG_INFO << "plugin list:";
  for (auto path : plugin_path_list) {
    MBLOG_INFO << " " << path;
    path = modelbox_full_path(path);
    auto plugin = ServerPlugin::MakePlugin(path);
    plugins_.push_back(plugin);
  }

  return modelbox::STATUS_OK;
}

}  // namespace modelbox