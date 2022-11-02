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

#include "server_plugin.h"

#include <dlfcn.h>

#include <functional>
#include <map>
#include <utility>

namespace modelbox {

typedef std::shared_ptr<Plugin> (*CreatePluginFunc)();

std::map<std::string, std::function<std::shared_ptr<ServerPlugin>(
                          const std::string &plugin_path)>>
    plugin_construct_map = {{".so",
                             [](const std::string &plugin_path) {
                               return std::make_shared<DlPlugin>(plugin_path);
                             }},
                            {".js", [](const std::string &plugin_path) {
                               return std::make_shared<JsPlugin>(plugin_path);
                             }}};

ServerPlugin::ServerPlugin(std::string plugin_path)
    : plugin_path_(std::move(plugin_path)) {}

ServerPlugin::~ServerPlugin() = default;

std::shared_ptr<ServerPlugin> ServerPlugin::MakePlugin(
    const std::string &plugin_path) {
  auto suffix_pos = plugin_path.find_last_of('.');
  if (suffix_pos != std::string::npos) {
    auto suffix = plugin_path.substr(suffix_pos);
    auto type_item = plugin_construct_map.find(suffix);
    if (type_item != plugin_construct_map.end()) {
      return type_item->second(plugin_path);
    }
  }

  return std::make_shared<DlPlugin>(plugin_path);
}

modelbox::Status ServerPlugin::Check() { return modelbox::STATUS_OK; }

std::string ServerPlugin::PluginFile() { return plugin_path_; }

bool ServerPlugin::IsInit() { return is_init_; }

void ServerPlugin::SetInit(bool init) { is_init_ = init; }

DlPlugin::DlPlugin(const std::string &plugin_path)
    : ServerPlugin(plugin_path) {}

DlPlugin::~DlPlugin() {
  plugin_.reset();
  if (plugin_handler_ == nullptr) {
    return;
  }

  dlclose(plugin_handler_);
}

modelbox::Status DlPlugin::Init(
    std::shared_ptr<modelbox::Configuration> config) {
  modelbox::Status ret = modelbox::STATUS_FAULT;

  plugin_handler_ = dlopen(plugin_path_.c_str(), RTLD_NOW);
  if (plugin_handler_ == nullptr) {
    std::string errmsg = "Open library " + plugin_path_ + " failed";
    auto *dlerr_msg = dlerror();
    if (dlerr_msg != nullptr) {
      errmsg += ", ";
      errmsg += dlerr_msg;
    }
    MBLOG_ERROR << errmsg;
    return {modelbox::STATUS_FAULT, errmsg};
  }

  Defer {
    if (!ret && plugin_handler_ != nullptr) {
      dlclose(plugin_handler_);
      plugin_handler_ = nullptr;
    }
  };

  CreatePluginFunc create_plugin_func;
  create_plugin_func = (CreatePluginFunc)dlsym(plugin_handler_, "CreatePlugin");

  if (create_plugin_func == nullptr) {
    std::string errmsg = "Cannot find symbol CreatePlugin";
    auto *dlerr_msg = dlerror();
    if (dlerr_msg != nullptr) {
      errmsg += ", ";
      errmsg += dlerr_msg;
    }
    MBLOG_ERROR << errmsg;
    return {modelbox::STATUS_FAULT, errmsg};
  }

  auto plugin = create_plugin_func();
  if (plugin == nullptr) {
    MBLOG_ERROR << "create plugin failed";
    return modelbox::STATUS_FAULT;
  }

  if (!plugin->Init(config)) {
    return modelbox::STATUS_FAULT;
  }

  plugin_ = plugin;
  ret = modelbox::STATUS_OK;

  return ret;
}

modelbox::Status DlPlugin::Start() {
  if (plugin_ == nullptr) {
    MBLOG_ERROR << "plugin is null";
    return modelbox::STATUS_FAULT;
  }

  return plugin_->Start();
}

modelbox::Status DlPlugin::Stop() {
  if (plugin_ == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  return plugin_->Stop();
}

modelbox::Status DlPlugin::Check() {
  if (plugin_ == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  if (plugin_->Check() == false) {
    return STATUS_FAULT;
  }

  return STATUS_OK;
}

}  // namespace modelbox