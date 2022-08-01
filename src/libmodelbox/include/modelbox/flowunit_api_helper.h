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


#ifndef MODELBOX_FLOW_UNIT_API_HELPER_H_
#define MODELBOX_FLOW_UNIT_API_HELPER_H_

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/utils.h"
#include "modelbox/flowunit.h"

#pragma GCC visibility push(hidden)
class FlowUnitPluginFactory;
class FlowUnitPluginBase {
 public:
  modelbox::FlowUnitDesc Desc;

  virtual std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit() = 0;
};

template <typename T>
class FlowUnitPlugin : public FlowUnitPluginBase {
 public:
  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit() override {
    return std::make_shared<T>();
  }
};

class FlowUnitList {
 public:
  std::vector<FlowUnitPluginBase *> GetFlowUnitPlugins() {
    return flowunit_plugin_;
  }

  FlowUnitPluginBase *GetFlowUnitPlugin(const std::string &name,
                                        const std::string &type) {
    for (auto &plugin : flowunit_plugin_) {
      if (plugin->Desc.GetFlowUnitName() == name &&
          plugin->Desc.GetDriverDesc()->GetType() == type) {
        return plugin;
      }
    }

    return nullptr;
  }

  void AddFlowUnitPlugin(FlowUnitPluginBase *plugin) {
    flowunit_plugin_.push_back(plugin);
  }

 private:
  std::vector<FlowUnitPluginBase *> flowunit_plugin_;
};

class FlowUnitPluginFactory : public modelbox::FlowUnitFactory {
 public:
  FlowUnitPluginFactory(FlowUnitList *plugin_list)
      : plugin_list_(plugin_list) {}
  ~FlowUnitPluginFactory() override = default;
  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) override {
    auto *plugin = plugin_list_->GetFlowUnitPlugin(unit_name, unit_type);
    if (plugin == nullptr) {
      return nullptr;
    }

    return plugin->CreateFlowUnit();
  }

  std::vector<std::string> GetFlowUnitNames() override {
    std::vector<std::string> result;

    auto plugins = plugin_list_->GetFlowUnitPlugins();
    for (auto &plugin : plugins) {
      auto flowunit_desc = std::make_shared<modelbox::FlowUnitDesc>(plugin->Desc);
      result.push_back(flowunit_desc->GetFlowUnitName());
    }

    return result;
  }

  std::string GetFlowUnitFactoryType() override {
    auto plugins = plugin_list_->GetFlowUnitPlugins();
    if (plugins.size() <= 0) {
      return "";
    }

    return plugins[0]->Desc.GetDriverDesc()->GetType();
  }

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> return_map;
    auto plugins = plugin_list_->GetFlowUnitPlugins();
    for (auto &plugin : plugins) {
      auto flowunit_desc = std::make_shared<modelbox::FlowUnitDesc>(plugin->Desc);
      return_map.insert(
          std::make_pair(flowunit_desc->GetFlowUnitName(), flowunit_desc));
    }
    return return_map;
  }

 private:
  FlowUnitList *plugin_list_;
};

#define MODELBOX_FLOWUNIT_LIST_VAR kFlowUnitList
extern FlowUnitList *ModelBoxGetFlowUnitPluginList() MODELBOX_DLL_LOCAL;
#define MODELBOX_FLOWUNIT_PLUGIN_LIST ModelBoxGetFlowUnitPluginList()
#define MODELBOX_FLOWUNIT_PLUGIN_LIST_DEFINE()                     \
  FlowUnitList MODELBOX_FLOWUNIT_LIST_VAR;                         \
  MODELBOX_DLL_LOCAL FlowUnitList *ModelBoxGetFlowUnitPluginList() { \
    return &MODELBOX_FLOWUNIT_LIST_VAR;                            \
  }

extern std::shared_ptr<modelbox::DriverFactory> FlowUnitCreateFactory()
    MODELBOX_DLL_LOCAL;
#define MODELBOX_FLOWUNIT_PLUGIN_FACTORY_DEFINE()                               \
  MODELBOX_DLL_LOCAL std::shared_ptr<modelbox::DriverFactory>                     \
  FlowUnitCreateFactory() {                                                   \
    std::shared_ptr<modelbox::DriverFactory> factory =                          \
        std::make_shared<FlowUnitPluginFactory>(MODELBOX_FLOWUNIT_PLUGIN_LIST); \
    return factory;                                                           \
  }

#define MODELBOX_FLOWUNIT_PLUGIN_DEFINE() \
  MODELBOX_FLOWUNIT_PLUGIN_LIST_DEFINE()  \
  MODELBOX_FLOWUNIT_PLUGIN_FACTORY_DEFINE()

#define MODELBOX_FLOWUINT_PLUGIN_VAR_NAME(clazz) kFlowUnitPlugin_##clazz
#define MODELBOX_FLOWUINT_PLUGIN_DECLEAR(clazz) \
  FlowUnitPlugin<clazz> MODELBOX_FLOWUINT_PLUGIN_VAR_NAME(clazz);

#define MODELBOX_FLOWUNIT_SETTER(clazz, desc)                              \
  void FlowUnitPluginInit_##clazz(modelbox::FlowUnitDesc &(desc));         \
  auto unused_##clazz = []() {                                             \
    auto func = []() {                                                     \
      FlowUnitPluginInit_##clazz(                                          \
          MODELBOX_FLOWUINT_PLUGIN_VAR_NAME(clazz).Desc);                  \
      auto driver_desc = std::make_shared<modelbox::DriverDesc>(           \
          MODELBOX_DRIVER_PLUGIN->Desc);                                   \
      MODELBOX_FLOWUINT_PLUGIN_VAR_NAME(clazz).Desc.SetDriverDesc(         \
          driver_desc);                                                    \
      MODELBOX_DRIVER_PLUGIN->SetCreateFacotryFunc(FlowUnitCreateFactory); \
      MODELBOX_FLOWUNIT_PLUGIN_LIST->AddFlowUnitPlugin(                    \
          &MODELBOX_FLOWUINT_PLUGIN_VAR_NAME(clazz));                      \
    };                                                                     \
    MODELBOX_DRIVER_PLUGIN_INIT_FUNC(func);                                \
    return true;                                                           \
  }();                                                                     \
  void FlowUnitPluginInit_##clazz(modelbox::FlowUnitDesc &(desc))

/**
 * @brief Define an new flowunit driver
 * @param desc driver description
 */
#define MODELBOX_DRIVER_FLOWUNIT(desc) \
  MODELBOX_FLOWUNIT_PLUGIN_DEFINE()    \
  MODELBOX_DRIVER(desc)

/**
 * @brief Define a new flowunit
 * @param clazz class of flowunit
 * @param desc flowunit description
 */
#define MODELBOX_FLOWUNIT(clazz, desc)     \
  MODELBOX_FLOWUINT_PLUGIN_DECLEAR(clazz); \
  MODELBOX_FLOWUNIT_SETTER(clazz, desc)
#pragma GCC visibility pop

#endif  // MODELBOX_FLOW_UNIT_API_HELPER_H_
