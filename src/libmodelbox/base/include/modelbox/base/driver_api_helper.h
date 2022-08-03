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

#ifndef MODELBOX_DRIVER_API_HELPER_H_
#define MODELBOX_DRIVER_API_HELPER_H_

#include <functional>
#include <utility>

#include "modelbox/base/driver.h"
#include "modelbox/base/utils.h"

#pragma GCC visibility push(hidden)

class DriverPlugin {
 public:
  virtual ~DriverPlugin() = default;

  DriverPlugin &Init(std::function<modelbox::Status()> func) {
    init_func_ = std::move(func);
    return *this;
  }

  std::function<modelbox::Status()> GetInit() { return init_func_; }

  DriverPlugin &Exit(std::function<void()> func) {
    fini_func_ = std::move(func);
    return *this;
  }

  std::function<void()> GetExit() { return fini_func_; }

  DriverPlugin &SetCreateFacotryFunc(
      std::function<std::shared_ptr<modelbox::DriverFactory>()> create_func) {
    create_factory_func_ = std::move(create_func);
    return *this;
  }

  virtual std::shared_ptr<modelbox::DriverFactory> CreateFactory() {
    if (create_factory_func_ == nullptr) {
      MBLOG_ERROR << "Factory is null";
      return nullptr;
    }

    return create_factory_func_();
  }
  modelbox::DriverDesc Desc;

  void AddPluginInitFunc(const std::function<void()> &func) {
    plugin_init_func_.push_back(func);
  }

  void RunPluginInitFunc() {
    for (auto &func : plugin_init_func_) {
      func();
    }
  }

 private:
  std::function<modelbox::Status()> init_func_;
  std::function<void()> fini_func_;
  std::vector<std::function<void()>> plugin_init_func_;
  std::function<std::shared_ptr<modelbox::DriverFactory>()>
      create_factory_func_;
};

extern std::shared_ptr<DriverPlugin> ModelBoxGetDriverPlugin()
    MODELBOX_DLL_LOCAL;

#define MODELBOX_DRIVER_CREATE_FACTORY()                                      \
  extern "C" std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() { \
    auto plugin = ModelBoxGetDriverPlugin();                                  \
    return plugin->CreateFactory();                                           \
  }

#define MODELBOX_DRIVER_DESCRIPTION()                             \
  extern "C" void DriverDescription(modelbox::DriverDesc *desc) { \
    ModelBoxDriverPluginInit();                                   \
    auto plugin = ModelBoxGetDriverPlugin();                      \
    *desc = plugin->Desc;                                         \
    return;                                                       \
  }

#define MODELBOX_DRIVER_INIT()               \
  extern "C" modelbox::Status DriverInit() { \
    ModelBoxDriverPluginInit();              \
    auto plugin = ModelBoxGetDriverPlugin(); \
    auto func = plugin->GetInit();           \
    if (func == nullptr) {                   \
      return modelbox::STATUS_OK;            \
    }                                        \
    return func();                           \
  }

#define MODELBOX_DRIVER_FINI()               \
  extern "C" void DriverFini() {             \
    auto plugin = ModelBoxGetDriverPlugin(); \
    auto func = plugin->GetExit();           \
    if (func == nullptr) {                   \
      return;                                \
    }                                        \
    func();                                  \
  }

#define MODELBOX_DRIVER_PLUGIN ModelBoxGetDriverPlugin()
#define MODELBOX_DRIVER_PLUGIN_INIT_FUNC(func) \
  MODELBOX_DRIVER_PLUGIN->AddPluginInitFunc(func)
#define MODELBOX_DRIVER_PLUGIN_DEFINE()                                        \
  void DriverPluginInit(DriverPlugin &desc);                                   \
  bool ModelBoxDriverPluginInit();                                             \
  MODELBOX_DLL_LOCAL std::shared_ptr<DriverPlugin> ModelBoxGetDriverPlugin() { \
    static std::shared_ptr<DriverPlugin> plugin =                              \
        std::make_shared<DriverPlugin>();                                      \
    return plugin;                                                             \
  }

#define MODELBOX_DRIVER_INIT_FUNC()                    \
  MODELBOX_DLL_LOCAL bool ModelBoxDriverPluginInit() { \
    static bool is_init = false;                       \
    if (is_init) {                                     \
      return true;                                     \
    }                                                  \
    is_init = true;                                    \
    DriverPluginInit(*(MODELBOX_DRIVER_PLUGIN));       \
    MODELBOX_DRIVER_PLUGIN->RunPluginInitFunc();       \
    return true;                                       \
  }

#define MODELBOX_DRIVER_DEFINE(desc) \
  MODELBOX_DRIVER_PLUGIN_DEFINE()    \
  MODELBOX_DRIVER_CREATE_FACTORY()   \
  MODELBOX_DRIVER_DESCRIPTION()      \
  MODELBOX_DRIVER_INIT()             \
  MODELBOX_DRIVER_FINI()

#define MODELBOX_DRIVER_SETTER(desc) void DriverPluginInit(DriverPlugin &(desc))

#define MODELBOX_DRIVER(desc)  \
  MODELBOX_DRIVER_DEFINE(desc) \
  MODELBOX_DRIVER_INIT_FUNC()  \
  MODELBOX_DRIVER_SETTER(desc)

#pragma GCC visibility pop

extern "C" {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

MODELBOX_DLL_PUBLIC std::shared_ptr<modelbox::DriverFactory>
CreateDriverFactory();

MODELBOX_DLL_PUBLIC modelbox::Status DriverInit();

MODELBOX_DLL_PUBLIC void DriverFini();

MODELBOX_DLL_PUBLIC void DriverDescription(modelbox::DriverDesc *desc);

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

#endif  // MODELBOX_DRIVER_API_HELPER_H_