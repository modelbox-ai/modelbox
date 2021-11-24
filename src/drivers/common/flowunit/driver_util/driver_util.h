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


#ifndef MODELBOX_FLOWUNIT_DRIVER_UTIL_H_
#define MODELBOX_FLOWUNIT_DRIVER_UTIL_H_

#include <modelbox/base/config.h>
#include <modelbox/base/device.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>

namespace driverutil {

template <class T>
modelbox::Status GetPlugin(
    const std::string &driver_class, std::shared_ptr<modelbox::Drivers> &drivers,
    std::vector<std::shared_ptr<modelbox::DriverFactory>> &factories,
    std::map<std::string, std::shared_ptr<T>> &plugins) {
  auto driver_list = drivers->GetDriverListByClass(driver_class);
  for (auto &driver : driver_list) {
    auto driver_desc = driver->GetDriverDesc();
    if (driver_desc == nullptr) {
      continue;
    }

    auto name = driver_desc->GetName();
    auto factory = driver->CreateFactory();
    if (factory == nullptr) {
      MBLOG_ERROR << "Plugin : " << name << " factory create failed";
      continue;
    }

    auto plugin = std::dynamic_pointer_cast<T>(factory->GetDriver());
    if (plugin == nullptr) {
      MBLOG_ERROR << "plugin : " << name << " is not derived from "
                  << typeid(T).name();
      continue;
    }

    plugins[name] = plugin;
    factories.push_back(factory);
    MBLOG_INFO << "Add plugin : " << name;
  }

  return modelbox::STATUS_SUCCESS;
}

}  // namespace driverutil

#endif  // MODELBOX_FLOWUNIT_DRIVER_UTIL_H_