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

#include <dlfcn.h>
#include <modelbox/base/config.h>

#include <memory>
#include <utility>

#include "modelbox/base/driver.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"
#include "toml.hpp"

namespace modelbox {

VirtualDriverManager::VirtualDriverManager() = default;

VirtualDriverManager::~VirtualDriverManager() = default;

Status VirtualDriverManager::Add(const std::string &file) { return STATUS_OK; };

Status VirtualDriverManager::Init(Drivers &driver) { return STATUS_OK; };

Status VirtualDriverManager::Scan(const std::vector<std::string> &scan_dirs) {
  auto ret = STATUS_OK;
  for (const auto &dir : scan_dirs) {
    ret = Scan(dir);
    if (ret != STATUS_OK) {
      MBLOG_WARN << "Scan " << dir << " failed, " << ret;
    }
    ret = STATUS_OK;
  }

  return ret;
}

Status VirtualDriverManager::Scan(const std::string &path) { return STATUS_OK; }

std::vector<std::shared_ptr<VirtualDriver>>
VirtualDriverManager::GetAllDriverList() {
  return drivers_list_;
}

void VirtualDriverManager::Clear() { drivers_list_.clear(); };

std::shared_ptr<VirtualDriverDesc> VirtualDriver::GetVirtualDriverDesc() {
  return virtual_driver_desc_;
}

void VirtualDriver::SetVirtualDriverDesc(
    std::shared_ptr<VirtualDriverDesc> desc) {
  virtual_driver_desc_ = std::move(desc);
}

std::vector<std::shared_ptr<Driver>> VirtualDriver::GetBindDriver() {
  return std::vector<std::shared_ptr<Driver>>();
}

std::shared_ptr<DriverFactory> VirtualDriver::CreateFactory() {
  return nullptr;
}

}  // namespace modelbox