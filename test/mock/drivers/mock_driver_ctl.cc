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


#include "mock_driver_ctl.h"

#include <dlfcn.h>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"

namespace modelbox {

constexpr const char *MOCK_DRIVER_DEVICE_LIB_PREFIX = "libmodelbox-device-";
constexpr const char *MOCK_DRIVER_UNIT_LIB_PREFIX = "libmodelbox-unit-";
constexpr const char *MOCK_DRIVER_GRAPHCONF_LIB_PREFIX = "libmodelbox-graphconf-";

MockDriverDescSetup::MockDriverDescSetup() {}

MockDriverDescSetup::~MockDriverDescSetup() {}

void MockDriverDescSetup::SetDriverDesc(std::shared_ptr<DriverDesc> desc) {
  desc_ = desc;
}

void MockDriverDescSetup::SetDriverFilePath(std::string filepath) {
  file_path_ = filepath;
}

void MockDriverDescSetup::SetDriverHandler(void *handler) {
  driver_handler_ = handler;
}

void MockDriverDescSetup::SetMockDriver(MockDriver *mock_driver) {
  mock_driver_ = mock_driver;
}

std::shared_ptr<modelbox::DriverDesc> MockDriverDescSetup::GetDriverDesc() {
  return desc_;
}

std::string MockDriverDescSetup::GetDriverFilePath() { return file_path_; }

void *MockDriverDescSetup::GetDriverHander() { return driver_handler_; }

MockDriver *MockDriverDescSetup::GetMockDriver() { return mock_driver_; }

void MockDriverDescSetup::Setup() { mock_driver_->SetDriverDesc(desc_); }

MockDriverCtl::MockDriverCtl() {}

MockDriverCtl::~MockDriverCtl() {
  RemoveAllMockDriverFlowUnit();
  RemoveAllMockDriverDevice();
}

std::string MockDriverCtl::GetMockDriverFlowUnitFilePath(
    const std::string &drive_name, const std::string &device_name,
    const std::string &flowunit_dir) {
  std::ostringstream otarget;
  otarget << flowunit_dir << "/" << MOCK_DRIVER_UNIT_LIB_PREFIX << device_name
          << "-" << drive_name << ".so";
  return otarget.str();
}

bool MockDriverCtl::AddMockDriverFlowUnit(std::string drive_name,
                                          std::string device_name,
                                          const MockFlowUnitDriverDesc &desc,
                                          const std::string &copy_path) {
  std::string drive_class_name;
  std::string driver_file =
      GetMockDriverFlowUnitFilePath(drive_name, device_name, copy_path);
  MockDriverDescSetup mock_desc;
  std::string key;
  void *driver_handler = nullptr;
  auto ptr_desc = std::make_shared<MockFlowUnitDriverDesc>();
  typedef MockDriver *(*GetDriverMock)();
  GetDriverMock driver_mock_func = nullptr;

  key = device_name + drive_name;
  if (device_.find(key) != device_.end()) {
    return false;
  }

  if (CopyFile(TEST_FLOWUNIT_MOCKFLOWUNIT_PATH, driver_file, 0, true) ==
      false) {
    MBLOG_ERROR << "Copy file " << TEST_FLOWUNIT_MOCKFLOWUNIT_PATH << " to "
                << driver_file << " failed";
    return false;
  }

  driver_handler = dlopen(driver_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    MBLOG_ERROR << "Open library " << driver_file.c_str() << " failed, "
                << dlerror();
    MBLOG_ERROR << driver_handler;
    goto errout;
  }

  driver_mock_func = (GetDriverMock)dlsym(driver_handler, "GetDriverMock");
  if (driver_handler == nullptr) {
    MBLOG_ERROR << "Cannot find symbol GetDriverMock, " << modelbox::StrError(errno);
    goto errout;
  }

  *ptr_desc = desc;
  mock_desc.SetDriverDesc(ptr_desc);
  mock_desc.SetDriverFilePath(driver_file);
  mock_desc.SetDriverHandler(driver_handler);
  mock_desc.SetMockDriver(driver_mock_func());
  flow_unit_[key] = mock_desc;
  mock_desc.Setup();

  return true;

errout:
  if (driver_handler) {
    dlclose(driver_handler);
  }
  remove(driver_file.c_str());
  return false;
}

bool MockDriverCtl::RemoveMockDriverFlowUnit(std::string drive_name,
                                             std::string device_name) {
  std::string key = device_name + drive_name;
  std::string driver_file;

  if (flow_unit_.find(key) == flow_unit_.end()) {
    return false;
  }

  auto mock_desc = flow_unit_[key];
  UnloadAndRemove(mock_desc);
  flow_unit_.erase(key);
  return false;
}

void MockDriverCtl::RemoveAllMockDriverFlowUnit() {
  for (auto it = flow_unit_.begin(); it != flow_unit_.end();) {
    auto mock_desc = it->second;
    flow_unit_.erase(it++);
    UnloadAndRemove(mock_desc);
  }
}

std::string MockDriverCtl::GetMockDriverDeviceFilePath(
    const std::string &device_name, const std::string &device_dir) {
  std::ostringstream otarget;
  otarget << device_dir << "/" << MOCK_DRIVER_DEVICE_LIB_PREFIX << device_name
          << ".so";
  return otarget.str();
}

bool MockDriverCtl::AddMockDriverDevice(std::string device_name,
                                        const modelbox::DriverDesc &desc,
                                        const std::string &copy_path) {
  std::string drive_class_name;
  std::string driver_file = GetMockDriverDeviceFilePath(device_name, copy_path);
  std::shared_ptr<modelbox::DriverDesc> ptr_desc =
      std::make_shared<MockFlowUnitDriverDesc>();
  MockDriverDescSetup mock_desc;
  std::string key;
  void *driver_handler = nullptr;
  typedef MockDriver *(*GetDriverMock)();
  GetDriverMock driver_mock_func = nullptr;

  key = device_name;
  if (device_.find(key) != device_.end()) {
    return false;
  }

  if (CopyFile(TEST_DEVICE_MOCKDEVICE_PATH, driver_file, 0, true) == false) {
    MBLOG_ERROR << "Copy file " << TEST_DEVICE_MOCKDEVICE_PATH << " to "
                << driver_file << " failed";
    return false;
  }

  driver_handler = dlopen(driver_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    MBLOG_ERROR << "Open library " << driver_file.c_str() << " failed. "
                << dlerror();
    goto errout;
  }

  driver_mock_func = (GetDriverMock)dlsym(driver_handler, "GetDriverMock");
  if (driver_mock_func == nullptr) {
    MBLOG_ERROR << "Cannot find symbol GetDriverMock, " << modelbox::StrError(errno);
    goto errout;
  }

  *ptr_desc = desc;
  mock_desc.SetDriverDesc(ptr_desc);
  mock_desc.SetDriverFilePath(driver_file);
  mock_desc.SetDriverHandler(driver_handler);
  mock_desc.SetMockDriver(driver_mock_func());
  device_[key] = mock_desc;
  mock_desc.Setup();

  return true;
errout:
  if (driver_handler) {
    dlclose(driver_handler);
  }
  remove(driver_file.c_str());
  return false;
}

bool MockDriverCtl::RemoveMockDriverDevice(std::string device_name) {
  std::string key = device_name;

  if (device_.find(key) == device_.end()) {
    return false;
  }

  auto mock_desc = flow_unit_[key];
  UnloadAndRemove(mock_desc);
  device_.erase(key);
  return false;
}

void MockDriverCtl::RemoveAllMockDriverDevice() {
  for (auto it = device_.begin(); it != device_.end();) {
    auto mock_desc = it->second;
    device_.erase(it++);
    UnloadAndRemove(mock_desc);
  }
}

void MockDriverCtl::UnloadAndRemove(MockDriverDescSetup &mock_desc) {
  std::string driver_file;
  void *driver_handler;

  driver_file = mock_desc.GetDriverFilePath();
  driver_handler = mock_desc.GetDriverHander();

  if (driver_handler) {
    dlclose(driver_handler);
    mock_desc.SetDriverHandler(nullptr);
  }
  remove(driver_file.c_str());
}

bool MockDriverCtl::AddMockDriverGraphConf(std::string drive_name,
                                           std::string device_name,
                                           const modelbox::DriverDesc &desc,
                                           const std::string &copy_path) {
  std::string drive_class_name;
  std::string driver_file =
      GetMockDriverGraphConfFilePath(drive_name, copy_path);
  std::shared_ptr<modelbox::DriverDesc> ptr_desc =
      std::make_shared<MockFlowUnitDriverDesc>();
  MockDriverDescSetup mock_desc;
  std::string key;
  void *driver_handler = nullptr;
  typedef MockDriver *(*GetDriverMock)();
  GetDriverMock driver_mock_func = nullptr;

  key = drive_name;
  if (graph_conf_.find(key) != graph_conf_.end()) {
    return false;
  }

  if (CopyFile(TEST_GRAPHCONF_MOCKGRAPHCONF_PATH, driver_file, 0, true) ==
      false) {
    MBLOG_ERROR << "Copy file " << TEST_GRAPHCONF_MOCKGRAPHCONF_PATH << " to "
                << driver_file << " failed";
    return false;
  }

  driver_handler = dlopen(driver_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    MBLOG_ERROR << "Open library " << driver_file.c_str() << " failed, "
                << dlerror();
    MBLOG_ERROR << driver_handler;
    goto errout;
  }

  driver_mock_func = (GetDriverMock)dlsym(driver_handler, "GetDriverMock");
  if (driver_mock_func == nullptr) {
    MBLOG_ERROR << "Cannot find symbol GetDriverMock, " << dlerror();
    goto errout;
  }

  *ptr_desc = desc;
  mock_desc.SetDriverDesc(ptr_desc);
  mock_desc.SetDriverFilePath(driver_file);
  mock_desc.SetDriverHandler(driver_handler);
  mock_desc.SetMockDriver(driver_mock_func());
  flow_unit_[key] = mock_desc;
  mock_desc.Setup();

  return true;

errout:
  if (driver_handler) {
    dlclose(driver_handler);
  }
  remove(driver_file.c_str());
  return false;
}

bool MockDriverCtl::RemoveMockDriverGraphConf(std::string drive_name,
                                              std::string device_name) {
  std::string key = drive_name;

  if (graph_conf_.find(key) == graph_conf_.end()) {
    return false;
  }

  auto mock_desc = graph_conf_[key];
  UnloadAndRemove(mock_desc);
  graph_conf_.erase(key);
  return true;
}

std::string MockDriverCtl::GetMockDriverGraphConfFilePath(
    const std::string &graph_conf_name, const std::string &graph_dir) {
  std::ostringstream otarget;
  otarget << graph_dir << "/" << MOCK_DRIVER_GRAPHCONF_LIB_PREFIX
          << graph_conf_name << ".so";
  return otarget.str();
}

void MockDriverCtl::RemoveAllMockDriverGraphConf() {
  for (auto it = graph_conf_.begin(); it != graph_conf_.end();) {
    auto mock_desc = it->second;
    graph_conf_.erase(it++);
    UnloadAndRemove(mock_desc);
  }
}

}  // namespace modelbox