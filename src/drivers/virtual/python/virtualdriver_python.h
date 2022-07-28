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


#ifndef MODELBOX_VIRTUAL_DRIVER_PYTHON_H_
#define MODELBOX_VIRTUAL_DRIVER_PYTHON_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/flowunit.h>

constexpr const char *BIND_PYTHON_FLOWUNIT_NAME = "python";
constexpr const char *BIND_PYTHON_FLOWUNIT_VERSION = "1.0.0";
const std::vector<std::string> BIND_PYTHON_FLOWUNIT_TYPE{"cpu"};

// Virtual
class PythonVirtualDriverDesc : public modelbox::VirtualDriverDesc {
 public:
  PythonVirtualDriverDesc() = default;
  virtual ~PythonVirtualDriverDesc() = default;
};

class VirtualPythonFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  VirtualPythonFlowUnitDesc() = default;
  virtual ~VirtualPythonFlowUnitDesc() = default;

  void SetPythonEntry(std::string python_entry);
  std::string GetPythonEntry();

  void SetConfiguration(std::shared_ptr<modelbox::Configuration> config);
  std::shared_ptr<modelbox::Configuration> GetConfiguration();

  void SetPythonFilePath(const std::string &path) { python_file_path_ = path; }
  const std::string &GetPythonFilePath() const { return python_file_path_; }

 protected:
  std::string python_entry_;
  std::shared_ptr<modelbox::Configuration> config_;
  std::string python_file_path_;
};

class PythonVirtualDriver
    : public modelbox::VirtualDriver,
      public std::enable_shared_from_this<PythonVirtualDriver> {
 public:
  PythonVirtualDriver() = default;
  virtual ~PythonVirtualDriver() = default;

  std::shared_ptr<modelbox::DriverFactory> CreateFactory();
  std::vector<std::shared_ptr<modelbox::Driver>> GetBindDriver();
  void SetBindDriver(std::vector<std::shared_ptr<modelbox::Driver>> driver_list);

 private:
  std::vector<std::shared_ptr<modelbox::Driver>> python_flowunit_driver_;
};

class VirtualPythonFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  VirtualPythonFlowUnitFactory() = default;
  virtual ~VirtualPythonFlowUnitFactory() = default;

  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type);

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe();

  void SetFlowUnitFactory(std::vector<std::shared_ptr<modelbox::DriverFactory>>
                              bind_flowunit_factory_list);

  std::shared_ptr<modelbox::Driver> GetDriver() { return driver_; };

  virtual void SetDriver(std::shared_ptr<modelbox::Driver> driver) {
    driver_ = driver;
  }

 private:
  modelbox::Status FillInput(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
      const std::string &device);
  modelbox::Status FillOutput(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
      const std::string &device);
  modelbox::Status FillBaseInfo(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc,
      const std::string &toml_file, std::string *device);
  void FillFlowUnitType(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualPythonFlowUnitDesc> &flowunit_desc);
  std::shared_ptr<modelbox::Driver> driver_;
  std::vector<std::shared_ptr<modelbox::DriverFactory>>
      bind_flowunit_factory_list_;
};

class PythonVirtualDriverManager : public modelbox::VirtualDriverManager {
 public:
  PythonVirtualDriverManager() = default;
  virtual ~PythonVirtualDriverManager() = default;

  modelbox::Status Scan(const std::string &path);
  modelbox::Status Add(const std::string &file);
  modelbox::Status Init(modelbox::Drivers &driver);

 private:
  modelbox::Status BindBaseDriver(modelbox::Drivers &driver);
  std::vector<std::shared_ptr<modelbox::Driver>> python_flowunit_driver_list_;
};

#endif  // MODELBOX_VIRTUAL_DRIVER_PYTHON_H_