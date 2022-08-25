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

#ifndef MODELBOX_VIRTUAL_DRIVER_INFERENCE_H_
#define MODELBOX_VIRTUAL_DRIVER_INFERENCE_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/flowunit.h>

constexpr const char *BIND_INFERENCE_FLOWUNIT_VERSION = "1.0.0";
constexpr const char *VIRTUAL_INFERENCE_FLOWUNIT = "inference";

// Virtual
class InferenceVirtualDriverDesc : public modelbox::VirtualDriverDesc {
 public:
  InferenceVirtualDriverDesc() = default;
  ~InferenceVirtualDriverDesc() override = default;
};

class VirtualInferenceFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  VirtualInferenceFlowUnitDesc() = default;
  ~VirtualInferenceFlowUnitDesc() override = default;

  void SetModelEntry(std::string model_entry);
  std::string GetModelEntry();

  void SetConfiguration(const std::shared_ptr<modelbox::Configuration> &config);
  std::shared_ptr<modelbox::Configuration> GetConfiguration();

 protected:
  std::string model_entry_;
  std::shared_ptr<modelbox::Configuration> config_;
};

class InferenceVirtualDriver : public modelbox::VirtualDriver {
 public:
  InferenceVirtualDriver() = default;
  ~InferenceVirtualDriver() override = default;

  std::shared_ptr<modelbox::DriverFactory> CreateFactory() override;
  std::vector<std::shared_ptr<modelbox::Driver>> GetBindDriver();
  void SetBindDriver(
      const std::vector<std::shared_ptr<modelbox::Driver>> &driver_list);

 private:
  std::vector<std::shared_ptr<modelbox::Driver>>
      inference_flowunit_driver_list_;
};

class VirtualInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  VirtualInferenceFlowUnitFactory() = default;
  ~VirtualInferenceFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override;

  void SetFlowUnitFactory(
      const std::vector<std::shared_ptr<modelbox::DriverFactory>>
          &bind_flowunit_factory_list) override;

  std::string GetVirtualType() override;
  void SetVirtualType(const std::string &virtual_type) override;

  std::shared_ptr<modelbox::Driver> GetDriver() override { return driver_; };

  void SetDriver(const std::shared_ptr<modelbox::Driver> &driver) override {
    driver_ = driver;
  }

 private:
  modelbox::Status FillItem(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc,
      const std::string &device, const std::string &type);
  modelbox::Status FillBaseInfo(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc,
      const std::string &toml_file, std::string *device);
  void FillFlowUnitType(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<VirtualInferenceFlowUnitDesc> &flowunit_desc);
  std::shared_ptr<modelbox::Driver> driver_;
  std::vector<std::shared_ptr<modelbox::DriverFactory>>
      bind_flowunit_factory_list_;
  std::string virtual_type_;
};

class InferenceVirtualDriverManager : public modelbox::VirtualDriverManager {
 public:
  InferenceVirtualDriverManager() = default;
  ~InferenceVirtualDriverManager() override = default;

  modelbox::Status Scan(const std::string &path) override;
  modelbox::Status Add(const std::string &file) override;
  modelbox::Status Init(modelbox::Drivers &driver) override;

 private:
  modelbox::Status BindBaseDriver(modelbox::Drivers &driver);
  std::vector<std::shared_ptr<modelbox::Driver>>
      inference_flowunit_driver_list_;
};

#endif  // MODELBOX_VIRTUAL_DRIVER_PYTHON_H_
