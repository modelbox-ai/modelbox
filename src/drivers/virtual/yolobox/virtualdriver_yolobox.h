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

#ifndef MODELBOX_VIRTUALDRIVER_YOLOBOX_H_
#define MODELBOX_VIRTUALDRIVER_YOLOBOX_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/flowunit.h>

constexpr const char *BIND_FLOWUNIT_NAME = "yolov3_postprocess";
constexpr const char *VIRTUAL_YOLO_POST_FLOWUNIT = "yolo_postprocess";
constexpr const char *BIND_FLOWUNIT_VERSION = "1.0.0";
const std::vector<std::string> BIND_FLOWUNIT_TYPE{"cpu"};
// Virtual
class YoloBoxVirtualDriverDesc : public modelbox::VirtualDriverDesc {
 public:
  YoloBoxVirtualDriverDesc() = default;
  ~YoloBoxVirtualDriverDesc() override = default;
};

class YoloBoxVirtualFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  YoloBoxVirtualFlowUnitDesc() = default;
  ~YoloBoxVirtualFlowUnitDesc() override = default;

  void SetConfiguration(std::shared_ptr<modelbox::Configuration> config);
  std::shared_ptr<modelbox::Configuration> GetConfiguration();

 protected:
  std::shared_ptr<modelbox::Configuration> config_;
};

class YoloBoxVirtualDriver
    : public modelbox::VirtualDriver,
      public std::enable_shared_from_this<YoloBoxVirtualDriver> {
 public:
  YoloBoxVirtualDriver() = default;
  ~YoloBoxVirtualDriver() override = default;

  std::shared_ptr<modelbox::DriverFactory> CreateFactory() override;
  std::vector<std::shared_ptr<modelbox::Driver>> GetBindDriver();
  void SetBindDriver(
      std::vector<std::shared_ptr<modelbox::Driver>> driver_list);

 private:
  std::vector<std::shared_ptr<modelbox::Driver>> flowunit_driver_list_;
};

class YoloBoxVirtualFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  YoloBoxVirtualFlowUnitFactory() = default;
  ~YoloBoxVirtualFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override;

  void SetFlowUnitFactory(std::vector<std::shared_ptr<modelbox::DriverFactory>>
                              bind_flowunit_factory_list) override;

  std::string GetVirtualType() override;

  void SetVirtualType(const std::string &virtual_type) override;

  std::shared_ptr<modelbox::Driver> GetDriver() override { return driver_; };

  void SetDriver(std::shared_ptr<modelbox::Driver> driver) override {
    driver_ = driver;
  }

 private:
  std::shared_ptr<modelbox::Driver> driver_;
  std::string virtual_type_;
  modelbox::Status FillInput(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<YoloBoxVirtualFlowUnitDesc> &flowunit_desc);

  modelbox::Status FillOutput(
      std::shared_ptr<modelbox::Configuration> &config,
      std::shared_ptr<YoloBoxVirtualFlowUnitDesc> &flowunit_desc);

  std::vector<std::shared_ptr<modelbox::DriverFactory>>
      bind_flowunit_factory_list_;
};

class YoloBoxVirtualDriverManager : public modelbox::VirtualDriverManager {
 public:
  YoloBoxVirtualDriverManager() = default;
  ~YoloBoxVirtualDriverManager() override = default;

  modelbox::Status Scan(const std::string &path) override;
  modelbox::Status Add(const std::string &file) override;
  modelbox::Status Init(modelbox::Drivers &driver) override;

 private:
  modelbox::Status GetTargetDriverList(modelbox::Drivers &drivers);

  std::vector<std::shared_ptr<modelbox::Driver>> flowunit_driver_list_;
};

#endif  // MODELBOX_VIRTUALDRIVER_YOLOBOX_H_