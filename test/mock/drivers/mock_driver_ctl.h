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


#ifndef MODELBOX_DRIVER_MOCK_CTRL_H_
#define MODELBOX_DRIVER_MOCK_CTRL_H_

#include <map>
#include <utility>

#include "modelbox/base/device.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/timer.h"
#include "modelbox/flow.h"
#include "modelbox/flowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_config.h"


namespace modelbox {

class MockFlowUnit;

class MockDriver {
 public:
  MockDriver() {
    EXPECT_CALL(*this, DriverInit).WillRepeatedly([]() {
      return STATUS_OK;
    });

    EXPECT_CALL(*this, DriverFini).WillRepeatedly([]() {});
  };
  virtual ~MockDriver() = default;
  virtual void SetDriverDesc(std::shared_ptr<modelbox::DriverDesc> desc) {
    desc_ = std::move(desc);
  };

  std::shared_ptr<modelbox::DriverDesc> GetDriverDesc() { return desc_; };
  MOCK_METHOD(modelbox::Status, DriverInit, ());
  MOCK_METHOD(void, DriverFini, ());

 private:
  std::shared_ptr<modelbox::DriverDesc> desc_;
};

class MockFlowUnitDriverDesc : public modelbox::DriverDesc {
 public:
  MockFlowUnitDriverDesc() = default;
  ~MockFlowUnitDriverDesc() override = default;

  void SetMockFlowUnit(std::shared_ptr<MockFlowUnit> mock_flowunit) {
    mock_flowunit_ = std::move(mock_flowunit);
  }

  void SetMockFlowUnit(
      std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string &name,
                                                      const std::string &type)>
          create_func,
      std::vector<std::shared_ptr<modelbox::FlowUnitDesc>> flowunit_descs) {
    flowunit_create_func_ = std::move(create_func);
    mock_flowunit_desc_ = std::move(flowunit_descs);
  }

  void SetMockFlowUnit(
      std::function<std::shared_ptr<modelbox::FlowUnit>(
          const std::string &name, const std::string &type)>
          create_func,
      const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc) {
    flowunit_create_func_ = std::move(create_func);
    mock_flowunit_desc_.push_back(flowunit_desc);
  }

  std::shared_ptr<MockFlowUnit> GetMockFlowUnit() { return mock_flowunit_; }

  std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string &name,
                                                  const std::string &type)>
  GetMockFlowCreateFunc() {
    return flowunit_create_func_;
  }

  std::vector<std::shared_ptr<modelbox::FlowUnitDesc>> GetMockFlowunitDesc() {
    return mock_flowunit_desc_;
  }

 private:
  std::shared_ptr<MockFlowUnit> mock_flowunit_;
  std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string &name,
                                                  const std::string &type)>
      flowunit_create_func_;
  std::vector<std::shared_ptr<modelbox::FlowUnitDesc>> mock_flowunit_desc_;
};

class MockDriverDescSetup {
 public:
  MockDriverDescSetup();
  virtual ~MockDriverDescSetup();
  std::shared_ptr<modelbox::DriverDesc> GetDriverDesc();
  std::string GetDriverFilePath();
  void *GetDriverHander();
  MockDriver *GetMockDriver();

  void SetDriverDesc(std::shared_ptr<modelbox::DriverDesc> desc);
  void SetDriverFilePath(std::string filepath);
  void SetDriverHandler(void *handler);
  void SetMockDriver(MockDriver *mock_driver);

  void Setup();

 private:
  std::string file_path_;
  std::shared_ptr<modelbox::DriverDesc> desc_;
  void *driver_handler_ = nullptr;
  MockDriver *mock_driver_;
};

class MockDriverCtl {
 public:
  MockDriverCtl();
  virtual ~MockDriverCtl();
  bool AddMockDriverFlowUnit(std::string drive_name, std::string device_name,
                             const modelbox::DriverDesc &desc);

  bool AddMockDriverFlowUnit(const std::string &drive_name,
                             const std::string &device_name,
                             const MockFlowUnitDriverDesc &desc,
                             const std::string &copy_path = TEST_LIB_DIR);

  bool RemoveMockDriverFlowUnit(const std::string &drive_name,
                                const std::string &device_name);

  std::string GetMockDriverFlowUnitFilePath(const std::string &drive_name,
                                            const std::string &device_name,
                                            const std::string &flowunit_dir);

  void RemoveAllMockDriverFlowUnit();

  bool AddMockDriverDevice(const std::string &device_name,
                           const modelbox::DriverDesc &desc,
                           const std::string &copy_path = TEST_LIB_DIR);

  bool RemoveMockDriverDevice(const std::string &device_name);

  std::string GetMockDriverDeviceFilePath(const std::string &device_name,
                                          const std::string &device_dir);

  void RemoveAllMockDriverDevice();

  bool AddMockDriverGraphConf(const std::string &drive_name,
                              const std::string &device_name,
                              const modelbox::DriverDesc &desc,
                              const std::string &copy_path = TEST_LIB_DIR);

  bool RemoveMockDriverGraphConf(const std::string &drive_name,
                                 const std::string &device_name);

  std::string GetMockDriverGraphConfFilePath(const std::string &graph_conf_name,
                                             const std::string &graph_dir);

  void RemoveAllMockDriverGraphConf();

 private:
  void UnloadAndRemove(MockDriverDescSetup &mock_desc);
  std::map<std::string, MockDriverDescSetup> flow_unit_;
  std::map<std::string, MockDriverDescSetup> device_;
  std::map<std::string, MockDriverDescSetup> graph_conf_;
};

}  // namespace modelbox
#endif  // MODELBOX_DRIVER_MOCK_CTRL_H_