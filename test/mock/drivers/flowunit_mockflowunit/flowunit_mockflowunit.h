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

#ifndef MODELBOX_FLOWUNIT_MOCK_CPU_H_
#define MODELBOX_FLOWUNIT_MOCK_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/flowunit.h"

class MockFlowUnit : public modelbox::FlowUnit {
 public:
  MockFlowUnit() = default;
  ~MockFlowUnit() override = default;

  MOCK_METHOD(modelbox::Status, Open,
              (const std::shared_ptr<modelbox::Configuration> &opts));
  MOCK_METHOD(modelbox::Status, Close, ());

  MOCK_METHOD(modelbox::Status, Process,
              (std::shared_ptr<modelbox::DataContext>));
  MOCK_METHOD(modelbox::Status, DataPre,
              (std::shared_ptr<modelbox::DataContext>));
  MOCK_METHOD(modelbox::Status, DataPost,
              (std::shared_ptr<modelbox::DataContext>));
  MOCK_METHOD(modelbox::Status, DataGroupPre,
              (std::shared_ptr<modelbox::DataContext>));
  MOCK_METHOD(modelbox::Status, DataGroupPost,
              (std::shared_ptr<modelbox::DataContext>));
};

class MockFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  MockFlowUnitFactory() = default;
  ~MockFlowUnitFactory() override = default;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override;

  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &name, const std::string &type) override;

  void SetMockFunctionFlowUnit(std::shared_ptr<MockFlowUnit> mock_flowunit);

  void SetMockCreateFlowUnitFunc(
      std::function<std::shared_ptr<modelbox::FlowUnit>(
          const std::string &name, const std::string &type)>
          create_func);
  void SetMockFlowUnitDesc(
      std::vector<std::shared_ptr<modelbox::FlowUnitDesc>> descs);

 private:
  std::shared_ptr<MockFlowUnit> bind_mock_flowunit_;
  std::vector<std::shared_ptr<modelbox::FlowUnitDesc>> flowunit_desc_;
  std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string &name,
                                                    const std::string &type)>
      flowunit_create_func_;
};

class MockDriverFlowUnit : public modelbox::MockDriver {
 public:
  MockDriverFlowUnit() = default;
  ~MockDriverFlowUnit() override = default;

  static MockDriverFlowUnit *Instance() { return &desc_; };

 private:
  static MockDriverFlowUnit desc_;
};

#endif  // MODELBOX_FLOWUNIT_MOCK_CPU_H_
