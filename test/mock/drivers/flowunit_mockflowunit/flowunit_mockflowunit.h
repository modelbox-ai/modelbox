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

namespace modelbox {

class MockFlowUnit : public FlowUnit {
 public:
  MockFlowUnit() = default;
  ~MockFlowUnit() override = default;

  MOCK_METHOD(Status, Open, (const std::shared_ptr<Configuration> &opts));
  MOCK_METHOD(Status, Close, ());

  MOCK_METHOD(Status, Process, (std::shared_ptr<DataContext>));
  MOCK_METHOD(Status, DataPre, (std::shared_ptr<DataContext>));
  MOCK_METHOD(Status, DataPost, (std::shared_ptr<DataContext>));
  MOCK_METHOD(Status, DataGroupPre, (std::shared_ptr<DataContext>));
  MOCK_METHOD(Status, DataGroupPost, (std::shared_ptr<DataContext>));
};

class MockFlowUnitFactory : public FlowUnitFactory {
 public:
  MockFlowUnitFactory() = default;
  ~MockFlowUnitFactory() override = default;

  std::map<std::string, std::shared_ptr<FlowUnitDesc>> FlowUnitProbe() override;

  std::shared_ptr<FlowUnit> CreateFlowUnit(const std::string &name,
                                           const std::string &type) override;

  void SetMockFunctionFlowUnit(std::shared_ptr<MockFlowUnit> mock_flowunit);

  void SetMockCreateFlowUnitFunc(
      std::function<std::shared_ptr<FlowUnit>(const std::string &name,
                                              const std::string &type)>
          create_func);
  void SetMockFlowUnitDesc(std::vector<std::shared_ptr<FlowUnitDesc>> descs);

 private:
  std::shared_ptr<MockFlowUnit> bind_mock_flowunit_;
  std::vector<std::shared_ptr<FlowUnitDesc>> flowunit_desc_;
  std::function<std::shared_ptr<FlowUnit>(const std::string &name,
                                          const std::string &type)>
      flowunit_create_func_;
};

class MockDriverFlowUnit : public MockDriver {
 public:
  MockDriverFlowUnit() = default;
  ~MockDriverFlowUnit() override = default;

  static MockDriverFlowUnit *Instance() { return &desc_; };

 private:
  static MockDriverFlowUnit desc_;
};
}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_MOCK_CPU_H_
