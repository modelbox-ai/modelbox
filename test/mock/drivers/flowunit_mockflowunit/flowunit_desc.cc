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


#include "flowunit_desc.h"

#include <stdio.h>

#include <memory>

#include "modelbox/base/status.h"
#include "modelbox/flowunit.h"
#include "modelbox/base/driver_api_helper.h"
#include "flowunit_mockflowunit.h"
#include "gtest/gtest.h"

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  auto factory = std::make_shared<MockFlowUnitFactory>();
  auto mock_flowunit =
      std::dynamic_pointer_cast<modelbox::MockFlowUnitDriverDesc>(
          MockDriverFlowUnit::Instance()->GetDriverDesc())
          ->GetMockFlowUnit();
  auto create_function =
      std::dynamic_pointer_cast<modelbox::MockFlowUnitDriverDesc>(
          MockDriverFlowUnit::Instance()->GetDriverDesc())
          ->GetMockFlowCreateFunc();
  auto flowunit_desc = std::dynamic_pointer_cast<modelbox::MockFlowUnitDriverDesc>(
          MockDriverFlowUnit::Instance()->GetDriverDesc())
          ->GetMockFlowunitDesc();
  
  factory->SetMockFunctionFlowUnit(mock_flowunit);
  factory->SetMockCreateFlowUnitFunc(create_function);
  factory->SetMockFlowUnitDesc(flowunit_desc);
  return factory;
}

MockDriverFlowUnit *GetDriverMock() { return MockDriverFlowUnit::Instance(); }

void DriverDescription(modelbox::DriverDesc *desc) {
  if (MockDriverFlowUnit::Instance()->GetDriverDesc() == nullptr) {
    printf("\x1B[31m===========================================================\x1B[0m\n");
    printf("\x1B[31m= WARNNING: Driver is not mocked. \x1B[0m\n");
    printf("\x1B[31m= please clean directory : %s \x1B[0m\n", TEST_LIB_DIR);
    printf("\x1B[31m===========================================================\x1B[0m\n");
    FAIL();
    return;
  }

  desc->SetName(MockDriverFlowUnit::Instance()->GetDriverDesc()->GetName());
  desc->SetClass(MockDriverFlowUnit::Instance()->GetDriverDesc()->GetClass());
  desc->SetType(MockDriverFlowUnit::Instance()->GetDriverDesc()->GetType());
  desc->SetDescription(
      MockDriverFlowUnit::Instance()->GetDriverDesc()->GetDescription());
  desc->SetVersion(
      MockDriverFlowUnit::Instance()->GetDriverDesc()->GetVersion());
  desc->SetFilePath(
      MockDriverFlowUnit::Instance()->GetDriverDesc()->GetFilePath());
}

modelbox::Status DriverInit() {
  // Driver Init.
  return MockDriverFlowUnit::Instance()->DriverInit();
}

void DriverFini() {
  // Driver Fini.
  MockDriverFlowUnit::Instance()->DriverFini();
}
