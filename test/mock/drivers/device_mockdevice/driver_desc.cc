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


#include "driver_desc.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"

#include <stdio.h>
#include <memory>

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<modelbox::MockDeviceFactory>();
  return factory;
}

modelbox::MockDriverDevice *GetDriverMock() { return modelbox::MockDriverDevice::Instance(); }

void DriverDescription(modelbox::DriverDesc *desc) {
  if (GetDriverMock() == nullptr) {
    MBLOG_WARN << "Mock is invalid.";
    return;
  }

  if (GetDriverMock()->GetDriverDesc() == nullptr) {
    MBLOG_WARN << "Mock driver is invalid.";
    return;
  }

  if (GetDriverMock()->GetDriverDesc() == nullptr) {
    MBLOG_WARN << "Mock driver is invalid.";
    return;
  }

  desc->SetName(GetDriverMock()->GetDriverDesc()->GetName());
  desc->SetClass(GetDriverMock()->GetDriverDesc()->GetClass());
  desc->SetType(GetDriverMock()->GetDriverDesc()->GetType());
  desc->SetDescription(
      GetDriverMock()->GetDriverDesc()->GetDescription());
  desc->SetVersion(GetDriverMock()->GetDriverDesc()->GetVersion());
  desc->SetFilePath(
      GetDriverMock()->GetDriverDesc()->GetFilePath());
}

modelbox::Status DriverInit() {
  // Driver Init.
  return modelbox::MockDriverDevice::Instance()->DriverInit();
}

void DriverFini() {
  // Driver Fini.
  modelbox::MockDriverDevice::Instance()->DriverFini();
}

