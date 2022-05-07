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


#include "modelbox/device/ascend/device_ascend.h"
#include "modelbox/base/driver_api_helper.h"
#include "modelbox/device/ascend/device_acl_module.h"

#include <stdio.h>
#include <memory>

std::shared_ptr<modelbox::Timer> kDeviceTimer;
std::shared_ptr<Acl> kAcl;

modelbox::Timer *GetTimer() { return kDeviceTimer.get(); }

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<modelbox::AscendFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetClass(modelbox::DRIVER_CLASS_DEVICE);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetName(modelbox::DEVICE_DRIVER_NAME);
  desc->SetDescription(modelbox::DEVICE_DRIVER_DESCRIPTION);
  //desc->SetNodelete(true);
  return;
}

modelbox::Status DriverInit() {
  if (kAcl == nullptr) {
    kAcl = std::make_shared<Acl>();
  }

  if (kDeviceTimer != nullptr) {
    return modelbox::STATUS_OK;
  } 

  kDeviceTimer = std::make_shared<modelbox::Timer>();
  kDeviceTimer->SetName("Ascend-Timer");
  kDeviceTimer->Start();
  return modelbox::STATUS_OK;
}

void DriverFini() {
  if (kDeviceTimer == nullptr) {
    return;
  }

  // Driver Fini.
  kDeviceTimer->Stop();
  kDeviceTimer = nullptr;
}

