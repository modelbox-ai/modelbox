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

#include <stdio.h>

#include <memory>

#include "modelbox/device/rockchip/device_rockchip.h"

namespace modelbox {

std::shared_ptr<Timer> kRKDeviceTimer;

Timer *GetTimer() { return kRKDeviceTimer.get(); }

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<RockChipFactory>();
  return factory;
}

void DriverDescription(DriverDesc *desc) {
  desc->SetClass(DRIVER_CLASS_DEVICE);
  desc->SetType(DEVICE_TYPE);
  desc->SetName(DEVICE_DRIVER_NAME);
  desc->SetDescription(DEVICE_DRIVER_DESCRIPTION);
}

Status DriverInit() {
  if (kRKDeviceTimer != nullptr) {
    return STATUS_OK;
  }

  kRKDeviceTimer = std::make_shared<Timer>();
  if (kRKDeviceTimer == nullptr) {
    auto msg = std::string("failed to make timer");
    MBLOG_ERROR << msg;
    return {STATUS_FAULT, msg};
  }

  kRKDeviceTimer->SetName("RockChip-Timer");
  kRKDeviceTimer->Start();
  return STATUS_OK;
}

void DriverFini() {
  if (kRKDeviceTimer == nullptr) {
    return;
  }

  // Driver Fini.
  kRKDeviceTimer->Stop();
  kRKDeviceTimer = nullptr;
}

}  // namespace modelbox
