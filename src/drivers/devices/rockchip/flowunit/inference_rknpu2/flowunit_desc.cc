/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#include <stdio.h>

#include <memory>

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/status.h"
#include "modelbox/device/rockchip/device_rockchip.h"
#include "modelbox/flowunit.h"
#include "rknpu2_inference_flowunit.h"

constexpr const char *RKNPU2_FLOWUNIT_NAME = "rknpu2_inference";
constexpr const char *RKNPU2_FLOWUNIT_DESC = "A rknpu2 inference flowunit";

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<RKNPU2InferenceFlowUnitFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetName(RKNPU2_FLOWUNIT_NAME);
  desc->SetClass(modelbox::DRIVER_CLASS_INFERENCE);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetDescription(RKNPU2_FLOWUNIT_DESC);
}

modelbox::Status DriverInit() {
  // Driver Init.
  return modelbox::STATUS_OK;
}

void DriverFini() {
  // Driver Fini.
}
