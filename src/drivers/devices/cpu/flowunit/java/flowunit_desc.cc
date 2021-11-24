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

#include <memory>
#include <mutex>

#include "java_flowunit.h"
#include "java_module.h"
#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/status.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "java";
constexpr const char *FLOWUNIT_DESC = "A java flowunit";

std::mutex kJavaInitLock;

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<JavaFlowUnitFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetName(FLOWUNIT_NAME);
  desc->SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetDescription(FLOWUNIT_DESC);
  desc->SetNodelete(true);
  desc->SetGlobal(true);
  return;
}

modelbox::Status DriverInit() {
  std::lock_guard<std::mutex> lock(kJavaInitLock);
  // Driver Init.
  if (kJavaJVM != nullptr) {
    return modelbox::STATUS_OK;
  }

  kJavaJVM = std::make_shared<JavaJVM>();
  auto ret = kJavaJVM->InitJVM();
  if (!ret) {
    kJavaJVM = nullptr;
  }

  return ret;
}

void DriverFini() {
  // Driver Fini.
  std::lock_guard<std::mutex> lock(kJavaInitLock);
  if (kJavaJVM) {
    kJavaJVM->ExitJVM();
    kJavaJVM = nullptr;
  }
}
