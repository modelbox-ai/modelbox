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

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/status.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "python_flowunit.h"
#include "python_module.h"

constexpr const char *FLOWUNIT_NAME = "python";
constexpr const char *FLOWUNIT_DESC = "A python flowunit";

std::mutex kPythonInitLock;
std::shared_ptr<PythonInterpreter> kpythonInterpreter = nullptr;

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<PythonFlowUnitFactory>();
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
  std::lock_guard<std::mutex> lock(kPythonInitLock);
  // Driver Init.
  if (kpythonInterpreter != nullptr) {
    return modelbox::STATUS_OK;
  }

  kpythonInterpreter = std::make_shared<PythonInterpreter>();
  auto ret = kpythonInterpreter->InitModule();
  if (!ret) {
    kpythonInterpreter = nullptr;
  }

  return ret;
}

void DriverFini() {
  // Driver Fini.
  std::lock_guard<std::mutex> lock(kPythonInitLock);
  if (kpythonInterpreter) {
    kpythonInterpreter->ExitModule();
    kpythonInterpreter = nullptr;
  }
}
