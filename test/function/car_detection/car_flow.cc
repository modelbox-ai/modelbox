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


#include "car_flow.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

namespace modelbox {

CarFlow::CarFlow() : flow_(std::make_shared<Flow>()) {}

CarFlow::~CarFlow() {
  flow_ = nullptr;
  ctl_ = nullptr;
}

void CarFlow::Clear() { flow_ = nullptr; }

Status CarFlow::Init(const std::string& graphFilePath) {
  ctl_ = GetMockFlowCtl();

  modelbox::DriverDesc desc;
  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl_->AddMockDriverDevice("cpu", desc, std::string(TEST_DRIVER_DIR));

  desc.SetClass("DRIVER-GRAPHCONF");
  desc.SetType("GRAPHVIZ");
  desc.SetName("GRAPHCONF-GRAPHVIZ");
  desc.SetDescription("graph config parse graphviz");
  desc.SetVersion("0.1.0");
  std::string file_path_graph =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-graphconf-graphviz.so";
  desc.SetFilePath(file_path_graph);

  ctl_->AddMockDriverGraphConf("graphviz", "", desc,
                               std::string(TEST_DRIVER_DIR));
  auto status = flow_->Init(graphFilePath);
  return status;
}

Status CarFlow::Build() { return flow_->Build(); }

void CarFlow::Run() { flow_->RunAsync(); }

void CarFlow::Wait(const uint64_t millisecond) { flow_->Wait(millisecond); }

void CarFlow::Destroy() {}

std::shared_ptr<MockDriverCtl> CarFlow::GetMockFlowCtl() {
  if (ctl_ == nullptr) {
    ctl_ = std::make_shared<MockDriverCtl>();
  }
  return ctl_;
}

}  // namespace modelbox