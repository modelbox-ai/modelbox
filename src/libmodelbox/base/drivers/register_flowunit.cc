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

#include "modelbox/base/register_flowunit.h"

constexpr const char *VIRTUAL_TYPE = "register_flowunit";

namespace modelbox {

RegisterFlowUnit::RegisterFlowUnit(const std::string &name) { name_ = name; }

RegisterFlowUnit::~RegisterFlowUnit() {}
Status RegisterFlowUnit::Open(const std::shared_ptr<Configuration> &config) {
  return STATUS_OK;
}

Status RegisterFlowUnit::Close() { return STATUS_OK; }

void RegisterFlowUnit::SetCallBack(
    std::function<StatusCode(std::shared_ptr<DataContext>)> callback) {
  callback_ = callback;
}

std::function<StatusCode(std::shared_ptr<DataContext>)>
RegisterFlowUnit::GetCallBack() {
  return callback_;
}

Status RegisterFlowUnit::Process(std::shared_ptr<DataContext> data_context) {
  auto callback = GetCallBack();
  if (callback) {
    return callback(data_context);
  }
  return STATUS_INVALID;
}

RegisterFlowUnitFactory::RegisterFlowUnitFactory(
    const std::string unit_name, std::set<std::string> inputs,
    std::set<std::string> outputs,
    std::function<StatusCode(std::shared_ptr<DataContext>)> &callback)
    : unit_name_(unit_name),
      input_ports_(inputs),
      output_ports_(outputs),
      callback_(callback) {
  if (STATUS_SUCCESS != Init()) {
    MBLOG_ERROR << "failed init RegisterFlowUnitFactory";
  }
}

std::shared_ptr<FlowUnit> RegisterFlowUnitFactory::CreateFlowUnit(
    const std::string &name, const std::string &unit_type) {
  auto register_flowunit = std::make_shared<RegisterFlowUnit>(name);
  auto iter = desc_map_.find(name);
  if (iter == desc_map_.end()) {
    MBLOG_ERROR << "failed find flowunit desc for " << name;
    return nullptr;
  }
  register_flowunit->SetFlowUnitDesc(desc_map_[name]);
  register_flowunit->SetCallBack(callback_);
  return register_flowunit;
}

Status RegisterFlowUnitFactory::Init() {
  std::shared_ptr<DriverDesc> driver_desc = std::make_shared<DriverDesc>();
  std::shared_ptr<Driver> driver = std::make_shared<Driver>();
  driver->SetDriverDesc(driver_desc);

  std::shared_ptr<FlowUnitDesc> desc = std::make_shared<FlowUnitDesc>();
  if (desc == nullptr) {
    return STATUS_FAULT;
  }
  SetDriver(driver);

  desc->SetFlowUnitName(unit_name_);
  for (auto &iter : input_ports_) {
    desc->AddFlowUnitInput(FlowUnitInput(iter, "cpu"));
  }

  for (auto &iter : output_ports_) {
    desc->AddFlowUnitOutput(FlowUnitOutput(iter, "cpu"));
  }
  desc->SetVirtualType(VIRTUAL_TYPE);
  desc_map_.emplace(unit_name_, desc);
  return STATUS_SUCCESS;
}

}  // namespace modelbox
