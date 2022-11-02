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

#include "java_flowunit.h"

#include "modelbox/device/cpu/device_cpu.h"

JavaFlowUnit::JavaFlowUnit() = default;
JavaFlowUnit::~JavaFlowUnit() = default;

modelbox::Status JavaFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  java_desc_ = std::dynamic_pointer_cast<VirtualJavaFlowUnitDesc>(
      this->GetFlowUnitDesc());

  auto java_entry = java_desc_->GetJarEntry();
  auto config = java_desc_->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  if (config != nullptr) {
    merge_config->Add(*config);
  }
  merge_config->Add(*opts);

  constexpr const char DELIM_CHAR = '@';
  constexpr size_t ENTRY_FILENAME_AND_CLASS_COUNT = 2;
  const auto& entry_list = modelbox::StringSplit(java_entry, DELIM_CHAR);
  if (entry_list.size() != ENTRY_FILENAME_AND_CLASS_COUNT) {
    return {modelbox::STATUS_INVALID, "invalid entry string: " + java_entry};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status JavaFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_FAULT;
}

modelbox::Status JavaFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_FAULT;
}

modelbox::Status JavaFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_FAULT;
}

modelbox::Status JavaFlowUnit::DataGroupPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_FAULT;
}

modelbox::Status JavaFlowUnit::DataGroupPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_FAULT;
}

modelbox::Status JavaFlowUnit::Close() { return modelbox::STATUS_OK; }

void JavaFlowUnit::SetFlowUnitDesc(
    std::shared_ptr<modelbox::FlowUnitDesc> desc) {}

std::shared_ptr<modelbox::FlowUnitDesc> JavaFlowUnit::GetFlowUnitDesc() {
  return nullptr;
}

JavaFlowUnitDesc::JavaFlowUnitDesc() = default;

JavaFlowUnitDesc::~JavaFlowUnitDesc() = default;

void JavaFlowUnitDesc::SetJavaEntry(const std::string& java_entry) {
  java_entry_ = java_entry;
}

std::string JavaFlowUnitDesc::GetJavaEntry() { return java_entry_; }

JavaFlowUnitFactory::JavaFlowUnitFactory() = default;

JavaFlowUnitFactory::~JavaFlowUnitFactory() = default;

std::shared_ptr<modelbox::FlowUnit> JavaFlowUnitFactory::CreateFlowUnit(
    const std::string& unit_name, const std::string& unit_type) {
  auto java_flowunit = std::make_shared<JavaFlowUnit>();
  return java_flowunit;
}

std::string JavaFlowUnitFactory::GetFlowUnitFactoryType() {
  return FLOWUNIT_TYPE;
}

std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
JavaFlowUnitFactory::FlowUnitProbe() {
  return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
}