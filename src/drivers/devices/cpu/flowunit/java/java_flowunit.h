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

#ifndef MODELBOX_FLOWUNIT_JAVA_H_
#define MODELBOX_FLOWUNIT_JAVA_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include "virtualdriver_java.h"

constexpr const char *FLOWUNIT_TYPE = "cpu";

class JavaFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  JavaFlowUnitDesc() = default;
  virtual ~JavaFlowUnitDesc() = default;

  void SetJavaEntry(const std::string java_entry);
  const std::string GetJavaEntry();

  std::string java_entry_;
};

class JavaFlowUnit : public modelbox::FlowUnit {
 public:
  JavaFlowUnit();
  virtual ~JavaFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx);

  virtual void SetFlowUnitDesc(std::shared_ptr<modelbox::FlowUnitDesc> desc);
  virtual std::shared_ptr<modelbox::FlowUnitDesc> GetFlowUnitDesc();

 private:
  std::shared_ptr<VirtualJavaFlowUnitDesc> java_desc_;
};

class JavaFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  JavaFlowUnitFactory() = default;
  virtual ~JavaFlowUnitFactory() = default;

  virtual std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) {
    auto java_flowunit = std::make_shared<JavaFlowUnit>();
    return java_flowunit;
  };

  std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_JAVA_H_
