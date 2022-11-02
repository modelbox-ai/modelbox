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

#ifndef MODELBOX_FLOWUNIT_PYTHON_H_
#define MODELBOX_FLOWUNIT_PYTHON_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <pybind11/pybind11.h>

#include "virtualdriver_python.h"

namespace py = pybind11;

constexpr const char *FLOWUNIT_TYPE = "cpu";

#pragma GCC visibility push(default)

class PythonFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  PythonFlowUnitDesc();
  ~PythonFlowUnitDesc() override;

  void SetPythonEntry(const std::string &python_entry);
  std::string GetPythonEntry();

  std::string python_entry_;
};

class PythonFlowUnit : public modelbox::FlowUnit {
 public:
  PythonFlowUnit();
  ~PythonFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  void SetFlowUnitDesc(std::shared_ptr<modelbox::FlowUnitDesc> desc) override;
  std::shared_ptr<modelbox::FlowUnitDesc> GetFlowUnitDesc() override;

 private:
  std::shared_ptr<VirtualPythonFlowUnitDesc> python_desc_;
  void EnablePythonDebug();

  py::object obj_;
  py::object pydevd_set_trace_;
  py::object python_process_;
  py::object python_data_pre_;
  py::object python_data_post_;
  py::object python_data_group_pre_;
  py::object python_data_group_post_;
  bool is_enable_debug_{false};
};

class PythonFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  PythonFlowUnitFactory();
  ~PythonFlowUnitFactory() override;

  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) override;

  std::string GetFlowUnitFactoryType() override;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override;
};
#pragma GCC visibility pop
#endif  // MODELBOX_FLOWUNIT_PYTHON_H_
