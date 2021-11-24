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


#ifndef MODELBOX_PYTHON_MODELBOX_API_LOG_H_
#define MODELBOX_PYTHON_MODELBOX_API_LOG_H_

#include <modelbox/base/log.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace modelbox {

class __attribute__((visibility("hidden"))) FlowUnitPythonLog {
 public:
  static void Init();
  static void Finish();
  static FlowUnitPythonLog &Instance();
  static void Debug(py::args args, py::kwargs kwargs);
  static void Info(py::args args, py::kwargs kwargs);
  static void Notice(py::args args, py::kwargs kwargs);
  static void Warn(py::args args, py::kwargs kwargs);
  static void Error(py::args args, py::kwargs kwargs);
  static void Fatal(py::args args, py::kwargs kwargs);

 private:
  FlowUnitPythonLog();
  FlowUnitPythonLog(FlowUnitPythonLog &) = delete;
  void operator=(FlowUnitPythonLog &) = delete;
  virtual ~FlowUnitPythonLog();

  void Log(LogLevel level, py::args args, py::kwargs kwargs);
  py::module inspect_module_;
};

}  // namespace modelbox

#endif  // MODELBOX_PYTHON_MODELBOX_API_LOG_H_
