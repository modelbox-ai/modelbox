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
  static void SetLogLevel(LogLevel level);
  static void Debug(const py::args& args, const py::kwargs& kwargs);
  static void Info(const py::args& args, const py::kwargs& kwargs);
  static void Notice(const py::args& args, const py::kwargs& kwargs);
  static void Warn(const py::args& args, const py::kwargs& kwargs);
  static void Error(const py::args& args, const py::kwargs& kwargs);
  static void Fatal(const py::args& args, const py::kwargs& kwargs);

 private:
  FlowUnitPythonLog();
  FlowUnitPythonLog(FlowUnitPythonLog &) = delete;
  void operator=(FlowUnitPythonLog &) = delete;
  virtual ~FlowUnitPythonLog();

  void Log(LogLevel level, const py::args &args, const py::kwargs &kwargs);
  py::module inspect_module_;
};

}  // namespace modelbox

#endif  // MODELBOX_PYTHON_MODELBOX_API_LOG_H_
