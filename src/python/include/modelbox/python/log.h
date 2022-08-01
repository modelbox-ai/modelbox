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


#ifndef MODELBOX_PYTHON_LIB_LOG_H_
#define MODELBOX_PYTHON_LIB_LOG_H_

#include <modelbox/base/log.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace modelbox {

class LoggerPython : public Logger {
 public:
  LoggerPython();
  ~LoggerPython() override;

  void Print(LogLevel level, const char *file, int lineno, const char *func,
             const char *msg) override;

  void SetLogLevel(LogLevel level) override;

  LogLevel GetLogLevel() override;

  void RegLogFunc(py::function pylog);

 private:
  py::function pylog_;
  LogLevel level_{LOG_OFF};
  bool has_exception_{false};
};

class LoggerPythonWapper {
 public:
  LoggerPythonWapper();
  virtual ~LoggerPythonWapper();

  void RegLogFunc(py::function pylog);

  void SetLogLevel(LogLevel level);

  std::shared_ptr<Logger> GetLogger();

  void SetLogger(std::shared_ptr<Logger> logger);

  void PrintExt(LogLevel level, const char *file, int lineno, const char *func,
             const char *msg);
  void Print(LogLevel level, const char *msg);

 private:
  std::shared_ptr<LoggerPython> logger_python_ =
      std::make_shared<LoggerPython>();
  py::module inspect_module_;
};

}  // namespace modelbox

#endif  // MODELBOX_PYTHON_LIB_LOG_H_
