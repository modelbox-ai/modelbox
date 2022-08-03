
#include "modelbox/python/log.h"

#include <pybind11/embed.h>

#include <utility>

namespace modelbox {

LoggerPython::LoggerPython() = default;
LoggerPython::~LoggerPython() = default;

void LoggerPython::Print(LogLevel level, const char *file, int lineno,
                         const char *func, const char *msg) {
  if (has_exception_ == true) {
    printf("%s\n", msg);
    return;
  }

  try {
    if (pylog_.is_none()) {
      py::print(level, file, lineno, msg);
      return;
    }

    py::gil_scoped_acquire acquire;
    pylog_(level, file, lineno, func, msg);
  } catch (py::error_already_set &ex) {
    if (has_exception_ == false) {
      printf("call function failed, %s, output log to console\n", ex.what());
      has_exception_ = true;
      printf("%s\n", msg);
    }
  }
}

void LoggerPython::SetLogLevel(LogLevel level) { level_ = level; }

LogLevel LoggerPython::GetLogLevel() { return level_; }

void LoggerPython::RegLogFunc(py::function pylog) {
  has_exception_ = false;
  pylog_ = std::move(pylog);
}

LoggerPythonWapper::LoggerPythonWapper() {
  inspect_module_ = py::module::import("inspect");
}

LoggerPythonWapper::~LoggerPythonWapper() { ModelBoxLogger.SetLogger(nullptr); }

void LoggerPythonWapper::RegLogFunc(py::function pylog) {
  logger_python_->RegLogFunc(std::move(pylog));
  ModelBoxLogger.SetLogger(logger_python_);
}

std::shared_ptr<Logger> LoggerPythonWapper::GetLogger() {
  return ModelBoxLogger.GetLogger();
}

void LoggerPythonWapper::SetLogger(const std::shared_ptr<Logger> &logger) {
  ModelBoxLogger.SetLogger(logger);
}

void LoggerPythonWapper::SetLogLevel(LogLevel level) {
  logger_python_->SetLogLevel(level);
}

void LoggerPythonWapper::Print(LogLevel level, const char *msg) {
  auto frame = inspect_module_.attr("currentframe")();
  auto info = inspect_module_.attr("getframeinfo")(frame);

  PrintExt(level, info.attr("filename").cast<std::string>().c_str(),
           info.attr("lineno").cast<int>(),
           info.attr("function").cast<std::string>().c_str(), msg);
}

void LoggerPythonWapper::PrintExt(LogLevel level, const char *file, int lineno,
                                  const char *func, const char *msg) {
  ModelBoxLogger.Print(level, file, lineno, func, "%s", msg);
}

}  // namespace modelbox
