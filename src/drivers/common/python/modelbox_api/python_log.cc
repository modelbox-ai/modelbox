
#include "python_log.h"

#include "modelbox/base/log.h"

namespace modelbox {

static std::shared_ptr<FlowUnitPythonLog> kInst = nullptr;

void FlowUnitPythonLog::Init() {
  if (!kInst) {
    auto* data = new FlowUnitPythonLog();
    py::gil_scoped_acquire interpreter_guard{};
    data->inspect_module_ = py::module::import("inspect");
    kInst = std::shared_ptr<FlowUnitPythonLog>(
        data, [](FlowUnitPythonLog* ptr) { delete ptr; });
  }
}

void FlowUnitPythonLog::Finish() { kInst = nullptr; }

FlowUnitPythonLog& FlowUnitPythonLog::Instance() {
  if (!kInst) {
    Init();
  }
  return *kInst;
}

FlowUnitPythonLog::FlowUnitPythonLog() = default;

FlowUnitPythonLog::~FlowUnitPythonLog() {
  // Avoid crash when log destroy.
  if (inspect_module_.ref_count() == 1) {
    inspect_module_.release();
  }
}

void FlowUnitPythonLog::SetLogLevel(LogLevel level) {
  ModelBoxLogger.GetLogger()->SetLogLevel(level);
}

void FlowUnitPythonLog::Debug(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_DEBUG, args, kwargs);
}

void FlowUnitPythonLog::Info(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_INFO, args, kwargs);
}

void FlowUnitPythonLog::Notice(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_NOTICE, args, kwargs);
}

void FlowUnitPythonLog::Warn(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_WARN, args, kwargs);
}

void FlowUnitPythonLog::Error(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_ERROR, args, kwargs);
}

void FlowUnitPythonLog::Fatal(py::args args, py::kwargs kwargs) {
  Instance().Log(modelbox::LOG_FATAL, args, kwargs);
}

void FlowUnitPythonLog::Log(LogLevel level, py::args args, py::kwargs kwargs) {
  if (ModelBoxLogger.CanLog(level) == false) {
    return;
  }

  std::string msg{};
  for (unsigned int i = 0; i < args.size(); i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += pybind11::str(args[i]);
  }

  auto frame = inspect_module_.attr("currentframe")();
  auto info = inspect_module_.attr("getframeinfo")(frame);

  auto filename = info.attr("filename").cast<std::string>();
  int last_slash_index = filename.find_last_of("/");
  const char* s_filename = filename.c_str();
  if (last_slash_index > 0) {
    s_filename = filename.c_str() + last_slash_index + 1;
  }

  ModelBoxLogger.Print(level, s_filename, info.attr("lineno").cast<int>(),
                       info.attr("function").cast<std::string>().c_str(), "%s",
                       msg.c_str());
}

}  // namespace modelbox
