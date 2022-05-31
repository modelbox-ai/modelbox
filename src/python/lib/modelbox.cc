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

#include <modelbox/flow.h>
#include <modelbox/flow_graph_desc.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>
#include <functional>

#include "modelbox/python/log.h"
#include "modelbox_api.h"

namespace modelbox {

class PyLogger : public Logger {
 public:
  using Logger::Logger;
  virtual ~PyLogger() = default;
  LogLevel GetLogLevel() {
    PYBIND11_OVERLOAD_PURE(LogLevel, Logger, GetLogLevel);
  }
};

void SetUpLog(pybind11::module &m) {
  py::class_<Logger, PyLogger, std::shared_ptr<Logger>>(m, "Logger")
      .def(py::init<>())
      .def("set_log_level", &Logger::SetLogLevel)
      .def("get_log_level", &Logger::GetLogLevel);

  auto c = py::class_<LoggerPythonWapper>(m, "Log")
               .def(py::init<>())
               .def("reg", &LoggerPythonWapper::RegLogFunc,
                    py::arg("Callable[[Level level, str file, int lineno, str "
                            "func, str msg], None]"))
               .def("get_logger", &LoggerPythonWapper::GetLogger)
               .def("set_logger", &LoggerPythonWapper::SetLogger)
               .def("set_log_level", &LoggerPythonWapper::SetLogLevel)
               .def("print_ext", &LoggerPythonWapper::PrintExt,
                    py::arg("level"), py::arg("file"), py::arg("lineno"),
                    py::arg("func"), py::arg("msg"))
               .def("print", &LoggerPythonWapper::Print, py::arg("level"),
                    py::arg("msg"));

  ModelboxPyApiSetUpLogLevel(c);

  ModelboxPyApiSetUpLog(m);
}

class ExtOutputBufferList {
 public:
  ExtOutputBufferList() = default;
  virtual ~ExtOutputBufferList() = default;

  OutputBufferList &GetOutputBufferList() { return out_data_; }
  std::shared_ptr<BufferList> GetBufferList(const std::string &key) {
    auto iter = out_data_.find(key);
    if (iter == out_data_.end()) {
      return nullptr;
    }

    return iter->second;
  }

 private:
  OutputBufferList out_data_;
};

template <typename T>
struct unique_ptr_nogil_deleter {
  void operator()(T *ptr) {
    pybind11::gil_scoped_release nogil;
    delete ptr;
  }
};

struct UniqueFlow : modelbox::Flow {};

void SetUpFlow(pybind11::module &m) {
  py::class_<ExtOutputBufferList, std::shared_ptr<ExtOutputBufferList>>(
      m, "ExtOutputBufferList", py::module_local())
      .def(py::init<>())
      .def("get_buffer_list", &ExtOutputBufferList::GetBufferList,
           py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>());

  py::class_<modelbox::ExternalDataMap,
             std::shared_ptr<modelbox::ExternalDataMap>>(m, "ExternalDataMap",
                                                         py::module_local())
      .def("create_buffer_list", &modelbox::ExternalDataMap::CreateBufferList,
           py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>())
      .def("send", &modelbox::ExternalDataMap::Send,
           py::call_guard<py::gil_scoped_release>())
      .def("recv",
           [](modelbox::ExternalDataMap &ext,
              std::shared_ptr<ExtOutputBufferList> &out_data)
               -> modelbox::Status {
             auto &map_data = out_data->GetOutputBufferList();
             auto status = ext.Recv(map_data);
             return status;
           },
           py::keep_alive<2, 1>(), py::call_guard<py::gil_scoped_release>())
      .def("close", &modelbox::ExternalDataMap::Close,
           py::call_guard<py::gil_scoped_release>())
      .def("shutdown", &modelbox::ExternalDataMap::Shutdown,
           py::call_guard<py::gil_scoped_release>())
      .def("set_output_meta", &modelbox::ExternalDataMap::SetOutputMeta,
           py::call_guard<py::gil_scoped_release>())
      .def("get_last_error", &modelbox::ExternalDataMap::GetLastError,
           py::call_guard<py::gil_scoped_release>());

  auto c = py::class_<
      modelbox::UniqueFlow,
      std::unique_ptr<UniqueFlow, unique_ptr_nogil_deleter<UniqueFlow>>>(
      m, "Flow");
  py::enum_<modelbox::Flow::Format>(c, "Format", py::arithmetic(),
                                    py::module_local())
      .value("FORMAT_AUTO", Flow::FORMAT_AUTO)
      .value("FORMAT_TOML", Flow::FORMAT_TOML)
      .value("FORMAT_JSON", Flow::FORMAT_JSON);

  c.def(py::init<>())
      .def("init",
           static_cast<modelbox::Status (modelbox::Flow::*)(
               const std::string &, modelbox::Flow::Format)>(
               &modelbox::Flow::Init),
           py::arg("conf_file"),
           py::arg("format") = modelbox::Flow::Format::FORMAT_AUTO,
           py::call_guard<py::gil_scoped_release>())
      .def("init",
           static_cast<modelbox::Status (modelbox::Flow::*)(
               const std::string &, const std::string &,
               modelbox::Flow::Format)>(&modelbox::Flow::Init),
           py::arg("name"), py::arg("graph"),
           py::arg("format") = modelbox::Flow::Format::FORMAT_AUTO,
           py::call_guard<py::gil_scoped_release>())
      .def("init",
           static_cast<modelbox::Status (modelbox::Flow::*)(
               std::shared_ptr<Configuration>)>(&modelbox::Flow::Init),
           py::call_guard<py::gil_scoped_release>())
      .def("init",
           static_cast<modelbox::Status (modelbox::Flow::*)(
               const Solution &solution)>(&modelbox::Flow::Init),
           py::call_guard<py::gil_scoped_release>())
      .def("init",
           static_cast<modelbox::Status (modelbox::Flow::*)(
               const std::shared_ptr<FlowGraphDesc> &)>(&modelbox::Flow::Init),
           py::keep_alive<1, 2>(), py::call_guard<py::gil_scoped_release>())
      .def("build", &modelbox::Flow::Build,
           py::call_guard<py::gil_scoped_release>())
      .def("run", &modelbox::Flow::Run,
           py::call_guard<py::gil_scoped_release>())
      .def("run_async", &modelbox::Flow::RunAsync,
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &modelbox::Flow::Wait, py::arg("timemout") = 0,
           py::arg("retval") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &modelbox::Flow::Stop,
           py::call_guard<py::gil_scoped_release>())
      .def("create_external_data_map", &modelbox::Flow::CreateExternalDataMap,
           py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>());
}

PYBIND11_MODULE(_modelbox, m) {
  m.doc() = R"pbdoc(
        modelbox module
    )pbdoc";

  SetUpLog(m);
  SetUpFlow(m);
  ModelboxPyApiSetUpStatus(m);
  ModelboxPyApiSetUpConfiguration(m);
  ModelboxPyApiSetUpBuffer(m);
  ModelboxPyApiSetUpBufferList(m);
  ModelboxPyApiSetUpGeneric(m);
  ModelboxPyApiSetUpEngine(m);
  ModelboxPyApiSetUpDataHandler(m);
  ModelboxPyApiSetUpNodeDesc(m);
  ModelboxPyApiSetUpFlowGraphDesc(m);
  ModelBoxPyApiSetUpSolution(m);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
}  // namespace modelbox