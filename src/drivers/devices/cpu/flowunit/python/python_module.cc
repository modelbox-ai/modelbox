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


#include "python_module.h"

#include <modelbox/base/log.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <functional>

#include "modelbox_api.h"
#include "python_log.h"

namespace py = pybind11;

#define PYBIND11_MODULE_INIT(name) PyInit_##name()

PythonInterpreter::PythonInterpreter() {
  if (!Py_IsInitialized()) {
    is_initialized_ = true;
    py::initialize_interpreter(0);
    // unlock GIL
    threadState_ = PyEval_SaveThread();
    return;
  }

  modelbox::FlowUnitPythonLog::Init();
}

PythonInterpreter::~PythonInterpreter() {
  modelbox::FlowUnitPythonLog::Finish();

  if (is_initialized_ == false) {
    return;
  }

  if (threadState_ != nullptr) {
    // lock GIL
    PyEval_RestoreThread(threadState_);
    threadState_ = nullptr;
  }

  // never release python interpreter
}

PYBIND11_MODULE(_flowunit, m) {
  modelbox::ModelboxPyApiSetUpLog(m);
  modelbox::ModelboxPyApiSetUpStatus(m);
  modelbox::ModelboxPyApiSetUpConfiguration(m);
  modelbox::ModelboxPyApiSetUpBuffer(m);
  modelbox::ModelboxPyApiSetUpBufferList(m);
  modelbox::ModelboxPyApiSetUpGeneric(m);
  modelbox::ModelboxPyApiSetUpFlowUnit(m);
}

modelbox::Status PythonInterpreter::InitModule() {
  py::gil_scoped_acquire acquire{};

  auto *m = PyImport_AddModule("_flowunit");
  if (m == nullptr) {
    MBLOG_ERROR << "Add python module failed.";
    return modelbox::STATUS_FAULT;
  }

  PyObject *module = PYBIND11_MODULE_INIT(_flowunit);
  if (module == nullptr) {
    MBLOG_ERROR << "Init python module failed.";
    return modelbox::STATUS_FAULT;
  }

  PyObject *sys_modules = PyImport_GetModuleDict();
  PyDict_SetItemString(sys_modules, "_flowunit", module);
  is_module_init_ = true;

  return modelbox::STATUS_OK;
}

modelbox::Status PythonInterpreter::ExitModule() {
  py::gil_scoped_acquire acquire{};

  if (is_module_init_ == false) {
    return modelbox::STATUS_OK;
  }

  is_module_init_ = false;
  PyObject *sys_modules = PyImport_GetModuleDict();
  PyDict_DelItemString(sys_modules, "_flowunit");
  return modelbox::STATUS_OK;
}
