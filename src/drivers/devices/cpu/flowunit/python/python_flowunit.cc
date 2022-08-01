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

#include "python_flowunit.h"

#include "modelbox/device/cpu/device_cpu.h"

using namespace pybind11::literals;

static std::mutex reload_mutex;

PythonFlowUnit::PythonFlowUnit() = default;
PythonFlowUnit::~PythonFlowUnit() {
  py::gil_scoped_acquire interpreter_guard{};
  python_process_.dec_ref();
  python_data_pre_.dec_ref();
  python_data_post_.dec_ref();
  python_data_group_pre_.dec_ref();
  python_data_group_post_.dec_ref();
  if (is_enable_debug_ == true) {
    pydevd_set_trace_.dec_ref();
  }
  obj_.dec_ref();
};

void PythonFlowUnit::EnablePythonDebug() {
  if (is_enable_debug_ == false) {
    return;
  }

  pydevd_set_trace_("suspend"_a = false, "trace_only_current_thread"_a = true);
}

modelbox::Status PythonFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  python_desc_ = std::dynamic_pointer_cast<VirtualPythonFlowUnitDesc>(
      this->GetFlowUnitDesc());

  auto python_entry = python_desc_->GetPythonEntry();
  auto config = python_desc_->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  if (config != nullptr) {
    merge_config->Add(*config);
  }
  merge_config->Add(*opts);

  constexpr const char DELIM_CHAR = '@';
  constexpr size_t ENTRY_FILENAME_AND_CLASS_COUNT = 2;
  const auto& entry_list = modelbox::StringSplit(python_entry, DELIM_CHAR);
  if (entry_list.size() != ENTRY_FILENAME_AND_CLASS_COUNT) {
    return {modelbox::STATUS_INVALID, "invalid entry string: " + python_entry};
  }

  const auto& python_path = python_desc_->GetPythonFilePath();

  // module reload mutex
  std::lock_guard<std::mutex> lck(reload_mutex);

  py::gil_scoped_acquire interpreter_guard{};

  // Avoid thread.lock assert after interpreter finish.
  PyGILState_Ensure();

  const char *enable_debug = getenv("MODELBOX_DEBUG_PYTHON");
  if (enable_debug != nullptr) {
    is_enable_debug_ = true;
  }

  try {
    auto sys = py::module::import("sys");
    if (is_enable_debug_ == true) {
      auto pydevd = py::module::import("pydevd");
      pydevd_set_trace_ = pydevd.attr("settrace");
    }

    sys.attr("path").cast<py::list>().append(python_path);

    auto python_module = py::module_::import(entry_list[0].c_str());
    python_module.reload();
    auto python_class = python_module.attr(entry_list[1].c_str());
    obj_ = python_class();
    python_process_ = obj_.attr("process");
    python_data_pre_ = obj_.attr("data_pre");
    python_data_post_ = obj_.attr("data_post");
    python_data_group_pre_ = obj_.attr("data_group_pre");
    python_data_group_post_ = obj_.attr("data_group_post");
  } catch (const std::exception& ex) {
    is_enable_debug_ = false;
    return {modelbox::STATUS_INVALID, "import " + python_desc_->GetPythonEntry() +
                                        " failed: " + ex.what()};
  }

  auto* fu = obj_.cast<modelbox::FlowUnit*>();
  fu->SetBindDevice(GetBindDevice());
  fu->SetExternalData(GetCreateExternalDataFunc());
  fu->SetFlowUnitDesc(GetFlowUnitDesc());

  py::object status;
  try {
    EnablePythonDebug();
    auto python_open = obj_.attr("open");
    status = python_open(merge_config);
  } catch (const std::exception& ex) {
    return {modelbox::STATUS_FAULT, python_desc_->GetPythonEntry() +
                                        " function open error: " + ex.what()};
  }

  try {
    // if return is modelbox::StatusCode
    return status.cast<modelbox::StatusCode>();
  } catch (...) {
    // do nothing
  }

  return status.cast<modelbox::Status>();
}

modelbox::Status PythonFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  py::gil_scoped_acquire interpreter_guard{};

  try {
    EnablePythonDebug();
    auto status = python_process_(data_ctx);
    try {
      // if return is modelbox::StatusCode
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      // do nothing
    }

    // if return modelbox::Status
    return status.cast<modelbox::Status>();
  } catch (py::error_already_set& ex) {
    MBLOG_WARN << python_desc_->GetPythonEntry()
               << " python function process catch exception: " << ex.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status PythonFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  py::gil_scoped_acquire interpreter_guard{};
  try {
    EnablePythonDebug();
    auto status = python_data_pre_(data_ctx);
    try {
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      // do nothing
    }

    return status.cast<modelbox::Status>();
  } catch (const std::exception& ex) {
    MBLOG_WARN << python_desc_->GetPythonEntry()
               << " python function data_pre catch exception: " << ex.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status PythonFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  py::gil_scoped_acquire interpreter_guard{};
  try {
    EnablePythonDebug();
    auto status = python_data_post_(data_ctx);
    try {
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      // do nothing
    }

    return status.cast<modelbox::Status>();
  } catch (const std::exception& ex) {
    MBLOG_WARN << python_desc_->GetPythonEntry()
               << " python function data_post catch exception: " << ex.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status PythonFlowUnit::DataGroupPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  py::gil_scoped_acquire interpreter_guard{};
  try {
    EnablePythonDebug();
    auto status = python_data_group_pre_(data_ctx);
    try {
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      // do nothing
    }

    return status.cast<modelbox::Status>();
  } catch (const std::exception& ex) {
    MBLOG_WARN << python_desc_->GetPythonEntry()
               << " python function data_group_pre catch exception: "
               << ex.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status PythonFlowUnit::DataGroupPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  py::gil_scoped_acquire interpreter_guard{};
  try {
    EnablePythonDebug();
    auto status = python_data_group_post_(data_ctx);
    try {
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      // do nothing
    }

    return status.cast<modelbox::Status>();
  } catch (const std::exception& ex) {
    MBLOG_WARN << python_desc_->GetPythonEntry()
               << " python function data_group_post catch exception: "
               << ex.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status PythonFlowUnit::Close() {
  py::gil_scoped_acquire interpreter_guard{};
  try {
    EnablePythonDebug();
    auto python_close = obj_.attr("close");
    auto status = python_close();
    try {
      return status.cast<modelbox::StatusCode>();
    } catch (...) {
      return modelbox::STATUS_OK;
    }

    return status.cast<modelbox::Status>();
  } catch (const std::exception& ex) {
    return modelbox::STATUS_OK;
  }
}

void PythonFlowUnit::SetFlowUnitDesc(
    std::shared_ptr<modelbox::FlowUnitDesc> desc) {
  python_desc_ = std::dynamic_pointer_cast<VirtualPythonFlowUnitDesc>(desc);
}

std::shared_ptr<modelbox::FlowUnitDesc> PythonFlowUnit::GetFlowUnitDesc() {
  return python_desc_;
}

void PythonFlowUnitDesc::SetPythonEntry(const std::string python_entry) {
  python_entry_ = python_entry;
}

std::string PythonFlowUnitDesc::GetPythonEntry() { return python_entry_; }
