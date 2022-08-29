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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

#include "modelbox/flow.h"
#include "modelbox/type.h"

namespace modelbox {

namespace py = pybind11;

static std::unordered_map<int, std::string> kTypeToNPType;
static std::unordered_map<std::string, ModelBoxDataType> kNPTypeToType;

void BuildTypeToNumpyType() {
  if (kTypeToNPType.empty() && kNPTypeToType.empty()) {
    kTypeToNPType[MODELBOX_UINT8] = py::format_descriptor<uint8_t>::format();
    kTypeToNPType[MODELBOX_INT8] = py::format_descriptor<int8_t>::format();
    kTypeToNPType[MODELBOX_BOOL] = py::format_descriptor<bool>::format();
    kTypeToNPType[MODELBOX_INT16] = py::format_descriptor<int16_t>::format();
    kTypeToNPType[MODELBOX_UINT16] = py::format_descriptor<uint16_t>::format();
    kTypeToNPType[MODELBOX_INT32] = py::format_descriptor<int32_t>::format();
    kTypeToNPType[MODELBOX_UINT32] = py::format_descriptor<uint32_t>::format();
    kTypeToNPType[MODELBOX_INT64] = py::format_descriptor<int64_t>::format();
    kTypeToNPType[MODELBOX_UINT64] = py::format_descriptor<uint64_t>::format();
    kTypeToNPType[MODELBOX_FLOAT] = py::format_descriptor<float>::format();
    kTypeToNPType[MODELBOX_DOUBLE] = py::format_descriptor<double>::format();
    kNPTypeToType[py::format_descriptor<uint8_t>::format()] = MODELBOX_UINT8;
    kNPTypeToType[py::format_descriptor<int8_t>::format()] = MODELBOX_INT8;
    kNPTypeToType[py::format_descriptor<bool>::format()] = MODELBOX_BOOL;
    kNPTypeToType[py::format_descriptor<int16_t>::format()] = MODELBOX_INT16;
    kNPTypeToType[py::format_descriptor<uint16_t>::format()] = MODELBOX_UINT16;
    kNPTypeToType[py::format_descriptor<int32_t>::format()] = MODELBOX_INT32;
    kNPTypeToType[py::format_descriptor<uint32_t>::format()] = MODELBOX_UINT32;
    kNPTypeToType[py::format_descriptor<int64_t>::format()] = MODELBOX_INT64;
    kNPTypeToType[py::format_descriptor<uint64_t>::format()] = MODELBOX_UINT64;
    kNPTypeToType[py::format_descriptor<float>::format()] = MODELBOX_FLOAT;
    kNPTypeToType[py::format_descriptor<double>::format()] = MODELBOX_DOUBLE;
    kNPTypeToType["e"] = MODELBOX_FLOAT;
    kNPTypeToType["l"] = MODELBOX_INT64;
  }
}

std::string FormatStrFromType(const modelbox::ModelBoxDataType &type) {
  BuildTypeToNumpyType();

  auto iter = kTypeToNPType.find(type);
  if (iter == kTypeToNPType.end()) {
    std::string errmsg = "invalid modelbox data type: ";
    errmsg += std::to_string(type);
    throw std::runtime_error(errmsg);
  }

  return iter->second;
}

modelbox::ModelBoxDataType TypeFromFormatStr(const std::string &format) {
  BuildTypeToNumpyType();

  auto iter = kNPTypeToType.find(format);
  if (iter == kNPTypeToType.end()) {
    throw std::runtime_error("invalid numpy data type: " + format);
  }

  return iter->second;
}

void PyBufferToBuffer(const std::shared_ptr<Buffer> &buffer,
                      const py::buffer &data) {
  py::buffer_info info = data.request();
  std::vector<size_t> i_shape;
  for (auto &dim : info.shape) {
    i_shape.push_back(dim);
  }

  if (info.shape.size() == 0) {
    throw std::runtime_error("can not accept empty numpy.");
  }

  size_t bytes = Volume(i_shape) * info.itemsize;
  if (PyBuffer_IsContiguous(info.view(), 'C')) {
    buffer->BuildFromHost(info.ptr, bytes);
  } else {
    // py_buffer is not C Contiguous, need convert
    buffer->Build(bytes);
    PyBuffer_ToContiguous(buffer->MutableData(), info.view(), bytes, 'C');
  }

  buffer->Set("shape", i_shape);
  buffer->Set("type", TypeFromFormatStr(info.format));
  buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
}

}  // namespace modelbox
