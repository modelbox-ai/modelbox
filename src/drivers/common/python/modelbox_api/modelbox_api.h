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

#ifndef MODELBOX_PYTHON_MODELBOX_API_H_
#define MODELBOX_PYTHON_MODELBOX_API_H_

#include <modelbox/base/log.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace modelbox {

template <typename T>
py::array mkarray_via_buffer(void *data, size_t n) {
  return py::array(py::buffer_info(data, sizeof(T),
                                   py::format_descriptor<T>::format(), 1, {n},
                                   {sizeof(T)}));
}

void ModelboxPyApiSetUpStatus(pybind11::module &m);

void ModelboxPyApiSetUpConfiguration(pybind11::module &m);

void ModelboxPyApiSetUpLogLevel(pybind11::handle &h);

void ModelboxPyApiSetUpLog(pybind11::module &m);

void ModelboxPyApiSetUpBuffer(pybind11::module &m);

void ModelboxPyApiSetUpBufferList(pybind11::module &m);

void ModelboxPyApiSetUpGeneric(pybind11::module &m);

void ModelboxPyApiSetUpFlowUnit(pybind11::module &m);

void ModelboxPyApiSetUpEngine(pybind11::module &m);

void ModelboxPyApiSetUpDataHandler(pybind11::module &m);

void ModelboxPyApiSetUpFlowGraphDesc(pybind11::module &m);

void ModelboxPyApiSetUpNodeDesc(pybind11::module &m);

void ModelBoxPyApiSetUpExternalDataMapSimple(pybind11::module &m);

void ModelBoxPyApiSetUpSolution(pybind11::module &m);
}  // namespace modelbox

#endif  // MODELBOX_PYTHON_MODELBOX_API_H_
