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

#ifndef MODELBOX_PYTHON_MODELBOX_API_COMMON_H_
#define MODELBOX_PYTHON_MODELBOX_API_COMMON_H_

#include <pybind11/pybind11.h>

#include <memory>

namespace modelbox {

std::string FormatStrFromType(const ModelBoxDataType &type);

ModelBoxDataType TypeFromFormatStr(const std::string &format);

void PyBufferToBuffer(const std::shared_ptr<Buffer> &buffer,
                      const py::buffer &data);

}  // namespace modelbox

#endif  // MODELBOX_PYTHON_MODELBOX_API_COMMON_H_
