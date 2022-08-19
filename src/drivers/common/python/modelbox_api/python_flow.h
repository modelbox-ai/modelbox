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

#ifndef MODELBOX_PYTHON_MODELBOX_API_FLOW_H_
#define MODELBOX_PYTHON_MODELBOX_API_FLOW_H_

#include <pybind11/pybind11.h>

#include "modelbox/flow.h"

namespace py = pybind11;

namespace modelbox {

class PythonFlowStreamIO {
 public:
  PythonFlowStreamIO(std::shared_ptr<modelbox::FlowStreamIO> io);

  virtual ~PythonFlowStreamIO();

  std::shared_ptr<modelbox::Buffer> CreateBuffer();

  modelbox::Status Send(const std::string &input_name,
                        const std::shared_ptr<modelbox::Buffer> &buffer);

  modelbox::Status Recv(const std::string &output_name,
                        std::shared_ptr<modelbox::Buffer> &buffer,
                        size_t timeout = 0);

  void CloseInput();

 private:
  std::shared_ptr<modelbox::FlowStreamIO> io_;
};

}  // namespace modelbox
#endif  // MODELBOX_PYTHON_MODELBOX_API_FLOW_H_
