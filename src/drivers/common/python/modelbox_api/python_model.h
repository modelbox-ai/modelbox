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

#ifndef MODELBOX_PYTHON_MODELBOX_API_MODEL_H_
#define MODELBOX_PYTHON_MODELBOX_API_MODEL_H_

#include <pybind11/pybind11.h>

#include "modelbox/flow.h"

namespace py = pybind11;

namespace modelbox {

class PythonModel {
 public:
  PythonModel(std::string path, std::string name,
              std::vector<std::string> in_names,
              std::vector<std::string> out_names, size_t max_batch_size,
              std::string device, std::string device_id);

  virtual ~PythonModel();

  void AddPath(const std::string &path);

  modelbox::Status Start();

  void Stop();

  std::vector<std::shared_ptr<Buffer>> Infer(
      const std::vector<py::buffer> &data_list);

  std::vector<std::vector<std::shared_ptr<Buffer>>> InferBatch(
      const std::vector<std::vector<py::buffer>> &data_list);

 private:
  std::vector<std::string> path_;
  std::string name_;
  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;
  std::string max_batch_size_;
  std::string device_;
  std::string device_id_;

  std::shared_ptr<modelbox::FlowGraphDesc> flow_graph_desc_;
  std::shared_ptr<modelbox::Flow> flow_;
};

}  // namespace modelbox

#endif  // MODELBOX_PYTHON_MODELBOX_API_MODEL_H_
