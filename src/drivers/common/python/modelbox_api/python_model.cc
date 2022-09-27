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

#include "python_model.h"

#include "python_common.h"

namespace modelbox {

PythonModel::PythonModel(std::string path, std::string name,
                         std::vector<std::string> in_names,
                         std::vector<std::string> out_names,
                         size_t max_batch_size, std::string device,
                         std::string device_id)
    : name_(std::move(name)),
      in_names_(std::move(in_names)),
      out_names_(std::move(out_names)),
      max_batch_size_(std::to_string(max_batch_size)),
      device_(std::move(device)),
      device_id_(std::move(device_id)) {
  path_.emplace_back(std::move(path));
}

PythonModel::~PythonModel() {
  // we need release gil before clear flow resource
  py::gil_scoped_release release;
  Stop();
}

void PythonModel::AddPath(const std::string &path) { path_.emplace_back(path); }

modelbox::Status PythonModel::Start() {
  flow_graph_desc_ = std::make_shared<modelbox::FlowGraphDesc>();
  flow_graph_desc_->SetDriversDir(path_);

  std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>> source_ports;
  for (const auto &in_name : in_names_) {
    auto in_port = flow_graph_desc_->AddInput(in_name);
    source_ports[in_name] = (*in_port)[0];
  }

  auto inference = flow_graph_desc_->AddNode(
      name_, device_,
      {"batch_size=" + max_batch_size_, "device_id=" + device_id_},
      source_ports);

  for (const auto &out_name : out_names_) {
    flow_graph_desc_->AddOutput(out_name, (*inference)[out_name]);
  }

  auto flow = std::make_shared<modelbox::Flow>();
  auto status = flow->Init(flow_graph_desc_);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "init flow failed, " << status;
    return status;
  }

  status = flow->StartRun();
  if (status != STATUS_OK) {
    MBLOG_ERROR << "start flow failed, " << status;
    return status;
  }

  flow_ = flow;
  return STATUS_SUCCESS;
}

void PythonModel::Stop() {
  if (flow_ != nullptr) {
    flow_->Stop();
  }

  flow_ = nullptr;
  flow_graph_desc_ = nullptr;
}

std::vector<std::shared_ptr<Buffer>> PythonModel::Infer(
    const std::vector<py::buffer> &data_list) {
  std::vector<std::shared_ptr<Buffer>> result_list;
  if (data_list.size() != in_names_.size()) {
    MBLOG_ERROR << "infer input data size != model input count";
    return result_list;
  }

  auto io = flow_->CreateStreamIO();
  auto in_port_count = in_names_.size();
  for (size_t i = 0; i < in_port_count; ++i) {
    auto buffer = io->CreateBuffer();
    {
      py::gil_scoped_acquire ac;
      PyBufferToBuffer(buffer, data_list[i]);
    }
    auto ret = io->Send(in_names_[i], buffer);
    if (ret != STATUS_OK) {
      MBLOG_ERROR << "infer send data failed, err " << ret;
      return result_list;
    }
  }
  io->CloseInput();

  result_list.reserve(out_names_.size());
  for (auto &out_name : out_names_) {
    std::shared_ptr<Buffer> out_buffer;
    io->Recv(out_name, out_buffer, 0);
    result_list.push_back(out_buffer);
  }

  return result_list;
}

std::vector<std::vector<std::shared_ptr<Buffer>>> PythonModel::InferBatch(
    const std::vector<std::vector<py::buffer>> &data_list) {
  // input[port1[batch1,batch2],port2[batch1,batch2]]
  // output[port1[batch1,batch2],port2[batch1,batch2]]
  std::vector<std::vector<std::shared_ptr<Buffer>>> result_list;
  if (data_list.size() != in_names_.size()) {
    MBLOG_ERROR << "infer input data size != model input count";
    return result_list;
  }

  const auto &first_port_batch = data_list.front();
  auto batch_size = first_port_batch.size();
  std::vector<std::shared_ptr<FlowStreamIO>> io_list;
  io_list.reserve(batch_size);
  auto in_port_count = in_names_.size();
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto io = flow_->CreateStreamIO();
    for (size_t i = 0; i < in_port_count; ++i) {
      auto buffer = io->CreateBuffer();
      {
        py::gil_scoped_acquire ac;
        PyBufferToBuffer(buffer, data_list[i][batch_idx]);
      }
      auto ret = io->Send(in_names_[i], buffer);
      if (ret != STATUS_OK) {
        MBLOG_ERROR << "infer send data failed, err " << ret;
        return result_list;
      }
    }
    io->CloseInput();
    io_list.push_back(io);
  }

  auto out_port_count = out_names_.size();
  result_list.resize(out_port_count);
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto io = io_list[batch_idx];
    for (size_t i = 0; i < out_port_count; ++i) {
      std::shared_ptr<Buffer> out_buffer;
      io->Recv(out_names_[i], out_buffer, 0);
      result_list[i].push_back(out_buffer);
    }
  }

  return result_list;
}

}  // namespace modelbox
