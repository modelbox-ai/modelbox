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

#include <modelbox/base/utils.h>

#include <regex>
#include <toml.hpp>

#include "python_common.h"

namespace modelbox {

PythonModel::PythonModel(std::string path, std::string name,
                         size_t max_batch_size, std::string device,
                         std::string device_id)
    : name_(std::move(name)),
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
  auto ret = ReadModelIO(in_names_, out_names_);
  if (!ret) {
    MBLOG_ERROR << "read model io failed";
    return ret;
  }

  flow_graph_desc_ = std::make_shared<modelbox::FlowGraphDesc>();
  flow_graph_desc_->SetDriversDir(path_);

  std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>> source_ports;
  for (const auto &in_name : in_names_) {
    auto in_port = flow_graph_desc_->AddInput(in_name);
    source_ports[in_name] = (*in_port)[0];
  }

  auto inference = flow_graph_desc_->AddNode(
      name_, device_,
      {"batch_size=" + max_batch_size_, "deviceid=" + device_id_},
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
  auto in_port_count = in_names_.size();
  auto io = flow_->CreateStreamIO();
  for (size_t i = 0; i < in_port_count; ++i) {
    std::vector<std::shared_ptr<Buffer>> input_list;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      auto buffer = io->CreateBuffer();
      {
        py::gil_scoped_acquire ac;
        PyBufferToBuffer(buffer, data_list[i][batch_idx]);
      }
      input_list.push_back(buffer);
    }

    auto ret = io->Send(in_names_[i], input_list);
    if (ret != STATUS_OK) {
      MBLOG_ERROR << "infer send " << in_names_[i] << " data failed, err "
                  << ret;
      return result_list;
    }
  }
  io->CloseInput();

  auto out_port_count = out_names_.size();
  result_list.resize(out_port_count);
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (size_t i = 0; i < out_port_count; ++i) {
      std::shared_ptr<Buffer> out_buffer;
      io->Recv(out_names_[i], out_buffer, 0);
      result_list[i].push_back(out_buffer);
    }
  }

  return result_list;
}

modelbox::Status PythonModel::ReadModelIO(std::vector<std::string> &in_names,
                                          std::vector<std::string> &out_names) {
  std::vector<std::string> files;
  auto ret = modelbox::ListSubDirectoryFiles(path_.front(), "*.toml", &files);
  if (!ret) {
    MBLOG_ERROR << "list file in path " << path_.front() << " failed, error "
                << ret;
    return ret;
  }

  if (files.empty()) {
    MBLOG_ERROR << "no valid model conf in path " << path_.front();
    return STATUS_BADCONF;
  }

  std::stringstream err_msg_cache;
  for (auto &file : files) {
    try {
      auto fu_config = toml::parse(file);
      auto name = toml::find<std::string>(fu_config, "base", "name");
      if (name != name_) {
        continue;
      }

      std::ifstream ifs(file);
      if (!ifs.good()) {
        err_msg_cache << "[" << file << "] read failed" << std::endl;
        continue;
      }

      Defer { ifs.close(); };

      // try to keep input and output order in config file
      std::string content((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
      std::smatch search_result;

      auto search_text = content;
      std::regex input_regex(R"(\[input\.(.*?)\])");
      std::vector<std::string> input_key_list;
      while (std::regex_search(search_text, search_result, input_regex)) {
        input_key_list.push_back(search_result[1]);
        search_text = search_result.suffix();
      }

      search_text = content;
      std::regex output_regex(R"(\[output\.(.*?)\])");
      std::vector<std::string> output_key_list;
      while (std::regex_search(search_text, search_result, output_regex)) {
        output_key_list.push_back(search_result[1]);
        search_text = search_result.suffix();
      }

      for (const auto &input_key : input_key_list) {
        auto input_name =
            toml::find<std::string>(fu_config, "input", input_key, "name");
        in_names.push_back(input_name);
      }

      for (const auto &output_key : output_key_list) {
        auto output_name =
            toml::find<std::string>(fu_config, "output", output_key, "name");
        out_names.push_back(output_name);
      }

      return STATUS_OK;
    } catch (std::exception &e) {
      err_msg_cache << "[" << file << "] parse toml failed, err: " << e.what()
                    << std::endl;
      continue;
    }
  }

  auto err_msg = err_msg_cache.str();
  if (err_msg.empty()) {
    err_msg = " target model not found";
  }

  MBLOG_ERROR << "can not load IO info for modle " << name_ << " in path "
              << path_.front() << ", detail: " << err_msg;
  return modelbox::STATUS_BADCONF;
}

}  // namespace modelbox
