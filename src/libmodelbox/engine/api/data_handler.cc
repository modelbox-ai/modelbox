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

#include "modelbox/data_handler.h"
#include "modelbox/modelbox_engine.h"

namespace modelbox {

DataHandler::DataHandler(const std::shared_ptr<ModelBoxEngine> &env) {
  engine_ = env;
}

DataHandler::~DataHandler() {
  MBLOG_INFO << "release datahandler for node: " << GetNodeName();
}

void DataHandler::SetEnv(const std::shared_ptr<ModelBoxEngine> &env) {
  engine_ = env;
}

std::shared_ptr<ModelBoxEngine> DataHandler::GetEnv() { return engine_.lock(); }

DataHandlerType DataHandler::GetDataHandlerType() { return data_handler_type_; }

void DataHandler::SetDataHandlerType(const DataHandlerType &type) {
  data_handler_type_ = type;
}

void DataHandler::SetNodeName(const std::string &name) { node_name_ = name; }

std::string DataHandler::GetNodeName() { return node_name_; }

std::set<std::string> DataHandler::GetPortNames() { return port_names_; }

std::shared_ptr<DataHandler> DataHandler::GetDataHandler(
    const std::string &key) {
  if (port_names_.find(key) == port_names_.end()) {
    MBLOG_ERROR << "faild find port name: " << key
                << " in node: " << node_name_;
    return nullptr;
  }
  if (engine_.lock() != nullptr) {
    auto data = std::make_shared<DataHandler>(engine_.lock());
    data->SetNodeName(node_name_);

    std::set<std::string> ports = {key};
    data->SetPortNames(ports);
    return data;
  }
  return nullptr;
}

Status DataHandler::SetDataHandler(
    const std::map<std::string, std::shared_ptr<DataHandler>> &data_map) {
  for (auto &iter : data_map) {
    auto ports = iter.second->GetPortNames();
    if (ports.size() != 1) {
      std::string err_msg = "input data handler has one more ports";
      return {STATUS_FAULT, err_msg};
    }

    auto in_port_name = iter.first;
    auto temp_data = iter.second;
    auto node_name = temp_data->GetNodeName();
    auto out_port_name = *(ports.begin());
    port_to_port_[in_port_name] = out_port_name;

    port_to_node_[in_port_name] = node_name;

    if (engine_.lock() != iter.second->engine_.lock()) {
      std::string msg = "sub datahandler bind different graph";
      MBLOG_ERROR << msg;
      return {STATUS_FAULT, msg};
    }
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<DataHandler> DataHandler::operator[](
    const std::string &port_name) {
  return GetDataHandler(port_name);
}

Status DataHandler::SetPortNames(std::set<std::string> &port_names) {
  port_names_ = port_names;
  return STATUS_OK;
}

}  // namespace modelbox
