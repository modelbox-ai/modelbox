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

#include "modelbox/flow_node_desc.h"
#include "modelbox/modelbox_engine.h"

namespace modelbox {

NodeDesc::NodeDesc() {}

NodeDesc::~NodeDesc() {
}

NodeDescType NodeDesc::GetNodeType() { return data_handler_type_; }

void NodeDesc::SetNodeType(const NodeDescType &type) {
  data_handler_type_ = type;
}

void NodeDesc::SetNodeName(const std::string &name) { node_name_ = name; }

std::string NodeDesc::GetNodeName() { return node_name_; }

std::set<std::string> NodeDesc::GetPortNames() { return port_names_; }

std::shared_ptr<NodeDesc> NodeDesc::GetNodeDesc(const std::string &key) {
  if (port_names_.find(key) == port_names_.end()) {
    MBLOG_ERROR << "faild find port name: " << key
                << " in node: " << node_name_;
    return nullptr;
  }

  auto data = std::make_shared<NodeDesc>();
  data->SetNodeName(node_name_);

  std::set<std::string> ports = {key};
  data->SetPortNames(ports);
  return data;
}

Status NodeDesc::SetNodeDesc(
    const std::map<std::string, std::shared_ptr<NodeDesc>> &data_map) {
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
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<NodeDesc> NodeDesc::operator[](const std::string &port_name) {
  return GetNodeDesc(port_name);
}

Status NodeDesc::SetPortNames(std::set<std::string> &port_names) {
  port_names_ = port_names;
  return STATUS_OK;
}

}  // namespace modelbox
