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

namespace modelbox {

FlowNodeDesc::FlowNodeDesc(const std::string &node_name)
    : node_name_(node_name) {}

FlowNodeDesc::~FlowNodeDesc() {}

void FlowNodeDesc::SetNodeName(const std::string &node_name) {
  node_name_ = node_name;
}

std::string FlowNodeDesc::GetNodeName() { return node_name_; }

std::shared_ptr<FlowPortDesc> FlowNodeDesc::operator[](
    const std::string &output_name) {
  for (auto &port_name : output_port_name_list_) {
    if (port_name == output_name) {
      return std::make_shared<FlowPortDesc>(node_name_, output_name);
    }
  }

  MBLOG_ERROR << "node " << node_name_ << " does not have output port "
              << output_name;
  return nullptr;
}

std::shared_ptr<FlowPortDesc> FlowNodeDesc::operator[](size_t port_idx) {
  if (output_port_name_list_.size() <= port_idx) {
    MBLOG_ERROR << "node " << node_name_ << " output number is "
                << output_port_name_list_.size() << ",  index " << port_idx
                << " is out of range";
    return nullptr;
  }

  return std::make_shared<FlowPortDesc>(node_name_,
                                        output_port_name_list_[port_idx]);
}

void FlowNodeDesc::SetNodeType(const std::string &type) { type_ = type; }

void FlowNodeDesc::SetFlowUnitName(const std::string &flowunit_name) {
  flowunit_name_ = flowunit_name;
}

void FlowNodeDesc::SetDevice(const std::string &device) { device_ = device; }

void FlowNodeDesc::SetConfig(const std::vector<std::string> &config) {
  config_ = config;
}

std::vector<std::string> FlowNodeDesc::GetNodeConfig() {
  auto node_config = config_;
  node_config.push_back("type=" + type_);
  if (type_ == GRAPH_NODE_FLOWUNIT) {
    node_config.push_back("flowunit=" + flowunit_name_);
  }
  node_config.push_back("device=" + device_);
  return node_config;
}

void FlowNodeDesc::SetOutputPortNames(
    const std::vector<std::string> &output_port_name_list) {
  output_port_name_list_ = output_port_name_list;
}

void FlowNodeDesc::SetInputLinks(
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &source_node_ports) {
  source_node_ports_ = source_node_ports;
}

const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
    &FlowNodeDesc::GetInputLinks() {
  return source_node_ports_;
}

void FlowNodeDesc::Clear() { source_node_ports_.clear(); }

}  // namespace modelbox
