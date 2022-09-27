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

#include <utility>

namespace modelbox {

FlowPortDesc::FlowPortDesc(std::shared_ptr<FlowNodeDesc> node,
                           std::string port_name)
    : node_(std::move(node)),
      is_in_name_{true},
      port_name_(std::move(port_name)) {}

FlowPortDesc::FlowPortDesc(std::shared_ptr<FlowNodeDesc> node, size_t port_idx)
    : node_(std::move(node)), is_in_name_{false}, port_idx_(port_idx) {}

std::shared_ptr<FlowNodeDesc> FlowPortDesc::GetNode() { return node_; }

std::string FlowPortDesc::GetNodeName() { return node_->GetNodeName(); }

bool FlowPortDesc::IsDescribeInName() { return is_in_name_; }

std::string FlowPortDesc::GetPortName() { return port_name_; }

size_t FlowPortDesc::GetPortIdx() { return port_idx_; }

FlowNodeDesc::FlowNodeDesc(std::string node_name)
    : node_name_(std::move(node_name)) {}

FlowNodeDesc::~FlowNodeDesc() = default;

void FlowNodeDesc::SetNodeName(const std::string &node_name) {
  node_name_ = node_name;
}

std::string FlowNodeDesc::GetNodeName() { return node_name_; }

std::shared_ptr<FlowPortDesc> FlowNodeDesc::operator[](
    const std::string &output_name) {
  if (type_ == GRAPH_NODE_INPUT) {
    return std::make_shared<FlowPortDesc>(shared_from_this(), output_name);
  }

  return std::make_shared<FlowPortDesc>(shared_from_this(), output_name);
}

std::shared_ptr<FlowPortDesc> FlowNodeDesc::operator[](size_t port_idx) {
  if (type_ == GRAPH_NODE_INPUT) {
    return std::make_shared<FlowPortDesc>(shared_from_this(), node_name_);
  }

  return std::make_shared<FlowPortDesc>(shared_from_this(), port_idx);
}

void FlowNodeDesc::SetNodeType(const std::string &type) { type_ = type; }

void FlowNodeDesc::SetFlowUnitName(const std::string &flowunit_name) {
  flowunit_name_ = flowunit_name;
}

std::string FlowNodeDesc::GetFlowUnitName() { return flowunit_name_; }

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
