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

#include "modelbox/flow_graph_desc.h"

#include <utility>

#include "modelbox/base/register_flowunit.h"

namespace modelbox {

FlowGraphFunctionInfo::FlowGraphFunctionInfo(
    std::string name, std::vector<std::string> input_name_list,
    std::vector<std::string> output_name_list,
    std::function<Status(std::shared_ptr<DataContext>)> func)
    : name_(std::move(name)),
      input_name_list_(std::move(input_name_list)),
      output_name_list_(std::move(output_name_list)),
      func_(std::move(func)) {}

std::string FlowGraphFunctionInfo::GetName() { return name_; }

std::vector<std::string> FlowGraphFunctionInfo::GetInputNameList() {
  return input_name_list_;
}

std::vector<std::string> FlowGraphFunctionInfo::GetOutputNameList() {
  return output_name_list_;
}

std::function<Status(std::shared_ptr<DataContext>)>
FlowGraphFunctionInfo::GetFunc() {
  return func_;
}

constexpr const char *CONFIG_KEY_QUEUE_SIZE = "graph.queue_size";
constexpr const char *CONFIG_KEY_BATCH_SIZE = "graph.batch_size";
constexpr const char *CONFIG_KEY_DRIVERS_DIR = "driver.dir";
constexpr const char *CONFIG_KEY_DRIVERS_SKIP_DEFAULT = "driver.skip-default";
constexpr const char *CONFIG_KEY_PROFILE_DIR = "profile.dir";
constexpr const char *CONFIG_KEY_PROFILE_TRACE_ENABLE = "profile.trace";

FlowGraphDesc::FlowGraphDesc() { config_ = ConfigurationBuilder().Build(); }

FlowGraphDesc::~FlowGraphDesc() {
  for (auto &node_desc : node_desc_list_) {
    node_desc->Clear();
  }
}

void FlowGraphDesc::SetQueueSize(size_t queue_size) {
  config_->SetProperty(CONFIG_KEY_QUEUE_SIZE, queue_size);
}

void FlowGraphDesc::SetBatchSize(size_t batch_size) {
  config_->SetProperty(CONFIG_KEY_BATCH_SIZE, batch_size);
}

void FlowGraphDesc::SetDriversDir(
    const std::vector<std::string> &drivers_dir_list) {
  config_->SetProperty(CONFIG_KEY_DRIVERS_DIR, drivers_dir_list);
}

void FlowGraphDesc::SetSkipDefaultDrivers(bool is_skip) {
  config_->SetProperty(CONFIG_KEY_DRIVERS_SKIP_DEFAULT, is_skip);
}

void FlowGraphDesc::SetProfileDir(const std::string &profile_dir) {
  config_->SetProperty(CONFIG_KEY_PROFILE_DIR, profile_dir);
}

void FlowGraphDesc::SetProfileTraceEnable(bool profile_trace_enable) {
  config_->SetProperty(CONFIG_KEY_PROFILE_TRACE_ENABLE, profile_trace_enable);
}

// add input
std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddInput(
    const std::string &input_name) {
  if (node_name_idx_map_[input_name] >= 1) {
    MBLOG_ERROR << "Input name " << input_name << " has been used";
    return nullptr;
  }

  ++node_name_idx_map_[input_name];
  auto node = std::make_shared<FlowNodeDesc>(input_name);
  node->SetNodeType(GRAPH_NODE_INPUT);
  node_desc_list_.push_back(node);
  return node;
}

// add output

void FlowGraphDesc::AddOutput(
    const std::string &output_name,
    const std::shared_ptr<FlowPortDesc> &source_node_port) {
  AddOutput(output_name, "cpu", source_node_port);
}

void FlowGraphDesc::AddOutput(
    const std::string &output_name,
    const std::shared_ptr<FlowNodeDesc> &source_node) {
  if (source_node == nullptr) {
    MBLOG_ERROR << "add output " << output_name
                << " failed, source_node is null";
    return;
  }

  AddOutput(output_name, "cpu", (*source_node)[0]);
}

// add flowunit

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::vector<std::string> &config,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &source_node_ports) {
  auto node_name = flowunit_name;
  ++node_name_idx_map_[node_name];
  auto idx = node_name_idx_map_[node_name];
  if (idx != 1) {
    node_name = node_name + std::to_string(idx);
  }

  auto node = std::make_shared<FlowNodeDesc>(node_name);
  node->SetNodeType(GRAPH_NODE_FLOWUNIT);
  node->SetFlowUnitName(flowunit_name);
  node->SetDevice(device);
  node->SetConfig(config);
  node->SetInputLinks(source_node_ports);
  node_desc_list_.push_back(node);
  return node;
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::vector<std::string> &config,
    const std::shared_ptr<FlowNodeDesc> &source_node) {
  if (source_node == nullptr) {
    MBLOG_ERROR << "source node is nullptr";
    return nullptr;
  }

  // all source node output connect to this node input in order
  return AddNode(flowunit_name, device, config, {{"*", (*source_node)["*"]}});
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &source_node_ports) {
  return AddNode(flowunit_name, device, {}, source_node_ports);
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::shared_ptr<FlowNodeDesc> &source_node) {
  if (source_node == nullptr) {
    MBLOG_ERROR << "source node is nullptr";
    return nullptr;
  }

  return AddNode(flowunit_name, device, {}, source_node);
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::vector<std::string> &config) {
  return AddNode(
      flowunit_name, device, config,
      std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>{});
}

// add function

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddFunction(
    const std::function<Status(std::shared_ptr<DataContext>)> &func,
    const std::vector<std::string> &input_name_list,
    const std::vector<std::string> &output_name_list,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &source_node_ports) {
  std::string flowunit_name =
      "register_func_" + std::to_string(function_node_idx_);
  ++function_node_idx_;
  const auto *device = "cpu";
  // register flowunit
  auto func_info = std::make_shared<FlowGraphFunctionInfo>(
      flowunit_name, input_name_list, output_name_list, func);
  function_list_.push_back(func_info);
  // add node
  auto node = std::make_shared<FlowNodeDesc>(flowunit_name);
  node->SetNodeType(GRAPH_NODE_FLOWUNIT);
  node->SetFlowUnitName(flowunit_name);
  node->SetDevice(device);
  node->SetInputLinks(source_node_ports);
  node_desc_list_.push_back(node);
  return node;
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddFunction(
    const std::function<Status(std::shared_ptr<DataContext>)> &func,
    const std::vector<std::string> &input_name_list,
    const std::vector<std::string> &output_name_list,
    const std::shared_ptr<FlowNodeDesc> &source_node) {
  if (source_node == nullptr) {
    MBLOG_ERROR << "function node source_node is null";
    return nullptr;
  }

  return AddFunction(func, input_name_list, output_name_list,
                     {{"*", (*source_node)["*"]}});
}

// inner interface

std::shared_ptr<Configuration> FlowGraphDesc::GetConfig() { return config_; }

void FlowGraphDesc::GetFuncFactoryList(
    std::list<std::shared_ptr<FlowUnitFactory>> &factory_list) {
  for (auto &func_info : function_list_) {
    factory_list.push_back(std::make_shared<RegisterFlowUnitFactory>(
        func_info->GetName(), func_info->GetInputNameList(),
        func_info->GetOutputNameList(), func_info->GetFunc()));
  }
}

std::shared_ptr<GCGraph> FlowGraphDesc::GenGCGraph(
    const std::shared_ptr<modelbox::FlowUnitManager> &flowunit_mgr) {
  auto gcgraph = std::make_shared<GCGraph>();
  gcgraph->Init(nullptr);
  auto graph_config = config_->GetSubConfig("graph");
  gcgraph->SetConfiguration(graph_config);
  auto ret = GenGCNodes(gcgraph);
  if (!ret) {
    return nullptr;
  }

  ret = GenGCEdges(gcgraph, flowunit_mgr);
  if (!ret) {
    return nullptr;
  }

  return gcgraph;
}

Status FlowGraphDesc::GenGCNodes(const std::shared_ptr<GCGraph> &gcgraph) {
  for (auto &node_desc : node_desc_list_) {
    auto gcnode = std::make_shared<GCNode>();
    gcnode->Init(node_desc->GetNodeName(), gcgraph);
    const auto &node_config = node_desc->GetNodeConfig();
    for (const auto &config_item : node_config) {
      auto split_pos = config_item.find('=');
      auto key = config_item.substr(0, split_pos);
      auto value = config_item.substr(split_pos + 1);
      gcnode->SetConfiguration(key, value);
    }
    gcgraph->AddNode(gcnode);
  }

  return STATUS_OK;
}

Status FlowGraphDesc::GenGCEdges(
    const std::shared_ptr<GCGraph> &gcgraph,
    const std::shared_ptr<FlowUnitManager> &flowunit_mgr) {
  for (auto &node_desc : node_desc_list_) {
    auto dest_node_name = node_desc->GetNodeName();
    std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>> input_links;
    auto status = GetInputLinks(node_desc, flowunit_mgr, input_links);
    if (!status) {
      return status;
    }

    for (const auto &link_item : input_links) {
      const auto &dest_port = link_item.first;
      const auto &src_node_port = link_item.second;
      auto dest_node = gcgraph->GetNode(dest_node_name);
      auto src_node = gcgraph->GetNode(src_node_port->GetNodeName());
      dest_node->SetInputPort(dest_port);
      src_node->SetOutputPort(src_node_port->port_name_);

      auto gcedge = std::make_shared<GCEdge>();
      gcedge->Init(gcgraph);
      gcedge->SetHeadNode(src_node);
      gcedge->SetHeadPort(src_node_port->port_name_);
      gcedge->SetTailNode(dest_node);
      gcedge->SetTailPort(dest_port);
      gcgraph->AddEdge(gcedge);
    }
  }

  return STATUS_OK;
}

Status FlowGraphDesc::GetInputLinks(
    const std::shared_ptr<FlowNodeDesc> &dest_node_desc,
    const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
    std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &input_links) {
  const auto &origin_input_links = dest_node_desc->GetInputLinks();
  auto generate_match = origin_input_links.find("*");
  if (!(origin_input_links.size() == 1 &&
        generate_match != origin_input_links.end())) {
    input_links = origin_input_links;
    return FormatInputLinks(flowunit_mgr, input_links);
  }

  /**
   * user set src_node -> dest_node
   * we need get port to port info
   **/
  auto dest_node_fu_desc = GetFlowUnitDesc(dest_node_desc, flowunit_mgr);
  if (dest_node_fu_desc == nullptr) {
    return STATUS_NOTFOUND;
  }
  const auto &dest_input_port_list = dest_node_fu_desc->GetFlowUnitInput();
  if (dest_input_port_list.empty()) {
    MBLOG_ERROR << "dest node " << dest_node_desc->GetNodeName()
                << " has no input";
    return STATUS_FAULT;
  }

  const auto &src_node_port = generate_match->second;
  auto src_node_desc = src_node_port->GetNode();
  if (src_node_desc->type_ == GRAPH_NODE_INPUT) {
    if (dest_input_port_list.size() != 1) {
      MBLOG_ERROR << "node " << dest_node_desc->GetNodeName()
                  << " has multi input port, please specify the port that "
                     "input node connect to";
      return STATUS_FAULT;
    }

    input_links[dest_input_port_list[0].GetPortName()] =
        std::make_shared<FlowPortDesc>(src_node_desc,
                                       src_node_desc->GetNodeName());
    return STATUS_OK;
  }

  auto src_node_fu_desc = GetFlowUnitDesc(src_node_desc, flowunit_mgr);
  if (src_node_fu_desc == nullptr) {
    return STATUS_NOTFOUND;
  }
  const auto &src_output_port_list = src_node_fu_desc->GetFlowUnitOutput();
  if (src_output_port_list.size() != dest_input_port_list.size()) {
    MBLOG_ERROR << "src node " << src_node_desc->GetNodeName()
                << " input port count and dest node "
                << dest_node_desc->GetNodeName()
                << " output port count not equal";
    return STATUS_FAULT;
  }

  for (size_t i = 0; i < src_output_port_list.size(); ++i) {
    auto src_output_port_name = src_output_port_list[i].GetPortName();
    auto dest_input_port_name = dest_input_port_list[i].GetPortName();
    input_links[dest_input_port_name] =
        std::make_shared<FlowPortDesc>(src_node_desc, src_output_port_name);
  }

  return STATUS_OK;
}

Status FlowGraphDesc::FormatInputLinks(
    const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
    std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &input_links) {
  for (auto &input_link : input_links) {
    const auto &dest_port_name = input_link.first;
    auto src_node_port = input_link.second;
    if (src_node_port == nullptr) {
      MBLOG_ERROR << "src port connect to " << dest_port_name << " is nullptr";
      return STATUS_FAULT;
    }

    if (src_node_port->IsDescribeInName()) {
      continue;
    }

    // need translate port id to port name
    auto port_idx = src_node_port->GetPortIdx();
    auto src_node_desc = src_node_port->GetNode();
    auto flowunit_desc = GetFlowUnitDesc(src_node_desc, flowunit_mgr);
    if (flowunit_desc == nullptr) {
      return STATUS_NOTFOUND;
    }

    const auto &outputs = flowunit_desc->GetFlowUnitOutput();
    if (outputs.size() <= port_idx) {
      MBLOG_ERROR << "node " << src_node_desc->GetNodeName() << " has "
                  << outputs.size() << " port, idx " << port_idx
                  << " is out of range";
      return STATUS_NOTFOUND;
    }

    auto format_src_node_port = std::make_shared<FlowPortDesc>(
        src_node_desc, outputs[port_idx].GetPortName());
    input_link.second = format_src_node_port;
  }

  return STATUS_OK;
}

std::shared_ptr<FlowUnitDesc> FlowGraphDesc::GetFlowUnitDesc(
    const std::shared_ptr<FlowNodeDesc> &node_desc,
    const std::shared_ptr<FlowUnitManager> &flowunit_mgr) {
  // to support multi device config
  auto node_fu_name = node_desc->GetFlowUnitName();
  auto device_info_list = StringSplit(node_desc->device_, ';');
  if (device_info_list.empty()) {
    MBLOG_ERROR << "flowunit: " << node_fu_name << ", device config error, ["
                << node_desc->device_ << "]";
    return nullptr;
  }

  std::string device_name;
  for (auto &device_info : device_info_list) {
    auto device_info_item = StringSplit(device_info, ':');
    if (device_info_item.empty()) {
      continue;
    }

    auto device_name = device_info_item[0];
    auto fu_desc = flowunit_mgr->GetFlowUnitDesc(device_name, node_fu_name);
    if (fu_desc != nullptr) {
      return fu_desc;
    }
  }

  MBLOG_ERROR << "can not find flowunit: " << node_fu_name
              << ", device: " << node_desc->device_;
  return nullptr;
}

void FlowGraphDesc::AddOutput(
    const std::string &output_name, const std::string &device,
    const std::shared_ptr<FlowPortDesc> &source_node_port) {
  if (source_node_port == nullptr) {
    MBLOG_ERROR << "add output " << output_name
                << " failed, source_node_port is null";
    return;
  }

  if (node_name_idx_map_[output_name] >= 1) {
    MBLOG_ERROR << "Output name " << output_name << " has been used";
    return;
  }

  ++node_name_idx_map_[output_name];
  auto node = std::make_shared<FlowNodeDesc>(output_name);
  node->SetNodeType(GRAPH_NODE_OUTPUT);
  node->SetDevice(device);
  node->SetInputLinks({{output_name, source_node_port}});
  node_desc_list_.push_back(node);
}

}  // namespace modelbox
