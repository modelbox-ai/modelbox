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

#include "modelbox/base/register_flowunit.h"

namespace modelbox {

constexpr const char *CONFIG_KEY_QUEUE_SIZE = "graph.queue_size";
constexpr const char *CONFIG_KEY_BATCH_SIZE = "graph.batch_size";
constexpr const char *CONFIG_KEY_DRIVERS_DIR = "drivers.dir";
constexpr const char *CONFIG_KEY_DRIVERS_SKIP_DEFAULT = "drivers.skip-default";

FlowConfig::FlowConfig() { content_ = ConfigurationBuilder().Build(); }

void FlowConfig::SetQueueSize(size_t queue_size) {
  content_->SetProperty(CONFIG_KEY_QUEUE_SIZE, queue_size);
}

void FlowConfig::SetBatchSize(size_t batch_size) {
  content_->SetProperty(CONFIG_KEY_BATCH_SIZE, batch_size);
}

void FlowConfig::SetDriversDir(
    const std::vector<std::string> &drivers_dir_list) {
  content_->SetProperty(CONFIG_KEY_DRIVERS_DIR, drivers_dir_list);
}

void FlowConfig::SetSkipDefaultDrivers(bool is_skip) {
  content_->SetProperty(CONFIG_KEY_DRIVERS_SKIP_DEFAULT, is_skip);
}

FlowGraphDesc::FlowGraphDesc() { config_ = ConfigurationBuilder().Build(); }

FlowGraphDesc::~FlowGraphDesc() {
  for (auto &node_desc : node_desc_list_) {
    node_desc->Clear();
  }
}

Status FlowGraphDesc::Init() {
  auto flow_cfg = std::make_shared<FlowConfig>();
  return Init(flow_cfg);
}

Status FlowGraphDesc::Init(std::shared_ptr<FlowConfig> config) {
  if (is_init_) {
    return STATUS_OK;
  }

  config_ = config->content_;
  drivers_ = std::make_shared<Drivers>();
  device_mgr_ = std::make_shared<DeviceManager>();
  flowunit_mgr_ = std::make_shared<FlowUnitManager>();

  auto ret = drivers_->Initialize(config_->GetSubConfig("drivers"));
  if (!ret) {
    MBLOG_ERROR << "init drivers failed, ret " << ret;
    return ret;
  }

  ret = drivers_->Scan();
  if (!ret) {
    MBLOG_ERROR << "scan driver failed, ret " << ret;
    return ret;
  }

  ret = device_mgr_->Initialize(drivers_, nullptr);
  if (!ret) {
    MBLOG_ERROR << "device mgr init failed, ret " << ret;
    return ret;
  }

  ret = flowunit_mgr_->Initialize(drivers_, device_mgr_, nullptr);
  if (!ret) {
    MBLOG_ERROR << "flowunit mgr init failed, ret " << ret;
    return ret;
  }

  build_status_ = STATUS_OK;
  is_init_ = true;
  return STATUS_OK;
}

// add input

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddInput(
    const std::string &input_name) {
  if (!is_init_) {
    MBLOG_ERROR << "Should call init first";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  if (node_name_idx_map_[input_name] >= 1) {
    MBLOG_ERROR << "Input name " << input_name << " has been used";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  ++node_name_idx_map_[input_name];
  auto node = std::make_shared<FlowNodeDesc>(input_name);
  node->SetNodeType(GRAPH_NODE_INPUT);
  node->SetOutputPortNames({input_name});
  node_desc_list_.push_back(node);
  return node;
}

// add output

void FlowGraphDesc::AddOutput(const std::string &output_name,
                              std::shared_ptr<FlowPortDesc> source_node_port) {
  AddOutput(output_name, "cpu", source_node_port);
}

void FlowGraphDesc::AddOutput(const std::string &output_name,
                              std::shared_ptr<FlowNodeDesc> source_node) {
  if (source_node == nullptr) {
    MBLOG_ERROR << "add output " << output_name
                << " failed, source_node is null";
    build_status_ = STATUS_FAULT;
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
  if (!is_init_) {
    MBLOG_ERROR << "Should call init first";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  auto node_name = flowunit_name;
  ++node_name_idx_map_[node_name];
  auto idx = node_name_idx_map_[node_name];
  if (idx != 1) {
    node_name = node_name + std::to_string(idx);
  }

  auto flowunit_desc = flowunit_mgr_->GetFlowUnitDesc(device, flowunit_name);
  if (flowunit_desc == nullptr) {
    MBLOG_ERROR << "Can not find flowunit " << flowunit_name << ", device "
                << device;
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
      format_source_node_ports;
  auto ret = FormatInputLinks(flowunit_name, flowunit_desc, source_node_ports,
                              format_source_node_ports);
  if (!ret) {
    return nullptr;
  }

  auto output_list = flowunit_desc->GetFlowUnitOutput();
  std::vector<std::string> output_name_list;
  output_name_list.reserve(output_list.size());
  for (auto &output_desc : output_list) {
    output_name_list.push_back(output_desc.GetPortName());
  }

  auto node = std::make_shared<FlowNodeDesc>(node_name);
  node->SetNodeType(GRAPH_NODE_FLOWUNIT);
  node->SetFlowUnitName(flowunit_name);
  node->SetDevice(device);
  node->SetConfig(config);
  node->SetInputLinks(format_source_node_ports);
  node->SetOutputPortNames(output_name_list);
  node_desc_list_.push_back(node);
  return node;
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::vector<std::string> &config,
    std::shared_ptr<FlowNodeDesc> source_node) {
  return AddNode(flowunit_name, device, config, {{"", (*source_node)[0]}});
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &source_node_ports) {
  return AddNode(flowunit_name, device, {}, source_node_ports);
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddNode(
    const std::string &flowunit_name, const std::string &device,
    std::shared_ptr<FlowNodeDesc> source_node) {
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
  if (!is_init_) {
    MBLOG_ERROR << "Should call init first";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  std::string flowunit_name =
      "register_func_" + std::to_string(function_node_idx_);
  ++function_node_idx_;
  auto device = "cpu";
  // register flowunit
  auto flowunit_factory = std::make_shared<RegisterFlowUnitFactory>(
      flowunit_name, input_name_list, output_name_list, func);
  flowunit_mgr_->InsertFlowUnitFactory(flowunit_name, device, flowunit_factory);
  // check link
  if (!CheckInputLinks(input_name_list, source_node_ports)) {
    return nullptr;
  }
  // add node
  auto node = std::make_shared<FlowNodeDesc>(flowunit_name);
  node->SetNodeType(GRAPH_NODE_FLOWUNIT);
  node->SetFlowUnitName(flowunit_name);
  node->SetDevice(device);
  node->SetInputLinks(source_node_ports);
  node->SetOutputPortNames(output_name_list);
  node_desc_list_.push_back(node);
  return node;
}

std::shared_ptr<FlowNodeDesc> FlowGraphDesc::AddFunction(
    const std::function<Status(std::shared_ptr<DataContext>)> &func,
    const std::vector<std::string> &input_name_list,
    const std::vector<std::string> &output_name_list,
    std::shared_ptr<FlowNodeDesc> source_node) {
  if (input_name_list.empty()) {
    MBLOG_ERROR << "function node input ports is not defined";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  if (source_node == nullptr) {
    MBLOG_ERROR << "function node source_node is null";
    build_status_ = STATUS_FAULT;
    return nullptr;
  }

  return AddFunction(func, input_name_list, output_name_list,
                     {{input_name_list.front(), (*source_node)[0]}});
}

// get status

Status FlowGraphDesc::GetStatus() { return build_status_; }

// inner interface

std::shared_ptr<Configuration> FlowGraphDesc::GetConfig() { return config_; }

std::shared_ptr<GCGraph> FlowGraphDesc::GetGCGraph() {
  auto gcgraph = std::make_shared<GCGraph>();
  gcgraph->Init(nullptr);
  auto graph_config = config_->GetSubConfig("graph");
  gcgraph->SetConfiguration(graph_config);
  GenGCNodes(gcgraph);
  GenGCEdges(gcgraph);
  return gcgraph;
}

void FlowGraphDesc::GenGCNodes(std::shared_ptr<GCGraph> gcgraph) {
  for (auto &node_desc : node_desc_list_) {
    auto gcnode = std::make_shared<GCNode>();
    gcnode->Init(node_desc->GetNodeName(), gcgraph);
    const auto &node_config = node_desc->GetNodeConfig();
    for (auto config_item : node_config) {
      auto split_pos = config_item.find('=');
      auto key = config_item.substr(0, split_pos);
      auto value = config_item.substr(split_pos + 1);
      gcnode->SetConfiguration(key, value);
    }
    gcgraph->AddNode(gcnode);
  }
}

void FlowGraphDesc::GenGCEdges(std::shared_ptr<GCGraph> gcgraph) {
  for (auto &node_desc : node_desc_list_) {
    auto dest_node_name = node_desc->GetNodeName();
    const auto &input_links = node_desc->GetInputLinks();
    for (auto &link_item : input_links) {
      auto &dest_port = link_item.first;
      auto &src_node_port = link_item.second;
      auto dest_node = gcgraph->GetNode(dest_node_name);
      auto src_node = gcgraph->GetNode(src_node_port->node_name_);
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
}

std::shared_ptr<Drivers> FlowGraphDesc::GetDrivers() { return drivers_; }

std::shared_ptr<DeviceManager> FlowGraphDesc::GetDeviceManager() {
  return device_mgr_;
}

std::shared_ptr<FlowUnitManager> FlowGraphDesc::GetFlowUnitManager() {
  return flowunit_mgr_;
}

void FlowGraphDesc::AddOutput(const std::string &output_name,
                              const std::string &device,
                              std::shared_ptr<FlowPortDesc> source_node_port) {
  if (!is_init_) {
    MBLOG_ERROR << "Should call init first";
    build_status_ = STATUS_FAULT;
    return;
  }

  if (source_node_port == nullptr) {
    MBLOG_ERROR << "add output " << output_name
                << " failed, source_node_port is null";
    build_status_ = STATUS_FAULT;
    return;
  }

  if (node_name_idx_map_[output_name] >= 1) {
    MBLOG_ERROR << "Output name " << output_name << " has been used";
    build_status_ = STATUS_FAULT;
    return;
  }

  ++node_name_idx_map_[output_name];
  auto node = std::make_shared<FlowNodeDesc>(output_name);
  node->SetNodeType(GRAPH_NODE_OUTPUT);
  node->SetDevice(device);
  node->SetInputLinks({{output_name, source_node_port}});
  node_desc_list_.push_back(node);
}

bool FlowGraphDesc::FormatInputLinks(
    const std::string &flowunit_name,
    std::shared_ptr<FlowUnitDesc> flowunit_desc,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &origin_source_node_ports,
    std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &format_source_node_ports) {
  auto input_list = flowunit_desc->GetFlowUnitInput();
  std::unordered_set<std::string> input_names;
  for (auto input_desc : input_list) {
    input_names.insert(input_desc.GetPortName());
  }
  size_t port_idx = 0;
  for (auto &source_node_port_item : origin_source_node_ports) {
    auto my_input_port_name = source_node_port_item.first;
    if (!my_input_port_name.empty() &&
        input_names.find(my_input_port_name) == input_names.end()) {
      MBLOG_ERROR << "flowunit " << flowunit_name
                  << " does not have input port " << my_input_port_name;
      build_status_ = STATUS_FAULT;
      return false;
    }

    if (my_input_port_name.empty()) {
      // auto fill by port index
      my_input_port_name = input_list[port_idx].GetPortName();
    }

    auto source_node_port = source_node_port_item.second;
    if (source_node_port == nullptr) {
      MBLOG_ERROR << "flowunit " << flowunit_name << ", port "
                  << my_input_port_name << ", link source_node_port is null";
      build_status_ = STATUS_FAULT;
      return false;
    }

    format_source_node_ports[my_input_port_name] = source_node_port;
    ++port_idx;
  }

  return true;
}

bool FlowGraphDesc::CheckInputLinks(
    const std::vector<std::string> &defined_ports,
    const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
        &input_links) {
  std::unordered_set<std::string> input_names;
  for (auto input_name : defined_ports) {
    input_names.insert(input_name);
  }

  for (auto &link_item : input_links) {
    auto &port_name = link_item.first;
    if (input_names.find(port_name) == input_names.end()) {
      MBLOG_ERROR << "function node, source_node_ports connect to a port ["
                  << port_name << "] not defined";
      build_status_ = STATUS_FAULT;
      return false;
    }
  }

  return true;
}

}  // namespace modelbox