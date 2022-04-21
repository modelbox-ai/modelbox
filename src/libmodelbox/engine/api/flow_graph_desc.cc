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
#include <securec.h>

#include "modelbox/base/log.h"
#include "modelbox/base/register_flowunit.h"
#include "modelbox/base/utils.h"
#include "modelbox/flow_node_desc.h"
#include "scheduler/flow_scheduler.h"

namespace modelbox {

constexpr const char *GRAPH_KEY_DEVICE = "device";
constexpr const char *GRAPH_KEY_DEVICE_ID = "deviceid";
constexpr const char *GRAPH_KEY_QUEUE_SIZE = "queue_size";
constexpr const char *GRAPH_KEY_BATCH_SIZE = "batch_size";
constexpr const char *GRAPH_KEY_CHECK_NODE_OUTPUT = "need_check_output";
constexpr const char *GRAPH_VIRTUAL_NODE = "inner_virtual_node_";
constexpr const char *GRAPH_NODE_REGISTER_FLOWUNIT = "register_flowunit";

static std::shared_ptr<NodeDesc> err_msg(const Status &status) {
  auto data_handler = std::make_shared<NodeDesc>();
  data_handler->SetError(status);
  MBLOG_ERROR << status.Errormsg();
  return data_handler;
}

FlowGraphDesc::FlowGraphDesc() {}

FlowGraphDesc::~FlowGraphDesc() { Close(); }

Status FlowGraphDesc::Init(std::shared_ptr<Configuration> &config) {
  config_ = config;
  drivers_ = std::make_shared<Drivers>();
  device_mgr_ = std::make_shared<DeviceManager>();
  flowunit_mgr_ = std::make_shared<FlowUnitManager>();
  std::string msg = "";
  if (!(drivers_ && device_mgr_ && flowunit_mgr_)) {
    msg = "drivers, flowunit_mgr , device is null, return";
    MBLOG_ERROR << msg;
    return STATUS_FAULT;
  }

  auto ret = drivers_->Initialize(config_->GetSubConfig("drivers"));
  if (!ret) {
    msg = "driver init failed";
    MBLOG_ERROR << msg << ", " << ret.WrapErrormsgs();
    return {ret, msg};
  }

  Defer {
    if (ret == STATUS_OK) {
      return;
    }
    Close();
  };

  ret = drivers_->Scan();
  if (!ret) {
    msg = "Scan driver failed.";
    MBLOG_ERROR << msg << ": " << ret.WrapErrormsgs();
    return {ret, msg};
  }

  ret = device_mgr_->Initialize(drivers_, nullptr);
  if (!ret) {
    msg = "Inital device failed";
    MBLOG_ERROR << msg << ", " << ret.WrapErrormsgs();
    return {ret, msg};
  }

  ret = flowunit_mgr_->Initialize(drivers_, device_mgr_, nullptr);
  if (!ret) {
    msg = "Initial flowunit manager failed ";
    MBLOG_ERROR << msg << ", " << ret.WrapErrormsgs();
    return {ret, msg};
  }

  return STATUS_OK;
}

std::shared_ptr<DeviceManager> FlowGraphDesc::GetDeviceManager() {
  return device_mgr_;
}
std::shared_ptr<FlowUnitManager> FlowGraphDesc::GetFlowUnitManager() {
  return flowunit_mgr_;
}
std::shared_ptr<Configuration> FlowGraphDesc::GetConfig() { return config_; }

std::shared_ptr<FlowUnitDesc> FlowGraphDesc::GetFlowunitDesc(
    const std::string &name, const std::map<std::string, std::string> &config) {
  std::string msg = "";
  if (flowunit_mgr_ == nullptr) {
    MBLOG_ERROR << "failed get flowunit manager, please init Unit firstly";
    return nullptr;
  }
  auto device_iter = config.find("device");
  if (device_iter == config.end()) {
    return nullptr;
  }
  auto device = device_iter->second;
  auto desc = flowunit_mgr_->GetFlowUnitDesc(device_iter->second, name);
  if (desc == nullptr) {
    MBLOG_ERROR << "failed find flowunit " << name << " description";
    return nullptr;
  }

  return desc;
}

Status FlowGraphDesc::AddCallBackFactory(
    const std::string unit_name, const std::set<std::string> input_ports,
    const std::set<std::string> output_ports,
    std::function<StatusCode(std::shared_ptr<DataContext>)> &callback) {
  auto factory = std::make_shared<RegisterFlowUnitFactory>(
      unit_name, input_ports, output_ports, callback);

  std::string type = "cpu";
  std::string name = unit_name;

  flowunit_mgr_->InsertFlowUnitFactory(
      name, type, std::dynamic_pointer_cast<FlowUnitFactory>(factory));

  return STATUS_SUCCESS;
}

Status FlowGraphDesc::InsertGraphEdge(std::shared_ptr<GCGraph> &root_graph,
                                      std::shared_ptr<GCNode> &input_node,
                                      std::string &input_port,
                                      std::shared_ptr<GCNode> &output_node,
                                      std::string &output_port) {
  if (root_graph == nullptr || input_node == nullptr ||
      output_node == nullptr) {
    return STATUS_FAULT;
  }

  auto gcedge = std::make_shared<GCEdge>();
  if (STATUS_OK != gcedge->Init(root_graph)) {
    return STATUS_FAULT;
  }

  gcedge->SetHeadPort(input_port);
  gcedge->SetTailPort(output_port);
  gcedge->SetHeadNode(input_node);
  gcedge->SetTailNode(output_node);
  root_graph->AddEdge(gcedge);
  return STATUS_OK;
}

Status FlowGraphDesc::CheckInputPort(
    const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
    const std::shared_ptr<NodeDesc> &data_handler) {
  auto flowunit_input = flowunit_desc->GetFlowUnitInput();
  if (data_handler->GetPortNames().size() == 1 && flowunit_input.size() == 1) {
    return STATUS_OK;
  }

  if (data_handler->GetPortNames().size() != 0) {
    auto inport_names = data_handler->GetPortNames();
    if (inport_names.size() != flowunit_input.size()) {
      MBLOG_ERROR << "not all input port has data";
      return STATUS_INVALID;
    }
    for (auto &iter : flowunit_input) {
      if (inport_names.find(iter.GetPortName()) == inport_names.end()) {
        MBLOG_ERROR << "port:" << iter.GetPortName() << " has link any node";
        return STATUS_INVALID;
      }
    }
  }
  return STATUS_OK;
}

std::shared_ptr<GCNode> FlowGraphDesc::CreateGCNode(
    const std::string name, std::set<std::string> input_ports,
    std::set<std::string> out_ports,
    const std::map<std::string, std::string> &config,
    const std::shared_ptr<NodeDesc> &data_handler) {
  if (gcgraph_ == nullptr) {
    MBLOG_ERROR << "gcgraph is not created";
    return nullptr;
  }

  auto gcnode = std::make_shared<GCNode>();

  for (auto &input : input_ports) {
    gcnode->SetInputPort(input);
  }

  for (auto &output : out_ports) {
    gcnode->SetOutputPort(output);
  }

  gcnode->Init(name, gcgraph_);
  for (auto &iter : config) {
    gcnode->SetConfiguration(iter.first, iter.second);
  }
  gcnode->SetNodeType(GRAPH_NODE_REGISTER_FLOWUNIT);
  gcgraph_->AddNode(gcnode);
  return gcnode;
}

Status FlowGraphDesc::AddToGCGraph(
    const std::string name, std::set<std::string> inputs,
    std::set<std::string> outputs,
    const std::map<std::string, std::string> &config,
    const std::shared_ptr<NodeDesc> &data_handler) {
  auto gcnode = CreateGCNode(name, inputs, outputs, config, data_handler);
  if (gcnode == nullptr) {
    MBLOG_ERROR << "Creat GCNode failed";
    return STATUS_FAULT;
  }

  if (data_handler == nullptr) {
    return STATUS_SUCCESS;
  }

  std::unordered_map<std::string, std::string> port_to_port_map;
  std::unordered_map<std::string, std::string> port_to_node_map;
  if (data_handler->GetNodeType() == NODE_OUTPUT) {
    auto port_name = gcnode->GetNodeName();
    auto last_port_name = *(data_handler->GetPortNames().begin());
    auto last_node_name = data_handler->GetNodeName();
    port_to_port_map[port_name] = last_port_name;
    port_to_node_map[port_name] = last_node_name;
  } else {
    port_to_port_map = data_handler->GetPortMap();
    port_to_node_map = data_handler->port_to_node_;
  }

  for (auto &iter : port_to_port_map) {
    std::string node_name = port_to_node_map[iter.first];
    auto outport_name = iter.first;
    auto inport_name = iter.second;
    auto input_node = gcgraph_->GetNode(node_name);
    auto output_node = gcnode;
    if (InsertGraphEdge(gcgraph_, input_node, inport_name, gcnode,
                        outport_name) != STATUS_OK) {
      MBLOG_ERROR << "InsertGrahEdge failed";
      return STATUS_FAULT;
    }
  }

  return STATUS_SUCCESS;
}

Status FlowGraphDesc::CheckBuffer(const std::shared_ptr<FlowUnitDesc> &desc,
                                  const std::shared_ptr<NodeDesc> &data) {
  if (data->GetNodeType() == NODE_OUTPUT) {
    auto input_num = desc->GetFlowUnitInput().size();
    auto output_num = data->GetPortNames().size();
    if (desc->GetFlowUnitInput().size() != 1 ||
        data->GetPortNames().size() != 1) {
      MBLOG_ERROR << "node: " << desc->GetFlowUnitName()
                  << "must use correct input: "
                  << "input_num(" << std::to_string(input_num)
                  << "),output_num(" << std::to_string(output_num) << ").";
      return STATUS_FAULT;
    }
  }
  return STATUS_OK;
}

void FlowGraphDesc::BindDataHanlder(std::shared_ptr<NodeDesc> &data_handler,
                                    std::shared_ptr<GCNode> &gcnode) {
  auto outports = gcnode->GetOutputPorts();
  auto outport_names =
      std::set<std::string>(outports->begin(), outports->end());
  auto name = gcnode->GetNodeName();
  data_handler->SetPortNames(outport_names);

  data_handler->SetNodeName(name);
  data_handler->SetNodeType(NODE_OUTPUT);
}

std::shared_ptr<NodeDesc> FlowGraphDesc::AddNode(
    const std::string &name, std::map<std::string, std::string> config_map,
    const std::map<std::string, std::shared_ptr<NodeDesc>> &buffers) {
  if (buffers.size() > 0) {
    auto data = std::make_shared<NodeDesc>();
    auto ret = data->SetNodeDesc(buffers);
    if (ret != STATUS_OK) {
      data->SetError(ret);
      return data;
    }

    return AddNode(name, config_map, data);
  } else {
    return AddNode(name, config_map);
  }
}

static void SetDefaultConfigValue(
    const std::string &name, std::map<std::string, std::string> &config_map) {
  if (config_map.find("type") == config_map.end()) {
    config_map["type"] = "flowunit";
  }
  if (config_map.find("device") == config_map.end()) {
    config_map["device"] = "cpu";
  }
  if (config_map.find("deviceid") == config_map.end()) {
    config_map["deviceid"] = "0";
  }
  if (config_map.find("flowunit") == config_map.end()) {
    config_map["flowunit"] = name;
  }
}

Status FlowGraphDesc::CheckInputFlowUnit(
    const std::string &name, std::map<std::string, std::string> &config_map,
    const std::shared_ptr<NodeDesc> &buffers,
    const std::shared_ptr<FlowUnitDesc> &desc) {
  if (buffers != nullptr) {
    if (buffers->GetError() != STATUS_SUCCESS) {
      auto err_msg = "node: " + name + ", input data has error.";
      MBLOG_ERROR << err_msg;
      return {STATUS_FAULT, err_msg};
    }

    if (graph_ != nullptr) {
      std::string err_msg = "graph has been build, flowunit " + name +
                            " cannot been linked to this graph";
      MBLOG_ERROR << err_msg;
      return {STATUS_FAULT, err_msg};
    }
  }

  if (desc->GetFlowUnitInput().size() > 0 && buffers == nullptr) {
    auto msg = "must set input for flowunit: " + name;
    MBLOG_ERROR << msg;
    return {STATUS_FAULT, msg};
  }

  return STATUS_OK;
}

std::shared_ptr<NodeDesc> FlowGraphDesc::ExecuteStreamNode(
    const std::shared_ptr<FlowUnitDesc> &desc,
    const std::shared_ptr<NodeDesc> &buffers,
    std::map<std::string, std::string> &config_map) {
  auto stream_data_handler = std::make_shared<NodeDesc>();
  if (buffers == nullptr) {
    if (gcgraph_ == nullptr) {
      gcgraph_ = std::make_shared<GCGraph>();
      gcgraph_->Init(nullptr);
    }

    auto graph_config = config_->GetSubConfig("graph");
    if (graph_config != nullptr) {
      gcgraph_->SetConfiguration(config_);
    }
    std::string name = desc->GetFlowUnitName();
    std::set<std::string> inports;
    std::set<std::string> outports;
    auto inputs = desc->GetFlowUnitInput();
    auto outputs = desc->GetFlowUnitOutput();
    for (auto &iter : inputs) {
      inports.emplace(iter.GetPortName());
    }
    for (auto &iter : outputs) {
      outports.emplace(iter.GetPortName());
    }

    stream_data_handler->SetPortNames(outports);

    auto gcnode =
        CreateGCNode(name, inports, outports, config_map, stream_data_handler);
    if (gcnode == nullptr) {
      auto msg = "Create GCNode failed";
      return err_msg({STATUS_FAULT, msg});
    }
    BindDataHanlder(stream_data_handler, gcnode);
    return stream_data_handler;
  }

  if (CheckInputPort(desc, buffers) != STATUS_OK) {
    return err_msg({STATUS_FAULT, "not all port has data"});
  }
  std::string name = desc->GetFlowUnitName();
  std::set<std::string> inports;
  std::set<std::string> outports;
  auto inputs = desc->GetFlowUnitInput();
  auto outputs = desc->GetFlowUnitOutput();
  for (auto &iter : inputs) {
    inports.emplace(iter.GetPortName());
  }
  for (auto &iter : outputs) {
    outports.emplace(iter.GetPortName());
  }

  auto status = AddToGCGraph(name, inports, outports, config_map, buffers);
  if (status != status) {
    MBLOG_ERROR << "add node to graph failed";
    stream_data_handler->SetError({STATUS_INVALID, status.Errormsg()});
  }
  auto gcnode = gcgraph_->GetNode(name);
  BindDataHanlder(stream_data_handler, gcnode);
  return stream_data_handler;
}

std::shared_ptr<NodeDesc> FlowGraphDesc::AddNode(
    const std::string &name, std::map<std::string, std::string> config_map,
    const std::shared_ptr<NodeDesc> &buffers) {
  SetDefaultConfigValue(name, config_map);

  auto flowunit_desc = GetFlowunitDesc(name, config_map);
  if (flowunit_desc == nullptr) {
    return err_msg(
        {STATUS_INVALID, "failed find flowunit " + name + " description"});
  }
  if (buffers != nullptr) {
    auto ret = CheckInputFlowUnit(name, config_map, buffers, flowunit_desc);
    if (ret != STATUS_OK) {
      return err_msg(ret);
    }
  }
  return ExecuteStreamNode(flowunit_desc, buffers, config_map);
}

std::shared_ptr<NodeDesc> FlowGraphDesc::AddNode(
    std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
    std::vector<std::string> inports, std::vector<std::string> outports,
    const std::map<std::string, std::shared_ptr<NodeDesc>> &buffers) {
  if (buffers.size() > 0) {
    auto data = std::make_shared<NodeDesc>();
    auto ret = data->SetNodeDesc(buffers);
    if (ret != STATUS_OK) {
      data->SetError(ret);
      return data;
    }

    return AddNode(callback, inports, outports, data);
  } else {
    return AddNode(callback, inports, outports);
  }
}

std::shared_ptr<NodeDesc> FlowGraphDesc::AddNode(
    std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
    std::vector<std::string> inports, std::vector<std::string> outports,
    const std::shared_ptr<NodeDesc> &data) {
  std::string name = "register_" + std::to_string(node_sequence_++);
  std::set<std::string> input_ports(inports.begin(), inports.end());
  std::set<std::string> output_ports(outports.begin(), outports.end());
  AddCallBackFactory(name, input_ports, output_ports, callback);
  auto data_handler = std::make_shared<NodeDesc>();
  data_handler->SetNodeName(name);

  if (data == nullptr) {
    if (gcgraph_ == nullptr) {
      gcgraph_ = std::make_shared<GCGraph>();
      gcgraph_->Init(nullptr);
    }

    auto graph_config = config_->GetSubConfig("graph");
    if (graph_config != nullptr) {
      gcgraph_->SetConfiguration(config_);
    }

    auto gcnode = CreateGCNode(
        name, input_ports, output_ports,
        {{"type", GRAPH_NODE_REGISTER_FLOWUNIT}, {"flowunit", name}},
        data_handler);
    BindDataHanlder(data_handler, gcnode);
    return data_handler;
  }

  auto status = AddToGCGraph(
      name, input_ports, output_ports,
      {{"type", GRAPH_NODE_REGISTER_FLOWUNIT}, {"flowunit", name}}, data);
  if (status != STATUS_SUCCESS) {
    auto msg = "create gcnode failed";
    MBLOG_ERROR << msg;
    data_handler->SetError({STATUS_INVALID, msg});
    return data_handler;
  }

  auto gcnode = gcgraph_->GetNode(name);
  BindDataHanlder(data_handler, gcnode);
  return data_handler;
}

void FlowGraphDesc::Close() {
  if (closed_) {
    return;
  }
  nodes_config_.clear();

  flowunit_mgr_->Clear();
  flowunit_mgr_ = nullptr;
  device_mgr_->Clear();
  device_mgr_ = nullptr;

  if (drivers_) {
    drivers_->Clear();
  }
  drivers_ = nullptr;

  closed_ = true;
}

std::shared_ptr<NodeDesc> FlowGraphDesc::BindOutput(
    std::shared_ptr<NodeDesc> &node, const std::string port_name) {
  auto virtual_node_name = node->GetNodeName() + "_" + port_name;
  auto gcnode = std::make_shared<GCNode>();
  gcnode->Init(virtual_node_name, gcgraph_);
  gcnode->SetConfiguration("type", "output");

  auto gcedge = std::make_shared<GCEdge>();
  auto input_node_name = node->GetNodeName();
  auto input_node = gcgraph_->GetNode(input_node_name);

  gcedge->SetTailNode(gcnode);
  gcedge->SetHeadNode(input_node);
  gcedge->SetHeadPort(port_name);
  gcedge->SetTailPort(virtual_node_name);
  if (port_name == "__default__outport__") {
    if (input_node->GetOutputPorts()->size() != 1) {
      MBLOG_ERROR << "node " << input_node->GetNodeName()
                  << "has not only one outport, please specify port name";
      return nullptr;
    }
    auto node_port_name = *(input_node->GetOutputPorts()->begin());
    gcedge->SetHeadPort(node_port_name);
  }

  auto output = std::make_shared<NodeDesc>();
  output->SetNodeName(virtual_node_name);
  output->SetNodeType(NODE_OUTPUT);
  gcgraph_->AddNode(gcnode);
  gcgraph_->AddEdge(gcedge);

  return output;
}

std::shared_ptr<NodeDesc> FlowGraphDesc::BindInput(
    std::shared_ptr<NodeDesc> &node, const std::string port_name) {
  auto virtual_node_name = node->GetNodeName() + "_" + port_name;
  auto gcnode = std::make_shared<GCNode>();
  gcnode->Init(virtual_node_name, gcgraph_);
  gcnode->SetConfiguration("type", "input");

  auto gcedge = std::make_shared<GCEdge>();
  auto input_node_name = node->GetNodeName();
  auto input_node = gcgraph_->GetNode(input_node_name);

  gcedge->SetHeadNode(gcnode);
  gcedge->SetTailNode(input_node);
  gcedge->SetHeadPort(virtual_node_name);

  auto linked_gcnode = gcgraph_->GetNode(node->GetNodeName());
  if (linked_gcnode == nullptr) {
    MBLOG_ERROR << "failed find node: " << node->GetNodeName() << " in graph";
    return nullptr;
  }

  if (port_name != "__default__inport__") {
    gcedge->SetTailPort(port_name);
  } else {
    if (linked_gcnode->GetInputPorts()->size() != 1) {
      MBLOG_ERROR << "node " << linked_gcnode->GetNodeName()
                  << "has not only one inport, please specify port name";
      return nullptr;
    }
    auto node_port_name = *(linked_gcnode->GetInputPorts()->begin());
    gcedge->SetTailPort(node_port_name);
  }

  auto input = std::make_shared<NodeDesc>();
  input->SetNodeType(NODE_INPUT);
  input->SetNodeName(virtual_node_name);
  gcgraph_->AddNode(gcnode);
  gcgraph_->SetFirstNode(gcnode);
  gcgraph_->AddEdge(gcedge);
  return input;
}

std::shared_ptr<GCGraph> FlowGraphDesc::GetGCGraph() { return gcgraph_; }

}  // namespace modelbox