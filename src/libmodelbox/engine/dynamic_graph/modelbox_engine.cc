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

#include "modelbox/modelbox_engine.h"
#include <securec.h>
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/data_handler.h"
#include "modelbox/single_node.h"
#include "scheduler/flow_scheduler.h"

namespace modelbox {

constexpr const char *GRAPH_VIRTUAL_NODE = "inner_virtual_node_";

static std::shared_ptr<DataHandler> err_msg(const Status &status) {
  auto data_handler = std::make_shared<DataHandler>();
  data_handler->SetError(status);
  MBLOG_ERROR << status.Errormsg();
  return data_handler;
}

ModelBoxEngine::ModelBoxEngine() {}

ModelBoxEngine::~ModelBoxEngine() { Close(); }

std::shared_ptr<DeviceManager> ModelBoxEngine::GetDeviceManager() {
  return device_mgr_;
}

std::shared_ptr<FlowUnitManager> ModelBoxEngine::GetFlowUnitManager() {
  return flowunit_mgr_;
}

std::shared_ptr<Scheduler> ModelBoxEngine::GetScheduler() { return scheduler_; }

std::shared_ptr<Profiler> ModelBoxEngine::GetProfiler() { return profiler_; }

Status ModelBoxEngine::Init(std::shared_ptr<Configuration> &config) {
  config_ = config;
  drivers_ = std::make_shared<Drivers>();
  device_mgr_ = std::make_shared<DeviceManager>();
  flowunit_mgr_ = std::make_shared<FlowUnitManager>();
  scheduler_ = std::make_shared<FlowScheduler>();
  profiler_ = nullptr;
  std::string msg = "";
  if (!(drivers_ && device_mgr_ && flowunit_mgr_)) {
    msg = "drivers, flowunit_mgr or device is null, return";
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

  ret = scheduler_->Init(config_);
  if (!ret) {
    msg = "scheduler_ init  failed";
    MBLOG_ERROR << msg << ", " << ret.WrapErrormsgs();
    return {ret, msg};
  }

  scheduler_->RunAsync();
  return STATUS_OK;
}

std::shared_ptr<FlowUnitDesc> ModelBoxEngine::GetFlowunitDesc(
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

bool CheckMapEquate(const std::map<std::string, std::string> &first_map,
                    const std::map<std::string, std::string> &second_map) {
  if (first_map.size() != second_map.size()) {
    return false;
  }

  for (auto &iter : first_map) {
    auto temp_iter = second_map.find(iter.first);
    if (temp_iter == second_map.end()) {
      return false;
    }
    if (temp_iter->second != iter.second) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<NodeBase> ModelBoxEngine::CheckNodeExist(
    const std::string &name, const std::map<std::string, std::string> &config) {
  auto iter = nodes_config_.find(name);
  if (iter == nodes_config_.end()) {
    return nullptr;
  }

  auto node_config_map = iter->second;
  if (node_config_map.size() == 0) {
    return nullptr;
  }

  for (auto &temp_iter : node_config_map) {
    if (CheckMapEquate(temp_iter.first, config)) {
      return temp_iter.second;
    }
  }

  return nullptr;
}

std::shared_ptr<NodeBase> ModelBoxEngine::CreateDynamicNormalNode(
    const std::string &name,
    const std::map<std::string, std::string> &config_map) {
  auto node = CheckNodeExist(name, config_map);
  if (node != nullptr) {
    return node;
  }

  ConfigurationBuilder builder;
  auto config = builder.Build();
  for (auto &iter : config_map) {
    config->SetProperty(iter.first, iter.second);
  }

  auto flowunit_desc = GetFlowunitDesc(name, config_map);
  if (flowunit_desc == nullptr) {
    MBLOG_ERROR << "failed find flowunit: " << name;
    return nullptr;
  }

  auto unit_type = config->GetString("device");
  auto unit_device_id = config->GetString("deviceid");
  auto flow_stats = Statistics::GetGlobalItem()->GetItem(STATISTICS_ITEM_FLOW);
  auto dynamic_node =
      std::make_shared<SingleNode>(name, unit_type, unit_device_id,
                                   flowunit_mgr_, config, nullptr, flow_stats);
  auto status = dynamic_node->Init();

  if (status != STATUS_OK) {
    return nullptr;
  }
  nodes_config_[name][config_map] = dynamic_node;
  return dynamic_node;
}

std::shared_ptr<DataHandler> ModelBoxEngine::CreateInput(
    const std::set<std::string> &port_map) {
  auto gcgraph = std::make_shared<GCGraph>();
  gcgraph->Init(nullptr);
  auto grah_config = config_->GetSubConfig("graph");
  if(grah_config != nullptr){
    gcgraph->SetConfiguration(grah_config);
  }
  auto data_handler =
      std::make_shared<DataHandler>(VIRTUAL_NODE, shared_from_this());
  data_handler->SetDataHandlerType(INPUT);
  data_handler->SetNodeName(GRAPH_VIRTUAL_NODE);
  for (auto &iter : port_map) {
    auto gcnode = std::make_shared<GCNode>();
    gcnode->Init(iter, gcgraph);
    gcnode->SetConfiguration("type", "input");
    gcgraph->AddNode(gcnode);
    gcgraph->SetFirstNode(gcnode);
    gcnode->SetOutDataHandler(data_handler);

    if (port_map.size() == 1) {
      data_handler->SetNodeName(iter);
    }
  }

  auto graph_state = std::make_shared<GraphState>();
  graph_state->gcgraph_ = gcgraph;
  data_handler->SetBindGraph(graph_state);

  return data_handler;
}

Status ModelBoxEngine::InsertGrahEdge(std::shared_ptr<GCGraph> &root_graph,
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

Status ModelBoxEngine::CheckInputPort(
    const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
    const std::shared_ptr<DataHandler> &data_handler) {
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
        MBLOG_ERROR << "port:" << iter.GetPortName() << " has no data";
        return STATUS_INVALID;
      }
    }
  }
  return STATUS_OK;
}

std::shared_ptr<GCNode> ModelBoxEngine::CreateDynamicStreamNode(
    const std::string &name, const std::map<std::string, std::string> &config,
    const std::shared_ptr<DataHandler> &data_handler) {
  auto root_graph = data_handler->GetBindGraph();
  if (root_graph == nullptr) {
    return nullptr;
  }

  auto flowunit_desc = GetFlowunitDesc(name, config);
  auto gcnode = std::make_shared<GCNode>();
  gcnode->Init(name, root_graph->gcgraph_);
  for (auto &iter : config) {
    gcnode->SetConfiguration(iter.first, iter.second);
  }

  auto flowunit_input = flowunit_desc->GetFlowUnitInput();
  for (auto &iter : flowunit_input) {
    gcnode->SetInputPort(iter.GetPortName());
  }

  std::set<std::string> outport_names;
  auto flowunit_outports = flowunit_desc->GetFlowUnitOutput();
  for (auto &iter : flowunit_outports) {
    outport_names.insert(iter.GetPortName());
    gcnode->SetOutputPort(iter.GetPortName());
  }

  root_graph->gcgraph_->AddNode(gcnode);
  return gcnode;
}

std::shared_ptr<GCNode> ModelBoxEngine::ProcessOutputHandler(
    const std::shared_ptr<DataHandler> &data_handler,
    std::shared_ptr<GCNode> &gcnode, std::shared_ptr<GCGraph> &root_graph) {
  auto inport_name = *(data_handler->GetPortNames().begin());
  auto input_node = root_graph->GetNode(data_handler->GetNodeName());
  if (input_node == nullptr) {
    MBLOG_ERROR << "failed find input node: " << data_handler->GetNodeName();
    return nullptr;
  }

  if (gcnode->GetInputPorts()->size() > 0) {
    auto outport_name = *(gcnode->GetInputPorts()->begin());
    if (InsertGrahEdge(root_graph, input_node, inport_name, gcnode,
                       outport_name) != STATUS_OK) {
      MBLOG_ERROR << "InsertGrahEdge failed";
      return nullptr;
    }
  }
  return gcnode;
}

std::shared_ptr<GCNode> ModelBoxEngine::ProcessVirtualHandler(
    std::shared_ptr<GCNode> &gcnode, std::shared_ptr<GCGraph> &root_graph) {
  auto virtual_nodes = root_graph->GetFirstNodes();
  for (auto &iter : virtual_nodes) {
    std::string inport_name = iter->GetNodeName();
    auto output_node = gcnode;
    auto outport_name = *(gcnode->GetInputPorts()->begin());
    if (InsertGrahEdge(root_graph, iter, inport_name, gcnode, outport_name) !=
        STATUS_OK) {
      MBLOG_ERROR << "InsertGrahEdge failed,inport_name:" << inport_name
                  << ", outport_name " << outport_name;
      return nullptr;
    }
  }
  return gcnode;
}
std::shared_ptr<GCNode> ModelBoxEngine::CreateDynamicGCGraph(
    const std::string &name, const std::map<std::string, std::string> &config,
    const std::shared_ptr<DataHandler> &data_handler) {
  auto gcnode = CreateDynamicStreamNode(name, config, data_handler);
  if (gcnode == nullptr) {
    MBLOG_ERROR << "CreateDynamicStreamNode failed";
    return nullptr;
  }

  if (data_handler == nullptr) {
    return gcnode;
  }
  auto root_graph = data_handler->GetBindGraph()->gcgraph_;
  if (data_handler->GetDataHandlerType() == OUTPUT) {
    return ProcessOutputHandler(data_handler, gcnode, root_graph);
  }

  if (data_handler->GetBindNodeType() == VIRTUAL_NODE) {
    return ProcessVirtualHandler(gcnode, root_graph);
  }

  auto port_map = data_handler->GetPortMap();
  for (auto &iter : port_map) {
    std::string node_name = data_handler->port_to_node_[iter.first];
    auto outport_name = iter.first;
    auto inport_name = iter.second;
    auto input_node = root_graph->GetNode(node_name);
    auto output_node = gcnode;
    if (InsertGrahEdge(root_graph, input_node, inport_name, gcnode,
                       outport_name) != STATUS_OK) {
      MBLOG_ERROR << "InsertGrahEdge failed";
      return nullptr;
    }
  }

  return gcnode;
}

Status ModelBoxEngine::CheckBuffer(const std::shared_ptr<FlowUnitDesc> &desc,
                                   const std::shared_ptr<DataHandler> &data) {
  if (data->GetDataHandlerType() == OUTPUT) {
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

std::shared_ptr<DataHandler> ModelBoxEngine::BindDataHanlder(
    std::shared_ptr<DataHandler> &data_handler,
    std::shared_ptr<GCNode> &gcnode) {
  gcnode->SetOutDataHandler(data_handler);
  auto outports = gcnode->GetOutputPorts();
  auto outport_names =
      std::set<std::string>(outports->begin(), outports->end());
  auto name = gcnode->GetNodeName();
  data_handler->SetPortNames(outport_names);

  data_handler->SetNodeName(name);
  data_handler->SetDataHandlerType(OUTPUT);
  data_handler->SetEnv(shared_from_this());
  data_handler->SetBindNodeType(STREAM_NODE);
  gcnode->SetOutDataHandler(data_handler);
  return data_handler;
}
Status ModelBoxEngine::RunGraph(std::shared_ptr<DataHandler> &data_handler) {
  Status ret = STATUS_OK;

  if (data_handler->context_->GetGraphState()->graph_) {
    MBLOG_WARN << "graph has been build and run";
    return STATUS_EXIST;
  }

  auto gcgraph = data_handler->context_->GetGraphState()->gcgraph_;
  if (gcgraph == nullptr) {
    MBLOG_WARN << "DataHandler has no bind graph";
    return STATUS_FAULT;
  }

  auto dynamic_graph = std::make_shared<DynamicGraph>();
  auto status =
      dynamic_graph->Initialize(flowunit_mgr_, device_mgr_, profiler_, config_);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "graph init failed";
    return STATUS_FAULT;
  }

  status = dynamic_graph->Build(gcgraph);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "build graph failed: " << status.Errormsg();
    return status;
  }
  graphs_.emplace(dynamic_graph);

  auto scheduler = GetScheduler();
  if (scheduler == nullptr) {
    MBLOG_ERROR << "scheduler has not been created";
    return STATUS_INVALID;
  }

  if (scheduler->Build(*dynamic_graph) != STATUS_OK) {
    MBLOG_ERROR << "failed build graph";
    return STATUS_FAULT;
  }
  auto first_nodes = gcgraph->GetFirstNodes();

  if (first_nodes.size() > 0) {
    FeedData(dynamic_graph, gcgraph);
  }
  return STATUS_OK;
}

std::shared_ptr<DynamicGraph> ModelBoxEngine::CreateDynamicGraph(
    std::shared_ptr<GCGraph> &graph) {
  auto dynamic_graph = std::make_shared<DynamicGraph>();
  auto ret = dynamic_graph->Initialize(GetFlowUnitManager(), GetDeviceManager(),
                                       GetProfiler(), GetConfig());
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "init graph failed";
    return nullptr;
  }
  graph->ShowAllNode();
  graph->ShowAllEdge();
  ret = dynamic_graph->Build(graph);
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "build dynmaic graph failed: " << ret.Errormsg();
    return nullptr;
  }

  graphs_.emplace(dynamic_graph);

  auto scheduler = GetScheduler();
  if (scheduler == nullptr) {
    MBLOG_ERROR << "scheduler is not inited";
    return nullptr;
  }

  if (STATUS_OK != scheduler->Build(*dynamic_graph)) {
    MBLOG_ERROR << "add graph to scheduler failed";
    return nullptr;
  }

  return dynamic_graph;
}

Status ModelBoxEngine::SendExternalData(
    std::shared_ptr<modelbox::ExternalDataMap> &extern_datamap,
    std::shared_ptr<modelbox::BufferList> &buffer_list,
    const std::shared_ptr<GCNode> &gcnode) {
  auto input_data = gcnode->GetBindDataHandler();
  if (input_data == nullptr) {
    MBLOG_ERROR << "failed find bind data handler for input.";
    return STATUS_FAULT;
  }
  auto node_name = gcnode->GetNodeName();
  bool send_data = false;
  auto input_buffer_list = input_data->context_->GetBufferList(node_name);
  if (input_buffer_list == nullptr || input_buffer_list->Size() == 0) {
    buffer_list->Build({1});
  } else {
    for (auto iter = input_buffer_list->begin();
         iter != input_buffer_list->end(); iter++) {
      buffer_list->PushBack(*iter);
    }
    send_data = true;
  }

  if (!input_data->context_->meta_.empty()) {
    auto data_meta = std::make_shared<DataMeta>();
    for (auto &iter : input_data->context_->meta_) {
      data_meta->SetMeta(iter.first,
                         std::make_shared<std::string>(iter.second));
    }
    extern_datamap->SetOutputMeta(node_name, data_meta);
    send_data = true;
  }
  if (send_data) {
    if (STATUS_OK != extern_datamap->Send(node_name, buffer_list)) {
      return STATUS_FAULT;
    }
  }

  return STATUS_SUCCESS;
}

Status ModelBoxEngine::FeedData(std::shared_ptr<DynamicGraph> &dynamic_graph,
                                std::shared_ptr<GCGraph> &gcgraph) {
  if (dynamic_graph == nullptr || gcgraph == nullptr) {
    MBLOG_ERROR << "graph or gcgraph is nullptr .";
    return STATUS_FAULT;
  }

  auto extern_data = dynamic_graph->CreateExternalDataMap();
  if (extern_data == nullptr) {
    return STATUS_FAULT;
  }
  auto buffer_list = extern_data->CreateBufferList();
  if (buffer_list == nullptr) {
    return STATUS_FAULT;
  }

  auto input_nodes = gcgraph->GetFirstNodes();
  for (auto &iter : input_nodes) {
    if (STATUS_SUCCESS != SendExternalData(extern_data, buffer_list, iter)) {
      return STATUS_FAULT;
    }
  }
  auto input_data = input_nodes.front()->GetBindDataHandler();
  auto input_context =
      std::static_pointer_cast<InputContext>(input_data->context_);
  input_context->SetExternPtr(extern_data, buffer_list);
  input_data->context_->GetGraphState()->external_data_ = extern_data;
  input_data->context_->GetGraphState()->graph_ = dynamic_graph;
  if (input_data->closed_) {
    extern_data->Shutdown();
  }
  return STATUS_OK;
}

std::shared_ptr<DataHandler> ModelBoxEngine::Execute(
    const std::string &name, std::map<std::string, std::string> config_map,
    const std::map<std::string, std::shared_ptr<DataHandler>> &buffers) {
  auto data = std::make_shared<DataHandler>(STREAM_NODE, shared_from_this());
  auto ret = data->SetDataHandler(buffers);
  if (ret != STATUS_OK) {
    data->SetError(ret);
    return data;
  }

  return Execute(name, config_map, data);
}

bool CheckPortisLinked(std::shared_ptr<GCGraph> &gcgraph,
                       std::shared_ptr<GCNode> &gcnode,
                       const std::string &port_name) {
  auto edges = gcgraph->GetAllEdges();
  for (auto &iter : edges) {
    auto edge = iter.second;
    if (edge->GetTailInPort() == port_name && edge->GetTailNode() == gcnode) {
      return true;
    }
  }
  return false;
}

bool CheckNodeIsLinked(std::shared_ptr<GCGraph> &gcgraph,
                       std::shared_ptr<GCNode> &gcnode) {
  bool result = true;
  auto inports = gcnode->GetInputPorts();
  for (auto iter = inports->begin(); iter != inports->end(); iter++) {
    result &= CheckPortisLinked(gcgraph, gcnode, *iter);
  }
  return result;
}

bool ModelBoxEngine::CheckisStream(
    const std::shared_ptr<FlowUnitDesc> &desc,
    const std::shared_ptr<DataHandler> &data_handler) {
  if (desc->GetFlowType() == STREAM) {
    return true;
  }

  if (data_handler == nullptr) {
    return true;
  }

  if (desc->GetFlowType() == NORMAL) {
    if (data_handler->GetBindNodeType() == STREAM_NODE) {
      return true;
    }

    if (data_handler->GetBindNodeType() == VIRTUAL_NODE) {
      return true;
    }
  }
  return false;
}

static void SetDefaultConfigValue(const std::string &name,
                           std::map<std::string, std::string> &config_map) {
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

Status ModelBoxEngine::CheckInputFlowUnit(
    const std::string &name, std::map<std::string, std::string> &config_map,
    const std::shared_ptr<DataHandler> &buffers,
    const std::shared_ptr<FlowUnitDesc> &desc) {
  if (buffers != nullptr) {
    if (buffers->GetError() != STATUS_SUCCESS) {
      auto err_msg = "node: " + name + ", input data has error.";
      MBLOG_ERROR << err_msg;
      return {STATUS_FAULT, err_msg};
    }

    if (buffers->GetBindGraph() != nullptr &&
        buffers->GetBindGraph()->graph_ != nullptr) {
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

  auto node_type = STREAM_NODE;
  if (buffers != nullptr) {
    if (CheckBuffer(desc, buffers) != STATUS_OK) {
      return {STATUS_INVALID, "input DataHandler is invalid"};
    }
    if (STATUS_OK != buffers->CheckInputType(node_type)) {
      auto msg = "check input failed, node name = " + name;
      return {STATUS_FAULT, msg};
    }
  }
  return STATUS_OK;
}

std::shared_ptr<DataHandler> ModelBoxEngine::ExecuteStreamNode(
    const std::shared_ptr<FlowUnitDesc> &desc,
    const std::shared_ptr<DataHandler> &buffers,
    std::map<std::string, std::string> &config_map) {
  auto stream_data_handler =
      std::make_shared<DataHandler>(STREAM_NODE, shared_from_this());
  stream_data_handler->context_->SetFlowUnitDesc(desc);
  if (buffers == nullptr) {
    stream_data_handler->SetBindGraph(std::make_shared<GraphState>());
    auto gcgraph = std::make_shared<GCGraph>();
    gcgraph->Init(nullptr);
    auto graph_config = config_->GetSubConfig("graph");
    if (graph_config != nullptr) {
      gcgraph->SetConfiguration(config_);
    }

    stream_data_handler->GetBindGraph()->gcgraph_ = gcgraph;
    auto root_graph = stream_data_handler->GetBindGraph()->gcgraph_;
    auto gcnode = CreateDynamicStreamNode(desc->GetFlowUnitName(), config_map,
                                          stream_data_handler);
    if (gcnode == nullptr) {
      auto msg = "CreateDynamicStreamNode failed";
      return err_msg({STATUS_FAULT, msg});
    }
    stream_data_handler = BindDataHanlder(stream_data_handler, gcnode);
    return stream_data_handler;
  }

  if (CheckInputPort(desc, buffers) != STATUS_OK) {
    return err_msg({STATUS_FAULT, "not all port has data"});
  }
  if (buffers->GetBindGraph() == nullptr) {
    return err_msg({STATUS_FAULT, "input datahandler has no valid graph"});
  }

  stream_data_handler->SetBindGraph(buffers->GetBindGraph());

  auto gcnode =
      CreateDynamicGCGraph(desc->GetFlowUnitName(), config_map, buffers);
  if (gcnode == nullptr) {
    auto msg = "create gcnode failed";
    stream_data_handler->SetError({STATUS_INVALID, msg});
    MBLOG_ERROR << msg;
    return stream_data_handler;
  }
  stream_data_handler = BindDataHanlder(stream_data_handler, gcnode);

  if (desc->GetFlowUnitOutput().size() == 0 &&
      CheckNodeIsLinked(stream_data_handler->GetBindGraph()->gcgraph_,
                        gcnode)) {
    if (STATUS_OK != RunGraph(stream_data_handler)) {
      auto msg = "build graph failed";
      stream_data_handler->SetError({STATUS_INVALID, msg});
      MBLOG_ERROR << msg;
    }
  }
  return stream_data_handler;
}

std::shared_ptr<DataHandler> ModelBoxEngine::ExecuteBufferListNode(
    const std::string &name, std::map<std::string, std::string> &config_map,
    const std::shared_ptr<DataHandler> &buffers) {
  auto node = CreateDynamicNormalNode(name, config_map);
  if (node == nullptr) {
    return err_msg({STATUS_INVALID, "create dynamic node " + name + " failed"});
  }

  auto dynamic_node = std::static_pointer_cast<SingleNode>(node);
  dynamic_node->Run(buffers);
  auto data_handler = std::make_shared<DataHandler>();
  data_handler->SetDataHandlerType(OUTPUT);

  if (STATUS_NODATA == dynamic_node->PushDataToDataHandler(data_handler)) {
    return err_msg({STATUS_NODATA, "recv no data from node"});
  }

  return data_handler;
}

std::shared_ptr<DataHandler> ModelBoxEngine::Execute(
    const std::string &name, std::map<std::string, std::string> config_map,
    const std::shared_ptr<DataHandler> &buffers) {
  SetDefaultConfigValue(name, config_map);
  auto flowunit_desc = GetFlowunitDesc(name, config_map);
  if (flowunit_desc == nullptr) {
    return err_msg(
        {STATUS_INVALID, "failed find flowunit " + name + " description"});
  }

  auto ret = CheckInputFlowUnit(name, config_map, buffers, flowunit_desc);
  if (ret != STATUS_OK) {
    return err_msg(ret);
  }

  // stream node, create gcgraph
  if (CheckisStream(flowunit_desc, buffers)) {
    return ExecuteStreamNode(flowunit_desc, buffers, config_map);
  }
  return ExecuteBufferListNode(name, config_map, buffers);
}

void ModelBoxEngine::ShutDown() {
  if (scheduler_) {
    scheduler_->Shutdown();
    scheduler_ = nullptr;
  }
}

void ModelBoxEngine::Close() {
  if (scheduler_) {
    if (graphs_.size() == 1) {
      Status status = STATUS_OK;
      scheduler_->Wait(0, &status);
    }

    scheduler_->Shutdown();
    scheduler_ = nullptr;
  }

  graphs_.clear();
  nodes_config_.clear();
  if (device_mgr_) device_mgr_->Clear();
  flowunit_mgr_ = nullptr;
  device_mgr_ = nullptr;
  profiler_ = nullptr;
  if (drivers_) {
    drivers_->Clear();
  }
  drivers_ = nullptr;
}

std::shared_ptr<Configuration> ModelBoxEngine::GetConfig() { return config_; }

}  // namespace modelbox