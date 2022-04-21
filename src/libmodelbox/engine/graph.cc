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

#include "modelbox/graph.h"

#include "modelbox/base/log.h"
#include "modelbox/base/uuid.h"
#include "modelbox/profiler.h"
#include "scheduler/flow_scheduler.h"

namespace modelbox {

constexpr const char *GRAPH_NODE_TYPE = "type";
constexpr const char *GRAPH_NODE_FLOWUNIT = "flowunit";
constexpr const char *GRAPH_NODE_INPUT = "input";
constexpr const char *GRAPH_NODE_OUTPUT = "output";
constexpr const char *GRAPH_KEY_DEVICE = "device";
constexpr const char *GRAPH_KEY_DEVICE_ID = "deviceid";
constexpr const char *GRAPH_KEY_QUEUE_SIZE = "queue_size";
constexpr const char *GRAPH_KEY_BATCH_SIZE = "batch_size";
constexpr const char *GRAPH_KEY_CHECK_NODE_OUTPUT = "need_check_output";
constexpr const char *GRAPH_NODE_REGISTER_FLOWUNIT = "register_flowunit";

Graph::Graph()
    : nodes_(),
      src_to_dst_(),
      dst_to_src_(),
      topo_order_(),
      scheduler_(nullptr) {}

Graph::~Graph() {
  CloseNodes();
  src_to_dst_.clear();
  dst_to_src_.clear();
  topo_order_.clear();
  nodes_.clear();
  if (flow_stats_ != nullptr) {
    flow_stats_->DelItem(id_);
  }
}

Status Graph::Initialize(std::shared_ptr<FlowUnitManager> flowunit_mgr,
                         std::shared_ptr<DeviceManager> device_mgr,
                         std::shared_ptr<Profiler> profiler,
                         std::shared_ptr<Configuration> config) {
  if (flowunit_mgr == nullptr || device_mgr == nullptr || config == nullptr) {
    auto msg = "argument is invalid";
    auto ret = Status(STATUS_INVALID, msg);
    MBLOG_ERROR << ret.WrapErrormsgs();
    return ret;
  }

  flowunit_mgr_ = flowunit_mgr;
  device_mgr_ = device_mgr;
  profiler_ = profiler;
  flow_stats_ = Statistics::GetGlobalItem()->GetItem(STATISTICS_ITEM_FLOW);
  config_ = config;
  auto ret = GetUUID(&id_);
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "Get uuid for graph failed";
    return ret;
  }

  if (flow_stats_ != nullptr) {
    graph_stats_ = flow_stats_->AddItem(id_);
    if (graph_stats_ == nullptr) {
      MBLOG_ERROR << "Get stats for graph " << id_
                  << " failed, err: " << StatusError.Errormsg();
    }
  }

  return STATUS_OK;
}

std::string Graph::GetId() const { return id_; }

std::string Graph::GetName() const { return name_; }

Status Graph::CheckLoopStructureNode() {
  Status status{STATUS_OK};
  for (auto &loop : loop_structures_) {
    auto pos = std::find(loop.begin(), loop.end(), *(loop.end() - 1));
    for (auto iter = pos; iter != loop.end(); iter++) {
      auto real_node = std::dynamic_pointer_cast<Node>(GetNode(*iter));
      if (real_node == nullptr) {
        auto err_msg = "invalid node " + *iter;
        status = {STATUS_FAULT, err_msg};
        break;
      }

      if (real_node->GetFlowType() == STREAM) {
        auto err_msg = "loop node " + *iter + " can not be stream node.";
        status = {STATUS_FAULT, err_msg};
        break;
      }
    }
  }
  return status;
}

Status Graph::Build(std::shared_ptr<GCGraph> g) {
  if (g == nullptr) {
    return STATUS_INVALID;
  }

  name_ = g->GetGraphName();
  MBLOG_INFO << "Build graph name:" << name_ << ", id:" << id_;
  g->ShowAllSubGraph();
  g->ShowAllNode();
  g->ShowAllEdge();

  if (flowunit_mgr_ == nullptr || device_mgr_ == nullptr ||
      config_ == nullptr) {
    auto msg = "graph is not initialized";
    auto ret = Status(STATUS_INVALID, msg);
    return ret;
  }

  // build node and add link
  Status status = BuildGraph(g);
  if (!status) {
    auto msg = "build graph from config fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = IsValidGraph();
  if (!status) {
    auto msg = "invalid graph.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = FindLoopStructure();
  if (!status) {
    auto msg = "there is no loop structure in the graph";
    MBLOG_DEBUG << msg;
    return status;
  }

  status = GenerateTopology();
  if (!status) {
    auto msg = "generate topology fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = UpdatePriority();
  if (!status) {
    auto msg = "update proiority fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = InitPort();
  if (!status) {
    auto msg = "init port fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = CheckStreamMatcher();
  if (!status) {
    auto msg = "check stream fail, msg: " + status.WrapErrormsgs();
    auto ret = Status(status, msg);
    return ret;
  }

  status = CheckLoopStructureNode();
  if (!status) {
    auto msg = "check loop node fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  status = InitScheduler();
  if (!status) {
    auto msg = "init scheduler fail.";
    auto ret = Status(status, msg);
    return ret;
  }

  return STATUS_OK;
}

Status Graph::AddNode(std::shared_ptr<NodeBase> node) {
  if (node == nullptr) {
    auto msg = "node is null pointer.";
    return {STATUS_INVALID, msg};
  }

  auto ite = nodes_.find(node->GetName());
  if (ite != nodes_.end()) {
    auto msg = "node is already exist. name: " + node->GetName();
    return {STATUS_INVALID, msg};
  }

  nodes_[node->GetName()] = node;
  return STATUS_OK;
}

std::shared_ptr<NodeBase> Graph::GetNode(const std::string &nodeName) const {
  auto ite = nodes_.find(nodeName);
  if (ite == nodes_.end()) {
    return nullptr;
  }

  return ite->second;
}

std::shared_ptr<InPort> Graph::GetInPort(const std::string &nodeName,
                                         const std::string &portName) const {
  auto ite = nodes_.find(nodeName);
  if (ite == nodes_.end()) {
    return nullptr;
  }

  auto node = ite->second;
  if (node == nullptr) {
    auto msg = "node is null pointer, never here. name: " + node->GetName();
    MBLOG_ERROR << msg;
    return nullptr;
  }

  return node->GetInputPort(portName);
}

std::shared_ptr<OutPort> Graph::GetOutPort(const std::string &nodeName,
                                           const std::string &portName) const {
  auto ite = nodes_.find(nodeName);
  if (ite == nodes_.end()) {
    return nullptr;
  }

  auto node = ite->second;
  if (node == nullptr) {
    auto msg = "node is null pointer, never here. name: " + node->GetName();
    MBLOG_ERROR << msg;
    return nullptr;
  }

  return node->GetOutputPort(portName);
}

const std::unordered_map<std::shared_ptr<NodeBase>,
                         std::vector<std::shared_ptr<IPort>>>
Graph::GetNotifyPort() const {
  std::unordered_map<std::shared_ptr<NodeBase>,
                     std::vector<std::shared_ptr<IPort>>>
      node_ports;
  for (auto &node : nodes_) {
    std::vector<std::shared_ptr<IPort>> ports;
    // add in ports
    const auto &in_ports = node.second->GetInputPorts();
    std::copy(in_ports.begin(), in_ports.end(), std::back_inserter(ports));

    // add event port
    const auto &event_port = node.second->GetEventPort();
    ports.push_back(std::dynamic_pointer_cast<IPort>(event_port));

    // add external port
    const auto &external_ports = node.second->GetExternalPorts();
    std::copy(external_ports.begin(), external_ports.end(),
              std::back_inserter(ports));

    node_ports.emplace(node.second, std::move(ports));
  }

  return node_ports;
}

Status Graph::AddLink(const std::string &srcNodeName,
                      const std::string &srcPortName,
                      const std::string &dstNodeName,
                      const std::string &dstPortName) {
  auto srcPort = GetOutPort(srcNodeName, srcPortName);
  if (srcPort == nullptr) {
    auto msg =
        "src port is not exist. node: " + srcNodeName + " port: " + srcPortName;
    return {STATUS_INVALID, msg};
  }

  auto dstPort = GetInPort(dstNodeName, dstPortName);
  if (dstPort == nullptr) {
    auto msg =
        "dst port is not exist. node: " + dstNodeName + " port: " + dstPortName;
    return {STATUS_INVALID, msg};
  }

  return AddLink(srcPort, dstPort);
}

Status Graph::AddLink(std::shared_ptr<OutPort> src,
                      std::shared_ptr<InPort> dst) {
  if (src == nullptr) {
    auto msg = "src port is null pointer.";
    return {STATUS_INVALID, msg};
  }

  if (dst == nullptr) {
    auto msg = "dst port is null pointer.";
    return {STATUS_INVALID, msg};
  }

  auto srcNode = src->GetNode();
  if (srcNode == nullptr) {
    auto msg = "src node is null point.";
    return {STATUS_INVALID, msg};
  }

  auto dstNode = dst->GetNode();
  if (dstNode == nullptr) {
    auto msg = "dst node is null point.";
    return {STATUS_INVALID, msg};
  }

  auto dstLinks = src_to_dst_.find(src);
  if (dstLinks != src_to_dst_.end()) {
    auto ite = dstLinks->second.find(dst);
    if (ite != dstLinks->second.end()) {
      auto msg = "link is already exist. srcNode: " + srcNode->GetName() +
                 " srcPort: " + src->GetName() + "->" +
                 " dstNode: " + dstNode->GetName() +
                 " dstPort: " + dst->GetName();
      return {STATUS_INVALID, msg};
    }
  }

  src_to_dst_[src].insert(dst);
  dst_to_src_[dst].insert(src);

  auto msg = "add link, " + srcNode->GetName() + ":" + src->GetName() + " -> " +
             dstNode->GetName() + ":" + dst->GetName();
  MBLOG_INFO << msg;

  return STATUS_OK;
}

std::set<std::shared_ptr<InPort>> Graph::GetDstPortsByPort(
    std::shared_ptr<OutPort> port) const {
  std::set<std::shared_ptr<InPort>> ports;
  if (port == nullptr) {
    return ports;
  }

  auto ite = src_to_dst_.find(port);
  if (ite == src_to_dst_.end()) {
    return ports;
  }

  ports = ite->second;
  return ports;
}

std::set<std::shared_ptr<OutPort>> Graph::GetSrcPortsByPort(
    std::shared_ptr<InPort> port) const {
  std::set<std::shared_ptr<OutPort>> ports;
  if (port == nullptr) {
    return ports;
  }

  auto ite = dst_to_src_.find(port);
  if (ite == dst_to_src_.end()) {
    return ports;
  }

  ports = ite->second;
  return ports;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetStartNodes() const {
  std::set<std::shared_ptr<NodeBase>> startNode;
  for (auto &node : nodes_) {
    auto inputNum = node.second->GetInputNum();
    if (inputNum == 0) {
      startNode.insert(node.second);
    }
  }

  return startNode;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetEndNodes() const {
  std::set<std::shared_ptr<NodeBase>> endNode;
  for (auto &node : nodes_) {
    auto inputNum = node.second->GetInputNum();
    if (inputNum == 0) {
      endNode.insert(node.second);
    }
  }

  return endNode;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetEndPointNodes() const {
  std::set<std::shared_ptr<NodeBase>> endNode;
  for (auto &node : nodes_) {
    auto outports = node.second->GetOutputPorts();
    for (auto iter : outports) {
      if (iter->GetConnectInPort().size() <= 0) {
        endNode.insert(node.second);
      }
    }
  }

  return endNode;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetAllNodes() const {
  std::set<std::shared_ptr<NodeBase>> allNode;
  for (auto &node : nodes_) {
    allNode.insert(node.second);
  }

  return allNode;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetDstNodesByNode(
    const std::string &nodeName) const {
  std::set<std::shared_ptr<NodeBase>> nodes;
  auto node = GetNode(nodeName);
  if (node == nullptr) {
    return nodes;
  }

  auto outports = node->GetOutputPorts();
  for (auto port : outports) {
    auto linkPorts = GetDstPortsByPort(port);
    for (auto linkport : linkPorts) {
      nodes.insert(linkport->GetNode());
    }
  }

  return nodes;
}

std::set<std::shared_ptr<NodeBase>> Graph::GetSrcNodesByNode(
    const std::string &nodeName) const {
  std::set<std::shared_ptr<NodeBase>> nodes;
  auto node = GetNode(nodeName);
  if (node == nullptr) {
    return nodes;
  }

  auto inports = node->GetInputPorts();
  for (auto port : inports) {
    auto linkPorts = GetSrcPortsByPort(port);
    for (auto linkport : linkPorts) {
      nodes.insert(linkport->GetNode());
    }
  }

  return nodes;
}

std::shared_ptr<ExternalDataMap> Graph::CreateExternalDataMap() {
  if (input_node_ == nullptr) {
    MBLOG_ERROR << "virtual input_node is nullptr";
    return nullptr;
  }
  auto extrern_data = std::make_shared<ExternalDataMapImpl>(
      input_node_, output_node_, graph_stats_);
  extrern_data->Init();
  return extrern_data;
}

Status Graph::UpdateGraphConfigToNode(std::shared_ptr<GCGraph> g,
                                      const std::shared_ptr<GCNode> node) {
  auto graph_config = g->GetConfiguration();
  auto node_config = node->GetConfiguration();
  auto update_node_config = [=](const std::string &key) {
    if (node_config->Contain(key) == true) {
      return;
    }

    if (graph_config->Contain(key) == false) {
      return;
    }

    node_config->Copy(*graph_config.get(), key);
  };

  update_node_config(GRAPH_KEY_BATCH_SIZE);
  update_node_config(GRAPH_KEY_QUEUE_SIZE);
  update_node_config(GRAPH_KEY_DEVICE_ID);
  update_node_config(GRAPH_KEY_CHECK_NODE_OUTPUT);

  return STATUS_OK;
}

Status Graph::BuildFlowunitNode(std::shared_ptr<GCGraph> g,
                                std::shared_ptr<GCNode> gcnode, bool strict) {
  auto name = gcnode->GetNodeName();
  auto node_config = gcnode->GetConfiguration();
  auto device = node_config->GetString(GRAPH_KEY_DEVICE, "");
  auto deviceid = node_config->GetString(GRAPH_KEY_DEVICE_ID, "");
  auto flowunit = node_config->GetString(GRAPH_NODE_FLOWUNIT, "");
  auto inports = gcnode->GetInputPorts();
  auto outports = gcnode->GetOutputPorts();

  if (flowunit.empty()) {
    auto msg = "node " + name + ": flowunit name is empty.";
    return {STATUS_INVALID, msg};
  }

  if (inports->size() == 0 && outports->size() == 0) {
    if (strict == false) {
      MBLOG_INFO << "skip orphan node: " << name;
      return STATUS_SUCCESS;
    }

    auto msg = "orphan node: '" + name +
               "', use graph.strict=false to disable orphan check";
    return {STATUS_BADCONF, msg};
  }

  if (UpdateGraphConfigToNode(g, gcnode) == false) {
    auto msg =
        "update node config failed, please check node config in graph scope";
    return {STATUS_BADCONF, msg};
  }

  auto node = std::make_shared<Node>(flowunit, device, deviceid, flowunit_mgr_,
                                     profiler_, graph_stats_);

  auto status = InitNode(node, *inports, *outports, node_config);
  if (!status) {
    return status;
  }

  node->SetName(name);
  status = AddNode(node);
  if (!status) {
    auto msg = "add node failed. name: '" + name + "'";
    return {status, msg};
  }

  return STATUS_SUCCESS;
}

Status Graph::BuildCommonRegisterNode(std::shared_ptr<GCGraph> g,
                                      std::shared_ptr<GCNode> gcnode) {
  auto name = gcnode->GetNodeName();
  auto node_config = gcnode->GetConfiguration();
  auto device = node_config->GetString(GRAPH_KEY_DEVICE, "cpu");
  auto deviceid = node_config->GetString(GRAPH_KEY_DEVICE_ID, "0");
  auto flowunit = node_config->GetString(GRAPH_NODE_FLOWUNIT, "");
  auto inports = gcnode->GetInputPorts();
  auto outports = gcnode->GetOutputPorts();

  if (inports->size() == 0 && outports->size() == 0) {
    auto msg =
        "callback flowunit has no input or output port, please must specify "
        "one port";
    return {STATUS_BADCONF, msg};
  }

  if (UpdateGraphConfigToNode(g, gcnode) == false) {
    auto msg =
        "update node config failed, please check node config in graph scope";
    return {STATUS_BADCONF, msg};
  }

  auto node = std::make_shared<Node>(flowunit, device, deviceid, flowunit_mgr_,
                                     profiler_, graph_stats_);

  auto status = InitNode(node, *inports, *outports, node_config);
  if (!status) {
    return status;
  }

  node->SetName(name);
  status = AddNode(node);
  if (!status) {
    auto msg = "add node failed. name: '" + name + "'";
    return {status, msg};
  }

  return STATUS_SUCCESS;
}

Status Graph::BuildInputNode(std::shared_ptr<GCNode> gcnode) {
  auto name = gcnode->GetNodeName();
  auto node_config = gcnode->GetConfiguration();
  if (input_node_ports_.find(name) != input_node_ports_.end()) {
    auto msg = "virtual input port is already exist. name: '" + name + "'";
    return {STATUS_INVALID, msg};
  }
  input_node_ports_.insert(name);
  input_node_config_map_.emplace(name, node_config);
  return STATUS_SUCCESS;
}

Status Graph::BuildOutputNode(std::shared_ptr<GCNode> gcnode) {
  auto name = gcnode->GetNodeName();
  auto node_config = gcnode->GetConfiguration();
  if (output_node_ports_.find(name) != output_node_ports_.end()) {
    auto msg = "virtual out port is already exist. name: '" + name + "'";
    return {STATUS_INVALID, msg};
  }
  output_node_ports_.insert(name);
  output_node_config_map_.emplace(name, node_config);
  return STATUS_SUCCESS;
}

Status Graph::BuildNode(std::shared_ptr<GCGraph> g,
                        std::shared_ptr<GCNode> gcnode, bool strict) {
  auto name = gcnode->GetNodeName();
  auto node_config = gcnode->GetConfiguration();
  auto type = node_config->GetString(GRAPH_NODE_TYPE, "");
  auto flowunit = node_config->GetString(GRAPH_NODE_FLOWUNIT, "");

  if (flowunit.length() > 0 && type.length() == 0) {
    type = GRAPH_NODE_FLOWUNIT;
  }

  Status status = STATUS_SUCCESS;
  if (type == GRAPH_NODE_FLOWUNIT) {
    status = BuildFlowunitNode(g, gcnode, strict);
  } else if (type == GRAPH_NODE_INPUT) {
    status = BuildInputNode(gcnode);
  } else if (type == GRAPH_NODE_OUTPUT) {
    status = BuildOutputNode(gcnode);
  } else if (type == GRAPH_NODE_REGISTER_FLOWUNIT) {
    status = BuildCommonRegisterNode(g, gcnode);
  } else {
    if (strict) {
      auto msg =
          "unsupport node type. name: '" + name + "' type: '" + type + "'";
      status = {STATUS_NOTSUPPORT, msg};
    }
  }

  return status;
}

Status Graph::BuildNodes(std::shared_ptr<GCGraph> g) {
  if (g == nullptr) {
    auto msg = "g is null pointer.";
    return {STATUS_INVALID, msg};
  }

  auto strict = config_->GetBool("graph.strict", true);

  auto nodes = g->GetAllNodes();

  for (auto &ite : nodes) {
    auto gcnode = ite.second;
    auto name = gcnode->GetNodeName();
    MBLOG_INFO << "begin build node " << name;
    auto status = BuildNode(g, gcnode, strict);
    if (!status) {
      MBLOG_ERROR << status;
      auto msg = "build node failed. name: '" + name + "'";
      return {STATUS_FAULT, msg};
    }
    MBLOG_INFO << "build node " << name << " success";
  }
  return STATUS_OK;
}

Status Graph::BuildVirtualNodes(std::shared_ptr<GCGraph> g) {
  if (!input_node_ports_.empty()) {
    input_node_name_ = *input_node_ports_.begin();
    input_node_ = std::make_shared<InputVirtualNode>("cpu", "0", device_mgr_);
    auto input_config = input_node_config_map_.begin()->second;
    auto status = input_node_->Init({}, input_node_ports_, input_config);
    if (!status) {
      auto msg = "init virtual input node failed.";
      return {status, msg};
    }
    input_node_->SetName(input_node_name_);
    status = AddNode(input_node_);
    if (!status) {
      auto msg = "add virtual input node failed.";
      return {status, msg};
    }
  }

  if (!output_node_ports_.empty()) {
    output_node_name_ = *output_node_ports_.begin();
    auto output_config = output_node_config_map_.begin()->second;
    auto output_type = output_config->GetString("output_type", "match");
    if (output_type == "match") {
      output_node_ =
          std::make_shared<OutputVirtualNode>("cpu", "0", device_mgr_);
    } else if (output_type == "unmatch") {
      output_node_ =
          std::make_shared<OutputUnmatchVirtualNode>("cpu", "0", device_mgr_);
    } else {
      return {STATUS_INVALID, "Invalid Output Type"};
    }

    auto status = output_node_->Init(output_node_ports_, {}, output_config);
    if (!status) {
      auto msg = "init virtual output node failed.";
      return {status, msg};
    }
    output_node_->SetName(output_node_name_);
    status = AddNode(output_node_);
    if (!status) {
      auto msg = "add virtual output node failed.";
      return {status, msg};
    }
  }
  return STATUS_OK;
}

Status Graph::BuildEdges(std::shared_ptr<GCGraph> g) {
  auto edges = g->GetAllEdges();
  for (auto &ite : edges) {
    auto gcedge = ite.second;
    auto srcNode = gcedge->GetHeadNode();
    auto srcNodeName = srcNode->GetNodeName();
    auto srcPortName = gcedge->GetHeadOutPort();
    auto dstNode = gcedge->GetTailNode();
    auto dstNodeName = dstNode->GetNodeName();
    auto dstPortName = gcedge->GetTailInPort();

    if (input_node_ports_.find(srcNodeName) != input_node_ports_.end()) {
      srcPortName = srcNodeName;
      srcNodeName = input_node_name_;
    }

    if (output_node_ports_.find(dstNodeName) != output_node_ports_.end()) {
      dstPortName = dstNodeName;
      dstNodeName = output_node_name_;
    }

    auto status = AddLink(srcNodeName, srcPortName, dstNodeName, dstPortName);
    if (!status) {
      auto msg = "add link failed.";
      return {status, msg};
    }
  }

  return STATUS_OK;
}

Status Graph::OpenNodes() {
  ThreadPool pool(std::thread::hardware_concurrency());
  pool.SetName("Node-Open");
  std::vector<std::future<Status>> result;
  for (auto &itr : nodes_) {
    auto node = itr.second;
    auto ret = pool.Submit(node->GetName(), &NodeBase::Open, node.get());
    result.push_back(std::move(ret));
  }

  for (auto &fut : result) {
    auto msg = "open node failed, please check log.";
    if (!fut.valid()) {
      return {STATUS_FAULT, msg};
    }

    auto ret = fut.get();
    if (!ret) {
      return Status(ret, msg);
    }
  }

  return STATUS_OK;
}

void Graph::CloseNodes() const {
  ThreadPool pool(std::thread::hardware_concurrency());
  pool.SetName("Node-Close");

  std::vector<std::future<void>> result;
  for (auto &itr : nodes_) {
    auto node = itr.second;
    auto ret =
        pool.Submit(node->GetName() + "_close", &NodeBase::Close, node.get());
    result.push_back(std::move(ret));
  }

  for (auto &fut : result) {
    if (!fut.valid()) {
      continue;
    }

    fut.get();
  }
}

Status Graph::BuildGraph(std::shared_ptr<GCGraph> g) {
  auto status = BuildNodes(g);
  if (!status) {
    return status;
  }

  status = BuildVirtualNodes(g);
  if (!status) {
    return status;
  }

  status = BuildEdges(g);
  if (!status) {
    return status;
  }

  status = OpenNodes();
  return status;
}

Status Graph::IsValidGraph() const {
  if (nodes_.empty()) {
    auto msg = "graph is empty, no node.";
    return {STATUS_BADCONF, msg};
  }

  auto status = IsAllPortConnect();
  if (!status) {
    auto msg = "not all port connect.";
    return {status, msg};
  }

  status = IsAllNodeConnect();
  if (!status) {
    auto msg = "not all node connect.";
    return {status, msg};
  }

  return STATUS_OK;
}

Status Graph::IsAllPortConnect() const {
  for (auto node : nodes_) {
    // 某些输入port、输出port是可选的, TODO
    auto inports = node.second->GetInputPorts();
    for (auto port : inports) {
      auto ite = dst_to_src_.find(port);
      if (ite == dst_to_src_.end()) {
        auto msg = "in port is not connect. node: " + node.second->GetName() +
                   " port: " + port->GetName();
        return {STATUS_BADCONF, msg};
      }
    }

    auto outports = node.second->GetOutputPorts();
    for (auto port : outports) {
      auto ite = src_to_dst_.find(port);
      if (ite == src_to_dst_.end()) {
        auto msg = "out port is not connect. node: " + node.second->GetName() +
                   " port: " + port->GetName();
        return {STATUS_BADCONF, msg};
      }
    }
  }
  return STATUS_OK;
}

Status Graph::IsAllNodeConnect() const {
  std::map<std::string, int> nodeType;
  int idx = 0;
  for (auto &node : nodes_) {
    nodeType[node.first] = idx++;
  }

  for (auto link : src_to_dst_) {
    for (auto linkport : link.second) {
      auto srcNode = link.first->GetNode();
      auto dstNode = linkport->GetNode();
      auto srcType = nodeType[srcNode->GetName()];
      auto dstType = nodeType[dstNode->GetName()];
      if (srcType == dstType) {
        continue;
      }

      auto msg = "node. srcNode: " + srcNode->GetName() + " #" +
                 std::to_string(srcType) + " dstNode: " + dstNode->GetName() +
                 " #" + std::to_string(dstType);
      MBLOG_DEBUG << msg;

      auto mergeType = srcType < dstType ? srcType : dstType;
      for (auto node : nodeType) {
        if (node.second == srcType || node.second == dstType) {
          nodeType[node.first] = mergeType;
          auto msg = "merge. src: " + node.first + ", " +
                     std::to_string(node.second) + " -> " +
                     std::to_string(mergeType);
          MBLOG_DEBUG << msg;
        }
      }
    }
  }

  for (auto node : nodeType) {
    auto msg = "node: " + node.first + " #" + std::to_string(node.second);
    MBLOG_INFO << msg;
  }
  auto firstType = nodeType.begin();
  for (auto node : nodeType) {
    if (node.second != firstType->second) {
      auto msg = "not all node union.";
      return Status(STATUS_BADCONF, msg);
    }
  }

  return STATUS_OK;
}

Status Graph::UpdatePriority() {
  auto callback = [](std::shared_ptr<NodeBase> node, int order) {
    auto inports = node->GetInputPorts();
    for (auto port : inports) {
      port->SetPriority(order);
      auto msg = "set priority. node: " + node->GetName() +
                 " port: " + port->GetName() +
                 " priority: " + std::to_string(order);
      MBLOG_INFO << msg;
    }
    return true;
  };

  return Topology(callback);
}

Status Graph::Topology(
    std::function<bool(std::shared_ptr<NodeBase> node, int order)> callback)
    const {
  int idx = 0;
  for (auto node : topo_order_) {
    auto ret = callback(node, idx);
    if (!ret) {
      auto msg = "callback fail. topo idx: " + std::to_string(idx) + ", " +
                 node->GetName();
      MBLOG_WARN << msg;
    } else {
      auto msg = "callback success. topo idx: " + std::to_string(idx) + ", " +
                 node->GetName();
      MBLOG_DEBUG << msg;
    }
    idx++;
  }

  return STATUS_OK;
}

Status Graph::CheckStreamMatcher() {
  auto checkingNodes = GetStartNodes();
  auto allNodes = GetAllNodes();
  allNodes.erase(output_node_);
  auto stream_matcher =
      std::make_shared<StreamMatcher>(checkingNodes, allNodes);
  return stream_matcher->StartCheck();
}

void Graph::FindLoopSeq(std::shared_ptr<NodeBase> &root_node,
                        std::vector<std::string> &vis) {
  auto dstNodes = GetDstNodesByNode(root_node->GetName());
  if (dstNodes.empty()) {
    return;
  }

  for (auto dstNode : dstNodes) {
    if (std::find(vis.begin(), vis.end(), dstNode->GetName()) != vis.end()) {
      vis.push_back(dstNode->GetName());
      loop_structures_.push_back(vis);
      vis.pop_back();
      return;
    }

    vis.push_back(dstNode->GetName());
    FindLoopSeq(dstNode, vis);
    vis.pop_back();
  }
}

void Graph::FindLoopWithNode(std::shared_ptr<NodeBase> &root_node,
                             std::vector<std::string> &vis) {
  vis.push_back(root_node->GetName());
  FindLoopSeq(root_node, vis);
}

void Graph::FillLoopLink() {
  for (auto &loop : loop_structures_) {
    auto loop_link_from = *(loop.end() - 2);
    auto loop_link_to = *(loop.end() - 1);
    loop_links_.insert(std::make_pair(loop_link_to, loop_link_from));
  }
}

Status Graph::CheckLoopNode() {
  Status status{STATUS_EOF};
  for (auto &node : nodes_) {
    auto tmp_node = std::dynamic_pointer_cast<Node>(node.second);
    // virtual node
    if (tmp_node == nullptr) {
      continue;
    }

    if (tmp_node->GetLoopType() != LOOP) {
      continue;
    }

    if (tmp_node->GetInputNum() != 1) {
      return {STATUS_FAULT, "loop node input should be one."};
    }

    if (tmp_node->GetOutputNum() != 2) {
      return {STATUS_FAULT, "loop node output shoulde be two."};
    }

    status = STATUS_OK;
  }

  return status;
}

Status Graph::FindLoopStructure() {
  auto status = CheckLoopNode();
  if (status == STATUS_FAULT) {
    return status;
  }

  if (status == STATUS_EOF) {
    MBLOG_DEBUG << "there is no loop node.";
    return STATUS_OK;
  }

  auto connectNode = GetStartNodes();
  if (connectNode.empty()) {
    auto msg = "start node is not exist.";
    return {STATUS_BADCONF, msg};
  }

  std::vector<std::string> vis;
  while (!connectNode.empty()) {
    auto nodeIte = connectNode.begin();
    auto node = *nodeIte;
    connectNode.erase(nodeIte);

    FindLoopWithNode(node, vis);
    vis.clear();
  }

  for (auto &loop : loop_structures_) {
    for (auto &item : loop) {
      MBLOG_INFO << "item: " << item;
    }
  }

  FillLoopLink();
  for (auto &loop : loop_links_) {
    MBLOG_INFO << loop.first << ", " << loop.second;
  }

  return STATUS_OK;
}

Status Graph::GenerateTopology() {
  std::vector<std::shared_ptr<NodeBase>> topoNode;
  auto connectNode = GetStartNodes();
  if (connectNode.empty()) {
    auto msg = "start node is not exist.";
    return {STATUS_BADCONF, msg};
  }

  while (!connectNode.empty()) {
    auto nodeIte = connectNode.begin();
    auto node = *nodeIte;
    connectNode.erase(nodeIte);
    auto ite = std::find(topoNode.begin(), topoNode.end(), node);
    if (ite != topoNode.end()) {
      continue;
    }

    // all inport is in topoNode set
    bool topo = true;
    auto srcNodes = GetSrcNodesByNode(node->GetName());
    for (const auto &srcNode : srcNodes) {
      if (loop_links_.find(node->GetName()) != loop_links_.end() &&
          loop_links_[node->GetName()] == srcNode->GetName()) {
        continue;
      }

      auto ite = std::find(topoNode.begin(), topoNode.end(), srcNode);
      if (ite == topoNode.end()) {
        auto msg = "srcNode not topnode. srcNode: " + node->GetName() +
                   " node: " + node->GetName();
        MBLOG_DEBUG << msg;
        topo = false;
      }
    }

    if (topo) {
      auto msg = "add new topnode. node: " + node->GetName();
      MBLOG_DEBUG << msg;
      topoNode.push_back(node);
      auto dstNodes = GetDstNodesByNode(node->GetName());
      for (auto dstNode : dstNodes) {
        connectNode.insert(dstNode);
        auto msg = "add connect node. " + node->GetName() + " -> " +
                   dstNode->GetName();
        MBLOG_DEBUG << msg;
      }
    }
  }

  auto i = 0;
  for (auto node : topoNode) {
    auto msg = "topo index: " + std::to_string(i) + ", " + node->GetName();
    MBLOG_INFO << msg;
    ++i;
  }

  if (topoNode.size() != nodes_.size()) {
    auto msg = "not all node connect.";
    return {STATUS_BADCONF, msg};
  }

  topo_order_ = topoNode;

  return STATUS_OK;
}

Status Graph::InitPort() {
  for (auto &portIte : dst_to_src_) {
    auto inport = portIte.first;
    for (const auto &outport : portIte.second) {
      auto msg = "port connect, " + outport->GetNode()->GetName() + ":" +
                 outport->GetName() + " -> " + inport->GetNode()->GetName() +
                 ":" + inport->GetName();
      MBLOG_INFO << msg;
      outport->AddPort(inport);
    }
  }

  return STATUS_OK;
}

Status Graph::InitNode(std::shared_ptr<Node> &node,
                       const std::set<std::string> &input_port_names,
                       const std::set<std::string> &output_port_names,
                       std::shared_ptr<Configuration> &config) {
  auto status = node->Init(input_port_names, output_port_names, config);
  return status;
}

Status Graph::InitScheduler() {
  scheduler_ = std::make_shared<FlowScheduler>();
  size_t thread_num = nodes_.size() * 2;
  if (thread_num < std::thread::hardware_concurrency()) {
    thread_num = std::thread::hardware_concurrency();
  }

  if (!config_->Contain("graph.thread-num")) {
    config_->SetProperty("graph.thread-num", thread_num);
  }

  if (!config_->Contain("graph.thread-num")) {
    config_->SetProperty("graph.max-thread-num", thread_num * 4);
  }

  auto status = scheduler_->Init(config_);
  if (!status) {
    auto msg = "init scheduler failed.";
    MBLOG_FATAL << msg;
    return {status, msg};
  }

  status = scheduler_->Build(*this);
  if (!status) {
    auto msg = "build scheduler failed.";
    MBLOG_FATAL << msg;
    return {status, msg};
  }

  return STATUS_OK;
}

Status Graph::Run() {
  if (scheduler_ == nullptr) {
    auto message = "scheduler is not initialized.";
    return {STATUS_SHUTDOWN, message};
  }

  if (profiler_ != nullptr) {
    profiler_->Start();
  }

  return scheduler_->Run();
}

void Graph::RunAsync() {
  if (scheduler_ == nullptr) {
    auto message = "scheduler is not initialized.";
    StatusError = {STATUS_SHUTDOWN, message};
    return;
  }

  if (profiler_ != nullptr) {
    profiler_->Start();
  }

  scheduler_->RunAsync();
}

Status Graph::Wait(int64_t milliseconds, Status *ret_val) {
  if (scheduler_ == nullptr) {
    auto message = "scheduler is not initialized.";
    return {STATUS_SHUTDOWN, message};
  }

  auto status = scheduler_->Wait(milliseconds, ret_val);
  return status;
}

Status Graph::Shutdown() {
  if (scheduler_ != nullptr) {
    scheduler_->Shutdown();
    scheduler_ = nullptr;
  }
  if (profiler_ != nullptr) {
    profiler_->Stop();
  }

  return STATUS_OK;
}

DynamicGraph::DynamicGraph() : Graph() {}
DynamicGraph::~DynamicGraph() { Shutdown(); }
Status DynamicGraph::Shutdown() { return STATUS_OK; }
Status DynamicGraph::IsValidGraph() const { return STATUS_OK; }
Status DynamicGraph::InitScheduler() { return STATUS_OK; }

Status DynamicGraph::InitNode(std::shared_ptr<Node> &node,
                              const std::set<std::string> &input_port_names,
                              const std::set<std::string> &output_port_names,
                              std::shared_ptr<Configuration> &config) {
  auto status = node->Init(input_port_names, output_port_names, config);
  return status;
}

}  // namespace modelbox
