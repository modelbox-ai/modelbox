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

#include <modelbox/base/graph_manager.h>

#include <utility>

namespace modelbox {

std::vector<std::string> SplitByCommaIgnoreQuotes(const std::string &str) {
  std::string value;
  std::vector<std::string> values;
  char quoteChar = 0;

  for (char ch : str) {
    if (quoteChar == '\\') {
      value.push_back(ch);
      quoteChar = 0;
      continue;
    }

    if (quoteChar && ch != quoteChar) {
      value.push_back(ch);
      continue;
    }

    switch (ch) {
      case '\'':
      case '\"':
      case '\\':
        quoteChar = quoteChar ? 0 : ch;
        break;
      case ',':
        if (!value.empty()) {
          values.push_back(value);
          value.clear();
        }
        break;
      default:
        value.push_back(ch);
        break;
    }
  }

  if (!value.empty()) {
    values.push_back(value);
  }

  return values;
}

GCNode::GCNode() = default;

GCNode::~GCNode() = default;

Status GCNode::Init(const std::string &name,
                    const std::shared_ptr<GCGraph> &root_graph) {
  name_ = name;
  root_graph_ = root_graph;
  ConfigurationBuilder builder;
  configuration_ = builder.Build();

  return STATUS_OK;
}

std::string GCNode::GetNodeName() const { return name_; }

std::shared_ptr<Configuration> GCNode::GetConfiguration() const {
  return configuration_;
}

std::shared_ptr<const std::set<std::string>> GCNode::GetInputPorts() const {
  auto input_ports =
      std::make_shared<const std::set<std::string>>(input_ports_);
  return input_ports;
}

std::shared_ptr<const std::set<std::string>> GCNode::GetOutputPorts() const {
  auto output_ports =
      std::make_shared<const std::set<std::string>>(output_ports_);
  return output_ports;
}

std::shared_ptr<GCGraph> GCNode::GetRootGraph() const {
  return root_graph_.lock();
}

std::string GCNode::GetNodeType() const { return type_; }

void GCNode::SetNodeType(std::string type) { type_ = std::move(type); }

void GCNode::SetConfiguration(const std::string &key,
                              const std::string &value) {
  auto sub_str_list = SplitByCommaIgnoreQuotes(value);
  for (auto &str : sub_str_list) {
    Configuration::Trim(&str);
  }
  configuration_->SetProperty(key, sub_str_list);
}

Status GCNode::SetInputPort(const std::string &port) {
  input_ports_.insert(port);
  return STATUS_OK;
}

Status GCNode::SetOutputPort(const std::string &port) {
  output_ports_.insert(port);
  return STATUS_OK;
}

void GCNode::SetOutDataHandler(std::shared_ptr<DataHandler> &data_handler) {
  out_data_handler_ = data_handler;
}

std::shared_ptr<DataHandler> GCNode::GetBindDataHandler() {
  return out_data_handler_.lock();
}

GCEdge::GCEdge() = default;

GCEdge::~GCEdge() = default;

Status GCEdge::Init(const std::shared_ptr<GCGraph> &root_graph) {
  root_graph_ = root_graph;
  ConfigurationBuilder builder;
  configuration_ = builder.Build();
  return STATUS_OK;
}

const std::string &GCEdge::GetHeadOutPort() const { return head_out_port_; }

const std::string &GCEdge::GetTailInPort() const { return tail_in_port_; }

std::shared_ptr<GCNode> GCEdge::GetHeadNode() const { return head_; }

std::shared_ptr<GCNode> GCEdge::GetTailNode() const { return tail_; }

std::shared_ptr<Configuration> GCEdge::GetConfiguration() const {
  return configuration_;
}

std::shared_ptr<GCGraph> GCEdge::GetRootGraph() const {
  return root_graph_.lock();
}

Status GCEdge::SetHeadNode(std::shared_ptr<GCNode> node) {
  head_ = std::move(node);
  return STATUS_OK;
}

Status GCEdge::SetTailNode(std::shared_ptr<GCNode> node) {
  tail_ = std::move(node);
  return STATUS_OK;
}

Status GCEdge::SetHeadPort(std::string port) {
  head_out_port_ = std::move(port);
  return STATUS_OK;
}

Status GCEdge::SetTailPort(std::string port) {
  tail_in_port_ = std::move(port);
  return STATUS_OK;
}

void GCEdge::SetConfiguration(const std::string &key,
                              const std::string &value) {
  auto sub_str_list = SplitByCommaIgnoreQuotes(value);
  for (auto &str : sub_str_list) {
    Configuration::Trim(&str);
  }
  configuration_->SetProperty(key, sub_str_list);
}

GCGraph::GCGraph() = default;

GCGraph::~GCGraph() = default;

Status GCGraph::Init(const std::shared_ptr<GCGraph> &root_graph) {
  root_graph_ = root_graph;
  ConfigurationBuilder builder;
  configuration_ = builder.Build();
  return STATUS_OK;
}

void GCGraph::SetGraphName(const std::string &name) { name_ = name; };

const std::string &GCGraph::GetGraphName() const { return name_; }

std::shared_ptr<GCGraph> GCGraph::GetRootGraph() const {
  return root_graph_.lock();
}

Status GCGraph::AddSubGraph(const std::shared_ptr<GCGraph> &subgraph) {
  std::string key = subgraph->GetGraphName();
  subgraphs_.insert(
      std::pair<std::string, const std::shared_ptr<GCGraph>>(key, subgraph));
  return STATUS_OK;
}

std::shared_ptr<GCGraph> GCGraph::GetSubGraph(const std::string &name) const {
  auto elem = subgraphs_.find(name);
  if (elem == subgraphs_.end()) {
    return nullptr;
  }
  return elem->second;
}

std::map<std::string, const std::shared_ptr<GCGraph>> GCGraph::GetAllSubGraphs()
    const {
  return subgraphs_;
}

void GCGraph::ShowAllSubGraph() const {}

Status GCGraph::AddNode(const std::shared_ptr<GCNode> &node) {
  std::string key = node->GetNodeName();
  nodes_.insert(
      std::pair<std::string, const std::shared_ptr<GCNode>>(key, node));
  return STATUS_OK;
}

Status GCGraph::SetFirstNode(const std::shared_ptr<GCNode> &node) {
  first_nodes_.push_back(node);
  return STATUS_OK;
}

std::shared_ptr<GCNode> GCGraph::GetNode(const std::string &name) const {
  auto elem = nodes_.find(name);
  if (elem == nodes_.end()) {
    return nullptr;
  }
  return elem->second;
}

std::vector<std::shared_ptr<GCNode>> GCGraph::GetFirstNodes() {
  return first_nodes_;
}

std::map<std::string, const std::shared_ptr<GCNode>> GCGraph::GetAllNodes()
    const {
  return nodes_;
}

void GCGraph::ShowAllNode() const {
  for (const auto &elem : nodes_) {
    MBLOG_INFO << "node name : " << elem.second->GetNodeName();

    std::shared_ptr<const std::set<std::string>> input_ports;
    input_ports = elem.second->GetInputPorts();
    for (const auto &input_port : *input_ports) {
      MBLOG_INFO << "input port : " << input_port;
    }

    std::shared_ptr<const std::set<std::string>> output_ports;
    output_ports = elem.second->GetOutputPorts();
    for (const auto &output_port : *output_ports) {
      MBLOG_INFO << "output port : " << output_port;
    }
  }
}

Status GCGraph::AddEdge(const std::shared_ptr<GCEdge> &edge) {
  std::string key =
      edge->GetHeadNode()->GetNodeName() + ":" + edge->GetHeadOutPort() + "-" +
      edge->GetTailNode()->GetNodeName() + ":" + edge->GetTailInPort();
  edges_.insert(
      std::pair<std::string, const std::shared_ptr<GCEdge>>(key, edge));
  return STATUS_OK;
}

std::shared_ptr<GCEdge> GCGraph::GetEdge(const std::string &name) const {
  auto elem = edges_.find(name);
  if (elem == edges_.end()) {
    return nullptr;
  }
  return elem->second;
}

std::map<std::string, const std::shared_ptr<GCEdge>> GCGraph::GetAllEdges()
    const {
  return edges_;
}

void GCGraph::ShowAllEdge() const {
  for (const auto &elem : edges_) {
    MBLOG_DEBUG << elem.second->GetHeadNode()->GetNodeName() << ":"
                << elem.second->GetHeadOutPort() << "->"
                << elem.second->GetTailNode()->GetNodeName() << ":"
                << elem.second->GetTailInPort();
  }
}

std::shared_ptr<Configuration> GCGraph::GetConfiguration() const {
  return configuration_;
}

void GCGraph::SetConfiguration(const std::string &key,
                               const std::string &value) {
  auto sub_str_list = SplitByCommaIgnoreQuotes(value);
  for (auto &str : sub_str_list) {
    Configuration::Trim(&str);
  }
  configuration_->SetProperty(key, sub_str_list);
}

GraphConfig::GraphConfig() = default;

GraphConfig::~GraphConfig() = default;

GraphConfigFactory::GraphConfigFactory() = default;

GraphConfigFactory::~GraphConfigFactory() = default;

GraphConfigManager::GraphConfigManager() = default;

GraphConfigManager::~GraphConfigManager() = default;

Status GraphConfigManager::Initialize(
    const std::shared_ptr<Drivers> &driver,
    const std::shared_ptr<Configuration> &config) {
  auto ret = InitGraphConfigFactory(driver);
  if (STATUS_OK != ret) {
    MBLOG_ERROR << "Init Graph config factory failed";
    return STATUS_FAULT;
  }
  return STATUS_OK;
}

std::shared_ptr<GraphConfig> GraphConfigManager::LoadGraphConfig(
    const std::shared_ptr<Configuration> &config) {
  std::shared_ptr<GraphConfig> graph_config;
  auto graph_format = config->GetString("graph.format", "");
  if (graph_format == "") {
    MBLOG_ERROR << "graph.format is empty.";
    StatusError = {STATUS_BADCONF, "graph.format is empty"};
    return nullptr;
  }

  MBLOG_INFO << "graph.format : " << graph_format;
  auto graph_conf_factory =
      GetGraphConfFactory(graph_format);  // from config get this type.
  if (graph_conf_factory == nullptr) {
    std::string types;
    for (auto &type : GetSupportTypes()) {
      if (types.length() == 0) {
        types = type;
        continue;
      }

      types += ", " + type;
    }
    MBLOG_ERROR << "Graph format not supported, support type: " << types;
    StatusError = {STATUS_NOTSUPPORT,
                   "Graph format not supported, support type:" + types};
    return nullptr;
  }

  auto graph_graphconf_array = config->GetStrings("graph.graphconf");
  std::string graph_graphconf;
  for (const auto &line : graph_graphconf_array) {
    graph_graphconf += line + "\n";
  }

  if (graph_graphconf != "") {
    graph_config = graph_conf_factory->CreateGraphConfigFromStr(
        graph_graphconf);  // from config get graph config value
    return graph_config;
  }

  auto graph_graphconf_file_path =
      config->GetString("graph.graphconffilepath", "");
  MBLOG_INFO << "graph.graphconffilepath : " << graph_graphconf_file_path;
  if (graph_graphconf_file_path == "") {
    MBLOG_ERROR << "get graph config and graph config file path all failed, "
                   "value is null";
    StatusError = {STATUS_NOTFOUND, "graph config path is null."};
    return nullptr;
  }

  graph_config = graph_conf_factory->CreateGraphConfigFromFile(
      graph_graphconf_file_path);  // from config get graph config file value

  return graph_config;
}

GraphConfigManager &GraphConfigManager::GetInstance() {
  static GraphConfigManager graph_config_manager;
  return graph_config_manager;
}

Status GraphConfigManager::Register(
    const std::shared_ptr<GraphConfigFactory> &factory) {
  graph_conf_factories_.insert(
      std::pair<std::string, std::shared_ptr<GraphConfigFactory>>(
          factory->GetGraphConfFactoryType(), factory));
  return STATUS_OK;
}

std::map<std::string, const std::shared_ptr<GraphConfigFactory>>
GraphConfigManager::GetGraphConfFactoryList() {
  return graph_conf_factories_;
}

std::vector<std::string> GraphConfigManager::GetSupportTypes() {
  std::vector<std::string> ret;
  for (auto &type : graph_conf_factories_) {
    ret.push_back(type.first);
  }

  return ret;
}

std::shared_ptr<GraphConfigFactory> GraphConfigManager::GetGraphConfFactory(
    const std::string &type) {
  auto graph_conf_map = graph_conf_factories_.find(type);
  if (graph_conf_map == graph_conf_factories_.end()) {
    MBLOG_ERROR << "do not find graph config factory type " << type;
    return nullptr;
  }
  return graph_conf_map->second;
}

Status GraphConfigManager::InitGraphConfigFactory(
    const std::shared_ptr<Drivers> &driver) {
  std::vector<std::shared_ptr<Driver>> driver_list =
      driver->GetDriverListByClass(DRIVER_CLASS_GRAPHCONF);
  std::shared_ptr<DriverDesc> desc;
  for (auto &device_driver : driver_list) {
    auto temp_factory = device_driver->CreateFactory();
    if (nullptr == temp_factory) {
      continue;
    }

    std::shared_ptr<GraphConfigFactory> graph_conf_factory =
        std::dynamic_pointer_cast<GraphConfigFactory>(temp_factory);

    graph_conf_factories_.insert(std::make_pair(
        graph_conf_factory->GetGraphConfFactoryType(), graph_conf_factory));
  }

  return STATUS_OK;
}

std::shared_ptr<GraphConfig> GraphConfigManager::GetGraphConfig(
    const std::string &graph_conf_name) {
  auto graph_conf = graph_conf_list_.find(graph_conf_name);
  return graph_conf->second;
}

std::map<std::string, const std::shared_ptr<GraphConfig>>
GraphConfigManager::GetGraphConfList() {
  return graph_conf_list_;
}

Status GraphConfigManager::DeleteGraphConfig(
    const std::string &graph_conf_name) {
  auto graph_conf = graph_conf_list_.find(graph_conf_name);
  if (graph_conf == graph_conf_list_.end()) {
    return STATUS_OK;
  }

  graph_conf_list_.erase(graph_conf);
  return STATUS_OK;
}

void GraphConfigManager::Clear() {
  graph_conf_list_.clear();
  graph_conf_factories_.clear();
}

void GCGraph::SetConfiguration(std::shared_ptr<Configuration> &config) {
  configuration_ = config;
}

}  // namespace modelbox