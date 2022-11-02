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

#ifndef MODELBOX_GRAPH_MANAGER_H
#define MODELBOX_GRAPH_MANAGER_H

#include <modelbox/base/configuration.h>
#include <modelbox/base/driver.h>
#include <modelbox/base/log.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace modelbox {

constexpr const char *DRIVER_CLASS_GRAPHCONF = "DRIVER-GRAPHCONF";
constexpr const char *GCGRAPH_NODE_TYPE_NODE = "node";
constexpr const char *GCGRAPH_NODE_TYPE_SUBGRAPH = "subgraph";

class GCGraph;
class DataHandler;
class GCNode {
 public:
  GCNode();
  virtual ~GCNode();

  Status Init(const std::string &name,
              const std::shared_ptr<GCGraph> &root_graph);

  std::string GetNodeName() const;
  std::shared_ptr<Configuration> GetConfiguration() const;
  std::shared_ptr<const std::set<std::string>> GetInputPorts() const;
  std::shared_ptr<const std::set<std::string>> GetOutputPorts() const;
  std::shared_ptr<GCGraph> GetRootGraph() const;
  std::string GetNodeType() const;

  void SetNodeType(std::string type);
  void SetConfiguration(const std::string &key, const std::string &value);
  Status SetInputPort(const std::string &port);
  Status SetOutputPort(const std::string &port);
  void SetOutDataHandler(std::shared_ptr<DataHandler> &data_handler);
  std::shared_ptr<DataHandler> GetBindDataHandler();

 private:
  std::string name_;
  std::string type_;
  std::weak_ptr<GCGraph> root_graph_;
  std::set<std::string> input_ports_;
  std::set<std::string> output_ports_;
  std::shared_ptr<Configuration> configuration_;
  std::weak_ptr<DataHandler> out_data_handler_;
};

class GCEdge {
 public:
  GCEdge();
  virtual ~GCEdge();

  Status Init(const std::shared_ptr<GCGraph> &root_graph);

  const std::string &GetHeadOutPort() const;
  const std::string &GetTailInPort() const;
  std::shared_ptr<GCNode> GetHeadNode() const;
  std::shared_ptr<GCNode> GetTailNode() const;
  std::shared_ptr<Configuration> GetConfiguration() const;
  std::shared_ptr<GCGraph> GetRootGraph() const;

  Status SetHeadNode(std::shared_ptr<GCNode> node);
  Status SetTailNode(std::shared_ptr<GCNode> node);
  Status SetHeadPort(std::string port);
  Status SetTailPort(std::string port);
  void SetConfiguration(const std::string &key, const std::string &value);

 private:
  std::shared_ptr<GCNode> head_;
  std::shared_ptr<GCNode> tail_;
  std::weak_ptr<GCGraph> root_graph_;
  std::string head_out_port_;
  std::string tail_in_port_;
  std::shared_ptr<Configuration> configuration_;
};

class GCGraph {
 public:
  GCGraph();
  virtual ~GCGraph();

  Status Init(const std::shared_ptr<GCGraph> &root_graph);

  void SetGraphName(const std::string &name);
  const std::string &GetGraphName() const;
  std::shared_ptr<GCGraph> GetRootGraph() const;

  Status AddSubGraph(const std::shared_ptr<GCGraph> &subgraph);
  std::shared_ptr<GCGraph> GetSubGraph(const std::string &name) const;
  std::map<std::string, const std::shared_ptr<GCGraph>> GetAllSubGraphs() const;
  void ShowAllSubGraph() const;

  Status AddNode(const std::shared_ptr<GCNode> &node);
  Status SetFirstNode(const std::shared_ptr<GCNode> &node);
  std::vector<std::shared_ptr<GCNode>> GetFirstNodes();
  std::shared_ptr<GCNode> GetNode(const std::string &name) const;
  std::map<std::string, const std::shared_ptr<GCNode>> GetAllNodes() const;
  void ShowAllNode() const;

  Status AddEdge(const std::shared_ptr<GCEdge> &edge);
  std::shared_ptr<GCEdge> GetEdge(const std::string &name) const;
  std::map<std::string, const std::shared_ptr<GCEdge>> GetAllEdges() const;
  void ShowAllEdge() const;

  std::shared_ptr<Configuration> GetConfiguration() const;
  void SetConfiguration(const std::string &key, const std::string &value);
  void SetConfiguration(std::shared_ptr<Configuration> &config);

 private:
  std::map<std::string, const std::shared_ptr<GCNode>> nodes_;
  std::map<std::string, const std::shared_ptr<GCEdge>> edges_;
  std::map<std::string, const std::shared_ptr<GCGraph>> subgraphs_;
  std::weak_ptr<GCGraph> root_graph_;
  std::vector<std::shared_ptr<GCNode>> first_nodes_;
  std::string name_;
  std::shared_ptr<Configuration> configuration_;
};

class GraphConfig {
 public:
  GraphConfig();
  virtual ~GraphConfig();

  virtual std::shared_ptr<GCGraph> Resolve() = 0;
};

class GraphConfigFactory : public DriverFactory {
 public:
  GraphConfigFactory();
  ~GraphConfigFactory() override;
  virtual std::shared_ptr<GraphConfig> CreateGraphConfigFromStr(
      const std::string &graph_config) = 0;
  virtual std::shared_ptr<GraphConfig> CreateGraphConfigFromFile(
      const std::string &file_path) = 0;
  virtual std::string GetGraphConfFactoryType() = 0;
};

class GraphConfigManager {
 public:
  GraphConfigManager();
  virtual ~GraphConfigManager();

  static GraphConfigManager &GetInstance();

  Status Register(const std::shared_ptr<GraphConfigFactory> &factory);

  Status Initialize(const std::shared_ptr<Drivers> &driver,
                    const std::shared_ptr<Configuration> &config);

  std::shared_ptr<GraphConfig> LoadGraphConfig(
      const std::shared_ptr<Configuration> &config);

  std::vector<std::string> GetSupportTypes();

  void Clear();

 private:
  Status InitGraphConfigFactory(const std::shared_ptr<Drivers> &driver);

  std::map<std::string, const std::shared_ptr<GraphConfigFactory>>
  GetGraphConfFactoryList();

  std::shared_ptr<GraphConfigFactory> GetGraphConfFactory(
      const std::string &type);

  std::shared_ptr<GraphConfig> CreateGraphConfig(std::string graph_conf_type,
                                                 std::string graph_conf_name);
  std::shared_ptr<GraphConfig> GetGraphConfig(
      const std::string &graph_conf_name);
  std::map<std::string, const std::shared_ptr<GraphConfig>> GetGraphConfList();
  Status DeleteGraphConfig(const std::string &graph_conf_name);

  std::map<std::string, const std::shared_ptr<GraphConfigFactory>>
      graph_conf_factories_;
  std::map<std::string, const std::shared_ptr<GraphConfig>> graph_conf_list_;
};

}  // namespace modelbox
#endif  // MODELBOX_GRAPH_MANAGER_H