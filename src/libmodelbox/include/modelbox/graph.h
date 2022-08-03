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

#ifndef MODELBOX_GRAPH_H_
#define MODELBOX_GRAPH_H_

#include <modelbox/flowunit.h>
#include <modelbox/scheduler.h>
#include <modelbox/session.h>

#include <memory>
#include <vector>

#include "modelbox/base/graph_manager.h"
#include "modelbox/node.h"
#include "modelbox/statistics.h"
#include "modelbox/virtual_node.h"
namespace modelbox {

class Graph {
 public:
  Graph();
  virtual ~Graph();

  /**
   * @brief Initialize graph
   * @param flowunit_mgr flowunit manager
   * @param device_mgr device manager
   * @param profiler profiler
   * @param config configuration
   * @return initialize result
   */
  Status Initialize(const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
                    const std::shared_ptr<DeviceManager> &device_mgr,
                    std::shared_ptr<Profiler> profiler,
                    const std::shared_ptr<Configuration> &config);
  /**
   * @brief Build a graph
   * @param g graph data pointer
   * @return initialize result
   */
  Status Build(const std::shared_ptr<GCGraph> &g);

  Status Topology(const std::function<bool(std::shared_ptr<NodeBase> node,
                                           int order)> &callback) const;

  Status AddNode(const std::shared_ptr<NodeBase> &node);

  std::shared_ptr<NodeBase> GetNode(const std::string &nodeName) const;

  std::shared_ptr<InPort> GetInPort(const std::string &nodeName,
                                    const std::string &portName) const;

  std::unordered_map<std::shared_ptr<NodeBase>,
                     std::vector<std::shared_ptr<IPort>>>
  GetNotifyPort() const;

  std::shared_ptr<OutPort> GetOutPort(const std::string &nodeName,
                                      const std::string &portName) const;

  Status AddLink(const std::string &srcNodeName, const std::string &srcPortName,
                 const std::string &dstNodeName,
                 const std::string &dstPortName);

  Status AddLink(const std::shared_ptr<OutPort> &src,
                 const std::shared_ptr<InPort> &dst);

  std::set<std::shared_ptr<InPort>> GetDstPortsByPort(
      const std::shared_ptr<OutPort> &port) const;

  std::set<std::shared_ptr<OutPort>> GetSrcPortsByPort(
      const std::shared_ptr<InPort> &port) const;

  std::set<std::shared_ptr<NodeBase>> GetStartNodes() const;

  std::set<std::shared_ptr<NodeBase>> GetEndNodes() const;

  std::set<std::shared_ptr<NodeBase>> GetAllNodes() const;

  std::set<std::shared_ptr<NodeBase>> GetDstNodesByNode(
      const std::string &nodeName) const;

  std::set<std::shared_ptr<NodeBase>> GetSrcNodesByNode(
      const std::string &nodeName) const;

  std::shared_ptr<ExternalDataMap> CreateExternalDataMap();

  Status Run();

  void RunAsync();

  virtual Status Shutdown();

  Status Wait(int64_t milliseconds = 0, Status *ret_val = nullptr);

  std::string GetId() const;

  std::string GetName() const;

  std::set<std::shared_ptr<NodeBase>> GetEndPointNodes() const;

 private:
  void ShowGraphInfo(const std::shared_ptr<GCGraph> &g);

  Status CheckGraph();

  Status BuildFlowunitNode(const std::shared_ptr<GCGraph> &g,
                           const std::shared_ptr<GCNode> &gcnode, bool strict);

  Status BuildInputNode(const std::shared_ptr<GCNode> &gcnode);

  Status BuildOutputNode(const std::shared_ptr<GCNode> &gcnode);

  Status BuildNode(const std::shared_ptr<GCGraph> &g,
                   const std::shared_ptr<GCNode> &gcnode, bool strict);

  Status BuildNodes(const std::shared_ptr<GCGraph> &g);

  Status BuildVirtualNodes(const std::shared_ptr<GCGraph> &g);

  Status BuildEdges(const std::shared_ptr<GCGraph> &g);

  Status BuildGraph(const std::shared_ptr<GCGraph> &g);

  Status OpenNodes();

  void CloseNodes() const;

  virtual Status IsValidGraph() const;
  void FindLoopWithNode(std::shared_ptr<NodeBase> &root_node,
                        std::vector<std::string> &vis);

  void FindLoopSeq(std::shared_ptr<NodeBase> &root_node,
                   std::vector<std::string> &vis);

  Status FindLoopStructure();

  void FillLoopLink();

  void FillLoopEndPort();

  Status CheckLoopNode();

  Status IsAllPortConnect() const;

  Status IsAllNodeConnect() const;

  Status UpdatePriority();

  Status GenerateTopology();

  Status CheckLoopStructureNode();

  Status InitPort();

  virtual Status InitScheduler();

  Status UpdateGraphConfigToNode(const std::shared_ptr<GCGraph> &g,
                                 const std::shared_ptr<GCNode> &node);

  virtual Status InitNode(std::shared_ptr<Node> &node,
                          const std::set<std::string> &input_port_names,
                          const std::set<std::string> &output_port_names,
                          std::shared_ptr<Configuration> &config);

  SessionManager session_manager_;

  std::map<std::string, std::shared_ptr<NodeBase>> nodes_;

  std::map<std::shared_ptr<OutPort>, std::set<std::shared_ptr<InPort>>>
      src_to_dst_;

  std::map<std::shared_ptr<InPort>, std::set<std::shared_ptr<OutPort>>>
      dst_to_src_;

  std::vector<std::shared_ptr<NodeBase>> topo_order_;

  std::shared_ptr<Scheduler> scheduler_;

  std::shared_ptr<FlowUnitManager> flowunit_mgr_;

  std::shared_ptr<DeviceManager> device_mgr_;

  std::shared_ptr<Profiler> profiler_;

  std::shared_ptr<StatisticsItem> flow_stats_;
  std::shared_ptr<StatisticsItem> graph_stats_;

  std::shared_ptr<Configuration> config_;

  std::string input_node_name_;

  std::set<std::string> input_node_ports_;

  std::unordered_map<std::string, std::shared_ptr<Configuration>>
      input_node_config_map_;

  std::shared_ptr<Node> input_node_;

  std::string output_node_name_;

  std::set<std::string> output_node_ports_;

  std::unordered_map<std::string, std::shared_ptr<Configuration>>
      output_node_config_map_;

  std::shared_ptr<NodeBase> output_node_;

  std::string name_;

  std::string id_;

  std::vector<std::vector<std::string>> loop_structures_;

  std::map<std::string, std::string> loop_links_;

  bool is_stop_{false};
};

class DynamicGraph : public Graph {
 public:
  DynamicGraph();
  ~DynamicGraph() override;

  Status Shutdown() override;
  Status IsValidGraph() const override;
  Status InitScheduler() override;

  Status InitNode(std::shared_ptr<Node> &node,
                  const std::set<std::string> &input_port_names,
                  const std::set<std::string> &output_port_names,
                  std::shared_ptr<Configuration> &config) override;
};

}  // namespace modelbox

#endif  // MODELBOX_GRAPH_H
