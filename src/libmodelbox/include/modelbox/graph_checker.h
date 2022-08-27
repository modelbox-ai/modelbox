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

#ifndef MODELBOX_GRAPH_CHECKER_H
#define MODELBOX_GRAPH_CHECKER_H

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "modelbox/base/status.h"
#include "modelbox/node.h"
#include "modelbox/virtual_node.h"

namespace modelbox {

class Graph;
class NodeBase;
class InputVirtualNode;
class OutputVirtualNode;
class OutputUnmatchVirtualNode;

enum IndexPortType { INPUT, OUTPUT, UNKNOWN };

class IndexPort {
 public:
  IndexPort();
  IndexPort(std::string node, std::string port,
            const IndexPortType &type = IndexPortType::UNKNOWN);
  virtual ~IndexPort();

  std::string ToString() const;

  void SetNodeName(std::string node_name);

  void SetPortName(std::string port_name);

  void SetPortType(IndexPortType port_type);

  const std::string &GetNodeName() const;

  const std::string &GetPortName() const;

  const IndexPortType &GetPortType() const;

 private:
  std::string node_name_;
  std::string port_name_;
  IndexPortType port_type_{IndexPortType::UNKNOWN};
};

using NodeStreamConnection = std::map<std::string, std::vector<IndexPort>>;

class LeastCommonAncestor {
 public:
  LeastCommonAncestor(
      std::unordered_map<std::string, std::shared_ptr<NodeBase>> all_nodes);
  virtual ~LeastCommonAncestor();
  void Update(const std::vector<IndexPort> &values,
              const std::unordered_map<std::string, std::string> &match_map);
  IndexPort Find(const IndexPort &node_a, const IndexPort &node_b);

 private:
  void InitMap();
  // match_a_name & match_b_name : match_node_name
  std::string GetMatchPortName(const std::string &match_a_name,
                               const std::string &match_b_name,
                               const std::string &match_node_name);
  std::string GetMatchPortName(const std::string &port_name,
                               const std::string &match_a_name,
                               const std::string &match_b_name,
                               const std::string &match_node_name);
  IndexPort ProcessSameNode(const IndexPort &node_a, const IndexPort &node_b);
  void FindMatchNode(const IndexPort &node_a, const IndexPort &node_b,
                     std::string &match_a_name, std::string &match_b_name,
                     std::string &match_node_name, std::string &port_name);
  void GetIndexPortType(const std::string &node_name,
                        const std::string &port_name, IndexPortType &port_type);

  std::unordered_map<std::string, std::shared_ptr<NodeBase>> all_nodes_;
  std::map<int, std::vector<int>> paths_;
  std::unordered_map<int, std::string> index_name_map_;
  std::unordered_map<std::string, int> name_index_map_;
};

class OverHierarchyCheck {
 public:
  OverHierarchyCheck(
      const std::unordered_map<std::string, std::shared_ptr<NodeBase>>
          &all_nodes,
      std::set<std::shared_ptr<NodeBase>> start_nodes,
      std::map<std::string, std::string> loop_links,
      std::vector<std::vector<std::string>> loop_structures,
      std::map<std::shared_ptr<OutPort>, std::set<std::shared_ptr<InPort>>>
          edges);
  virtual ~OverHierarchyCheck();

  Status Check(
      const std::unordered_map<std::string, std::string> &graph_match_map,
      const std::unordered_map<std::string,
                               std::unordered_map<std::string, std::string>>
          &graph_single_port_match_map,
      const std::unordered_map<std::string, std::string> &end_if_map);

 private:
  void InitFirstNode(const std::shared_ptr<Node> &node);
  Status CheckInputPortsColorReady(
      std::shared_ptr<IndexPort> &index_port,
      const std::vector<std::shared_ptr<InPort>> &input_ports);
  Status CheckInputPorts(
      const std::shared_ptr<Node> &node,
      const std::unordered_map<std::string,
                               std::unordered_map<std::string, std::string>>
          &graph_single_port_match_map);
  void GetColorMap(
      const std::shared_ptr<Node> &node,
      const std::vector<std::shared_ptr<OutPort>> &output_ports,
      const std::unordered_map<std::string, std::string> &graph_match_map,
      const std::unordered_map<std::string,
                               std::unordered_map<std::string, std::string>>
          &graph_single_port_match_map,
      const std::unordered_map<std::string, std::string> &end_if_map);
  std::shared_ptr<NodeBase> FindLoopLinkNode(const std::shared_ptr<Node> &node);
  void SetOutPortColor(const std::shared_ptr<Node> &node,
                       const std::vector<std::shared_ptr<OutPort>> &out_ports,
                       const std::vector<int> &new_color);
  bool CheckEndIfPort(
      const std::shared_ptr<InPort> &input_port,
      const std::shared_ptr<IndexPort> &index_port,
      const std::unordered_map<std::string,
                               std::unordered_map<std::string, std::string>>
          &graph_single_port_match_map);
  bool CheckEndIfNode(
      const std::shared_ptr<Node> &node,
      const std::unordered_map<std::string, std::string> &end_if_map);

  std::unordered_map<std::string, std::shared_ptr<NodeBase>> all_nodes_;
  std::set<std::shared_ptr<NodeBase>> start_nodes_;
  std::map<std::string, std::string> loop_links_;
  std::vector<std::vector<std::string>> loop_structures_;
  std::map<std::shared_ptr<OutPort>, std::set<std::shared_ptr<InPort>>> edges_;
  std::unordered_map<std::string, std::vector<int>> color_map_;
  std::unordered_map<std::string, bool> visited_;
  int max_color_{0};
};

class GraphChecker {
 public:
  GraphChecker(const std::vector<std::shared_ptr<NodeBase>> &nodes,
               const std::set<std::shared_ptr<NodeBase>> &start_nodes,
               std::map<std::string, std::string> loop_links,
               std::vector<std::vector<std::string>> loop_structures,
               const std::map<std::shared_ptr<OutPort>,
                              std::set<std::shared_ptr<InPort>>> &edges);
  virtual ~GraphChecker();

  void SetMatchNodes();
  void ShowMatchNodes();
  Status Check();

 private:
  Status CalNodeStreamMap(const std::shared_ptr<NodeBase> &node,
                          NodeStreamConnection &node_stream_map);
  Status CheckNodeMatch(const std::shared_ptr<Node> &node,
                        const NodeStreamConnection &node_stream_map);
  Status CheckCollapseMatch(const std::shared_ptr<Node> &node);
  Status CheckBranchPathMatch(const std::string &start, const std::string &end);
  Status CheckOverHierarchyMatch();
  Status CheckUnmatchExpands(size_t size);
  Status CheckLeastCommonAncestorsAnyTwoNodes(
      const std::vector<IndexPort> &match_nodes,
      std::vector<IndexPort> &res_nodes);
  Status LeastCommonAncestors(const std::vector<IndexPort> &match_nodes,
                              IndexPort &res_match_node);
  std::unordered_map<std::string, std::string> GetGraphMatchMap();
  bool CheckPortMatch(const IndexPort &match_pair);
  void FindNearestNeighborMatchExpand(const std::string &node,
                                      std::string &match_node);
  void UpdateAncestorPath(const std::vector<IndexPort> &values);

  std::vector<std::shared_ptr<NodeBase>> nodes_;
  std::map<std::string, std::string> loop_links_;
  std::vector<std::vector<std::string>> loop_structures_;
  std::shared_ptr<LeastCommonAncestor> lca_;
  std::shared_ptr<OverHierarchyCheck> ovc_;
  std::unordered_map<std::string, std::shared_ptr<NodeBase>> all_nodes_;
  std::unordered_map<std::string, std::string> graph_match_map_;
  std::map<std::string, NodeStreamConnection> node_stream_connection_map_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      graph_single_port_match_map_;
  std::unordered_map<std::string, std::string> end_if_map_;
  size_t expands_{0};
};

}  // namespace modelbox

#endif