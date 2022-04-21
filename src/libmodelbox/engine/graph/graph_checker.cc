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

#include "modelbox/graph_checker.h"

#include <cmath>
#include <queue>
#include <stack>

namespace modelbox {

constexpr const char *EXTERNAL = "external";

static std::shared_ptr<Node> CastNode(std::shared_ptr<NodeBase> node_base) {
  return std::dynamic_pointer_cast<Node>(node_base);
};

void LeastCommonAncestor::InitMap() {
  int index = 0;
  for (auto &all_node : all_nodes_) {
    index_name_map_[index] = all_node.second->GetName();
    name_index_map_[all_node.second->GetName()] = index;
    index++;
  }
}

LeastCommonAncestor::LeastCommonAncestor(
    const std::unordered_map<std::string, std::shared_ptr<NodeBase>> &all_nodes)
    : all_nodes_(all_nodes) {
  InitMap();
}

LeastCommonAncestor::~LeastCommonAncestor() {
  index_name_map_.clear();
  name_index_map_.clear();
}

void LeastCommonAncestor::Update(
    const std::vector<IndexPort> &values,
    const std::unordered_map<std::string, std::string> &match_map) {
  for (auto &value : values) {
    auto cur_name = value.node_name;
    auto index = name_index_map_[cur_name];
    std::vector<int> path{index};
    auto pre_name = match_map.at(cur_name);
    while (pre_name != EXTERNAL) {
      path.push_back(name_index_map_[pre_name]);
      cur_name = pre_name;
      pre_name = match_map.at(cur_name);
    }

    path.push_back(-1);
    paths_[index] = path;
  }
}

std::string LeastCommonAncestor::GetMatchPortName(
    const std::string &match_a_name, const std::string &match_b_name,
    const std::string &match_node_name) {
  std::shared_ptr<Node> node_a = CastNode(all_nodes_[match_a_name]);
  std::shared_ptr<Node> node_b = CastNode(all_nodes_[match_b_name]);
  std::shared_ptr<Node> match = CastNode(all_nodes_[match_node_name]);
  std::shared_ptr<InputVirtualNode> match_virtual_node;
  if (match == nullptr) {
    match_virtual_node = std::dynamic_pointer_cast<InputVirtualNode>(
        all_nodes_[match_node_name]);
  }

  auto input_port_a = node_a->GetInputPorts()[0];
  auto input_port_b = node_b->GetInputPorts()[0];
  std::string output_port_a =
      input_port_a->GetAllOutPort()[0].lock()->GetName();
  std::string output_port_b =
      input_port_b->GetAllOutPort()[0].lock()->GetName();
  if (output_port_a != output_port_b) {
    std::vector<std::shared_ptr<InPort>> match_input_ports;
    if (match == nullptr) {
      match_input_ports = match_virtual_node->GetInputPorts();
    } else {
      match_input_ports = match->GetInputPorts();
    }

    if (match_input_ports.size() == 0) {
      return EXTERNAL;
    }
    return match_input_ports[0]->GetName();
  }

  return output_port_a;
}

IndexPort LeastCommonAncestor::ProcessSameNode(const IndexPort &node_a,
                                               const IndexPort &node_b) {
  if (node_a.port_name == node_b.port_name) {
    return IndexPort(node_a.node_name, node_a.port_name);
  } else {
    std::string match_port_name;
    auto input_nums = all_nodes_[node_a.node_name]->GetInputNum();
    if (input_nums == 0) {
      match_port_name = EXTERNAL;
    } else {
      match_port_name =
          all_nodes_[node_a.node_name]->GetInputPorts()[0]->GetName();
    }
    return IndexPort(node_a.node_name, match_port_name);
  }
}

void LeastCommonAncestor::FindMatchNode(const IndexPort &node_a,
                                        const IndexPort &node_b,
                                        std::string &match_a_name,
                                        std::string &match_b_name,
                                        std::string &match_node_name,
                                        std::string &port_name) {
  int index_a = name_index_map_[node_a.node_name];
  int index_b = name_index_map_[node_b.node_name];
  auto path_a = paths_[index_a];
  auto path_b = paths_[index_b];
  int res = -1;
  bool swap_flag = false;
  if (path_a.size() > path_b.size()) {
    swap_flag = true;
    std::swap(path_a, path_b);
  }

  int begin_b = path_b.size() - path_a.size();
  size_t index = -1;
  for (size_t i = 0; i < path_a.size(); ++i) {
    if (path_a[i] != path_b[begin_b + i]) {
      continue;
    }

    res = path_a[i];
    index = i;
    break;
  }

  if (res == -1) {
    return;
  }

  match_node_name = index_name_map_[res];
  if (index == 0) {
    match_a_name = match_node_name;
    match_b_name = index_name_map_[path_b[begin_b - 1]];
    if (swap_flag) {
      port_name = node_b.port_name;
    } else {
      port_name = node_a.port_name;
    }
    return;
  }

  match_a_name = index_name_map_[path_a[index - 1]];
  match_b_name = index_name_map_[path_b[begin_b + index - 1]];
}

std::string LeastCommonAncestor::GetMatchPortName(
    const std::string &port_name, const std::string &match_a_name,
    const std::string &match_b_name, const std::string &match_node_name) {
  std::string match_port_name;
  auto matching_node = all_nodes_[match_node_name];
  auto input_ports = all_nodes_[match_b_name]->GetInputPorts();
  for (auto &input_port : input_ports) {
    auto connected_out_ports = input_port->GetAllOutPort();
    for (auto &connected_out_port : connected_out_ports) {
      if (connected_out_port.lock()->GetNode()->GetName() != match_node_name) {
        continue;
      }

      if (connected_out_port.lock()->GetName() == port_name) {
        match_port_name = port_name;
        return match_port_name;
      }
    }
  }

  if (matching_node->GetInputPorts().size() == 0) {
    match_port_name = EXTERNAL;
  } else {
    match_port_name = matching_node->GetInputPorts()[0]->GetName();
  }

  return match_port_name;
}

IndexPort LeastCommonAncestor::Find(const IndexPort &node_a,
                                    const IndexPort &node_b) {
  if (node_a.node_name == node_b.node_name) {
    return ProcessSameNode(node_a, node_b);
  }

  IndexPort ans;
  std::string match_node_name;
  std::string match_a_name;
  std::string match_b_name;
  std::string port_name;
  FindMatchNode(node_a, node_b, match_a_name, match_b_name, match_node_name,
                port_name);
  if (match_node_name.empty()) {
    return ans;
  }

  std::string match_port_name;
  if (match_node_name == match_a_name) {
    match_port_name = GetMatchPortName(port_name, match_a_name, match_b_name,
                                       match_node_name);
    ans = IndexPort(match_node_name, match_port_name);
    return ans;
  }

  match_port_name =
      GetMatchPortName(match_a_name, match_b_name, match_node_name);
  ans = IndexPort(match_node_name, match_port_name);
  return ans;
}

OverHierarchyCheck::OverHierarchyCheck(
    const std::unordered_map<std::string, std::shared_ptr<NodeBase>> &all_nodes,
    const std::set<std::shared_ptr<NodeBase>> &start_nodes,
    const std::map<std::string, std::string> &loop_links,
    const std::vector<std::vector<std::string>> &loop_structures,
    const std::map<std::shared_ptr<OutPort>, std::set<std::shared_ptr<InPort>>>
        &edges)
    : all_nodes_(all_nodes),
      start_nodes_(start_nodes),
      loop_links_(loop_links),
      loop_structures_(loop_structures),
      edges_(edges) {
  for (auto &all_node : all_nodes) {
    visited_[all_node.first] = false;
  }
}

OverHierarchyCheck::~OverHierarchyCheck() {
  all_nodes_.clear();
  loop_links_.clear();
  start_nodes_.clear();
  edges_.clear();
}

void OverHierarchyCheck::InitFirstNode(std::shared_ptr<Node> node) {
  auto node_name = node->GetName();
  auto index_port = std::make_shared<IndexPort>(node_name, EXTERNAL);
  color_map_[index_port->ToString()] = {0};

  for (auto &output_port : node->GetOutputPorts()) {
    auto index_port =
        std::make_shared<IndexPort>(node_name, output_port->GetName());
    color_map_[index_port->ToString()] = {0};
  }
}

bool OverHierarchyCheck::CheckEndIfPort(
    std::shared_ptr<InPort> input_port,
    const std::shared_ptr<IndexPort> &index_port,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>
        &graph_single_port_match_map) {
  auto connect_ports = input_port->GetAllOutPort();
  if (connect_ports.size() <= 1) {
    return false;
  }

  if (graph_single_port_match_map.find(index_port->node_name) ==
      graph_single_port_match_map.end()) {
    return false;
  }

  if (graph_single_port_match_map.at(index_port->node_name)
          .find(index_port->port_name) ==
      graph_single_port_match_map.at(index_port->node_name).end()) {
    return false;
  }

  return true;
}

Status OverHierarchyCheck::CheckInputPorts(
    std::shared_ptr<Node> node,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>
        &graph_single_port_match_map) {
  Status status{STATUS_OK};
  auto input_ports = node->GetInputPorts();
  std::vector<int> color;
  std::shared_ptr<IndexPort> index_port = std::make_shared<IndexPort>();
  index_port->node_name = node->GetName();
  for (auto &input_port : input_ports) {
    index_port->port_name = input_port->GetName();
    if (color_map_.find(index_port->ToString()) == color_map_.end()) {
      return STATUS_NODATA;
    }

    if (CheckEndIfPort(input_port, index_port, graph_single_port_match_map)) {
      color_map_[index_port->ToString()].pop_back();
    }

    if (color.size() == 0) {
      color = color_map_[index_port->ToString()];
      continue;
    }

    if (color != color_map_[index_port->ToString()]) {
      status = {STATUS_BADCONF,
                "node:" + node->GetName() +
                    " has different level links, pls check the input links."};
      return status;
    }
  }

  return status;
}

std::shared_ptr<NodeBase> OverHierarchyCheck::FindLoopLinkNode(
    std::shared_ptr<Node> node) {
  auto node_name = node->GetName();
  std::shared_ptr<NodeBase> res;
  for (auto &loop : loop_structures_) {
    for (size_t i = 0; i < loop.size(); ++i) {
      if (loop[i] != node_name) {
        continue;
      }

      res = all_nodes_[loop[i + 1]];
      break;
    }

    if (res != nullptr) {
      break;
    }
  }

  return res;
}

void OverHierarchyCheck::GetColorMap(
    std::shared_ptr<Node> node,
    const std::vector<std::shared_ptr<OutPort>> &output_ports,
    const std::unordered_map<std::string, std::string> &graph_match_map) {
  std::string node_name = node->GetName();
  std::vector<int> new_color;
  auto input_ports = node->GetInputPorts();
  std::shared_ptr<IndexPort> input_index_port = std::make_shared<IndexPort>();
  input_index_port->node_name = node->GetName();
  if (input_ports.size() == 0) {
    input_index_port->port_name = EXTERNAL;
  } else {
    input_index_port->port_name = input_ports[0]->GetName();
  }

  auto input_color = color_map_[input_index_port->ToString()];

  new_color.assign(input_color.begin(), input_color.end());
  if (node->GetConditionType() == ConditionType::IF_ELSE ||
      node->GetOutputType() == FlowOutputType::EXPAND) {
    ++max_color_;
    new_color.push_back(max_color_);
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  if (node->GetOutputType() == FlowOutputType::COLLAPSE) {
    new_color.pop_back();
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  if (node->GetLoopType() == LoopType::LOOP) {
    auto link_node = FindLoopLinkNode(node);
    if (link_node == node) {
      SetOutPortColor(node, output_ports, new_color);
      return;
    }

    for (auto &out_port : output_ports) {
      std::shared_ptr<IndexPort> index_output_port =
          std::make_shared<IndexPort>();
      index_output_port->node_name = node_name;
      index_output_port->port_name = out_port->GetName();
      auto inport = *out_port->GetConnectInPort().begin();
      auto link_node_name = inport->GetNode()->GetName();
      if (link_node->GetName() == link_node_name) {
        new_color.push_back(++max_color_);
        color_map_[index_output_port->ToString()] = new_color;
        new_color.pop_back();
      } else {
        color_map_[index_output_port->ToString()] = new_color;
      }
    }
    return;
  }

  auto match_node_name = graph_match_map.at(node->GetName());
  if (match_node_name == EXTERNAL) {
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  auto pre_match_real_node = CastNode(all_nodes_.at(match_node_name));
  if (pre_match_real_node == nullptr) {
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  for (auto &links : loop_links_) {
    if (links.second == node->GetName()) {
      for (auto &out_port : output_ports) {
        std::shared_ptr<IndexPort> index_output_port =
            std::make_shared<IndexPort>();
        index_output_port->node_name = node_name;
        index_output_port->port_name = out_port->GetName();
        auto inport = *out_port->GetConnectInPort().begin();
        auto link_node_name = inport->GetNode()->GetName();
        if (links.first == link_node_name) {
          auto color = new_color[new_color.size() - 1];
          new_color.pop_back();
          color_map_[index_output_port->ToString()] = new_color;
          new_color.push_back(color);
        } else {
          color_map_[index_output_port->ToString()] = new_color;
        }
      }
      return;
    }
  }

  bool single_port_has_multi_links = false;
  for (auto &input_port : input_ports) {
    auto outputs = input_port->GetAllOutPort();
    if (outputs.size() > 1) {
      single_port_has_multi_links = true;
      break;
    }
  }

  if (!single_port_has_multi_links) {
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  if (pre_match_real_node->GetConditionType() == ConditionType::IF_ELSE) {
    new_color.pop_back();
    SetOutPortColor(node, output_ports, new_color);
    return;
  }

  SetOutPortColor(node, output_ports, new_color);
}

void OverHierarchyCheck::SetOutPortColor(
    std::shared_ptr<Node> node,
    const std::vector<std::shared_ptr<OutPort>> &out_ports,
    const std::vector<int> &new_color) {
  auto node_name = node->GetName();
  for (auto &output_port : out_ports) {
    std::shared_ptr<IndexPort> index_output_port =
        std::make_shared<IndexPort>();
    index_output_port->node_name = node_name;
    index_output_port->port_name = output_port->GetName();
    color_map_[index_output_port->ToString()] = new_color;
  }
}

Status OverHierarchyCheck::Check(
    const std::unordered_map<std::string, std::string> &graph_match_map,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>
        &graph_single_port_match_map) {
  Status status{STATUS_OK};
  for (auto &start_node : start_nodes_) {
    auto real_node = CastNode(start_node);
    if (real_node == nullptr) {
      continue;
    }

    std::queue<std::shared_ptr<Node>> queue;
    queue.push(real_node);
    InitFirstNode(real_node);

    while (!queue.empty()) {
      auto node = queue.front();
      auto node_name = node->GetName();
      queue.pop();
      visited_[node_name] = true;
      status = CheckInputPorts(node, graph_single_port_match_map);
      if (status == STATUS_BADCONF) {
        status = {STATUS_BADCONF, status.WrapErrormsgs()};
        return status;
      }

      if (status == STATUS_NODATA) {
        queue.push(node);
        continue;
      }

      auto output_ports = node->GetOutputPorts();
      GetColorMap(node, output_ports, graph_match_map);

      for (auto &output_port : output_ports) {
        std::shared_ptr<IndexPort> index_output_port =
            std::make_shared<IndexPort>();
        index_output_port->node_name = node_name;
        index_output_port->port_name = output_port->GetName();
        auto input_ports = output_port->GetConnectInPort();
        for (auto &input_port : input_ports) {
          std::shared_ptr<IndexPort> index_input_port =
              std::make_shared<IndexPort>();
          auto inport_node_name = input_port->GetNode()->GetName();
          index_input_port->node_name = inport_node_name;
          index_input_port->port_name = input_port->GetName();

          if (!visited_[inport_node_name]) {
            auto connect_node = CastNode(all_nodes_[inport_node_name]);
            if (connect_node != nullptr) {
              queue.push(connect_node);
              visited_[inport_node_name] = true;
            }
          }

          if (color_map_.find(index_input_port->ToString()) ==
                  color_map_.end() ||
              color_map_[index_input_port->ToString()].empty()) {
            color_map_[index_input_port->ToString()] =
                color_map_[index_output_port->ToString()];
            continue;
          }

          if (CheckEndIfPort(input_port, index_input_port, graph_single_port_match_map)) {
            auto color_level = color_map_[index_output_port->ToString()];
            if (color_level == color_map_[index_input_port->ToString()]) {
              continue;
            }

            color_level.pop_back();
            if (color_level == color_map_[index_input_port->ToString()]) {
              continue;
            }
          }

          if (color_map_[index_input_port->ToString()] !=
              color_map_[index_output_port->ToString()]) {
            status = {STATUS_BADCONF,
                      index_output_port->node_name + ":" +
                          index_output_port->port_name + " links " +
                          index_input_port->node_name + ":" +
                          index_input_port->port_name + " failed. "};
            return status;
          }
        }
      }
    }
  }
  return status;
}

GraphChecker::GraphChecker(
    const std::vector<std::shared_ptr<NodeBase>> &nodes,
    const std::set<std::shared_ptr<NodeBase>> &start_nodes,
    const std::map<std::string, std::string> &loop_links,
    const std::vector<std::vector<std::string>> &loop_structures,
    const std::map<std::shared_ptr<OutPort>, std::set<std::shared_ptr<InPort>>>
        &edges)
    : nodes_(nodes),
      loop_links_(loop_links),
      loop_structures_(loop_structures) {
  for (auto node : nodes) {
    all_nodes_[node->GetName()] = node;
  }

  lca_ = std::make_shared<LeastCommonAncestor>(all_nodes_);
  ovc_ = std::make_shared<OverHierarchyCheck>(
      all_nodes_, start_nodes, loop_links_, loop_structures_, edges);
}

GraphChecker::~GraphChecker() {
  all_nodes_.clear();
  lca_ = nullptr;
  ovc_ = nullptr;
}

void GraphChecker::SetMatchNodes() {
  for (auto &node : all_nodes_) {
    auto real_node = CastNode(node.second);
    if (real_node == nullptr) {
      continue;
    }

    if (real_node->GetInputNum() < 1) {
      continue;
    }

    auto match_node = CastNode(all_nodes_.at(graph_match_map_[node.first]));
    if (match_node == nullptr) {
      continue;
    }

    if (real_node->GetInputNum() == 1 &&
        real_node->GetOutputType() != FlowOutputType::COLLAPSE &&
        match_node->GetConditionType() != ConditionType::IF_ELSE) {
      continue;
    }

    real_node->SetMatchNode("match_node", match_node);

    for (auto &single_node_match : graph_single_port_match_map_[node.first]) {
      auto match_condition_node =
          CastNode(all_nodes_.at(single_node_match.second));
      if (match_condition_node == nullptr) {
        modelbox::Abort("cast match condition node failed.");
      }

      real_node->SetMatchNode(single_node_match.first, match_condition_node);
    }
  }
}

void GraphChecker::ShowMatchNodes() {
  for (auto &node : all_nodes_) {
    auto real_node = CastNode(node.second);
    if (real_node == nullptr) {
      continue;
    }

    auto match_nodes = real_node->GetMatchNodes();
    for (auto &match_node : match_nodes) {
      MBLOG_INFO << "node: " << node.first << ", key: " << match_node.first
                 << ", value: " << match_node.second->GetName();
    }
  }
}

Status GraphChecker::Check() {
  for (auto &check_node : nodes_) {
    NodeStreamConnection node_stream_map;
    auto status = CalNodeStreamMap(check_node, node_stream_map);
    if (status != STATUS_SUCCESS) {
      auto msg = "caculate node stream map failed";
      MBLOG_ERROR << msg;
      return {status, msg};
    }

    auto cur_real_node = CastNode(check_node);
    // virtual node
    if (cur_real_node == nullptr) {
      continue;
    }

    auto name = cur_real_node->GetName();
    status = CheckNodeMatch(cur_real_node, node_stream_map);
    if (status != STATUS_SUCCESS) {
      auto msg = "check node " + name + " link connect failed.";
      MBLOG_ERROR << msg << ", " << status.WrapErrormsgs();
      return {status, msg};
    }

    status = CheckCollapseMatch(cur_real_node);
    if (status != STATUS_SUCCESS) {
      auto msg = "check node " + name + "branch match CollapseMatch failed";
      MBLOG_ERROR << msg << ", " << status.WrapErrormsgs();
      return {status, msg};
    }

    node_stream_connection_map_[cur_real_node->GetName()] = node_stream_map;
  }

  auto status = CheckOverHierarchyMatch();
  if (status != STATUS_SUCCESS) {
    auto msg = "check over hierarchy match failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << msg;
    return {status, msg};
  }

  return STATUS_SUCCESS;
}

Status GraphChecker::CalNodeStreamMap(std::shared_ptr<NodeBase> node,
                                      NodeStreamConnection &node_stream_map) {
  Status status{STATUS_SUCCESS};
  auto input_ports = node->GetInputPorts();

  // no input
  if (input_ports.empty()) {
    auto external = IndexPort(EXTERNAL, EXTERNAL);
    node_stream_map["p1"] = {external};
    graph_match_map_[node->GetName()] = EXTERNAL;
    return status;
  }

  for (auto &input_port : input_ports) {
    auto pre_output_ports = input_port->GetAllOutPort();
    auto key = input_port->GetName();
    for (auto &pre_output_port : pre_output_ports) {
      std::string output_port_name = pre_output_port.lock()->GetName();
      std::string pre_node_name = pre_output_port.lock()->GetNode()->GetName();
      auto value = IndexPort(pre_node_name, output_port_name);
      node_stream_map[key].emplace_back(value);
    }
  }

  if (node_stream_map.empty()) {
    status = {STATUS_BADCONF, "cal node stream connection failed."};
  }

  return status;
}

Status GraphChecker::CheckBranchPathMatch(const std::string &start,
                                          const std::string &end) {
  Status status{STATUS_SUCCESS};
  int expand_collapse_flag = 0;
  if (end == start) {
    auto end_node = CastNode(all_nodes_[end]);
    if (end_node == nullptr) {
      return status;
    }

    if (end_node->GetOutputType() == FlowOutputType::EXPAND) {
      expands_++;
    }

    return status;
  }

  std::string tmp{start};
  do {
    auto tmp_node = CastNode(all_nodes_[tmp]);

    if (tmp_node == nullptr) {
      break;
    }

    if (tmp_node->GetOutputType() == FlowOutputType::COLLAPSE) {
      expand_collapse_flag++;
    }

    if (tmp_node->GetOutputType() == FlowOutputType::EXPAND) {
      expand_collapse_flag--;
    }

    tmp = graph_match_map_[tmp];
  } while (tmp != graph_match_map_[end]);

  auto end_node = CastNode(all_nodes_[end]);
  // match node is virtual node
  if (end_node == nullptr && expand_collapse_flag != 0) {
    status = {STATUS_BADCONF, "from node:" + start + " to node:" + end +
                                  " has unmatched expand or collapse nodes"};
    return status;
  }

  // maybe the end expand node match at the checking node
  if (expand_collapse_flag == -1 &&
      end_node->GetOutputType() == FlowOutputType::EXPAND) {
    expands_++;
    return status;
  }

  // the end collapse node match at pre path.
  if (expand_collapse_flag == 1 &&
      end_node->GetOutputType() == FlowOutputType::COLLAPSE) {
    return status;
  }

  if (expand_collapse_flag != 0) {
    status = {STATUS_BADCONF, "from node:" + start + " to node:" + end +
                                  " has unmatched expand or collapse nodes"};
  }

  return status;
}

bool GraphChecker::CheckPortMatch(const IndexPort &match_pair) {
  std::shared_ptr<Node> node = CastNode(all_nodes_[match_pair.node_name]);
  if (node == nullptr) {
    if (match_pair.port_name == EXTERNAL) {
      return false;
    }

    return true;
  }

  auto port = node->GetOutputPort(match_pair.port_name);

  // input port
  if (port == nullptr) {
    return false;
  }

  // output port
  return true;
}

void GraphChecker::UpdateAncestorPath(const std::vector<IndexPort> &values) {
  lca_->Update(values, graph_match_map_);
}

Status GraphChecker::CheckUnmatchExpands(size_t size) {
  if (expands_ == 0) {
    return STATUS_OK;
  }

  if (expands_ == size) {
    expands_ = 0;
    return STATUS_OK;
  }

  expands_ = 0;
  return {STATUS_BADCONF, "unmatch expands are not the same."};
}

Status GraphChecker::CheckNodeMatch(
    std::shared_ptr<Node> node, const NodeStreamConnection &node_stream_map) {
  Status status{STATUS_SUCCESS};
  auto node_name = node->GetName();
  if (node->GetInputPorts().empty()) {
    graph_match_map_[node_name] = EXTERNAL;
    return status;
  }

  std::vector<IndexPort> single_match_result;
  std::unordered_map<std::string, std::string> single_port_match_map_;
  for (auto &iter: node_stream_map) {
    auto values = iter.second;

    // in: {d.output}
    // one input port and one edge links the port
    if (values.size() == 1) {
      single_match_result.emplace_back(values[0]);
      graph_match_map_[node_name] = values[0].node_name;
      UpdateAncestorPath(values);
      continue;
    }

    // in: {d.output, e.output}
    // one input port and multi edges link the port
    if (node->GetLoopType() == LoopType::LOOP) {
      if (values.size() != 2) {
        status = {STATUS_BADCONF, "loop node can only link 2 edges."};
        return status;
      }

      for (auto &loop_link : loop_links_) {
        if (loop_link.first != node_name) {
          continue;
        }

        for (auto &value : values) {
          if (value.node_name == loop_link.second) {
            continue;
          }

          single_match_result.emplace_back(value);
          graph_match_map_[node_name] = value.node_name;
          return status;
        }
      }
    }

    UpdateAncestorPath(values);
    std::vector<IndexPort> lca_nodes;
    auto status_single_lca =
        CheckLeastCommonAncestorsAnyTwoNodes(values, lca_nodes);
    if (status_single_lca != STATUS_SUCCESS) {
      status = {STATUS_BADCONF,
                node_name + ": " + iter.first +
                    " port match failed, err: " + status_single_lca.Errormsg()};
      return status;
    }

    IndexPort single_match_node;
    status_single_lca = LeastCommonAncestors(lca_nodes, single_match_node);
    if (status_single_lca != STATUS_SUCCESS) {
      status = {STATUS_BADCONF,
                node_name + ": " + iter.first +
                    " port match failed, err: " + status_single_lca.Errormsg()};
      return status;
    }

    // true: output port; false: input port
    // scene 2)
    if (CheckPortMatch(single_match_node)) {
      status = {STATUS_BADCONF,
                node_name + ": " + iter.first + " match at " +
                    single_match_node.node_name + ": " +
                    single_match_node.port_name +
                    ". One port links multi edges can not match at one port."};
      return status;
    }

    // scene 4)
    auto single_match_real_node =
        CastNode(all_nodes_[single_match_node.node_name]);
    if (single_match_real_node != nullptr &&
        single_match_real_node->GetConditionType() != ConditionType::IF_ELSE) {
      status = {STATUS_BADCONF,
                node_name + ": " + iter.first + " match at " +
                    single_match_node.node_name + ": " +
                    single_match_node.port_name +
                    ". One port links multi edges can not match at multi ports "
                    "when the match node is condition node."};
      return status;
    }
    single_port_match_map_[iter.first] = single_match_node.node_name;

    for (auto &value : values) {
      status =
          CheckBranchPathMatch(value.node_name, single_match_node.node_name);
      if (status != STATUS_SUCCESS) {
        return status;
      }
    }

    if (!CheckUnmatchExpands(values.size())) {
      status = {STATUS_BADCONF,
                "from " + node_name + " to " + single_match_node.node_name +
                    " path branches have unmatched expand node."};
      return status;
    }

    graph_match_map_[node_name] = single_match_real_node->GetName();
    single_match_result.emplace_back(single_match_node);
  }

  // multi branch match at one node
  if (single_match_result.size() == 1) {
    graph_match_map_[node_name] = single_match_result[0].node_name;
    return status;
  }

  UpdateAncestorPath(single_match_result);
  IndexPort multi_match_node;
  auto status_multi_lca =
      LeastCommonAncestors(single_match_result, multi_match_node);
  if (status_multi_lca != STATUS_SUCCESS) {
    status = {STATUS_BADCONF, node_name + " match failed at multi ports err: " +
                                  status_multi_lca.Errormsg()};
    return status;
  }

  auto output_port = CheckPortMatch(multi_match_node);
  auto multi_match_real_node = CastNode(all_nodes_[multi_match_node.node_name]);

  if (multi_match_real_node != nullptr) {
    if (!output_port &&
        multi_match_real_node->GetConditionType() == ConditionType::IF_ELSE) {
      auto err_msg = node_name + " match from multi ports at " +
                     multi_match_node.node_name + ":" +
                     multi_match_node.port_name + ". " +
                     multi_match_node.node_name + " can not be if-else node";
      status = {STATUS_BADCONF, err_msg};
      return status;
    }
  }

  for (auto &single_match : single_match_result) {
    status = CheckBranchPathMatch(single_match.node_name,
                                  multi_match_node.node_name);
    if (status != STATUS_SUCCESS) {
      return status;
    }
  }

  if (!CheckUnmatchExpands(single_match_result.size())) {
    status = {STATUS_BADCONF, "from " + node_name + " to " +
                                  multi_match_node.node_name +
                                  " path branches have unmatched expand node."};
    return status;
  }

  // scene 5) 6) 7) 8)
  graph_match_map_[node_name] =
      all_nodes_[multi_match_node.node_name]->GetName();
  graph_single_port_match_map_[node_name] = single_port_match_map_;
  return status;
}

Status GraphChecker::CheckOverHierarchyMatch() {
  return ovc_->Check(graph_match_map_, graph_single_port_match_map_);
}

void GraphChecker::FindNearestNeighborMatchExpand(const std::string &node,
                                                  std::string &match_node) {
  int expand_collapse_flag = 1;
  std::string tmp{node}, pre_node_name;
  std::shared_ptr<Node> pre_node;
  while (true) {
    pre_node_name = graph_match_map_[tmp];

    if (!pre_node_name.empty() && pre_node_name == EXTERNAL) {
      break;
    }

    pre_node = CastNode(all_nodes_[pre_node_name]);

    if (pre_node->GetOutputType() == FlowOutputType::COLLAPSE) {
      expand_collapse_flag++;
    }

    if (pre_node->GetOutputType() == FlowOutputType::EXPAND) {
      expand_collapse_flag--;
    }

    if (expand_collapse_flag == 0) {
      break;
    }

    tmp = pre_node_name;
  };

  if (expand_collapse_flag != 0) {
    return;
  }

  match_node = pre_node_name;
  graph_match_map_[node] = pre_node_name;
  return;
}

Status GraphChecker::CheckCollapseMatch(std::shared_ptr<Node> node) {
  Status status{STATUS_SUCCESS};
  if (node->GetInputNum() == 0) {
    return status;
  }

  if (node->GetOutputType() != FlowOutputType::COLLAPSE) {
    return status;
  }

  std::string match_node;
  FindNearestNeighborMatchExpand(node->GetName(), match_node);
  if (match_node.empty()) {
    status = {STATUS_BADCONF,
              "can't find a expand node for " + node->GetName()};
    return status;
  }

  return status;
}

std::unordered_map<std::string, std::string> GraphChecker::GetGraphMatchMap() {
  return graph_match_map_;
}

Status GraphChecker::CheckLeastCommonAncestorsAnyTwoNodes(
    const std::vector<IndexPort> &match_nodes,
    std::vector<IndexPort> &res_nodes) {
  Status status{STATUS_SUCCESS};
  for (size_t i = 0; i < match_nodes.size(); ++i) {
    auto first_node = match_nodes[i];
    for (size_t j = i + 1; j < match_nodes.size(); ++j) {
      auto second_node = match_nodes[j];
      auto res = lca_->Find(first_node, second_node);
      if (res.node_name.empty() && res.port_name.empty()) {
        std::string err_msg =
            "can not find LeastCommonAncestors node between " +
            first_node.node_name + ":" + first_node.port_name + " and " +
            second_node.node_name + ":" + second_node.port_name;
        status = {STATUS_BADCONF, err_msg};
        return status;
      }

      if (CheckPortMatch(res)) {
        status = {
            STATUS_BADCONF,
            first_node.node_name + ": " + first_node.port_name + " and " +
                second_node.node_name + ":" + second_node.port_name +
                " match at " + res.node_name + ": " + res.node_name +
                ". One port links multi edges can not match at one port."};
        return status;
      }

      if (j == i + 1) {
        res_nodes.emplace_back(res);
      }
    }
  }

  return status;
}

Status GraphChecker::LeastCommonAncestors(
    const std::vector<IndexPort> &match_nodes, IndexPort &res_match_node) {
  Status status{STATUS_OK};
  res_match_node = match_nodes[0];
  for (size_t i = 1; i < match_nodes.size(); ++i) {
    auto res = lca_->Find(res_match_node, match_nodes[i]);
    auto tmp_node = res_match_node;
    res_match_node = res;
    if (res.node_name.empty() && res.port_name.empty()) {
      std::string err_msg = "can not find LeastCommonAncestors node between " +
                            tmp_node.node_name + ":" + tmp_node.port_name +
                            " and " + match_nodes[i].node_name + ":" +
                            match_nodes[i].port_name;
      status = {STATUS_BADCONF, err_msg};
      return status;
    }
  }

  return status;
}

}  // namespace modelbox
