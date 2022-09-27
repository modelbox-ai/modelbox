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

#ifndef FLOW_NODE_DESC_H_
#define FLOW_NODE_DESC_H_
#include <utility>

#include "buffer.h"
#include "buffer_list.h"
#include "graph.h"
#include "node.h"

namespace modelbox {

constexpr const char *GRAPH_NODE_INPUT = "input";
constexpr const char *GRAPH_NODE_OUTPUT = "output";
constexpr const char *GRAPH_NODE_FLOWUNIT = "flowunit";

class FlowNodeDesc;

/**
 * @brief port of node
 **/
class FlowPortDesc {
  friend class FlowGraphDesc;

 public:
  FlowPortDesc(std::shared_ptr<FlowNodeDesc> node, std::string port_name);

  FlowPortDesc(std::shared_ptr<FlowNodeDesc> node, size_t port_idx);

  /**
   * @brief get node
   * @return node
   **/
  std::shared_ptr<FlowNodeDesc> GetNode();

  /**
   * @brief get node name
   * @return node name
   **/
  std::string GetNodeName();

  /**
   * @brief describe port in name or idx
   **/
  bool IsDescribeInName();

  /**
   * @brief get port name
   * @return port name
   **/
  std::string GetPortName();

  /**
   * @brief get port index
   * @return port index
   **/
  size_t GetPortIdx();

 private:
  std::shared_ptr<FlowNodeDesc> node_;

  bool is_in_name_;
  std::string port_name_;
  size_t port_idx_{0};
};

class FlowNodeDesc : public std::enable_shared_from_this<FlowNodeDesc> {
  friend class FlowGraphDesc;

 public:
  FlowNodeDesc(std::string node_name);

  virtual ~FlowNodeDesc();

  /**
   * @brief set custom node name to override default node name
   * @param node_name custom node name
   **/
  void SetNodeName(const std::string &node_name);

  /**
   * @brief get node name
   * @return node name
   **/
  std::string GetNodeName();

  /**
   * @brief get output port by output_name
   * @param output_name name for node output port
   **/
  std::shared_ptr<FlowPortDesc> operator[](const std::string &output_name);

  /**
   * @brief get output port by port index
   * @param port_idx index for node output port
   **/
  std::shared_ptr<FlowPortDesc> operator[](size_t port_idx);

 private:
  void SetNodeType(const std::string &type);

  void SetFlowUnitName(const std::string &flowunit_name);

  std::string GetFlowUnitName();

  void SetDevice(const std::string &device);

  void SetConfig(const std::vector<std::string> &config);

  std::vector<std::string> GetNodeConfig();

  void SetInputLinks(
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &source_node_ports);

  const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
      &GetInputLinks();

  void Clear();

  std::string node_name_;
  std::string flowunit_name_;
  std::string type_;
  std::string device_;
  std::vector<std::string> config_;
  std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
      source_node_ports_;
};

}  // namespace modelbox
#endif  // FLOW_NODE_DESC_H_