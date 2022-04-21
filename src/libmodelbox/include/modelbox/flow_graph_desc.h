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

#ifndef FLOW_GRAPH_DESC_H_
#define FLOW_GRAPH_DESC_H_

#include <queue>
#include "flow_node_desc.h"
#include "node.h"
#include "modelbox/base/error_info.h"
#include "scheduler.h"
namespace modelbox {

using ConfigNodeMap =
    std::map<std::map<std::string, std::string>, std::shared_ptr<NodeBase>>;



/**
 * @brief API mode interface
 * */
class FlowGraphDesc {
 public:
  FlowGraphDesc();
  virtual ~FlowGraphDesc();

  /**
   * @brief  init global config
   * @param config  global configuration
   * @return return result code
   */
  Status Init(std::shared_ptr<Configuration> &config);

  /**
   * @brief link a node to existing graph with name and config
   * @param name flowunit name
   * @param config flowunit config
   * @param data input data
   * @return NodeDesc bind to current node
   */
  std::shared_ptr<NodeDesc> AddNode(
      const std::string &name, std::map<std::string, std::string> config,
      const std::shared_ptr<NodeDesc> &data = nullptr);

  /**
   * @brief link a node to existing graph with name and config
   * @param name flowunit name
   * @param config flowunit config
   * @param data input data map
   * @return process result
   */
  std::shared_ptr<NodeDesc> AddNode(
      const std::string &name, std::map<std::string, std::string> config,
      const std::map<std::string, std::shared_ptr<NodeDesc>> &data);

  /**
   * @brief link a node to existing graph with name and process callback
   * @param callback process callback
   * @param inports inport names
   * @param outports outport names
   * @param data input port and node info
   * @return process result
   */
  std::shared_ptr<NodeDesc> AddNode(
      std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
      std::vector<std::string> inports, std::vector<std::string> outports,
      const std::shared_ptr<NodeDesc> &data = nullptr);

  std::shared_ptr<NodeDesc> AddNode(
      std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
      std::vector<std::string> inports, std::vector<std::string> outports,
      const std::map<std::string, std::shared_ptr<NodeDesc>> &buffers);

  /**
   * @brief bind input node to a inport of the node. port name can be ignored if
   * node has only one input port.
   * @param node NodeDesc bind for the node
   * @param port_name inport name of the node
   * @return NodeDesc bind the input node
   **/
  std::shared_ptr<NodeDesc> BindInput(
      std::shared_ptr<NodeDesc> &node,
      const std::string port_name = "__default__inport__");

  /**
   * @brief bind output node to a outport of the node.port name can be ignored
   * if node has only one out port.
   * @param node NodeDesc bind for the node
   * @param port_name outport name of the node
   * @return NodeDesc bind the output node
   **/
  std::shared_ptr<NodeDesc> BindOutput(
      std::shared_ptr<NodeDesc> &node,
      const std::string port_name = "__default__outport__");

  void Close();
  void ShutDown();
  Status Run();
  Status Wait(int64_t timeout);

  std::shared_ptr<GCGraph> GetGCGraph();
  std::shared_ptr<DeviceManager> GetDeviceManager();
  std::shared_ptr<FlowUnitManager> GetFlowUnitManager();
  std::shared_ptr<Configuration> GetConfig();
  /**
   * @brief get error info from graph
   * @return error information
   */
  std::shared_ptr<ErrorInfo> GetErrorInfo() { return error_info_; }

 private:
  void SetConfig(std::string &, std::string &);
  Status AddCallBackFactory(
      const std::string unit_name, const std::set<std::string> input_ports,
      const std::set<std::string> output_ports,
      std::function<StatusCode(std::shared_ptr<DataContext>)> &callback);
  Status AddToGCGraph(const std::string name, std::set<std::string> inputs,
                      std::set<std::string> outputs,
                      const std::map<std::string, std::string> &config,
                      const std::shared_ptr<NodeDesc> &data_handler);

  Status CheckBuffer(const std::shared_ptr<FlowUnitDesc> &desc,
                     const std::shared_ptr<NodeDesc> &data);

  std::shared_ptr<FlowUnitDesc> GetFlowunitDesc(
      const std::string &name,
      const std::map<std::string, std::string> &config);

  std::shared_ptr<GCNode> CreateGCNode(
      const std::string name, std::set<std::string> input_ports,
      std::set<std::string> out_ports,
      const std::map<std::string, std::string> &config,
      const std::shared_ptr<NodeDesc> &data_handler);

  Status InsertGraphEdge(std::shared_ptr<GCGraph> &root_graph,
                         std::shared_ptr<GCNode> &input_node,
                         std::string &input_port,
                         std::shared_ptr<GCNode> &output_node,
                         std::string &output_port);


  Status RunGraph(std::shared_ptr<NodeDesc> &data_handler);
  void BindDataHanlder(std::shared_ptr<NodeDesc> &data_handler,
                       std::shared_ptr<GCNode> &gcnode);
  Status CheckInputPort(const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
                        const std::shared_ptr<NodeDesc> &data_handler);
 
  Status CheckInputFlowUnit(const std::string &,
                            std::map<std::string, std::string> &,
                            const std::shared_ptr<NodeDesc> &,
                            const std::shared_ptr<FlowUnitDesc> &);
  std::shared_ptr<NodeDesc> ExecuteStreamNode(
      const std::shared_ptr<FlowUnitDesc> &,
      const std::shared_ptr<NodeDesc> &,
      std::map<std::string, std::string> &);



 private:
  friend class NodeDesc;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<Drivers> drivers_;
  std::shared_ptr<DeviceManager> device_mgr_;
  std::shared_ptr<FlowUnitManager> flowunit_mgr_;
  std::shared_ptr<Scheduler> scheduler_;
  std::shared_ptr<Profiler> profiler_;
  std::unordered_map<std::string, std::string> global_config_map_;
  std::set<std::shared_ptr<NodeBase>> stream_nodes_;
  std::map<std::string, ConfigNodeMap> nodes_config_;
  std::shared_ptr<ErrorInfo> error_info_;
  std::vector<std::shared_ptr<NodeDesc>> node_handlers_;
  std::shared_ptr<GCGraph> gcgraph_;

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<ExternalDataMap> externdata_map_;
  int node_sequence_{0};
  bool closed_{false};
};

}  // namespace modelbox

#endif