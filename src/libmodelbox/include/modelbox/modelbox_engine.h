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

#ifndef MODELBOX_ENGINE_H_
#define MODELBOX_ENGINE_H_

#include <queue>
#include "data_handler.h"
#include "node.h"
#include "scheduler.h"
namespace modelbox {

using ConfigNodeMap =
    std::map<std::map<std::string, std::string>, std::shared_ptr<NodeBase>>;

/**
 * @brief Job error info
 */
struct ErrorInfo {
  std::string error_code_;
  std::string error_msg_;
};

/**
 * @brief API mode interface
 * */
class ModelBoxEngine : public std::enable_shared_from_this<ModelBoxEngine> {
 public:
  ModelBoxEngine();
  virtual ~ModelBoxEngine();

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
   * @return datahandler bind to current node
   */
  std::shared_ptr<DataHandler> Execute(
      const std::string &name, std::map<std::string, std::string> config,
      const std::shared_ptr<DataHandler> &data = nullptr);

  /**
   * @brief link a node to existing graph with name and config
   * @param name flowunit name
   * @param config flowunit config
   * @param data input data map
   * @return process result
   */
  std::shared_ptr<DataHandler> Execute(
      const std::string &name, std::map<std::string, std::string> config,
      const std::map<std::string, std::shared_ptr<DataHandler>> &data);

  /**
   * @brief link a node to existing graph with name and process callback
   * @param callback process callback
   * @param inports inport names
   * @param outports outport names
   * @param data input port and node info
   * @return process result
   */
  std::shared_ptr<DataHandler> Execute(
      std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
      std::vector<std::string> inports, std::vector<std::string> outports,
      const std::shared_ptr<DataHandler> &data = nullptr);
  std::shared_ptr<DataHandler> Execute(
      std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
      std::vector<std::string> inports, std::vector<std::string> outports,
      const std::map<std::string, std::shared_ptr<DataHandler>> &buffers);
  /**
   * @brief Create external data map for flow
   * @return share pointer for external data map
   * */
  std::shared_ptr<ExternalDataMap> CreateExternalDataMap();
  /**
   * @brief bind input node to a inport of the node. port name can be ignored if
   * node has only one input port.
   * @param node datahandler bind for the node
   * @param port_name inport name of the node
   * @return datahandler bind the input node
   **/
  std::shared_ptr<DataHandler> BindInput(
      std::shared_ptr<DataHandler> &node,
      const std::string port_name = "__default__inport__");

  /**
   * @brief bind output node to a outport of the node.port name can be ignored
   * if node has only one out port.
   * @param node datahandler bind for the node
   * @param port_name outport name of the node
   * @return datahandler bind the output node
   **/
  std::shared_ptr<DataHandler> BindOutput(
      std::shared_ptr<DataHandler> &node,
      const std::string port_name = "__default__outport__");

  void Close();
  void ShutDown();
  Status Run();
  Status Wait(int64_t timeout);

  std::shared_ptr<Graph> GetGraph();

  /**
   * @brief get error info from graph
   * @return error information
   */
  std::shared_ptr<ErrorInfo> GetErrorInfo() { return error_info_; }

 private:
  void SetConfig(std::string &, std::string &);
  std::shared_ptr<Configuration> GetConfig();

  Status AddCallBackFactory(
      const std::string unit_name, const std::set<std::string> input_ports,
      const std::set<std::string> output_ports,
      std::function<StatusCode(std::shared_ptr<DataContext>)> &callback);
  Status AddToGCGraph(const std::string name, std::set<std::string> inputs,
                      std::set<std::string> outputs,
                      const std::map<std::string, std::string> &config,
                      const std::shared_ptr<DataHandler> &data_handler);

  Status CheckBuffer(const std::shared_ptr<FlowUnitDesc> &desc,
                     const std::shared_ptr<DataHandler> &data);

  std::shared_ptr<FlowUnitDesc> GetFlowunitDesc(
      const std::string &name,
      const std::map<std::string, std::string> &config);

  std::shared_ptr<GCNode> CreateGCNode(
      const std::string name, std::set<std::string> input_ports,
      std::set<std::string> out_ports,
      const std::map<std::string, std::string> &config,
      const std::shared_ptr<DataHandler> &data_handler);

  Status InsertGrahEdge(std::shared_ptr<GCGraph> &root_graph,
                        std::shared_ptr<GCNode> &input_node,
                        std::string &input_port,
                        std::shared_ptr<GCNode> &output_node,
                        std::string &output_port);
  std::shared_ptr<NodeBase> CheckNodeExist(
      const std::string &name,
      const std::map<std::string, std::string> &config);
  std::shared_ptr<DeviceManager> GetDeviceManager();
  std::shared_ptr<FlowUnitManager> GetFlowUnitManager();
  std::shared_ptr<Scheduler> GetScheduler();
  std::shared_ptr<Profiler> GetProfiler();
  Status RunGraph(std::shared_ptr<DataHandler> &data_handler);
  void BindDataHanlder(std::shared_ptr<DataHandler> &data_handler,
                       std::shared_ptr<GCNode> &gcnode);
  Status CheckInputPort(const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
                        const std::shared_ptr<DataHandler> &data_handler);
  bool CheckisStream(const std::shared_ptr<FlowUnitDesc> &desc,
                     const std::shared_ptr<DataHandler> &data_handler);
  Status CheckInputFlowUnit(const std::string &,
                            std::map<std::string, std::string> &,
                            const std::shared_ptr<DataHandler> &,
                            const std::shared_ptr<FlowUnitDesc> &);
  std::shared_ptr<DataHandler> ExecuteStreamNode(
      const std::shared_ptr<FlowUnitDesc> &,
      const std::shared_ptr<DataHandler> &,
      std::map<std::string, std::string> &);

  std::shared_ptr<GCNode> ProcessOutputHandler(
      const std::shared_ptr<DataHandler> &, std::shared_ptr<GCNode> &,
      std::shared_ptr<GCGraph> &);
  std::shared_ptr<GCNode> ProcessVirtualHandler(std::shared_ptr<GCNode> &,
                                                std::shared_ptr<GCGraph> &);

 private:
  friend class DataHandler;
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
  std::vector<std::shared_ptr<DataHandler>> node_handlers_;
  std::shared_ptr<GCGraph> gcgraph_;

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<ExternalDataMap> externdata_map_;
  int node_sequence_{0};
  bool closed_{false};
};

}  // namespace modelbox

#endif