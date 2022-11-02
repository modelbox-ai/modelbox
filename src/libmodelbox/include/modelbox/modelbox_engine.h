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

#include "data_handler.h"
#include "modelbox/base/error_info.h"
#include "node.h"
#include "scheduler.h"

namespace modelbox {
using ConfigNodeMap =
    std::map<std::map<std::string, std::string>, std::shared_ptr<NodeBase>>;

/**
 * @brief dynamic graph manager,graph start ,run and stop
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
   * @brief create input stream for external input
   * @return datahandler bind to extern input
   */
  std::shared_ptr<DataHandler> CreateInput(
      const std::set<std::string> &port_map);

  /**
   * @brief  choose right node to create graph and run graph
   * @param name flowunit name
   * @param config_map flowunit config
   * @param data input data
   * @return process result
   */
  std::shared_ptr<DataHandler> Execute(
      const std::string &name, std::map<std::string, std::string> config_map,
      const std::shared_ptr<DataHandler> &data = nullptr);
  /**
   * @brief choose right node to create graph and run graph
   * @param name flowunit name
   * @param config_map flowunit config
   * @param data input data map
   * @return process result
   */
  std::shared_ptr<DataHandler> Execute(
      const std::string &name, std::map<std::string, std::string> config_map,
      const std::map<std::string, std::shared_ptr<DataHandler>> &data);

  /**
   * @brief close the graph
   */
  void Close();
  void ShutDown();

  /**
   * @brief get error info from graph
   * @return error information
   */
  std::shared_ptr<ErrorInfo> GetErrorInfo();

  void SetConfig(std::string &, std::string &);
  std::shared_ptr<Configuration> GetConfig();

  std::shared_ptr<GCNode> CreateDynamicGCGraph(
      const std::string &name, const std::map<std::string, std::string> &config,
      const std::shared_ptr<DataHandler> &data_handler);

  Status CheckBuffer(const std::shared_ptr<FlowUnitDesc> &desc,
                     const std::shared_ptr<DataHandler> &data);

  std::shared_ptr<FlowUnitDesc> GetFlowunitDesc(
      const std::string &name,
      const std::map<std::string, std::string> &config);

  std::shared_ptr<GCNode> CreateDynamicStreamNode(
      const std::string &name, const std::map<std::string, std::string> &config,
      const std::shared_ptr<DataHandler> &data_handler);

  Status InsertGrahEdge(std::shared_ptr<GCGraph> &root_graph,
                        std::shared_ptr<GCNode> &input_node,
                        std::string &input_port,
                        std::shared_ptr<GCNode> &output_node,
                        std::string &output_port);
  std::shared_ptr<NodeBase> CheckNodeExist(
      const std::string &name,
      const std::map<std::string, std::string> &config);

  std::shared_ptr<NodeBase> CreateDynamicNormalNode(
      const std::string &name,
      const std::map<std::string, std::string> &config_map);

  /*
   feed data to graph
   */
  Status FeedData(std::shared_ptr<DynamicGraph> &dynamic_graph,
                  std::shared_ptr<GCGraph> &gcgraph);
  /*
  create a graph for gcgraph
  */
  std::shared_ptr<DynamicGraph> CreateDynamicGraph(
      std::shared_ptr<GCGraph> &graph);
  std::shared_ptr<DeviceManager> GetDeviceManager();
  std::shared_ptr<FlowUnitManager> GetFlowUnitManager();
  std::shared_ptr<Scheduler> GetScheduler();
  std::shared_ptr<Profiler> GetProfiler();
  Status RunGraph(std::shared_ptr<DataHandler> &data_handler);
  std::shared_ptr<DataHandler> BindDataHanlder(
      std::shared_ptr<DataHandler> &data_handler,
      std::shared_ptr<GCNode> &gcnode);
  Status CheckInputPort(const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
                        const std::shared_ptr<DataHandler> &data_handler);
  bool CheckisStream(const std::shared_ptr<FlowUnitDesc> &desc,
                     const std::shared_ptr<DataHandler> &data_handler);
  Status CheckInputFlowUnit(const std::string &name,
                            std::map<std::string, std::string> &config_map,
                            const std::shared_ptr<DataHandler> &buffers,
                            const std::shared_ptr<FlowUnitDesc> &desc);
  std::shared_ptr<DataHandler> ExecuteStreamNode(
      const std::shared_ptr<FlowUnitDesc> &desc,
      const std::shared_ptr<DataHandler> &buffers,
      std::map<std::string, std::string> &config_map);
  std::shared_ptr<DataHandler> ExecuteBufferListNode(
      const std::string &name, std::map<std::string, std::string> &config_map,
      const std::shared_ptr<DataHandler> &buffers);
  Status SendExternalData(std::shared_ptr<ExternalDataMap> &extern_datamap,
                          std::shared_ptr<BufferList> &buffer_list,
                          const std::shared_ptr<GCNode> &gcnode);
  std::shared_ptr<GCNode> ProcessOutputHandler(
      const std::shared_ptr<DataHandler> &data_handler,
      std::shared_ptr<GCNode> &gcnode, std::shared_ptr<GCGraph> &root_graph);
  std::shared_ptr<GCNode> ProcessVirtualHandler(
      std::shared_ptr<GCNode> &gcnode, std::shared_ptr<GCGraph> &root_graph);

 private:
  friend class DataHandler;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<Drivers> drivers_;
  std::shared_ptr<DeviceManager> device_mgr_;
  std::shared_ptr<FlowUnitManager> flowunit_mgr_;
  std::shared_ptr<Scheduler> scheduler_;
  std::shared_ptr<Profiler> profiler_;
  std::set<std::shared_ptr<Graph>> graphs_;
  std::unordered_map<std::string, std::string> global_config_map_;
  std::set<std::shared_ptr<NodeBase>> stream_nodes_;
  std::map<std::string, ConfigNodeMap> nodes_config_;
  std::shared_ptr<ErrorInfo> error_info_;
};

}  // namespace modelbox

#endif