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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flow_node_desc.h"
#include "modelbox/base/configuration.h"
#include "modelbox/base/graph_manager.h"

namespace modelbox {

/**
 * @brief Flow configuration
 **/
class FlowConfig {
  friend class FlowGraphDesc;

 public:
  FlowConfig();

  /**
   * @brief set graph scope queue size
   * @param queue_size for node input cache
   */
  void SetQueueSize(size_t queue_size);

  /**
   * @brief set graph scope batch size
   * @param batch_size for node process batch
   **/
  void SetBatchSize(size_t batch_size);

  /**
   * @brief set custom drivers scan directory
   * @param drivers_dir_list Dir list to scan custom drivers
   **/
  void SetDriversDir(const std::vector<std::string> &drivers_dir_list);

  /**
   * @brief skip modelbox default drivers
   * @param is_skip True if skip modelbox default drivers
   **/
  void SetSkipDefaultDrivers(bool is_skip);

 private:
  std::shared_ptr<Configuration> content_;
};

/**
 * @brief To describe a graph in api mode
 **/
class FlowGraphDesc {
  friend class Flow;

 public:
  FlowGraphDesc();

  virtual ~FlowGraphDesc();

  /**
   * @brief call first, will load drivers to complete graph
   * @return Status
   **/
  Status Init();

  /**
   * @brief call first, will load drivers to complete graph
   * @param config for flow init
   * @return Status
   **/
  Status Init(const std::shared_ptr<FlowConfig> &config);

  /**
   * @brief add input port for flow
   * @param input_name input port name
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddInput(const std::string &input_name);

  /**
   * @brief add output port for flow
   * @param output_name output port name
   * @param source_node_port node output port connect to this output port
   **/
  void AddOutput(const std::string &output_name,
                 const std::shared_ptr<FlowPortDesc> &source_node_port);

  /**
   * @brief add output port for flow
   * @param output_name output port name
   * @param source_node output port [0] of node will connect to this output port
   **/
  void AddOutput(const std::string &output_name,
                 const std::shared_ptr<FlowNodeDesc> &source_node);

  /**
   * @brief add node for flow
   * @param flowunit_name flowunit name, like resize, crop
   * @param device choose flowunit implementation
   * @param config flowunit configuration
   * @param source_node_ports node output ports connect to this node input ports
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddNode(
      const std::string &flowunit_name, const std::string &device,
      const std::vector<std::string> &config,
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &source_node_ports);

  /**
   * @brief add node for flow
   * @param flowunit_name flowunit name, like resize, crop
   * @param device choose flowunit implementation
   * @param config flowunit configuration
   * @param source_node output port [0] of node will connect to this output port
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddNode(
      const std::string &flowunit_name, const std::string &device,
      const std::vector<std::string> &config,
      const std::shared_ptr<FlowNodeDesc> &source_node);

  /**
   * @brief add node for flow
   * @param flowunit_name flowunit name, like resize, crop
   * @param device choose flowunit implementation
   * @param source_node_ports node output ports connect to this node input ports
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddNode(
      const std::string &flowunit_name, const std::string &device,
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &source_node_ports);

  /**
   * @brief add node for flow
   * @param flowunit_name flowunit name, like resize, crop
   * @param device choose flowunit implementation
   * @param source_node output port [0] of node will connect to this output port
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddNode(
      const std::string &flowunit_name, const std::string &device,
      const std::shared_ptr<FlowNodeDesc> &source_node);

  /**
   * @brief add node for flow
   * @param flowunit_name flowunit name, like resize, crop
   * @param device choose flowunit implementation
   * @param config flowunit configuration
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddNode(
      const std::string &flowunit_name, const std::string &device = "cpu",
      const std::vector<std::string> &config = {});

  /**
   * @brief add function node for flow
   * @param func func to insert as node
   * @param input_name_list define input port for node
   * @param output_name_list define output port for node
   * @param source_node_ports node output ports connect to this node input ports
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddFunction(
      const std::function<Status(std::shared_ptr<DataContext>)> &func,
      const std::vector<std::string> &input_name_list,
      const std::vector<std::string> &output_name_list,
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &source_node_ports);

  /**
   * @brief add function node for flow
   * @param func func to insert as node
   * @param input_name_list define input port for node
   * @param output_name_list define output port for node
   * @param source_node output port [0] of node will connect to this output port
   * @return a node in graph
   **/
  std::shared_ptr<FlowNodeDesc> AddFunction(
      const std::function<Status(std::shared_ptr<DataContext>)> &func,
      const std::vector<std::string> &input_name_list,
      const std::vector<std::string> &output_name_list,
      const std::shared_ptr<FlowNodeDesc> &source_node);

  /**
   * @brief get graph build status
   * @return Status of graph build
   **/
  Status GetStatus();

 private:
  bool is_init_{false};
  std::unordered_map<std::string, size_t> node_name_idx_map_;
  size_t function_node_idx_{0};
  std::unordered_map<std::string, size_t> model_node_idx_map_;
  std::list<std::shared_ptr<FlowNodeDesc>> node_desc_list_;
  Status build_status_{STATUS_FAULT};

  std::shared_ptr<Configuration> GetConfig();

  std::shared_ptr<GCGraph> GetGCGraph();

  std::shared_ptr<Drivers> GetDrivers();

  std::shared_ptr<DeviceManager> GetDeviceManager();

  std::shared_ptr<FlowUnitManager> GetFlowUnitManager();

  std::shared_ptr<Configuration> config_;
  std::shared_ptr<Drivers> drivers_;
  std::shared_ptr<DeviceManager> device_mgr_;
  std::shared_ptr<FlowUnitManager> flowunit_mgr_;

  void AddOutput(const std::string &output_name, const std::string &device,
                 const std::shared_ptr<FlowPortDesc> &source_node_port);

  bool FormatInputLinks(
      const std::string &flowunit_name,
      const std::shared_ptr<FlowUnitDesc> &flowunit_desc,
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &origin_source_node_ports,
      std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &format_source_node_ports);

  void GenGCNodes(const std::shared_ptr<GCGraph> &gcgraph);

  void GenGCEdges(const std::shared_ptr<GCGraph> &gcgraph);

  bool CheckInputLinks(
      const std::vector<std::string> &defined_ports,
      const std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &input_links);
};

}  // namespace modelbox

#endif  // FLOW_GRAPH_DESC_H_