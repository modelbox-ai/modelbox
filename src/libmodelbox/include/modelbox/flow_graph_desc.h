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

class FlowGraphFunctionInfo {
 public:
  FlowGraphFunctionInfo(
      std::string name, std::vector<std::string> input_name_list,
      std::vector<std::string> output_name_list,
      std::function<Status(std::shared_ptr<DataContext>)> func);

  std::string GetName();

  std::vector<std::string> GetInputNameList();

  std::vector<std::string> GetOutputNameList();

  std::function<Status(std::shared_ptr<DataContext>)> GetFunc();

 private:
  std::string name_;
  std::vector<std::string> input_name_list_;
  std::vector<std::string> output_name_list_;
  std::function<Status(std::shared_ptr<DataContext>)> func_;
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

  /**
   * @brief set directory to save profile info
   * @param profile_dir directory to write profile info
   **/
  void SetProfileDir(const std::string &profile_dir);

  /**
   * @brief set profile trace on or off
   * @param profile_trace_enable true to enable profile trace
   **/
  void SetProfileTraceEnable(bool profile_trace_enable);

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

 private:
  std::unordered_map<std::string, size_t> node_name_idx_map_;
  size_t function_node_idx_{0};

  std::shared_ptr<Configuration> config_;
  std::list<std::shared_ptr<FlowGraphFunctionInfo>> function_list_;
  std::list<std::shared_ptr<FlowNodeDesc>> node_desc_list_;

  std::shared_ptr<Configuration> GetConfig();

  void GetFuncFactoryList(
      std::list<std::shared_ptr<FlowUnitFactory>> &factory_list);

  std::shared_ptr<GCGraph> GenGCGraph(
      const std::shared_ptr<modelbox::FlowUnitManager> &flowunit_mgr);

  void AddOutput(const std::string &output_name, const std::string &device,
                 const std::shared_ptr<FlowPortDesc> &source_node_port);

  Status GenGCNodes(const std::shared_ptr<GCGraph> &gcgraph);

  Status GenGCEdges(
      const std::shared_ptr<GCGraph> &gcgraph,
      const std::shared_ptr<modelbox::FlowUnitManager> &flowunit_mgr);

  Status GetInputLinks(
      const std::shared_ptr<FlowNodeDesc> &dest_node_desc,
      const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
      std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &input_links);

  Status FormatInputLinks(
      const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
      std::unordered_map<std::string, std::shared_ptr<FlowPortDesc>>
          &input_links);

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc(
      const std::shared_ptr<FlowNodeDesc> &node_desc,
      const std::shared_ptr<FlowUnitManager> &flowunit_mgr);
};

}  // namespace modelbox

#endif  // FLOW_GRAPH_DESC_H_