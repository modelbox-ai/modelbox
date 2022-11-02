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

#ifndef MODELBOX_NODE_H_
#define MODELBOX_NODE_H_

#include <list>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

#include "modelbox/base/status.h"
#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_group.h"
#include "modelbox/inner_event.h"
#include "modelbox/match_stream.h"
#include "modelbox/profiler.h"
#include "modelbox/statistics.h"

namespace modelbox {

using PortsDataList =
    std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>;

enum RunType {
  DATA = 0,
  EVENT = 1,
};

class EventPort;
class InPort;
class OutPort;

class NodeBase : public std::enable_shared_from_this<NodeBase> {
 public:
  NodeBase();
  virtual ~NodeBase();

  virtual Status Init(const std::set<std::string>& input_port_names,
                      const std::set<std::string>& output_port_names,
                      const std::shared_ptr<Configuration>& config);

  virtual Status Run(RunType type) = 0;

  virtual Status Open() = 0;

  virtual std::shared_ptr<Device> GetDevice() = 0;

  virtual void Close();

  void SetName(const std::string& name);

  std::string GetName() const;

  void SetPriority(int32_t priortity);

  int32_t GetPriority() const;

  void SetQueueSize(int32_t queue_size);

  int32_t GetQueueSize() const;

  size_t GetInputNum();

  size_t GetExternNum();

  size_t GetOutputNum();

  std::set<std::string> GetInputNames();

  std::set<std::string> GetExternNames();

  std::set<std::string> GetOutputNames();

  std::vector<std::shared_ptr<InPort>> GetInputPorts() const;

  std::vector<std::shared_ptr<OutPort>> GetOutputPorts() const;

  std::vector<std::shared_ptr<InPort>> GetExternalPorts() const;

  std::shared_ptr<InPort> GetInputPort(const std::string& port_name);

  std::shared_ptr<InPort> GetExternalPort(const std::string& port_name);

  std::shared_ptr<OutPort> GetOutputPort(const std::string& port_name);

  std::shared_ptr<EventPort> GetEventPort();

  void SetAllInportActivated(bool flag);

  Status SendBatchEvent(
      std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list,
      bool update_active_time = true);

  Status SendEvent(std::shared_ptr<FlowUnitInnerEvent>& event,
                   bool update_active_time = true);

  void Shutdown();

 protected:
  Status InitPorts(const std::set<std::string>& input_port_names,
                   const std::set<std::string>& output_port_names,
                   const std::shared_ptr<Configuration>& config);

  std::string name_;

  std::shared_ptr<Configuration> config_;

  std::vector<std::shared_ptr<InPort>> input_ports_;

  std::shared_ptr<EventPort> event_port_;

  std::vector<std::shared_ptr<InPort>> extern_ports_;

  std::vector<std::shared_ptr<OutPort>> output_ports_;

  int32_t priority_{0};

  size_t queue_size_{0};

  size_t event_queue_size_{0};
};

class SessionManager;

class Node : public NodeBase {
 public:
  Node();

  ~Node() override;

  /**
   * @brief Init the node
   *
   * @param input_port_names {set} the input port name
   * @param output_port_names {set} the output port name
   * @param config node configuration
   * @return Status {status} if success return STATUS_SUCCESS
   */
  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              const std::shared_ptr<Configuration>& config) override;

  void SetFlowUnitInfo(const std::string& flowunit_name,
                       const std::string& flowunit_type,
                       const std::string& flowunit_device_id,
                       std::shared_ptr<FlowUnitManager> flowunit_manager);

  std::shared_ptr<FlowUnitGroup> GetFlowUnitGroup();

  void SetProfiler(std::shared_ptr<Profiler> profiler);

  void SetStats(std::shared_ptr<StatisticsItem> graph_stats);

  /**
   * @brief Open node
   * @return open result
   */
  Status Open() override;

  std::shared_ptr<Device> GetDevice() override { return nullptr; };

  /**
   * @brief close node
   */
  void Close() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  void SetOutputType(FlowOutputType type);

  void SetFlowType(FlowType type);

  void SetConditionType(ConditionType type);

  void SetLoopType(LoopType type);

  void SetInputContiguous(bool is_input_contiguous);

  void SetExceptionVisible(bool is_exception_visible);

  FlowOutputType GetOutputType();

  FlowType GetFlowType();

  ConditionType GetConditionType();

  LoopType GetLoopType();

  bool IsInputContiguous();

  bool IsExceptionVisible();

  std::unordered_map<std::string, std::shared_ptr<Node>> GetMatchNodes();

  void SetMatchNode(const std::string& name, std::shared_ptr<Node> match_node);

  std::shared_ptr<Node> GetMatchNode();

  std::shared_ptr<Node> GetMatchNode(const std::string& port_name);

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc();

  void SetSessionManager(SessionManager* session_mgr);

  void SetLoopOutPortName(const std::string& port_name);

  std::string GetLoopOutPortName();

 protected:
  virtual Status Recv(
      RunType type,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  virtual Status GenInputMatchStreamData(
      RunType type,
      std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list);

  Status GenMatchStreamFromDataPorts(
      std::vector<std::shared_ptr<InPort>>& data_ports,
      std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list);

  Status GenMatchStreamFromEventPorts(
      std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list);

  virtual Status GenDataContextList(
      std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status AppendDataContextByEvent(
      const std::shared_ptr<MatchStreamData>& match_stream_data,
      std::set<std::shared_ptr<FlowUnitDataContext>>& data_ctx_set);

  Status AppendDataContextByData(
      const std::shared_ptr<MatchStreamData>& match_stream_data,
      std::set<std::shared_ptr<FlowUnitDataContext>>& data_ctx_set);

  std::shared_ptr<FlowUnitDataContext> GetDataContext(MatchKey* key);

  std::shared_ptr<FlowUnitDataContext> CreateDataContext(
      MatchKey* key, const std::shared_ptr<Session>& session);

  std::shared_ptr<FlowUnitDataContext> AppendDataToDataContext(
      MatchKey* key, const std::shared_ptr<MatchStreamData>& match_stream_data,
      bool append_single_buffer = false, size_t buffer_index = 0);

  Status Process(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status Send(std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void SetLastError(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void Clean(std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void CleanDataContext();

  virtual Status InitNodeProperties();

  void UpdatePropConstrain(const std::shared_ptr<FlowUnitDesc>& flowunit_desc);

  std::shared_ptr<ExternalData> CreateExternalData(
      const std::shared_ptr<Device>& device);

  bool NeedNewIndex();

  std::unordered_map<std::string, size_t> GetStreamCountEachPort();

  std::shared_ptr<FlowUnitManager> flowunit_manager_;
  std::shared_ptr<FlowUnitGroup> flowunit_group_;
  bool is_flowunit_opened_{false};
  std::string flowunit_name_;
  std::string flowunit_type_;
  std::string flowunit_device_id_;

  FlowOutputType output_type_{FlowOutputType::ORIGIN};
  FlowType flow_type_{FlowType::STREAM};
  ConditionType condition_type_{ConditionType::NONE};
  LoopType loop_type_{LoopType::NOT_LOOP};
  bool is_input_contiguous_{false};
  bool is_exception_visible_{false};

  std::shared_ptr<Profiler> profiler_;
  std::shared_ptr<StatisticsItem> graph_stats_;
  SessionManager* session_mgr_{nullptr};

  std::unordered_map<std::string, std::shared_ptr<Node>> port_match_at_node_;
  std::once_flag input_stream_count_update_flag_;
  std::shared_ptr<InputMatchStreamManager> input_match_stream_mgr_;
  std::unordered_map<MatchKey*, std::shared_ptr<FlowUnitDataContext>>
      data_ctx_map_;
  std::shared_ptr<OutputMatchStreamManager> output_match_stream_mgr_;

  std::unordered_map<std::string, std::shared_ptr<Node>> match_node_;
  std::string loop_out_port_name_;
};

}  // namespace modelbox

#endif  // MODELBOX_NODE_H_