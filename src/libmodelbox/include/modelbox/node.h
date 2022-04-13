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

#include "modelbox/flowunit_group.h"
#include "modelbox/match_buffer.h"
#include "modelbox/output_buffer.h"
#include "modelbox/port.h"
#include "modelbox/profiler.h"
#include "modelbox/statistics.h"

namespace modelbox {
class SchedulerEvent;
class InPort;
class EventPort;
class ExternPort;
class OutPort;

struct CmpByInputKey {
  bool operator()(const std::shared_ptr<BufferGroup>& k1,
                  const std::shared_ptr<BufferGroup>& k2) {
    if (k1->GetOrder() == k2->GetOrder()) {
      return k1.get() < k2.get();
    }
    return k1->GetOrder() < k2->GetOrder();
  }
};

using SingleMatch = std::map<std::shared_ptr<BufferGroup>,
                             std::shared_ptr<MatchBuffer>, CmpByInputKey>;
using StreamMatch = std::map<std::tuple<std::shared_ptr<BufferGroup>, uint32_t>,
                             std::vector<std::shared_ptr<MatchBuffer>>>;

using ErrorMatch = std::map<std::tuple<std::shared_ptr<BufferGroup>, uint32_t>,
                            std::shared_ptr<FlowUnitError>>;
using StreamOrder =
    std::map<std::tuple<std::shared_ptr<BufferGroup>, uint32_t>, uint32_t>;

enum RunType {
  DATA = 0,
  EVENT = 1,
};

class StreamMatchCache {
 public:
  StreamMatchCache();
  virtual ~StreamMatchCache();
  std::shared_ptr<StreamMatch> GetStreamReceiveBuffer();
  std::shared_ptr<StreamOrder> GetStreamOrder();

 private:
  std::shared_ptr<StreamMatch> stream_match_buffer_;
  std::shared_ptr<StreamOrder> stream_order_;
};

enum UpdateType {
  UNDEFINED = 0,
  CONSTANT = 1,
  REDUCE = 2,
  ENLARGE = 3,
};

class NodeBase;
class SingleMatchCache {
 public:
  SingleMatchCache(std::shared_ptr<NodeBase> node);
  virtual ~SingleMatchCache() = default;
  uint32_t GetLeftBufferSize(std::string port);
  Status LoadCache(std::string port,
                   std::vector<std::shared_ptr<IndexBuffer>>& buffer_vector);
  void UnloadCache(std::shared_ptr<StreamMatchCache> stream_match_cache,
                   bool is_input_order);

  bool IsFull();

  UpdateType GetUpdataType();

  void EnlargeBufferCache();

  void ReduceBufferCache();

  std::shared_ptr<SingleMatch> GetReceiveBuffer();

  int GetLimitCount();

 private:
  std::shared_ptr<SingleMatch> single_match_buffer_;
  std::weak_ptr<NodeBase> node_;
  int origin_limit_counts_;
  int limit_counts_;
};

class NodeBase : public std::enable_shared_from_this<NodeBase> {
 public:
  virtual ~NodeBase() = default;

  virtual Status Init(const std::set<std::string>& input_port_names,
                      const std::set<std::string>& output_port_names,
                      std::shared_ptr<Configuration> config);

  virtual Status Run(RunType type) = 0;

  virtual void SetError(Status status) { status_ = status; }

  virtual Status GetError() { return status_; }

  /**
   * @brief Open node
   * @return open result
   */
  virtual Status Open() = 0;

  /**
   * @brief Close node
   * @return close result
   */
  virtual void Close();

  /**
   * @brief Shutdown node
   */
  virtual void Shutdown();

  /**
   * @brief Get the Node Device
   *
   * @return the device std::shared_ptr<Device>
   */
  virtual std::shared_ptr<Device> GetDevice() = 0;

  /**
   * @brief Set the Name object
   *
   * @return std::string
   */
  void SetName(const std::string& name) { name_ = name; }

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string GetName() const { return name_; }

  /**
   * @brief Set the Priority
   *
   * @param priortity
   */
  void SetPriority(int32_t priortity) { priority_ = priortity; }

  /**
   * @brief Get the Priority
   *
   * @return int32_t
   */

  int32_t GetPriority() const { return priority_; }

  void SetQueueSize(int32_t queue_size) { queue_size_ = queue_size; }

  int32_t GetQueueSize() const { return queue_size_; }

  Status SendBatchEvent(
      std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list,
      bool update_active_time = true);

  Status SendEvent(std::shared_ptr<FlowUnitInnerEvent>& event,
                   bool update_active_time = true);

  /**
   * @brief Get the Input Port Num
   *
   * @return uint32_t
   */
  uint32_t GetInputNum();

  /**
   * @brief Get the Output Port object
   *
   * @return uint32_t
   */
  uint32_t GetOutputNum();

  /**
   * @brief Get the Input Keys
   *
   * @return std::vector<std::string>
   */
  std::set<std::string> GetInputNames();

  std::set<std::string> GetExternNames();

  std::set<std::string> GetOutputNames();

  std::vector<std::shared_ptr<InPort>> GetInputPorts() const;

  std::vector<std::shared_ptr<OutPort>> GetOutputPorts() const;

  const std::vector<std::shared_ptr<InPort>>& GetExternalPorts() const;

  std::shared_ptr<InPort> GetInputPort(const std::string& port_name);

  std::shared_ptr<InPort> GetExternalPort(const std::string& port_name);

  std::shared_ptr<OutPort> GetOutputPort(const std::string& port_name);

  /**
   * @brief Get the event Port object
   *
   * @return std::shared_ptr<EventPort>
   */
  inline std::shared_ptr<EventPort> GetEventPort() { return event_port_; }

  void SetAllInportActivated(bool flag);

  /**
   * @brief Create a Output Buffer
   *
   * @return Output IndexBuffer
   */
  OutputIndexBuffer CreateOutputBuffer();

  /**
   * @brief Create the Input Buffer
   *
   * @return Input IndexBuffer
   */
  InputIndexBuffer CreateInputBuffer();

 protected:
  /**
   * @brief Send the output
   *
   * @param output_buffer
   * @return Status {status} if success return STATUS_SUCCESS
   */
  Status Send(OutputIndexBuffer* output_buffer);

  std::string name_;

  Status status_{STATUS_OK};

  std::shared_ptr<Configuration> config_;

  std::vector<std::shared_ptr<InPort>> input_ports_;

  std::shared_ptr<EventPort> event_port_;

  std::vector<std::shared_ptr<InPort>> extern_ports_;

  std::vector<std::shared_ptr<OutPort>> output_ports_;

  int32_t priority_{0};
  int32_t queue_size_{0};
};

class DataMatcherNode : public NodeBase {
 public:
  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              std::shared_ptr<Configuration> config) override;

  Status RecvDataQueue(InputIndexBuffer* input_buffer);

  Status RecvExternalDataQueue(InputIndexBuffer* input_buffer);

  std::shared_ptr<SingleMatchCache> GetSingleMatchCache();

  std::shared_ptr<StreamMatchCache> GetStreamMatchCache();

  void SetInputGatherAll(bool need_garther_all);

  void SetInputOrder(bool is_input_order);

  bool IsInputOrder();

  Status ReceiveGroupBuffer();

  Status ReceiveBuffer(std::shared_ptr<InPort>& input_port);

 protected:
  Status GenerateFromStreamPool(InputIndexBuffer* single_map);
  Status FillOneMatchVector(
      InputIndexBuffer* single_map,
      std::tuple<std::shared_ptr<modelbox::BufferGroup>, uint32_t> order_key,
      std::vector<std::shared_ptr<MatchBuffer>>& match_vector);

  Status FillVectorMap(
      std::tuple<std::shared_ptr<modelbox::BufferGroup>, uint32_t> order_key,
      std::vector<std::shared_ptr<MatchBuffer>>& match_vector,
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<IndexBuffer>>>&
          group_buffer_vector_map);

  bool need_garther_all_;
  bool is_input_order_;
  std::shared_ptr<SingleMatchCache> single_match_cache_;
  std::shared_ptr<StreamMatchCache> stream_match_cache_;
};

class ExternalDataImpl;
class Node : public DataMatcherNode {
 public:
  /**
   * @brief Construct a new Node object
   *
   * @param unit_name flowunit name
   * @param unit_type flowunit type
   * @param unit_device_id flowunit device id
   * @param flowunit_mgr flowunit manager
   * @param profiler profiler pointer
   * @param graph_stats stat pointer
   */
  Node(const std::string& unit_name, const std::string& unit_type,
       const std::string& unit_device_id,
       std::shared_ptr<FlowUnitManager> flowunit_mgr,
       std::shared_ptr<Profiler> profiler,
       std::shared_ptr<StatisticsItem> graph_stats = nullptr);

  virtual ~Node() override;

  /**
   * @brief Init the node
   *
   * @param input_port_names {set} the input port name
   * @param output_port_names {set} the output port name
   * @param config node configuration
   * @return Status {status} if success return STATUS_SUCCESS
   */
  virtual Status Init(const std::set<std::string>& input_port_names,
                      const std::set<std::string>& output_port_names,
                      std::shared_ptr<Configuration> config) override;

  Status RecvData(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status RecvEvent(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status RecvExternalData(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  std::shared_ptr<FlowUnitDataContext> GenDataContext(
      std::shared_ptr<BufferGroup> stream_info);

  Status FillDataContext(
      std::vector<std::shared_ptr<InputData>>& input_data_list,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status GenerateDataContext(
      InputIndexBuffer& input_buffer,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void SetOutputType(FlowOutputType type);

  void SetFlowType(FlowType type);

  void SetConditionType(ConditionType type);

  void SetLoopType(LoopType type);

  void SetStreamSameCount(bool is_stream_same_count);

  void SetInputContiguous(bool is_input_contiguous);

  FlowOutputType GetOutputType();

  FlowType GetFlowType();

  ConditionType GetConditionType();

  LoopType GetLoopType();
  bool IsStreamSameCount();

  bool IsInputContiguous();

  /**
   * @brief Open node
   * @return open result
   */
  Status Open() override;

  /**
   * @brief close node
   */
  virtual void Close() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  virtual Status Run(RunType type) override;

  /**
   * @brief Send the output
   *
   * @param output_buffer
   * @return Status {status} if success return STATUS_SUCCESS
   */
  Status Send(OutputIndexBuffer* output_buffer);

  std::shared_ptr<Device> GetDevice() override;

  void SetCondition(bool is_condition);

  void SetExceptionVisible(bool is_exception_visible_);
  bool IsExceptionVisible();

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc();

  Status GenerateOutputIndexBuffer(
      OutputIndexBuffer* output_map_index_buffer,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void ClearDataContext(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  InputIndexBuffer CreateExternalBuffer();

  std::shared_ptr<FlowUnitGroup> GetFlowUnitGroup() { return flowunit_group_; }

  std::shared_ptr<Configuration> GetConfiguration() { return config_; }

  std::shared_ptr<Profiler> GetProfiler() { return profiler_; };

  std::unordered_map<std::string, std::shared_ptr<Node>> GetMatchNodes() {
    return match_node_;
  }
  
  void SetMatchNode(const std::string& name, std::shared_ptr<Node> match_node) {
    match_node_[name] = match_node;
  }

  std::shared_ptr<Node> GetMatchNode() { return match_node_["match_node"]; }

  std::shared_ptr<Node> GetMatchNode(const std::string& port_name) {
    return match_node_[port_name];
  }

 private:
  friend class InPort;
  friend class SingleNode;

  void InitNodeWithFlowunit();

  Status Recv(RunType type,
              std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  Status GenDataContextFromEvent(
      std::shared_ptr<modelbox::FlowUnitInnerEvent> event,
      std::shared_ptr<FlowUnitDataContext>& data_ctx);

  // If exception can be see by the node.if not true,the node can not see the
  // exception
  bool is_exception_visible_;

  // The Node FlowUnitGroup
  std::shared_ptr<FlowUnitGroup> flowunit_group_;

  bool is_fug_opened_;

  std::string unit_name_;

  std::string unit_type_;

  std::string unit_device_id_;

  std::shared_ptr<Profiler> profiler_;

  std::shared_ptr<StatisticsItem> graph_stats_;

  std::shared_ptr<FlowUnitManager> flowunit_mgr_;

  FlowOutputType output_type_;

  FlowType flow_type_;

  ConditionType condition_type_;

  LoopType loop_type_;

  bool is_stream_same_count_;

  bool is_input_contiguous_{true};

  std::unordered_map<std::string, std::shared_ptr<Node>> match_node_;

  int GetRecvLimit();

  void GenDataContextFromInput(
      std::shared_ptr<InputData>& input_data,
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  void SetDataContextInput(std::shared_ptr<FlowUnitDataContext> data_context,
                           std::shared_ptr<InputData>& input_data,
                           uint32_t index);
  void PostProcessEvent(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list);

  std::shared_ptr<FlowUnitDataContext> GetDataContextFromKey(
      std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list,
      std::shared_ptr<IndexBuffer> buffer);

  std::vector<std::shared_ptr<FlowUnitDataContext>> GetDataContext();

  std::shared_ptr<ExternalData> CreateExternalData(
      std::shared_ptr<Device> device);

  std::map<std::shared_ptr<BufferGroup>, std::shared_ptr<FlowUnitDataContext>>
      data_context_map_;
};

}  // namespace modelbox
#endif
