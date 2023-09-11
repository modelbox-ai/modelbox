/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "modelbox/node.h"

#include <functional>
#include <utility>

#include "modelbox/port.h"
#include "modelbox/session.h"

namespace modelbox {

#define ReturnPortNames(port_list)     \
  std::set<std::string> name_list;     \
  for (auto& port : (port_list)) {     \
    name_list.insert(port->GetName()); \
  }                                    \
  return name_list;

#define ReturnPort(port_list, target_name)  \
  for (auto& port : (port_list)) {          \
    if (port->GetName() == (target_name)) { \
      return port;                          \
    }                                       \
  }                                         \
  return nullptr;

NodeBase::NodeBase() = default;

NodeBase::~NodeBase() = default;

void NodeBase::SetName(const std::string& name) { name_ = name; }

std::string NodeBase::GetName() const { return name_; }

void NodeBase::Close(){};

void NodeBase::SetPriority(int32_t priortity) { priority_ = priortity; }

int32_t NodeBase::GetPriority() const { return priority_; }

void NodeBase::SetQueueSize(int32_t queue_size) { queue_size_ = queue_size; }

int32_t NodeBase::GetQueueSize() const { return queue_size_; }

std::shared_ptr<EventPort> NodeBase::GetEventPort() { return event_port_; }

Status NodeBase::Init(const std::set<std::string>& input_port_names,
                      const std::set<std::string>& output_port_names,
                      const std::shared_ptr<Configuration>& config) {
  config_ = config;
  queue_size_ = config_->GetUint64("queue_size", DEFAULT_QUEUE_SIZE);
  if (0 == queue_size_) {
    MBLOG_ERROR << "queue size config is zero";
    return STATUS_INVALID;
  }
  event_queue_size_ = DEFAULT_QUEUE_EVENT;
  return InitPorts(input_port_names, output_port_names, config);
}

Status NodeBase::InitPorts(const std::set<std::string>& input_port_names,
                           const std::set<std::string>& output_port_names,
                           const std::shared_ptr<Configuration>& config) {
  // create event port
  event_port_ = std::make_shared<EventPort>(EVENT_PORT_NAME, shared_from_this(),
                                            GetPriority(), event_queue_size_);
  // create input port
  input_ports_.clear();
  input_ports_.reserve(input_port_names.size());
  for (const auto& input_port_name : input_port_names) {
    auto port_queue_size =
        config->GetUint64("queue_size_" + input_port_name, queue_size_);
    if (0 == port_queue_size) {
      MBLOG_ERROR << "queue size in zero for input " << input_port_name;
      return STATUS_INVALID;
    }

    input_ports_.push_back(std::make_shared<InPort>(
        input_port_name, shared_from_this(), GetPriority(), port_queue_size));
  }
  // create default external port if node has no input port
  if (input_port_names.empty()) {
    auto extern_queue_size =
        config_->GetUint64("queue_size_external", DEFAULT_QUEUE_SIZE_EXTERNAL);
    if (extern_queue_size == 0) {
      MBLOG_ERROR << "queue_size_external config is zero";
      return STATUS_INVALID;
    }
    extern_ports_.push_back(
        std::make_shared<InPort>(EXTERNAL_PORT_NAME, shared_from_this(),
                                 GetPriority(), extern_queue_size));
  }
  // create output port
  output_ports_.clear();
  output_ports_.reserve(output_port_names.size());
  for (const auto& output_port_name : output_port_names) {
    auto output_port =
        std::make_shared<OutPort>(output_port_name, shared_from_this());
    output_ports_.push_back(output_port);
  }

  return STATUS_SUCCESS;
}

size_t NodeBase::GetInputNum() { return input_ports_.size(); }

size_t NodeBase::GetExternNum() { return extern_ports_.size(); }

size_t NodeBase::GetOutputNum() { return output_ports_.size(); }

std::set<std::string> NodeBase::GetInputNames() {
  ReturnPortNames(input_ports_);
}

std::set<std::string> NodeBase::GetExternNames() {
  ReturnPortNames(extern_ports_);
}

std::set<std::string> NodeBase::GetOutputNames() {
  ReturnPortNames(output_ports_);
}

std::vector<std::shared_ptr<InPort>> NodeBase::GetInputPorts() const {
  return input_ports_;
}

std::vector<std::shared_ptr<OutPort>> NodeBase::GetOutputPorts() const {
  return output_ports_;
}

std::vector<std::shared_ptr<InPort>> NodeBase::GetExternalPorts() const {
  return extern_ports_;
}

std::shared_ptr<InPort> NodeBase::GetInputPort(const std::string& port_name) {
  ReturnPort(input_ports_, port_name);
}

std::shared_ptr<InPort> NodeBase::GetExternalPort(
    const std::string& port_name) {
  ReturnPort(extern_ports_, port_name);
}

std::shared_ptr<OutPort> NodeBase::GetOutputPort(const std::string& port_name) {
  ReturnPort(output_ports_, port_name);
}

void NodeBase::SetAllInportActivated(bool flag) {
  for (auto& port : input_ports_) {
    port->SetActiveState(flag);
  }
}

Status NodeBase::SendBatchEvent(
    std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list,
    bool update_active_time) {
  if (!event_port_) {
    MBLOG_ERROR << "Event port in null";
    return STATUS_FAULT;
  }

  auto status = event_port_->SendBatch(event_list);
  if (!status) {
    return status;
  }

  event_port_->NotifyPushEvent(update_active_time);
  return STATUS_SUCCESS;
}

Status NodeBase::SendEvent(std::shared_ptr<FlowUnitInnerEvent>& event,
                           bool update_active_time) {
  if (!event_port_) {
    MBLOG_ERROR << "Event port in null";
    return STATUS_FAULT;
  }

  auto status = event_port_->Send(event);
  if (!status) {
    return status;
  }

  event_port_->NotifyPushEvent(update_active_time);
  return STATUS_SUCCESS;
}

void NodeBase::Shutdown() {
  for (auto& port : input_ports_) {
    port->Shutdown();
  }

  for (auto& port : extern_ports_) {
    port->Shutdown();
  }

  event_port_->Shutdown();
}

Node::Node() = default;

Node::~Node() = default;

Status Node::Init(const std::set<std::string>& input_port_names,
                  const std::set<std::string>& output_port_names,
                  const std::shared_ptr<Configuration>& config) {
  auto ret = NodeBase::Init(input_port_names, output_port_names, config);
  if (!ret) {
    return ret;
  }

  flowunit_group_ = std::make_shared<FlowUnitGroup>(
      flowunit_name_, flowunit_type_, flowunit_device_id_, config, profiler_);
  if (flowunit_group_ == nullptr) {
    return STATUS_INVALID;
  }

  ret = flowunit_group_->Init(input_port_names, output_port_names,
                              flowunit_manager_);
  if (!ret) {
    return ret;
  }

  flowunit_group_->SetNode(std::dynamic_pointer_cast<Node>(shared_from_this()));

  auto port_count = GetInputNum();
  if (port_count == 0) {
    port_count = GetExternNum();
  }

  input_match_stream_mgr_ =
      std::make_shared<InputMatchStreamManager>(name_, queue_size_, port_count);
  output_match_stream_mgr_ =
      std::make_shared<OutputMatchStreamManager>(name_, GetOutputNames());
  ret = InitNodeProperties();
  if (!ret) {
    return ret;
  }

  return STATUS_OK;
}

std::unordered_map<std::string, std::shared_ptr<Node>> Node::GetMatchNodes() {
  return match_node_;
}

void Node::SetMatchNode(const std::string& name,
                        std::shared_ptr<Node> match_node) {
  match_node_[name] = std::move(match_node);
}

std::shared_ptr<Node> Node::GetMatchNode() { return match_node_["match_node"]; }

std::shared_ptr<Node> Node::GetMatchNode(const std::string& port_name) {
  return match_node_[port_name];
}

std::shared_ptr<FlowUnitDesc> Node::GetFlowUnitDesc() {
  return flowunit_group_->GetExecutorUnit()->GetFlowUnitDesc();
}

Status Node::InitNodeProperties() {
  // read flowunit desc
  auto flowunit_desc = flowunit_group_->GetExecutorUnit()->GetFlowUnitDesc();

  SetExceptionVisible(flowunit_desc->IsExceptionVisible());
  SetInputContiguous(flowunit_desc->IsInputContiguous());

  SetFlowType(flowunit_desc->GetFlowType());
  SetOutputType(flowunit_desc->GetOutputType());
  SetConditionType(flowunit_desc->GetConditionType());
  SetLoopType(flowunit_desc->GetLoopType());

  input_match_stream_mgr_->SetInputStreamGatherAll(
      GetOutputType() == COLLAPSE && flowunit_desc->IsCollapseAll());

  // update constrain
  UpdatePropConstrain(flowunit_desc);

  // Set input & output stream options
  if (GetFlowType() == STREAM || GetOutputType() == COLLAPSE) {
    input_match_stream_mgr_->SetInputBufferInOrder(true);
  }

  if (GetConditionType() != ConditionType::NONE) {
    input_match_stream_mgr_->SetInputBufferInOrder(true);
  }

  if (GetLoopType() != LoopType::NOT_LOOP) {
    input_match_stream_mgr_->SetInputStreamGatherAll(true);
  }

  output_match_stream_mgr_->SetNeedNewIndex(NeedNewIndex());
  return STATUS_OK;
}

void Node::UpdatePropConstrain(
    const std::shared_ptr<FlowUnitDesc>& flowunit_desc) {
  /**
   * constrain, Take effect by order
   * 1. expand: default normal
   * 2. collapse: default normal
   * 3. condition: only normal
   * 4. loop: only normal
   * 5. origin: default stream
   **/
  // constrain expand & collapse, not recommand to set flow type when use expand
  // & collapse
  auto output_type = GetOutputType();
  if (output_type != FlowOutputType::ORIGIN) {
    SetConditionType(ConditionType::NONE);
    SetLoopType(LoopType::NOT_LOOP);
    if (!flowunit_desc->IsUserSetFlowType()) {
      SetFlowType(FlowType::NORMAL);
    }
    return;
  }

  // constrain condition
  if (GetConditionType() != ConditionType::NONE) {
    SetFlowType(NORMAL);
    SetLoopType(LoopType::NOT_LOOP);
    return;
  }

  // constrain loop
  if (GetLoopType() != LoopType::NOT_LOOP) {
    SetFlowType(NORMAL);
    return;
  }

  // constrain origin
  if (!flowunit_desc->IsUserSetFlowType()) {
    SetFlowType(FlowType::STREAM);
  }
}

void Node::SetFlowUnitInfo(const std::string& flowunit_name,
                           const std::string& flowunit_type,
                           const std::string& flowunit_device_id,
                           std::shared_ptr<FlowUnitManager> flowunit_manager) {
  flowunit_name_ = flowunit_name;
  flowunit_type_ = flowunit_type;
  flowunit_device_id_ = flowunit_device_id;
  flowunit_manager_ = std::move(flowunit_manager);
}

std::shared_ptr<FlowUnitGroup> Node::GetFlowUnitGroup() {
  return flowunit_group_;
}

void Node::SetProfiler(std::shared_ptr<Profiler> profiler) {
  profiler_ = std::move(profiler);
}

void Node::SetStats(std::shared_ptr<StatisticsItem> graph_stats) {
  graph_stats_ = std::move(graph_stats);
}

std::shared_ptr<ExternalData> Node::CreateExternalData(
    const std::shared_ptr<Device>& device) {
  if (session_mgr_ == nullptr) {
    MBLOG_ERROR << "session manager is null";
    return nullptr;
  }

  auto port = GetExternalPort(EXTERNAL_PORT_NAME);
  if (!port) {
    MBLOG_WARN << "node has no external port";
    return nullptr;
  }

  auto session = session_mgr_->CreateSession(graph_stats_);
  auto init_stream = std::make_shared<Stream>(session);
  return std::make_shared<ExternalDataImpl>(port, device, init_stream);
}

bool Node::NeedNewIndex() {
  if (GetOutputType() == EXPAND ||
      (GetOutputType() == ORIGIN &&
       (GetFlowType() == STREAM || GetConditionType() == IF_ELSE))) {
    return true;
  }

  return false;
}

std::unordered_map<std::string, size_t> Node::GetStreamCountEachPort() {
  std::unordered_map<std::string, size_t> stream_count_each_port;
  for (auto& in_port : input_ports_) {
    auto port_count = in_port->GetConnectedPortNumber();
    if (port_count == 0) {
      continue;
    }

    stream_count_each_port[in_port->GetName()] = port_count;
    if (GetLoopType() == LOOP) {
      stream_count_each_port[in_port->GetName()] = 1;
    }
  }
  return stream_count_each_port;
}

Status Node::Open() {
  auto external_data_create_func =
      std::bind(&Node::CreateExternalData, this, std::placeholders::_1);
  auto ret = flowunit_group_->Open(external_data_create_func);
  if (!ret) {
    MBLOG_ERROR << "open flowunit " << flowunit_name_ << " failed";
    return ret;
  }

  is_flowunit_opened_ = true;
  return STATUS_SUCCESS;
}

void Node::Close() {
  if (flowunit_group_ == nullptr || !is_flowunit_opened_) {
    return;
  }

  is_flowunit_opened_ = false;
  auto ret = flowunit_group_->Close();
  if (!ret) {
    MBLOG_ERROR << "close flowunit " << flowunit_name_ << " failed, error "
                << ret;
  }
}

Status Node::GenDataContextList(
    std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  // one data context only generate one output match stream at time
  std::set<std::shared_ptr<FlowUnitDataContext>> data_ctx_set;
  for (auto& match_stream_data : match_stream_data_list) {
    Status ret = STATUS_SUCCESS;
    if (match_stream_data->GetEvent() != nullptr) {
      ret = AppendDataContextByEvent(match_stream_data, data_ctx_set);
    } else {
      ret = AppendDataContextByData(match_stream_data, data_ctx_set);
    }

    if (!ret) {
      MBLOG_ERROR << "append data context failed, err: " << ret;
      return STATUS_FAULT;
    }
  }

  data_ctx_list.assign(data_ctx_set.begin(), data_ctx_set.end());
  return STATUS_SUCCESS;
}

Status Node::AppendDataContextByEvent(
    const std::shared_ptr<MatchStreamData>& match_stream_data,
    std::set<std::shared_ptr<FlowUnitDataContext>>& data_ctx_set) {
  auto event = match_stream_data->GetEvent();
  auto* data_ctx_match_key = event->GetDataCtxMatchKey();
  auto data_ctx_item = data_ctx_map_.find(data_ctx_match_key);
  if (data_ctx_item == data_ctx_map_.end()) {
    // might be finished
    return STATUS_OK;
  }

  auto data_ctx = data_ctx_item->second;
  data_ctx_set.insert(data_ctx);
  if (GetOutputType() == COLLAPSE) {
    // collapse sub streams to one stream, each sub stream collpase to one
    // buffer stream collapse node process one sub stream at one process
    if (event->GetEventCode() != FlowUnitInnerEvent::COLLAPSE_NEXT_STREAM) {
      return {
          STATUS_INVALID,
          "only support collpase next stream event at collapse node " + name_};
    }

    auto stream_collapse_ctx =
        std::dynamic_pointer_cast<StreamCollapseFlowUnitDataContext>(data_ctx);
    stream_collapse_ctx->CollapseNextStream();
  } else if (GetOutputType() == EXPAND) {
    if (event->GetEventCode() == FlowUnitInnerEvent::EXPAND_UNFINISH_DATA) {
      // one buffer expand to one stream, the stream still has data
      // sent by flowunit developer
      data_ctx->SetEvent(event->GetUserEvent());
    } else if (event->GetEventCode() ==
               FlowUnitInnerEvent::EXPAND_NEXT_STREAM) {
      // expand next buffer
      // sent by stream expand node
      auto stream_expand_ctx =
          std::dynamic_pointer_cast<StreamExpandFlowUnitDataContext>(data_ctx);
      stream_expand_ctx->ExpandNextBuffer();
    } else {
      return {STATUS_INVALID, "not support event " +
                                  std::to_string(event->GetEventCode()) +
                                  " at expand node " + name_};
    }
  } else {
    // usecase: notify flowunit to process last data again
    if (event->GetEventCode() != FlowUnitInnerEvent::EXPAND_UNFINISH_DATA) {
      return {STATUS_INVALID, "only support user event at node " + name_};
    }

    if (GetFlowType() != STREAM) {
      return {
          STATUS_INVALID,
          "only support user event at stream node, not normal node " + name_};
    }

    data_ctx->SetEvent(event->GetUserEvent());
  }
  return STATUS_OK;
}

Status Node::AppendDataContextByData(
    const std::shared_ptr<MatchStreamData>& match_stream_data,
    std::set<std::shared_ptr<FlowUnitDataContext>>& data_ctx_set) {
  MatchKey* data_ctx_match_key = nullptr;
  if (GetFlowType() == STREAM) {
    if (GetOutputType() == COLLAPSE) {
      // collapse will match at expand, child stream after expand match at one
      // buffer in parent stream, we need parent stream info to gather all child
      // stream
      auto* match_at_ancestor_buffer =
          (BufferIndexInfo*)match_stream_data->GetStreamMatchKey();
      auto ancestor_stream = match_at_ancestor_buffer->GetStream();
      data_ctx_match_key = MatchKey::AsKey(ancestor_stream.get());
    } else {  // EXPAND, ORIGIN
      /**
       * expand: will expand one buffer for each node run, other data left in
       * ctx
       * origin: one match input to one match output
       **/
      data_ctx_match_key = match_stream_data->GetStreamMatchKey();
    }
  } else {  // NORMAL
    if (GetOutputType() == EXPAND) {
      /** expand buffer concurrently, will generate multi output
       * match_stream
       **/
      auto data_count = match_stream_data->GetDataCount();
      auto first_port_data =
          match_stream_data->GetBufferList()->begin()->second;
      for (size_t i = 0; i < data_count; ++i) {
        auto& buffer = first_port_data[i];
        data_ctx_match_key =
            MatchKey::AsKey(BufferManageView::GetIndexInfo(buffer).get());
        auto data_ctx = AppendDataToDataContext(data_ctx_match_key,
                                                match_stream_data, true, i);
        data_ctx_set.insert(data_ctx);
      }
      return STATUS_OK;
    }

    /**
     * collapse: collapse dirrerent match_stream concurrently
     * origin: one match input to one match output
     **/
    data_ctx_match_key = match_stream_data->GetStreamMatchKey();
  }

  auto data_ctx =
      AppendDataToDataContext(data_ctx_match_key, match_stream_data);
  data_ctx_set.insert(data_ctx);
  return STATUS_SUCCESS;
}

std::shared_ptr<FlowUnitDataContext> Node::GetDataContext(MatchKey* key) {
  auto item = data_ctx_map_.find(key);
  if (item != data_ctx_map_.end()) {
    return item->second;
  }

  return nullptr;
}

std::shared_ptr<FlowUnitDataContext> Node::CreateDataContext(
    MatchKey* key, const std::shared_ptr<Session>& session) {
  std::shared_ptr<FlowUnitDataContext> data_ctx;
  if (GetFlowType() == STREAM) {
    if (GetOutputType() == EXPAND) {
      data_ctx =
          std::make_shared<StreamExpandFlowUnitDataContext>(this, key, session);
    } else if (GetOutputType() == COLLAPSE) {
      data_ctx = std::make_shared<StreamCollapseFlowUnitDataContext>(this, key,
                                                                     session);
    } else {
      data_ctx =
          std::make_shared<StreamFlowUnitDataContext>(this, key, session);
      session->AddStateListener(data_ctx);
    }
  } else {  // NORMAL
    if (GetOutputType() == EXPAND) {
      data_ctx =
          std::make_shared<NormalExpandFlowUnitDataContext>(this, key, session);
    } else if (GetOutputType() == COLLAPSE) {
      data_ctx = std::make_shared<NormalCollapseFlowUnitDataContext>(this, key,
                                                                     session);
    } else if (GetLoopType() == LOOP) {
      data_ctx =
          std::make_shared<LoopNormalFlowUnitDataContext>(this, key, session);
    } else {
      data_ctx =
          std::make_shared<NormalFlowUnitDataContext>(this, key, session);
    }
  }

  data_ctx_map_[key] = data_ctx;
  return data_ctx;
}

std::shared_ptr<FlowUnitDataContext> Node::AppendDataToDataContext(
    MatchKey* key, const std::shared_ptr<MatchStreamData>& match_stream_data,
    bool append_single_buffer, size_t buffer_index) {
  auto data_ctx = GetDataContext(key);
  if (data_ctx == nullptr) {
    data_ctx = CreateDataContext(key, match_stream_data->GetSession());
  }

  auto stream_data_map = match_stream_data->GetBufferList();
  if (append_single_buffer == false) {
    data_ctx->WriteInputData(stream_data_map);
    return data_ctx;
  }

  auto split_stream_data_map = std::make_shared<PortDataMap>();
  for (auto& port_data_item : *stream_data_map) {
    const auto& port_name = port_data_item.first;
    auto& data_list = port_data_item.second;
    (*split_stream_data_map)[port_name].push_back(data_list[buffer_index]);
  }

  data_ctx->WriteInputData(split_stream_data_map);
  return data_ctx;
}

Status Node::Recv(
    RunType type,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  std::list<std::shared_ptr<MatchStreamData>> match_stream_data_list;
  auto ret = GenInputMatchStreamData(type, match_stream_data_list);
  if (!ret) {
    MBLOG_ERROR << "node " << name_ << " generate match stream failed, error "
                << ret;
    return ret;
  }

  if (match_stream_data_list.empty()) {
    return STATUS_SUCCESS;
  }

  ret = GenDataContextList(match_stream_data_list, data_ctx_list);
  if (!ret) {
    return ret;
  }

  return STATUS_SUCCESS;
}

Status Node::Process(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  auto ret = flowunit_group_->Run(data_ctx_list);
  if (!ret) {
    MBLOG_ERROR << "node " << name_ << " run flowunit group failed, error "
                << ret;
    return ret;
  }

  return STATUS_SUCCESS;
}

Status Node::Send(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  for (auto& data_ctx : data_ctx_list) {
    std::unordered_map<std::string, BufferPtrList> stream_data_map;
    data_ctx->PopOutputData(stream_data_map);
    auto ret = output_match_stream_mgr_->UpdateStreamInfo(
        stream_data_map, data_ctx->GetOutputPortStreamMeta(),
        data_ctx->GetSession());
    if (!ret) {
      return ret;
    }

    for (auto& output_port : output_ports_) {
      const auto& port_name = output_port->GetName();
      auto item = stream_data_map.find(port_name);
      if (item == stream_data_map.end()) {
        if (GetLoopType() == LoopType::LOOP) {
          // only one port has data for loop node
          continue;
        }

        MBLOG_ERROR << "node " << name_ << ", missing output for port "
                    << port_name;
        return STATUS_FAULT;
      }

      auto& output_datas = item->second;
      std::vector<std::shared_ptr<Buffer>> valid_output;
      valid_output.reserve(output_datas.size());
      for (auto& buffer : output_datas) {
        if (buffer == nullptr) {
          continue;
        }

        valid_output.push_back(buffer);
      }
      output_port->Send(valid_output);
    }
  }
  return STATUS_SUCCESS;
}

void Node::Clean(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  // clear data for this run
  for (auto& data_ctx : data_ctx_list) {
    data_ctx->ClearData();
  }

  input_match_stream_mgr_->Clean();
  CleanDataContext();
  output_match_stream_mgr_->Clean();

  MBLOG_DEBUG << "node: " << name_
              << ", resource state after run, input stream "
              << input_match_stream_mgr_->GetInputStreamCount() << ", data ctx "
              << data_ctx_map_.size() << ", output stream "
              << output_match_stream_mgr_->GetOutputStreamCount();
}

void Node::CleanDataContext() {
  // remove finished & closed data for this node
  for (auto data_ctx_iter = data_ctx_map_.begin();
       data_ctx_iter != data_ctx_map_.end();) {
    auto& data_ctx = data_ctx_iter->second;
    if (!data_ctx->IsFinished() && !data_ctx->GetSession()->IsAbort()) {
      ++data_ctx_iter;
      continue;
    }

    auto sess_ctx = data_ctx->GetSessionContext();
    if (GetFlowType() == STREAM && sess_ctx != nullptr) {
      MBLOG_INFO << "node: " << name_
                 << ", sess id: " << sess_ctx->GetSessionId()
                 << ", data ctx finished";
    }

    data_ctx->Dispose();
    data_ctx_iter = data_ctx_map_.erase(data_ctx_iter);
  }
}

Status Node::Run(RunType type) {
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;
  size_t process_count = 0;
  auto ret = Recv(type, data_ctx_list);

  if (!ret) {
    return ret;
  }

  std::list<std::shared_ptr<FlowUnitDataContext>> process_ctx_list;

  auto output_names_is_empty = GetOutputNames().empty();

  for (auto& ctx : data_ctx_list) {
    // process data according to batch size
    process_count++;
    process_ctx_list.push_back(ctx);

    if (process_ctx_list.size() < flowunit_group_->GetBatchSize()) {
      if (process_count < data_ctx_list.size()) {
        continue;
      }
    }

    ret = Process(process_ctx_list);
    if (!ret) {
      return ret;
    }

    if (!output_names_is_empty) {
      ret = Send(process_ctx_list);
      if (!ret) {
        return ret;
      }
    } else {
      SetLastError(process_ctx_list);
    }

    process_ctx_list.clear();
  }

  Clean(data_ctx_list);
  return STATUS_SUCCESS;
}

void Node::SetLastError(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  for (auto& data_ctx : data_ctx_list) {
    auto sess = data_ctx->GetSession();
    auto last_status = data_ctx->GetLastStatus();
    if (last_status != modelbox::STATUS_OK &&
        last_status != modelbox::STATUS_CONTINUE) {
      sess->SetError(std::make_shared<FlowUnitError>(last_status.Errormsg()));
      continue;
    }

    for (const auto& input_map : data_ctx->GetErrorInputs()) {
      auto error_buffer_list = input_map.second;
      if (!error_buffer_list.empty()) {
        sess->SetError(std::make_shared<FlowUnitError>(
            error_buffer_list[0]->GetErrorMsg()));
      }
    }
  }
}

Status Node::GenInputMatchStreamData(
    RunType type,
    std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list) {
  switch (type) {
    case RunType::DATA:
      if (GetInputNum() == 0) {
        return GenMatchStreamFromDataPorts(extern_ports_,
                                           match_stream_data_list);
      }

      std::call_once(input_stream_count_update_flag_, [this]() {
        input_match_stream_mgr_->UpdateStreamCountEachPort(
            GetStreamCountEachPort());
      });
      return GenMatchStreamFromDataPorts(input_ports_, match_stream_data_list);

    case RunType::EVENT:
      return GenMatchStreamFromEventPorts(match_stream_data_list);

    default:
      MBLOG_ERROR << "Invalid node run type " << type;
      return STATUS_INVALID;
  }
}

Status Node::GenMatchStreamFromDataPorts(
    std::vector<std::shared_ptr<InPort>>& data_ports,
    std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list) {
  auto ret = input_match_stream_mgr_->LoadData(data_ports);
  if (!ret) {
    return ret;
  }

  return input_match_stream_mgr_->GenMatchStreamData(match_stream_data_list);
}

Status Node::GenMatchStreamFromEventPorts(
    std::list<std::shared_ptr<MatchStreamData>>& match_stream_data_list) {
  FlowunitEventList events;
  auto status = event_port_->Recv(events);
  if (!events || events->empty()) {
    return STATUS_SUCCESS;
  }

  event_port_->NotifyPopEvent();
  for (auto& event : *events) {
    auto match_stream = std::make_shared<MatchStreamData>();
    match_stream->SetEvent(event);
    match_stream_data_list.push_back(match_stream);
  }

  return STATUS_SUCCESS;
}

void Node::SetOutputType(FlowOutputType type) { output_type_ = type; }

void Node::SetFlowType(FlowType type) { flow_type_ = type; }

void Node::SetConditionType(ConditionType type) { condition_type_ = type; }

void Node::SetLoopType(LoopType type) { loop_type_ = type; }

void Node::SetInputContiguous(bool is_input_contiguous) {
  is_input_contiguous_ = is_input_contiguous;
}

void Node::SetExceptionVisible(bool is_exception_visible) {
  is_exception_visible_ = is_exception_visible;
}

FlowOutputType Node::GetOutputType() { return output_type_; }

FlowType Node::GetFlowType() { return flow_type_; }

ConditionType Node::GetConditionType() { return condition_type_; }

LoopType Node::GetLoopType() { return loop_type_; }

bool Node::IsInputContiguous() { return is_input_contiguous_; }

bool Node::IsExceptionVisible() { return is_exception_visible_; }

void Node::SetSessionManager(SessionManager* session_mgr) {
  session_mgr_ = session_mgr;
}

void Node::SetLoopOutPortName(const std::string& port_name) {
  loop_out_port_name_ = port_name;
}

std::string Node::GetLoopOutPortName() { return loop_out_port_name_; }

}  // namespace modelbox
