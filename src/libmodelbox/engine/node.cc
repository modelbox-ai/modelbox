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

#include "modelbox/node.h"

namespace modelbox {

#define DEFAULT_QUEUE_SIZE 8192
#define DEFAULT_QUEUE_SIZE_EVENT 8192

SingleMatchCache::SingleMatchCache(std::shared_ptr<NodeBase> node) {
  single_match_buffer_ = std::make_shared<SingleMatch>();
  node_ = node;
  limit_counts_ = node->GetQueueSize();
  origin_limit_counts_ = limit_counts_;
}

uint32_t SingleMatchCache::GetLeftBufferSize(std::string port) {
  int exist_counts = 0;
  auto single_match_iter = single_match_buffer_->begin();
  while (single_match_iter != single_match_buffer_->end()) {
    auto match_buffer = single_match_iter->second;
    if (match_buffer->GetBuffer(port) != nullptr) {
      exist_counts++;
    }
    single_match_iter++;
  }

  if (limit_counts_ < 0) {
    return -1;
  }

  if (limit_counts_ < exist_counts) {
    return 0;
  }

  return limit_counts_ - exist_counts;
}

Status SingleMatchCache::LoadCache(
    std::string port,
    std::vector<std::shared_ptr<IndexBuffer>>& buffer_vector) {
  auto node = node_.lock();
  if (node == nullptr) {
    return STATUS_NOTFOUND;
  }

  uint32_t input_num = 0;
  if (node->GetInputNum() > 0) {
    input_num = node->GetInputNum();
  } else {
    input_num = node->GetExternNames().size();
  }

  for (auto buffer : buffer_vector) {
    auto buffer_group_key = buffer->GetSameLevelGroup();
    auto key_it = single_match_buffer_->find(buffer_group_key);
    if (key_it == single_match_buffer_->end()) {
      single_match_buffer_->emplace(buffer_group_key,
                                    std::make_shared<MatchBuffer>(input_num));
    }

    if (!single_match_buffer_->at(buffer_group_key)->SetBuffer(port, buffer)) {
      return STATUS_INVALID;
    }
  }
  return STATUS_SUCCESS;
}

void SingleMatchCache::UnloadCache(
    std::shared_ptr<StreamMatchCache> stream_match_cache, bool is_input_order) {
  auto group_match_cache = stream_match_cache->GetStreamReceiveBuffer();
  auto group_order = stream_match_cache->GetStreamOrder();

  auto it = single_match_buffer_->begin();
  while (it != single_match_buffer_->end()) {
    if (!it->second->IsMatch()) {
      it++;
      continue;
    }
    auto parent_group_ptr = it->first->GetGroup();

    auto port_id = it->first->GetPortId();
    auto key = std::make_tuple(parent_group_ptr, port_id);

    auto map_it = group_match_cache->find(key);
    if (map_it == group_match_cache->end()) {
      std::vector<std::shared_ptr<MatchBuffer>> group_vector;
      group_match_cache->emplace(key, group_vector);
    }

    auto order_it = group_order->find(key);
    if (order_it == group_order->end()) {
      uint32_t order = parent_group_ptr->GetGroupOrder(port_id);
      MBLOG_DEBUG << "group_order emplace group_ptr" << std::get<0>(key)
                  << " port_id " << std::get<1>(key) << " order " << order;
      if (is_input_order) {
        group_order->emplace(key, order);
      }
    }

    group_match_cache->at(key).push_back(it->second);
    it = single_match_buffer_->erase(it);
  }
}

UpdateType GetResult(std::unordered_map<std::string, int32_t>& map_limit,
                     int32_t limit_count, int32_t queue_size) {
  UpdateType result = UNDEFINED;
  auto iter = map_limit.begin();
  while (iter != map_limit.end()) {
    auto limit = iter->second;
    if (limit >= limit_count) {
      if ((result == ENLARGE) || (result == UNDEFINED)) {
        result = ENLARGE;
      } else {
        result = CONSTANT;
        break;
      }
    } else if (limit < limit_count - queue_size) {
      if ((result == REDUCE) || (result == UNDEFINED)) {
        result = REDUCE;
      } else {
        result = CONSTANT;
        break;
      }
    } else {
      result = CONSTANT;
      break;
    }
    iter++;
  }
  return result;
}

UpdateType SingleMatchCache::GetUpdataType() {
  UpdateType result = UNDEFINED;

  auto node = node_.lock();
  if (node == nullptr) {
    return result;
  }

  std::unordered_map<std::string, int32_t> map_limit;
  auto input_names = std::set<std::string>();

  if (node->GetInputNum() > 0) {
    input_names = node->GetInputNames();
  } else {
    input_names = node->GetExternNames();
  }

  for (auto key : input_names) {
    map_limit[key] = 0;
  }

  auto it = single_match_buffer_->begin();
  while (it != single_match_buffer_->end()) {
    auto match_buffer = it->second;
    for (auto key : input_names) {
      auto buffer = match_buffer->GetBuffer(key);
      if (buffer != nullptr) {
        map_limit[key] = map_limit[key] + 1;
      }
    }
    it++;
  }

  // when all port design to enlarge or reduce,we can enlarge or reduce
  auto queue_size = node->GetQueueSize();
  result = GetResult(map_limit, limit_counts_, queue_size);
  return result;
}

int SingleMatchCache::GetLimitCount() { return limit_counts_; }

void SingleMatchCache::EnlargeBufferCache() {
  auto node = node_.lock();
  if (node == nullptr) {
    return;
  }
  if (origin_limit_counts_ > 0) {
    limit_counts_ += node->GetQueueSize();
  }
}

void SingleMatchCache::ReduceBufferCache() {
  auto node = node_.lock();
  if (node == nullptr) {
    return;
  }
  if (origin_limit_counts_ > 0) {
    if (limit_counts_ > origin_limit_counts_) {
      limit_counts_ -= node->GetQueueSize();
    }
  }
}

std::shared_ptr<SingleMatch> SingleMatchCache::GetReceiveBuffer() {
  return single_match_buffer_;
}

StreamMatchCache::StreamMatchCache() {
  stream_match_buffer_ = std::make_shared<StreamMatch>();
  stream_order_ = std::make_shared<StreamOrder>();
}

StreamMatchCache::~StreamMatchCache() {}

std::shared_ptr<StreamMatch> StreamMatchCache::GetStreamReceiveBuffer() {
  return stream_match_buffer_;
}
std::shared_ptr<StreamOrder> StreamMatchCache::GetStreamOrder() {
  return stream_order_;
}

void NodeBase::Shutdown() {
  for (auto& port : input_ports_) {
    port->Shutdown();
  }

  for (auto& port : output_ports_) {
    port->Shutdown();
  }

  for (auto& port : extern_ports_) {
    port->Shutdown();
  }

  event_port_->Shutdown();
}

uint32_t NodeBase::GetInputNum() { return input_ports_.size(); }

uint32_t NodeBase::GetOutputNum() { return output_ports_.size(); }

std::set<std::string> NodeBase::GetInputNames() {
  std::set<std::string> names;
  for (auto& input_port : input_ports_) {
    names.insert(input_port->GetName());
  }
  return names;
}

std::set<std::string> NodeBase::GetExternNames() {
  std::set<std::string> names;
  for (auto& extern_port : extern_ports_) {
    names.insert(extern_port->GetName());
  }
  return names;
}

std::set<std::string> NodeBase::GetOutputNames() {
  std::set<std::string> names;
  for (auto& output_port : output_ports_) {
    names.insert(output_port->GetName());
  }
  return names;
}

std::shared_ptr<InPort> NodeBase::GetInputPort(const std::string& port_name) {
  for (auto& input_port : input_ports_) {
    if (input_port->GetName() == port_name) {
      return input_port;
    }
  }

  return nullptr;
}

std::vector<std::shared_ptr<InPort>> NodeBase::GetInputPorts() const {
  return input_ports_;
}

std::vector<std::shared_ptr<OutPort>> NodeBase::GetOutputPorts() const {
  return output_ports_;
}

std::shared_ptr<OutPort> NodeBase::GetOutputPort(const std::string& port_name) {
  for (auto& output_port : output_ports_) {
    if (output_port->GetName() == port_name) {
      return output_port;
    }
  }
  return nullptr;
}

std::shared_ptr<InPort> NodeBase::GetExternalPort(
    const std::string& port_name) {
  for (auto& ext_port : extern_ports_) {
    if (ext_port->GetName() == port_name) {
      return ext_port;
    }
  }

  return nullptr;
}

const std::vector<std::shared_ptr<InPort>>& NodeBase::GetExternalPorts() const {
  return extern_ports_;
}

void NodeBase::SetAllInportActivated(bool flag) {
  for (auto port : GetInputPorts()) {
    port->SetActiveState(flag);
  }
}

Status NodeBase::SendBatchEvent(
    std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list,
    bool update_active_time) {
  if (!event_port_) {
    MBLOG_ERROR << "event port must not be nullptr.";
    return STATUS_FAULT;
  }

  auto status = event_port_->SendBatch(event_list);
  event_port_->NotifyPushEvent(update_active_time);

  return status;
}

Status NodeBase::SendEvent(std::shared_ptr<FlowUnitInnerEvent>& event,
                           bool update_active_time) {
  if (!event_port_) {
    MBLOG_ERROR << "event port must not be nullptr.";
    return STATUS_FAULT;
  }

  auto status = event_port_->Send(event);
  if (status != STATUS_SUCCESS) {
    MBLOG_ERROR << "node " << name_ << " event port send faild";
    return status;
  }
  event_port_->NotifyPushEvent(update_active_time);
  return status;
}

Status NodeBase::Init(const std::set<std::string>& input_port_names,
                      const std::set<std::string>& output_port_names,
                      std::shared_ptr<Configuration> config) {
  config_ = config;

  queue_size_ = config_->GetUint64("queue_size", DEFAULT_QUEUE_SIZE);
  if (0 == queue_size_) {
    return {STATUS_INVALID, "invalid queue_size config: 0"};
  }

  auto event_queue_size =
      config_->GetUint64("queue_size_event", DEFAULT_QUEUE_SIZE_EVENT);
  if (0 == event_queue_size) {
    return {STATUS_INVALID, "invalid queue_size_event config: 0"};
  }

  event_port_ = std::make_shared<EventPort>(EVENT_PORT_NAME, shared_from_this(),
                                            GetPriority(), event_queue_size);

  if ((input_ports_.size() != 0) && (output_ports_.size() != 0)) {
    auto errmsg = "input port and output port is not empty.the node " + name_ +
                  " is already inited";
    MBLOG_ERROR << errmsg;
    return {STATUS_INTERNAL, errmsg};
  }

  // Input Port Initialization
  for (auto input_port_name : input_port_names) {
    auto in_queue_size =
        config_->GetUint64("queue_size_" + input_port_name, queue_size_);
    if (0 == in_queue_size) {
      return {STATUS_INVALID,
              "invalid queue_size_" + input_port_name + " config: 0"};
    }

    input_ports_.emplace_back(std::make_shared<InPort>(
        input_port_name,
        std::dynamic_pointer_cast<NodeBase>(shared_from_this()), GetPriority(),
        in_queue_size));
  }

  for (auto& input_port : input_ports_) {
    auto result = input_port->Init();
    if (result != STATUS_SUCCESS) {
      input_ports_.clear();
      return {result, "node init failed."};
    }
  }

  // Output Port Initialization
  for (auto output_port_name : output_port_names) {
    auto output_port = std::shared_ptr<OutPort>(
        new OutPort(output_port_name,
                    std::dynamic_pointer_cast<NodeBase>(shared_from_this())));
    output_ports_.push_back(output_port);
  }

  for (auto& output_port : output_ports_) {
    auto result = output_port->Init();
    if (result != STATUS_SUCCESS) {
      output_ports_.clear();
      return result;
    }
  }

  return STATUS_OK;
}

void NodeBase::Close() {}

OutputIndexBuffer NodeBase::CreateOutputBuffer() {
  OutputIndexBuffer output_map;
  for (auto& output_port : output_ports_) {
    std::vector<std::shared_ptr<IndexBuffer>> output_vector;
    output_map[output_port->GetName()] = output_vector;
  }
  return output_map;
}

Status NodeBase::Send(OutputIndexBuffer* output_buffer) {
  for (auto& output_port : output_ports_) {
    auto port_name = output_port->GetName();
    if (output_buffer->find(port_name) == output_buffer->end()) {
      return STATUS_INVALID;
    }

    auto buffer_vector = output_buffer->find(port_name)->second;
    output_port->Send(buffer_vector);
  }

  return STATUS_SUCCESS;
}

InputIndexBuffer NodeBase::CreateInputBuffer() {
  InputIndexBuffer input_map;
  for (auto& input_port : input_ports_) {
    std::vector<std::shared_ptr<IndexBufferList>> input_vector;
    input_map[input_port->GetName()] = input_vector;
  }
  return input_map;
}

Status DataMatcherNode::ReceiveGroupBuffer() {
  auto before_size = single_match_cache_->GetReceiveBuffer()->size();
  single_match_cache_->UnloadCache(stream_match_cache_, IsInputOrder());
  auto after_size = single_match_cache_->GetReceiveBuffer()->size();
  if (before_size != after_size) {
    SetAllInportActivated(true);
  }

  auto update_type = single_match_cache_->GetUpdataType();
  if (update_type == ENLARGE) {
    single_match_cache_->EnlargeBufferCache();
    SetAllInportActivated(true);
  } else if (update_type == REDUCE) {
    single_match_cache_->ReduceBufferCache();
  }

  return STATUS_SUCCESS;
}

bool DataMatcherNode::IsInputOrder() { return is_input_order_; }

Status DataMatcherNode::FillVectorMap(
    std::tuple<std::shared_ptr<modelbox::BufferGroup>, uint32_t> order_key,
    std::vector<std::shared_ptr<MatchBuffer>>& match_vector,
    std::unordered_map<std::string, std::vector<std::shared_ptr<IndexBuffer>>>&
        group_buffer_vector_map) {
  auto group_order = stream_match_cache_->GetStreamOrder();
  auto vector_size = match_vector.size();
  if (IsInputOrder()) {
    // for we use pop_back so here we should reverse the order
    std::sort(
        match_vector.begin(), match_vector.end(),
        [](std::shared_ptr<MatchBuffer> a, std::shared_ptr<MatchBuffer> b) {
          return (a->GetOrder() < b->GetOrder());
        });
  }

  while (!match_vector.empty()) {
    auto match = *(match_vector.begin());
    uint32_t sum = 0;

    if (need_garther_all_ == true) {
      if ((match->GetGroupSum(&sum) != STATUS_SUCCESS) ||
          (vector_size != sum)) {
        break;
      }
    }

    if (IsInputOrder() && (match->GetOrder() > group_order->at(order_key))) {
      break;
    }

    if (IsInputOrder() && match->GetOrder() == group_order->at(order_key)) {
      if ((match->GetGroupSum(&sum) == STATUS_SUCCESS) &&
          (sum == group_order->at(order_key))) {
        group_order->erase(order_key);
      } else {
        group_order->at(order_key)++;
      }
    }

    auto map_iter = group_buffer_vector_map.begin();
    while (map_iter != group_buffer_vector_map.end()) {
      auto input_name = map_iter->first;
      if (match->GetBuffer(input_name) == nullptr) {
        return STATUS_INVALID;
      }
      auto buffer = match->GetBuffer(input_name);
      map_iter->second.push_back(buffer);
      map_iter++;
    }

    match_vector.erase(match_vector.begin());
  }
  return STATUS_SUCCESS;
}

Status DataMatcherNode::FillOneMatchVector(
    InputIndexBuffer* single_map,
    std::tuple<std::shared_ptr<modelbox::BufferGroup>, uint32_t> order_key,
    std::vector<std::shared_ptr<MatchBuffer>>& match_vector) {
  std::unordered_map<std::string, std::vector<std::shared_ptr<IndexBuffer>>>
      group_buffer_vector_map;
  auto input_names = GetInputNames();
  if (input_names.size() == 0) {
    input_names = GetExternNames();
  }

  for (auto& input_name : input_names) {
    std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
    group_buffer_vector_map[input_name] = buffer_vector;
  }

  FillVectorMap(order_key, match_vector, group_buffer_vector_map);

  for (auto input_name : input_names) {
    if (group_buffer_vector_map[input_name].size() != 0) {
      single_map->at(input_name)
          .push_back(std::make_shared<IndexBufferList>(
              group_buffer_vector_map[input_name]));
    }
  }

  return STATUS_SUCCESS;
}

Status DataMatcherNode::GenerateFromStreamPool(InputIndexBuffer* single_map) {
  auto group_match_buffer = stream_match_cache_->GetStreamReceiveBuffer();
  auto it = group_match_buffer->begin();
  while (it != group_match_buffer->end()) {
    auto order_key = it->first;
    auto match_vector = &(it->second);
    FillOneMatchVector(single_map, order_key, *match_vector);
    if (match_vector->size() == 0) {
      it = group_match_buffer->erase(it);
    } else {
      it++;
    }
  }
  return STATUS_SUCCESS;
}

void DataMatcherNode::SetInputGatherAll(bool need_garther_all) {
  need_garther_all_ = need_garther_all;
}

void DataMatcherNode::SetInputOrder(bool is_input_order) {
  is_input_order_ = is_input_order;
}

Status DataMatcherNode::ReceiveBuffer(std::shared_ptr<InPort>& input_port) {
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
  uint32_t left_buffer_num =
      single_match_cache_->GetLeftBufferSize(input_port->GetName());
  input_port->Recv(buffer_vector, left_buffer_num);

  auto status =
      single_match_cache_->LoadCache(input_port->GetName(), buffer_vector);
  return status;
}

Status DataMatcherNode::RecvDataQueue(InputIndexBuffer* input_buffer) {
  for (auto& input_port : input_ports_) {
    auto result = ReceiveBuffer(input_port);
    if (result != STATUS_SUCCESS) {
      return result;
    }
  }
  auto result = ReceiveGroupBuffer();
  if (result != STATUS_SUCCESS) {
    return result;
  }

  GenerateFromStreamPool(input_buffer);

  return STATUS_SUCCESS;
}

Status DataMatcherNode::RecvExternalDataQueue(InputIndexBuffer* input_buffer) {
  for (auto& ext_port : extern_ports_) {
    auto result = ReceiveBuffer(ext_port);
    if (result != STATUS_SUCCESS) {
      return result;
    }
  }

  ReceiveGroupBuffer();
  GenerateFromStreamPool(input_buffer);

  return STATUS_SUCCESS;
}

Status DataMatcherNode::Init(const std::set<std::string>& input_port_names,
                             const std::set<std::string>& output_port_names,
                             std::shared_ptr<Configuration> config) {
  auto status = NodeBase::Init(input_port_names, output_port_names, config);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  single_match_cache_ = std::make_shared<SingleMatchCache>(shared_from_this());
  stream_match_cache_ = std::make_shared<StreamMatchCache>();

  return STATUS_SUCCESS;
}

std::shared_ptr<SingleMatchCache> DataMatcherNode::GetSingleMatchCache() {
  return single_match_cache_;
}

std::shared_ptr<StreamMatchCache> DataMatcherNode::GetStreamMatchCache() {
  return stream_match_cache_;
}

Node::Node(const std::string& unit_name, const std::string& unit_type,
           const std::string& unit_device_id,
           std::shared_ptr<FlowUnitManager> flowunit_mgr,
           std::shared_ptr<Profiler> profiler,
           std::shared_ptr<StatisticsItem> graph_stats)
    : is_exception_visible_(false),
      is_fug_opened_(false),
      unit_name_(unit_name),
      unit_type_(unit_type),
      unit_device_id_(unit_device_id),
      graph_stats_(graph_stats) {
  output_type_ = ORIGIN;
  flow_type_ = NORMAL;
  condition_type_ = NONE;
  loop_type_ = NOT_LOOP;
  is_stream_same_count_ = true;
  priority_ = 0;
  is_input_order_ = false;
  need_garther_all_ = false;
  flowunit_mgr_ = flowunit_mgr;
  profiler_ = profiler;
}

Node::~Node() {
  if (flowunit_group_) {
    if (is_fug_opened_) {
      is_fug_opened_ = false;
      auto status = flowunit_group_->Close();
      if (!status) {
        MBLOG_ERROR << "flow unit group close failed: " << status;
      }
    }
    auto status = flowunit_group_->Destory();
    if (!status) {
      MBLOG_ERROR << "flow unit group destory failed: " << status;
    }
  }

  input_ports_.clear();
  output_ports_.clear();
  extern_ports_.clear();
}

Status Node::Open() {
  auto extern_data_func =
      std::bind(&Node::CreateExternalData, this, std::placeholders::_1);

  auto status = flowunit_group_->Open(extern_data_func);
  if (!status) {
    auto errmsg = "flowunit group open " + name_ + " failed.";
    MBLOG_ERROR << errmsg << status;
    return {status, errmsg};
  }

  is_fug_opened_ = true;

  return status;
}

void Node::Close() {
  if (flowunit_group_ == nullptr) {
    return;
  }

  if (!is_fug_opened_) {
    return;
  }

  is_fug_opened_ = false;
  auto status = flowunit_group_->Close();
  if (!status) {
    MBLOG_ERROR << "flow unit group close failed: " << status;
  }
}

void Node::SetExceptionVisible(bool is_exception_visible) {
  is_exception_visible_ = is_exception_visible;
}

bool Node::IsExceptionVisible() { return is_exception_visible_; }

Status Node::Send(OutputIndexBuffer* output_buffer) {
  for (auto& output_port : output_ports_) {
    auto port_name = output_port->GetName();
    if (output_buffer->find(port_name) == output_buffer->end()) {
      MBLOG_ERROR << "Can not find " << port_name << " in output";
      return STATUS_INVALID;
    }

    auto buffer_vector = output_buffer->find(port_name)->second;
    output_port->Send(buffer_vector);
  }

  return STATUS_SUCCESS;
}

std::shared_ptr<FlowUnitDesc> Node::GetFlowUnitDesc() {
  return flowunit_group_->GetExecutorUnit()->GetFlowUnitDesc();
}

Status Node::GenerateOutputIndexBuffer(
    OutputIndexBuffer* output_map_index_buffer,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  for (auto& data_ctx : data_ctx_list) {
    data_ctx->AppendOutputMap(output_map_index_buffer);
  }

  return STATUS_SUCCESS;
}

void Node::InitNodeWithFlowunit() {
  auto flowunit_desc = flowunit_group_->GetExecutorUnit()->GetFlowUnitDesc();

  auto exception_visible = flowunit_desc->IsExceptionVisible();

  if (exception_visible) {
    SetExceptionVisible(config_->GetBool("is_exception_visible", false));
  }

  if (flowunit_desc->GetOutputType() == COLLAPSE) {
    SetOutputType(COLLAPSE);
    SetInputGatherAll(flowunit_desc->IsCollapseAll());
  } else if (flowunit_desc->GetOutputType() == EXPAND) {
    SetOutputType(EXPAND);
  } else {
    SetOutputType(ORIGIN);
  }

  if (flowunit_desc->GetFlowType() == STREAM) {
    SetFlowType(STREAM);
    SetStreamSameCount(flowunit_desc->IsStreamSameCount());
  } else {
    SetFlowType(NORMAL);
  }

  if (flowunit_desc->GetConditionType() == IF_ELSE) {
    SetConditionType(IF_ELSE);
  } else {
    SetConditionType(NONE);
  }

  if (flowunit_desc->GetLoopType() == LOOP) {
    SetLoopType(LOOP);
    SetInputGatherAll(flowunit_desc->IsCollapseAll());
  } else {
    SetLoopType(NOT_LOOP);
  }

  SetInputContiguous(flowunit_desc->IsInputContiguous());

  if (flow_type_ == STREAM) {
    is_input_order_ = true;
  }

  if (output_type_ == COLLAPSE) {
    is_input_order_ = true;
  }
}

Status Node::Init(const std::set<std::string>& input_port_names,
                  const std::set<std::string>& output_port_names,
                  std::shared_ptr<Configuration> config) {
  Status status =
      DataMatcherNode::Init(input_port_names, output_port_names, config);

  if (status != STATUS_SUCCESS) {
    return status;
  }

  flowunit_group_ = std::make_shared<FlowUnitGroup>(
      unit_name_, unit_type_, unit_device_id_, config, profiler_);
  if (flowunit_group_ == nullptr) {
    input_ports_.clear();
    output_ports_.clear();
    return STATUS_INVALID;
  }

  status =
      flowunit_group_->Init(input_port_names, output_port_names, flowunit_mgr_);
  if (status != STATUS_SUCCESS) {
    input_ports_.clear();
    output_ports_.clear();
    return status;
  }

  flowunit_group_->SetNode(std::dynamic_pointer_cast<Node>(shared_from_this()));

  InitNodeWithFlowunit();

  data_context_map_ = std::map<std::shared_ptr<BufferGroup>,
                               std::shared_ptr<FlowUnitDataContext>>();

  if (input_port_names.empty()) {
    auto ext_queue_size =
        config_->GetUint64("queue_size_external", queue_size_);
    if (0 == ext_queue_size) {
      return {STATUS_INVALID, "invalid queue_size_external config: 0"};
    }

    extern_ports_.emplace_back(std::make_shared<InPort>(
        EXTERNAL_PORT_NAME, shared_from_this(), GetPriority(), ext_queue_size));
    extern_ports_[0]->Init();
  }

  return STATUS_OK;
}

std::shared_ptr<Device> Node::GetDevice() {
  if (flowunit_mgr_ == nullptr) {
    MBLOG_ERROR << "flowunit_mgr is nullptr ";
    return nullptr;
  }

  auto device_mgr = flowunit_mgr_->GetDeviceManager();
  if (device_mgr == nullptr) {
    MBLOG_ERROR << "device_mgr is nullptr ";
    return nullptr;
  }

  auto device = device_mgr->GetDevice(unit_type_, unit_device_id_);
  if (device == nullptr) {
    MBLOG_ERROR << "device is nullptr."
                << " device_name: " << unit_type_
                << " device_id_: " << unit_device_id_;
    return nullptr;
  }
  return device;
}

Status Node::RecvExternalData(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  auto external_buffer = CreateExternalBuffer();

  auto status = RecvExternalDataQueue(&external_buffer);
  if (status != STATUS_SUCCESS) {
    MBLOG_WARN << "node '" << name_ << "' recv data failed: " << status << "!";
    return {status, name_ + " recv external failed."};
  }

  GenerateDataContext(external_buffer, data_ctx_list);
  return STATUS_SUCCESS;
}

InputIndexBuffer Node::CreateExternalBuffer() {
  auto external_name = "External_Port";
  InputIndexBuffer external_map;

  std::vector<std::shared_ptr<IndexBufferList>> external_vector;
  external_map[external_name] = external_vector;
  return external_map;
}

std::shared_ptr<FlowUnitDataContext> Node::GenDataContext(
    std::shared_ptr<BufferGroup> stream_info) {
  std::shared_ptr<FlowUnitDataContext> data_ctx = nullptr;
  if (GetOutputType() == COLLAPSE) {
    if (GetFlowType() == STREAM) {
      data_ctx = std::make_shared<StreamCollapseFlowUnitDataContext>(
          stream_info, this);
    } else {
      data_ctx = std::make_shared<NormalCollapseFlowUnitDataContext>(
          stream_info, this);
    }
  } else if (GetOutputType() == EXPAND) {
    if (GetFlowType() == STREAM) {
      data_ctx =
          std::make_shared<StreamExpandFlowUnitDataContext>(stream_info, this);
    } else {
      data_ctx =
          std::make_shared<NormalExpandFlowUnitDataContext>(stream_info, this);
    }

  } else {
    if (GetFlowType() == STREAM) {
      data_ctx = std::make_shared<StreamFlowUnitDataContext>(stream_info, this);
    } else {
      data_ctx = std::make_shared<NormalFlowUnitDataContext>(stream_info, this);
    }
  }

  return data_ctx;
}

std::shared_ptr<FlowUnitDataContext> Node::GetDataContextFromKey(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list,
    std::shared_ptr<IndexBuffer> buffer) {
  std::shared_ptr<BufferGroup> key = nullptr;
  auto stream_info = buffer->GetStreamLevelGroup();
  if (GetFlowType() == STREAM) {
    if (GetOutputType() == COLLAPSE) {
      key = buffer->GetStreamLevelGroup()->GetOneLevelGroup();
    } else {
      key = buffer->GetStreamLevelGroup();
    }
  } else {
    if (GetOutputType() == EXPAND) {
      key = buffer->GetSameLevelGroup();
    } else {
      key = buffer->GetStreamLevelGroup();
    }
  }

  auto data_context_iter = data_context_map_.find(key);
  if (data_context_iter == data_context_map_.end()) {
    auto new_data_ctx = GenDataContext(stream_info);
    data_context_map_.emplace(key, new_data_ctx);
  }

  auto data_ctx = data_context_map_[key];
  return data_ctx;
}

void Node::SetDataContextInput(std::shared_ptr<FlowUnitDataContext> data_ctx,
                               std::shared_ptr<InputData>& input_data,
                               uint32_t index) {
  if (GetOutputType() == EXPAND && GetFlowType() == NORMAL) {
    auto input_spilt = std::make_shared<InputData>();
    auto origin_data_iter = input_data->begin();
    while (origin_data_iter != input_data->end()) {
      auto key = origin_data_iter->first;
      auto index_buffer_list = origin_data_iter->second;
      auto split_index_buffer_list = std::make_shared<IndexBufferList>();
      split_index_buffer_list->PushBack(index_buffer_list->GetBuffer(index));
      input_spilt->emplace(key, split_index_buffer_list);
      origin_data_iter++;
    }
    data_ctx->SetInputData(input_spilt);
  } else {
    data_ctx->SetInputData(input_data);
  }
}

void Node::GenDataContextFromInput(
    std::shared_ptr<InputData>& input_data,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  auto first_buffer_list = input_data->begin()->second;
  for (uint32_t j = 0; j < first_buffer_list->GetBufferNum(); j++) {
    auto buffer = first_buffer_list->GetBuffer(j);
    auto data_ctx = GetDataContextFromKey(data_ctx_list, buffer);
    SetDataContextInput(data_ctx, input_data, j);

    bool insert_flag = true;
    for (auto& exist_data_ctx : data_ctx_list) {
      if (data_ctx == exist_data_ctx) {
        insert_flag = false;
        continue;
      }
    }

    if (data_ctx->GetInputs().size() == 0) {
      insert_flag = false;
    }

    if (insert_flag) {
      data_ctx_list.push_back(data_ctx);
    }

    if (GetOutputType() != EXPAND || GetFlowType() != NORMAL) {
      break;
    }
  }
}

Status Node::FillDataContext(
    std::vector<std::shared_ptr<InputData>>& input_data_list,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  for (uint32_t i = 0; i < input_data_list.size(); i++) {
    GenDataContextFromInput(input_data_list.at(i), data_ctx_list);
  }
  return STATUS_OK;
}

Status Node::GenerateDataContext(
    InputIndexBuffer& input_buffer,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  auto input_size = input_buffer.begin()->second.size();
  // if we receive no buffer,return immediately
  if (input_size == 0) {
    return STATUS_SUCCESS;
  }

  std::vector<std::shared_ptr<InputData>> input_data_list;
  for (uint32_t i = 0; i < input_size; i++) {
    auto input_data = std::make_shared<InputData>();
    input_data_list.push_back(input_data);
  }

  for (auto& input_iter : input_buffer) {
    auto key = input_iter.first;
    auto data_list = input_iter.second;
    for (uint32_t i = 0; i < input_size; i++) {
      input_data_list.at(i)->emplace(key, data_list.at(i));
    }
  }

  FillDataContext(input_data_list, data_ctx_list);

  return STATUS_SUCCESS;
}

Status Node::RecvData(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  auto input_buffer = CreateInputBuffer();

  auto status = RecvDataQueue(&input_buffer);
  if (status != STATUS_SUCCESS) {
    MBLOG_WARN << "node '" << name_ << "' recv data failed: " << status << "!";
    return {status, name_ + " recve data failed."};
  }

  GenerateDataContext(input_buffer, data_ctx_list);

  return STATUS_SUCCESS;
}

Status Node::GenDataContextFromEvent(
    std::shared_ptr<modelbox::FlowUnitInnerEvent> event,
    std::shared_ptr<FlowUnitDataContext>& data_context) {
  auto bg = event->GetBufferGroup();
  if (data_context_map_.find(bg) == data_context_map_.end()) {
    return STATUS_SUCCESS;
  }
  data_context = data_context_map_.find(bg)->second;

  if (GetOutputType() == COLLAPSE) {
    if (event->GetEventCode() == FlowUnitInnerEvent::COLLAPSE_NEXT_STREAM) {
      auto ctx = std::dynamic_pointer_cast<StreamCollapseFlowUnitDataContext>(
          data_context);
      ctx->UpdateCurrentOrder();
      ctx->CollapseNextStream();
    } else {
      return {STATUS_INVALID, "cannot collapse data"};
    }
  } else if (GetOutputType() == EXPAND) {
    if (event->GetEventCode() == FlowUnitInnerEvent::EXPAND_UNFINISH_DATA) {
      data_context->SetEvent(event->GetUserEvent());
    } else if (event->GetEventCode() ==
               FlowUnitInnerEvent::EXPAND_NEXT_STREAM) {
      auto ctx = std::dynamic_pointer_cast<StreamExpandFlowUnitDataContext>(
          data_context);
      ctx->UpdateCurrentOrder();
      ctx->ExpandNextStream();
    } else {
      return {STATUS_INVALID, "cannot expand data"};
    }
  } else if (!IsStreamSameCount()) {
    if (event->GetEventCode() == FlowUnitInnerEvent::EXPAND_UNFINISH_DATA) {
      data_context->SetEvent(event->GetUserEvent());
    }
  } else {
    return {STATUS_INVALID, "data mode is invalid."};
  }

  return STATUS_SUCCESS;
}

Status Node::RecvEvent(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  FlowunitEventList events = nullptr;

  auto status = event_port_->Recv(events);
  if (!events || events->empty()) {
    MBLOG_DEBUG << "can not receive any event for port.";
    return STATUS_SUCCESS;
  }

  event_port_->NotifyPopEvent();

  for (uint32_t i = 0; i < events->size(); i++) {
    auto event = events->at(i);
    std::shared_ptr<FlowUnitDataContext> data_ctx;
    auto status = GenDataContextFromEvent(event, data_ctx);
    if (status != STATUS_SUCCESS) {
      return status;
    }

    if ((data_ctx != nullptr) && ((data_ctx->GetInputs().size() != 0) ||
                                  (data_ctx->Event() != nullptr))) {
      data_ctx_list.push_back(data_ctx);
    }
  }

  return STATUS_SUCCESS;
}

void Node::ClearDataContext(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  for (auto& data_context : data_ctx_list) {
    data_context->ClearData();
  }

  auto data_context_iter = data_context_map_.begin();
  while (data_context_iter != data_context_map_.end()) {
    if (data_context_iter->second->IsFinished()) {
      auto sess_ctx = data_context_iter->second->GetSessionContext();
      if (sess_ctx != nullptr && GetFlowType() == STREAM) {
        MBLOG_INFO << name_
                   << " data_ctx finished se id:" << sess_ctx->GetSessionId();
      }

      data_context_iter = data_context_map_.erase(data_context_iter);
      MBLOG_DEBUG << name_ << " data_context_map_ left size is "
                  << data_context_map_.size() << " single match size is "
                  << GetSingleMatchCache()->GetReceiveBuffer()->size()
                  << " stream match size is "
                  << GetStreamMatchCache()->GetStreamReceiveBuffer()->size()
                  << " stream order size is "
                  << GetStreamMatchCache()->GetStreamOrder()->size();

    } else {
      data_context_iter++;
    }
  }
}

Status Node::Recv(
    RunType type,
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  Status status = STATUS_SUCCESS;
  if (type == DATA) {
    if (GetInputNum() != 0) {
      status = RecvData(data_ctx_list);
    } else {
      status = RecvExternalData(data_ctx_list);
    }
  } else {
    status = RecvEvent(data_ctx_list);
  }
  return status;
}

Status Node::Run(RunType type) {
  auto data_ctx_list = std::list<std::shared_ptr<FlowUnitDataContext>>();
  Status status = Recv(type, data_ctx_list);

  if (status != STATUS_SUCCESS) {
    return status;
  }

  if (data_ctx_list.empty()) {
    return STATUS_OK;
  }

  for (auto& data_ctx : data_ctx_list) {
    auto error = data_ctx->GetInputError();
    if (error != nullptr) {
      data_ctx->SetError(error);
    }
  }

  auto run_status = flowunit_group_->Run(data_ctx_list);
  MBLOG_DEBUG << "run node: " << name_;

  if (run_status != STATUS_SUCCESS) {
    return {run_status, "flowunit group run failed."};
  }

  for (auto& data_ctx : data_ctx_list) {
    data_ctx->CloseStreamIfNecessary();
  }

  if (!GetOutputNames().empty()) {
    auto output_index_buffer = CreateOutputBuffer();
    GenerateOutputIndexBuffer(&output_index_buffer, data_ctx_list);
    status = Send(&output_index_buffer);
    if (status != STATUS_SUCCESS) {
      return status;
    }
  } else {
    for (auto& data_ctx : data_ctx_list) {
      auto sess_ctx = data_ctx->GetSessionContext();
      if (data_ctx->IsFinished()) {
        sess_ctx->SetError(data_ctx->GetError());
        sess_ctx->UnBindExtenalData();
      }
    }
  }

  ClearDataContext(data_ctx_list);

  return STATUS_SUCCESS;
}

std::shared_ptr<ExternalData> Node::CreateExternalData(
    std::shared_ptr<Device> device) {
  auto port = GetExternalPort(EXTERNAL_PORT_NAME);
  if (!port) {
    MBLOG_WARN << "invalid name, can not find " << port->GetName();
    return nullptr;
  }

  return std::make_shared<ExternalDataImpl>(port, device, graph_stats_);
}

void Node::PostProcessEvent(
    std::list<std::shared_ptr<FlowUnitDataContext>>& data_ctx_list) {
  std::vector<std::shared_ptr<FlowUnitInnerEvent>> event_vector;
  for (auto& data_ctx : data_ctx_list) {
    auto event = data_ctx->GenerateSendEvent();
    if (event != nullptr) {
      event_vector.push_back(event);
    }
  }
  SendBatchEvent(event_vector);
}

FlowOutputType Node::GetOutputType() { return output_type_; }

FlowType Node::GetFlowType() { return flow_type_; }

ConditionType Node::GetConditionType() {
  if (output_type_ == ORIGIN && flow_type_ == NORMAL) {
    return condition_type_;
  } else {
    return NONE;
  }
}

LoopType Node::GetLoopType() { return loop_type_; }

bool Node::IsStreamSameCount() {
  if (output_type_ == ORIGIN && flow_type_ == STREAM) {
    return is_stream_same_count_;
  } else {
    return true;
  }
}

void Node::SetStreamSameCount(bool is_stream_same_count) {
  if (output_type_ == ORIGIN && flow_type_ == STREAM) {
    is_stream_same_count_ = is_stream_same_count;
  }
}

bool Node::IsInputContiguous() { return is_input_contiguous_; }

void Node::SetInputContiguous(bool is_input_contiguous) {
  is_input_contiguous_ = is_input_contiguous;
}

void Node::SetOutputType(FlowOutputType type) { output_type_ = type; }

void Node::SetFlowType(FlowType type) { flow_type_ = type; }

void Node::SetConditionType(ConditionType type) {
  if (output_type_ == ORIGIN && flow_type_ == NORMAL) {
    condition_type_ = type;
  }
}

void Node::SetLoopType(LoopType type) { loop_type_ = type; }

}  // namespace modelbox
