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


#include "modelbox/virtual_node.h"

#include <modelbox/session_context.h>
#include <stdint.h>

namespace modelbox {

ExternalDataMapImpl::ExternalDataMapImpl(
    std::shared_ptr<NodeBase> input_node, std::shared_ptr<NodeBase> output_node,
    std::shared_ptr<StatisticsItem> graph_stats) {
  MBLOG_DEBUG << "ExternalDataMapImpl";
  virtual_stream_ = std::make_shared<VirtualStream>(nullptr, 0);
  auto session_context = std::make_shared<SessionContext>(graph_stats);
  virtual_stream_->SetSessionContext(session_context);
  session_context_ = session_context;
  device_ = input_node->GetDevice();
  for (auto& ext_port : input_node->GetExternalPorts()) {
    input_ports_[ext_port->GetName()] = ext_port;
    input_buffer_cache_[ext_port->GetName()] =
        std::list<std::shared_ptr<Buffer>>();
  }
  output_buffer_cache_ = std::make_shared<BlockingQueue<OutputBufferList>>();
  error_ = nullptr;
  end_flag_ = false;
}

void ExternalDataMapImpl::Init() {
  auto session_context = session_context_.lock();
  if (session_context == nullptr) {
    return;
  }
  session_context->SetExternalData(shared_from_this());
}

Status ExternalDataMapImpl::SetOutputMeta(std::string port_name,
                                          std::shared_ptr<DataMeta> meta) {
  if (input_meta_.find(port_name) != input_meta_.end()) {
    return STATUS_NOTFOUND;
  }
  input_meta_.emplace(port_name, meta);
  return STATUS_OK;
}

void ExternalDataMapImpl::UpdateInputMeta(
    std::string port_name, std::shared_ptr<IndexBufferList> index_buffer_list) {
  if (input_meta_.find(port_name) == input_meta_.end()) {
    return;
  }

  if (index_buffer_list != nullptr) {
    index_buffer_list->SetDataMeta(input_meta_[port_name]);
  }
}

void ExternalDataMapImpl::SetSelector(
    std::shared_ptr<ExternalDataSelect> selector) {
  if (selector == nullptr) {
    return;
  }
  selector_ = selector;
}

bool ExternalDataMapImpl::GetReadyFlag() {
  if (end_flag_) {
    return true;
  }
  return !(output_buffer_cache_->Empty());
}

void ExternalDataMapImpl::UnbindSession() {
  std::unique_lock<std::mutex> guard(lock_);
  virtual_stream_ = nullptr;
}

std::shared_ptr<BufferList> ExternalDataMapImpl::CreateBufferList() {
  if (!device_) {
    MBLOG_ERROR << "device_ must not be nullptr";
    return nullptr;
  }

  return std::make_shared<BufferList>(device_);
}

Status ExternalDataMapImpl::Send(std::string port_name,
                                 std::shared_ptr<BufferList> buffer_list) {
  // before we close the ExternalDataMap we don't really send the data we only
  // cache them
  if (input_buffer_cache_.find(port_name) == input_buffer_cache_.end()) {
    return {STATUS_INVALID, "Send Port " + port_name + " is not exist"};
  }
  OriginDataMap data;
  for (auto& buffer : *buffer_list) {
    input_buffer_cache_[port_name].push_back(buffer);
  }

  uint32_t min_len = UINT32_MAX;
  for (auto output_iter : input_buffer_cache_) {
    auto len = output_iter.second.size();
    if (len < min_len) {
      min_len = len;
    }
  }

  for (auto& output_iter : input_buffer_cache_) {
    auto key = output_iter.first;
    auto& buffer_list = output_iter.second;
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    uint32_t i = 0;
    auto buffer_iter = buffer_list.begin();
    while (i < min_len) {
      buffer_vector.push_back(*buffer_iter);
      buffer_iter = buffer_list.erase(buffer_iter);
      i++;
    }
    data.emplace(key, std::make_shared<BufferList>(buffer_vector));
  }

  auto status = SendData(data);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  return STATUS_OK;
};

Status ExternalDataMapImpl::Recv(OutputBufferList& map_buffer_list,
                                 int timeout) {
  if (output_buffer_cache_ == nullptr) {
    return STATUS_NODATA;
  }

  std::vector<OutputBufferList> output_bufferlist_vector;

  auto size = output_buffer_cache_->Pop(&output_bufferlist_vector, timeout);
  if (size == 0) {
    if (end_flag_) {
      MBLOG_DEBUG << "output_buffer_cache_ pop "
                  << output_buffer_cache_->Size();
      auto external_selector = selector_.lock();
      if (external_selector != nullptr) {
        auto extenral =
            std::dynamic_pointer_cast<ExternalDataMap>(shared_from_this());
        external_selector->RemoveExternalData(extenral);
      }
      if (error_ == nullptr) {
        return STATUS_EOF;
      } else {
        return STATUS_INVALID;
      }
    }

    if (errno == ETIMEDOUT) {
      return STATUS_TIMEDOUT;
    }

    return STATUS_SUCCESS;
  }

  for (auto output_bufferlist : output_bufferlist_vector) {
    if (output_bufferlist.size() == 0) {
      map_buffer_list.clear();
      return STATUS_NODATA;
    }

    auto bufferlist_iter = output_bufferlist.begin();
    while (bufferlist_iter != output_bufferlist.end()) {
      auto key = bufferlist_iter->first;
      auto buffer_list = bufferlist_iter->second;
      if (map_buffer_list.find(key) == map_buffer_list.end()) {
        map_buffer_list.emplace(key, std::make_shared<BufferList>());
      }

      for (uint32_t i = 0; i < buffer_list->Size(); i++) {
        if (map_buffer_list[key] != nullptr) {
          map_buffer_list[key]->PushBack(buffer_list->At(i));
        }
      }
      bufferlist_iter++;
    }
  }
  return STATUS_SUCCESS;
}

Status ExternalDataMapImpl::Close() {
  std::unique_lock<std::mutex> guard(lock_);
  if (virtual_stream_ == nullptr) {
    return {STATUS_INVALID, "The extenral data is already closed"};
  }

  auto port_iter = input_ports_.begin();
  while (port_iter != input_ports_.end()) {
    auto ext_port = port_iter->second;

    auto buf_1 = std::make_shared<Buffer>();
    auto error = std::make_shared<FlowUnitError>("EOF");
    buf_1->SetError(error);
    std::vector<std::shared_ptr<IndexBuffer>> index_bf_list(1);
    index_bf_list[0] = std::make_shared<IndexBuffer>(buf_1);
    auto last_bg =
        virtual_stream_->GetLastBufferGroup()->GenerateSameLevelGroup();
    index_bf_list[0]->SetBufferGroup(last_bg);
    auto index_buffer_list = std::make_shared<IndexBufferList>(index_bf_list);
    auto error_index = index_buffer_list->GetDataErrorIndex();
    last_bg->GetStreamLevelGroup()->SetDataError(error_index, error);
    ext_port->Send(index_buffer_list->GetBuffer(0));
    port_iter++;
  }
  auto ext_port = input_ports_.begin()->second;
  ext_port->NotifyPushEvent();
  virtual_stream_ = nullptr;

  MBLOG_INFO << "external data close";

  return STATUS_OK;
}

std::shared_ptr<SessionContext> ExternalDataMapImpl::GetSessionContext() {
  return session_context_.lock();
};

Status ExternalDataMapImpl::SendData(OriginDataMap& data) {
  std::unique_lock<std::mutex> guard(lock_);
  auto out_put_rings = std::make_shared<OutputRings>(data);
  auto status = out_put_rings->IsValid();
  if (status != STATUS_SUCCESS) {
    return status;
  }

  auto first_buffer_list = out_put_rings->GetOneBufferList();

  if (virtual_stream_ == nullptr) {
    return {STATUS_INVALID, "virtual_stream already closed"};
  }

  if (virtual_stream_->IsClosed()) {
    return {STATUS_INVALID, "virtual_stream already closed"};
  }
  virtual_stream_->LabelIndexBuffer(first_buffer_list);
  out_put_rings->BroadcastMetaToAll();
  auto port_iter = input_ports_.begin();
  while (port_iter != input_ports_.end()) {
    auto key = port_iter->first;
    auto ext_port = port_iter->second;
    auto index_buffer_list = out_put_rings->GetBufferList(key);
    UpdateInputMeta(key, index_buffer_list);
    for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
      ext_port->Send(index_buffer_list->GetBuffer(i));
    }
    port_iter++;
  }
  auto ext_port = input_ports_.begin()->second;
  ext_port->NotifyPushEvent();
  return STATUS_OK;
}

Status ExternalDataMapImpl::SetOutputBuffer(OutputBufferList& output) {
  auto size = output_buffer_cache_->Size();
  if (!output_buffer_cache_->Push(output)) {
    return STATUS_INVALID;
  }

  if (size == 0) {
    auto selector = selector_.lock();
    if (selector != nullptr) {
      selector->NotifySelect();
    }
  }

  return STATUS_OK;
}

void ExternalDataMapImpl::SetEndError(std::shared_ptr<FlowUnitError> error) {
  end_flag_ = true;
  MBLOG_INFO << "ExternalDataMapImpl end_flag  true";
  error_ = error;
  auto selector = selector_.lock();
  if (selector != nullptr) {
    selector->NotifySelect();
  }
  output_buffer_cache_->Shutdown();
}

std::shared_ptr<FlowUnitError> ExternalDataMapImpl::GetLastError() {
  return error_;
}

Status ExternalDataMapImpl::Shutdown() {
  // when we shutdown the external data,we should send a last placeholder buffer
  // Prepare the buffer
  std::unique_lock<std::mutex> guard(lock_);
  OriginDataMap data;
  for (auto& output_iter : input_buffer_cache_) {
    auto key = output_iter.first;
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    data.emplace(key, std::make_shared<BufferList>(buffer_vector));
  }
  auto out_put_rings = std::make_shared<OutputRings>(data);
  out_put_rings->BackfillOutput(std::set<uint32_t>{0});

  auto first_buffer_list = out_put_rings->GetOneBufferList();

  if (virtual_stream_->IsClosed()) {
    return {STATUS_INVALID, "virtual_stream already closed"};
  }

  virtual_stream_->LabelIndexBuffer(first_buffer_list);
  out_put_rings->BroadcastMetaToAll();
  virtual_stream_->Close();

  // Send the end buffer

  auto port_iter = input_ports_.begin();
  while (port_iter != input_ports_.end()) {
    auto key = port_iter->first;
    auto ext_port = port_iter->second;
    auto index_buffer_list = out_put_rings->GetBufferList(key);
    UpdateInputMeta(key, index_buffer_list);
    for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
      ext_port->Send(index_buffer_list->GetBuffer(i));
    }
    port_iter++;
  }
  auto ext_port = input_ports_.begin()->second;
  ext_port->NotifyPushEvent();

  return STATUS_OK;
};

std::shared_ptr<Configuration> ExternalDataMapImpl::GetSessionConfig() {
  auto session_context = session_context_.lock();
  if (session_context == nullptr) {
    return nullptr;
  }
  return session_context->GetConfig();
};

ExternalDataSelect::ExternalDataSelect() {}

ExternalDataSelect::~ExternalDataSelect() {}

void ExternalDataSelect::RegisterExternalData(
    std::shared_ptr<ExternalDataMap> externl) {
  std::shared_ptr<ExternalDataMapImpl> externl_data =
      std::dynamic_pointer_cast<ExternalDataMapImpl>(externl);
  external_list_.push_back(externl_data);
  externl_data->SetSelector(shared_from_this());
}

void ExternalDataSelect::RemoveExternalData(
    std::shared_ptr<ExternalDataMap>& externl_data) {
  auto iter = external_list_.begin();
  while (iter != external_list_.end()) {
    if (*iter == std::dynamic_pointer_cast<ExternalDataMapImpl>(externl_data)) {
      iter = external_list_.erase(iter);
      break;
    } else {
      iter++;
    }
  }
}

bool ExternalDataSelect::IsExtenalDataReady() {
  bool ready_flag = false;
  for (auto external_data : external_list_) {
    if (external_data->GetReadyFlag()) {
      ready_flag = true;
      break;
    }
  }
  return ready_flag;
}

Status ExternalDataSelect::SelectExternalData(
    std::list<std::shared_ptr<ExternalDataMap>>& external_list,
    std::chrono::duration<long, std::milli> waittime) {
  MBLOG_DEBUG << "SelectExternalData";
  std::unique_lock<std::mutex> lck(mtx_);
  if (external_list_.empty()) {
    if (waittime < std::chrono::milliseconds(0)) {
      cv_.wait(lck);
    } else {
      if (cv_.wait_for(lck, waittime) == std::cv_status::timeout) {
        return STATUS_TIMEDOUT;
      }
    }
  }

  bool ready_flag = IsExtenalDataReady();

  while (!ready_flag) {
    if (waittime <= std::chrono::milliseconds(0)) {
      cv_.wait(lck);
    } else {
      if (cv_.wait_for(lck, waittime) == std::cv_status::timeout) {
        return STATUS_TIMEDOUT;
      }
    }

    ready_flag = IsExtenalDataReady();
  }

  for (auto external_data : external_list_) {
    if (external_data->GetReadyFlag()) {
      external_list.push_back(external_data);
    }
  }
  return STATUS_SUCCESS;
}

void ExternalDataSelect::NotifySelect() {
  std::unique_lock<std::mutex> lck(mtx_);
  cv_.notify_one();
  MBLOG_DEBUG << "NotifySelect";
}

InputVirtualNode::InputVirtualNode(
    const std::string& device_name, const std::string& device_id,
    std::shared_ptr<DeviceManager> device_manager)
    : device_name_(device_name), device_id_(device_id) {
  queue_size_ = -1;
  priority_ = 0;
  is_input_order_ = true;
  need_garther_all_ = false;
  device_mgr_ = device_manager;
}

InputVirtualNode::~InputVirtualNode() {}

Status InputVirtualNode::Init(const std::set<std::string>& input_port_names,
                              const std::set<std::string>& output_port_names,
                              std::shared_ptr<Configuration> config) {
  auto status =
      DataMatcherNode::Init(input_port_names, output_port_names, config);
  if (status != STATUS_SUCCESS) {
    return status;
  }

  auto ext_queue_size = config->GetUint64("queue_size_external", queue_size_);
  for (auto& output_port_name : output_port_names) {
    auto port = std::make_shared<InPort>(
        output_port_name,
        std::dynamic_pointer_cast<NodeBase>(shared_from_this()), GetPriority(),
        ext_queue_size);
    extern_ports_.emplace_back(port);
  }

  for (auto& port : extern_ports_) {
    port->Init();
  }

  return STATUS_OK;
}

Status InputVirtualNode::Open() { return STATUS_OK; }

std::shared_ptr<Device> InputVirtualNode::GetDevice() {
  if (device_mgr_ == nullptr) {
    MBLOG_ERROR << "device_mgr is nullptr ";
    return nullptr;
  }

  auto device = device_mgr_->CreateDevice(device_name_, device_id_);
  if (device == nullptr) {
    MBLOG_ERROR << "device is nullptr."
                << " device_name: " << device_name_
                << " device_id_: " << device_id_;
    return nullptr;
  }
  return device;
}

Status InputVirtualNode::RecvExternalData(InputIndexBuffer& external_buffer) {
  external_buffer = CreateExternalBuffer();

  auto status = RecvExternalDataQueue(&external_buffer);
  if (status != STATUS_SUCCESS) {
    MBLOG_WARN << "Node (" << name_ << ") Recv data failed: " << status << "!";
    return {status, name_ + " recv external failed."};
  }
  return STATUS_SUCCESS;
}

InputIndexBuffer InputVirtualNode::CreateExternalBuffer() {
  auto external_names = GetExternNames();
  InputIndexBuffer external_map;
  for (auto& external_name : external_names) {
    std::vector<std::shared_ptr<IndexBufferList>> input_vector;
    external_map[external_name] = input_vector;
  }
  return external_map;
}

bool InputVirtualNode::ExternalToOutput(InputIndexBuffer& ext_buffer,
                                        OutputIndexBuffer& output) {
  auto ext_names = GetExternNames();
  if (ext_names.empty()) {
    return true;
  }

  for (auto& index_buffer_list_vec : ext_buffer) {
    auto name = index_buffer_list_vec.first;
    std::vector<std::shared_ptr<IndexBuffer>> output_vector;
    for (auto& index_buffer_list : index_buffer_list_vec.second) {
      auto index_buffer_vec = index_buffer_list->GetIndexBufferVector();
      output_vector.insert(output_vector.end(), index_buffer_vec.begin(),
                           index_buffer_vec.end());
    }
    output[index_buffer_list_vec.first] = output_vector;
  }
  return true;
}

Status InputVirtualNode::Run(RunType type) {
  InputIndexBuffer input_buffer;
  InputIndexBuffer ext_buffer;

  auto status = RecvExternalData(ext_buffer);
  if (!status) {
    MBLOG_ERROR << "recv external data fail. " << status.Errormsg();
    return status;
  }
  OutputIndexBuffer ext_to_output_buffer;
  ExternalToOutput(ext_buffer, ext_to_output_buffer);

  status = Send(&ext_to_output_buffer);
  if (status != STATUS_SUCCESS) {
    return status;
  }

  return STATUS_SUCCESS;
}

OutputVirtualNode::OutputVirtualNode(
    const std::string& device_name, const std::string& device_id,
    std::shared_ptr<DeviceManager> device_manager)
    : device_name_(device_name), device_id_(device_id) {
  queue_size_ = -1;
  priority_ = 0;
  is_input_order_ = true;
  need_garther_all_ = false;
  device_mgr_ = device_manager;
}

OutputVirtualNode::~OutputVirtualNode() {}

Status OutputVirtualNode::Init(const std::set<std::string>& input_port_names,
                               const std::set<std::string>& output_port_names,
                               std::shared_ptr<Configuration> config) {
  auto status =
      DataMatcherNode::Init(input_port_names, output_port_names, config);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  return STATUS_SUCCESS;
}

Status OutputVirtualNode::Open() { return STATUS_SUCCESS; }

Status OutputVirtualNode::RecvData(InputIndexBuffer& input_buffer) {
  input_buffer = CreateInputBuffer();

  auto status = RecvDataQueue(&input_buffer);

  if (status != STATUS_SUCCESS) {
    MBLOG_WARN << "Node (" << name_ << ") Recv data failed: " << status << "!";
    return {status, name_ + " recve data failed."};
  }

  return STATUS_SUCCESS;
}

Status OutputVirtualNode::Run(RunType type) {
  Status status;

  InputIndexBuffer input_buffer;
  InputIndexBuffer ext_buffer;

  status = RecvData(input_buffer);
  if (!status) {
    MBLOG_ERROR << "recv data fail. " << status.Errormsg();
    return status;
  }

  auto buffer_list_num = input_buffer.begin()->second.size();
  if (buffer_list_num == 0) {
    return STATUS_OK;
  }
  std::vector<OutputBufferList> data_vector(buffer_list_num);
  std::vector<std::shared_ptr<SessionContext>> sess_ctx_vector(buffer_list_num);
  std::vector<std::shared_ptr<FlowUnitError>> error_vector(buffer_list_num);
  std::vector<bool> flag_vector(buffer_list_num);

  for (auto input_buffer_iter : input_buffer) {
    auto key = input_buffer_iter.first;
    auto buffer_vector = input_buffer_iter.second;
    auto buffer_size = buffer_vector.size();

    auto last_buffer = buffer_vector[buffer_size - 1];

    for (uint32_t i = 0; i < buffer_size; i++) {
      auto session_ctx = buffer_vector[i]
                             ->GetBuffer(0)
                             ->GetStreamLevelGroup()
                             ->GetSessionContext();
      auto last_buffer =
          buffer_vector[i]->GetBuffer(buffer_vector[i]->GetBufferNum() - 1);
      session_ctx->SetError(buffer_vector[i]->GetDataError());

      auto flag = last_buffer->GetSameLevelGroup()->IsEndGroup();
      if (flag) {
        session_ctx->UnBindExtenalData();
      }
      sess_ctx_vector[i] = session_ctx;
      auto buffer_vec = buffer_vector[i]->GetBufferPtrList();
      data_vector[i][key] = std::make_shared<BufferList>(buffer_vec);
    }
  }
  input_buffer.clear();

  for (uint32_t i = 0; i < buffer_list_num; i++) {
    sess_ctx_vector[i]->SetOutputBuffer(data_vector[i]);
  }
  sess_ctx_vector.clear();

  return STATUS_SUCCESS;
}

std::shared_ptr<Device> OutputVirtualNode::GetDevice() {
  if (device_mgr_ == nullptr) {
    MBLOG_ERROR << "device_mgr is nullptr ";
    return nullptr;
  }

  auto device = device_mgr_->CreateDevice(device_name_, device_id_);
  if (device == nullptr) {
    MBLOG_ERROR << "device is nullptr."
                << " device_name: " << device_name_
                << " device_id_: " << device_id_;
    return nullptr;
  }
  return device;
}

MultiLevelCache::MultiLevelCache() : order_(1) {}

MultiLevelCache::~MultiLevelCache() {}

std::string GenerateKeyFromSeq(std::vector<uint32_t>& order_seq) {
  std::string result = "";
  for (auto order : order_seq) {
    result = result + std::to_string(order) + "_";
  }
  return result;
}

void MultiLevelCache::PushBack(std::shared_ptr<IndexBuffer>& buffer) {
  auto bg = buffer->GetSameLevelGroup()->GetOneLevelGroup();
  if (cache_.find(bg) == cache_.end()) {
    cache_.emplace(bg, std::vector<std::shared_ptr<IndexBuffer>>());
  }

  cache_[bg].emplace_back(buffer);
  auto seq_order = buffer->GetSeqOrder();

  if (cur_order_seq_.size() == 0) {
    cur_order_seq_ = std::vector<uint32_t>(seq_order.size(), 1);
  }

  auto key = GenerateKeyFromSeq(seq_order);
  if (key_bg_map_.find(key) == key_bg_map_.end()) {
    key_bg_map_.emplace(key, bg);
  }
}

bool MultiLevelCache::PopOneGroup(
    std::string cur_key, std::vector<std::shared_ptr<Buffer>>& buffer_vector) {
  auto bg = key_bg_map_[cur_key];
  auto& index_buffer_vector = cache_[bg];
  bool result = false;

  if (index_buffer_vector.size() == 0) {
    return result;
  }

  auto index_buffer_list =
      std::make_shared<IndexBufferList>(index_buffer_vector);
  error_ = index_buffer_list->GetDataError();

  std::sort(index_buffer_vector.begin(), index_buffer_vector.end(),
            [](std::shared_ptr<IndexBuffer> a, std::shared_ptr<IndexBuffer> b) {
              return (a->GetSameLevelGroup()->GetOrder() <
                      b->GetSameLevelGroup()->GetOrder());
            });

  while (!index_buffer_vector.empty()) {
    auto iter = index_buffer_vector.begin();
    if ((*iter)->GetSameLevelGroup()->GetOrder() == order_) {
      order_++;
      if (!(*iter)->IsPlaceholder()) {
        buffer_vector.emplace_back((*iter)->GetBufferPtr());
      }
      if ((*iter)->GetSameLevelGroup()->IsEndGroup()) {
        result = true;
        order_ = 1;
      }
      iter = index_buffer_vector.erase(iter);

    } else {
      break;
    }
  }
  return result;
}

bool MultiLevelCache::UpdateNextGroup() {
  uint32_t i = 0;
  auto cur_key = GenerateKeyFromSeq(cur_order_seq_);
  auto bg = key_bg_map_[cur_key];
  while (bg->IsEndGroup()) {
    if (bg->IsRoot()) {
      return true;
    }
    bg = bg->GetOneLevelGroup();
    cur_order_seq_[i] = 1;
    i++;
  }
  cur_order_seq_[i]++;
  return false;
}

bool MultiLevelCache::PopOut(
    std::vector<std::shared_ptr<Buffer>>& buffer_vector) {
  while (true) {
    auto cur_key = GenerateKeyFromSeq(cur_order_seq_);

    if (key_bg_map_.find(cur_key) == key_bg_map_.end()) {
      return false;
    }

    if (PopOneGroup(cur_key, buffer_vector)) {
      if (UpdateNextGroup()) {
        return true;
      }
    } else {
      return false;
    }
  }
}

std::shared_ptr<FlowUnitError> MultiLevelCache::GetError() { return error_; }

OutputUnmatchVirtualNode::OutputUnmatchVirtualNode(
    const std::string& device_name, const std::string& device_id,
    std::shared_ptr<DeviceManager> device_manager)
    : device_name_(device_name), device_id_(device_id) {
  queue_size_ = -1;
  priority_ = 0;
  device_mgr_ = device_manager;
}

OutputUnmatchVirtualNode::~OutputUnmatchVirtualNode() {}

Status OutputUnmatchVirtualNode::Init(
    const std::set<std::string>& input_port_names,
    const std::set<std::string>& output_port_names,
    std::shared_ptr<Configuration> config) {
  return NodeBase::Init(input_port_names, output_port_names, config);
}

Status OutputUnmatchVirtualNode::Open() { return STATUS_SUCCESS; }

Status OutputUnmatchVirtualNode::Run(RunType type) {
  Status status;

  InputIndexBuffer input_buffer = CreateInputBuffer();

  for (auto& input_port : input_ports_) {
    std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
    input_port->Recv(buffer_vector, -1);

    for (auto buffer : buffer_vector) {
      auto session_ctx = buffer->GetStreamLevelGroup()->GetSessionContext();
      if (buffer->GetSameLevelGroup()->IsEndGroup()) {
        MBLOG_ERROR << "recv data fail. ";
      }

      if (cache_map_.find(session_ctx) == cache_map_.end()) {
        std::unordered_map<std::string, std::shared_ptr<MultiLevelCache>>
            map_cache;
        cache_map_.emplace(session_ctx, map_cache);
      }

      if (cache_map_[session_ctx].find(input_port->GetName()) ==
          cache_map_[session_ctx].end()) {
        auto multi_level_cache = std::make_shared<MultiLevelCache>();
        cache_map_[session_ctx][input_port->GetName()] = multi_level_cache;
      }
      cache_map_[session_ctx][input_port->GetName()]->PushBack(buffer);
    }
  }

  auto cache_iter = cache_map_.begin();
  while (cache_iter != cache_map_.end()) {
    auto session_ctx = cache_iter->first;
    if (session_sucess_count_.find(session_ctx) ==
        session_sucess_count_.end()) {
      session_sucess_count_[session_ctx] = 0;
    }
    OutputBufferList output_buffer_list;
    auto input_cache_map = cache_iter->second;
    std::shared_ptr<FlowUnitError> error;

    for (auto input_cache : input_cache_map) {
      auto input_name = input_cache.first;
      auto cache = input_cache.second;
      std::vector<std::shared_ptr<Buffer>> buffer_vector;
      auto flag = cache->PopOut(buffer_vector);

      if (flag) {
        error = cache->GetError();
        session_ctx->SetError(error);
        session_sucess_count_[session_ctx]++;
      }
      auto buffer_list = std::make_shared<BufferList>(buffer_vector);
      output_buffer_list.emplace(input_name, buffer_list);
    }

    if (session_sucess_count_[session_ctx] == GetInputNum()) {
      session_sucess_count_.erase(session_ctx);
      cache_iter = cache_map_.erase(cache_iter);
      session_ctx->UnBindExtenalData();
    } else {
      cache_iter++;
    }

    session_ctx->SetOutputBuffer(output_buffer_list);
  }

  return STATUS_SUCCESS;
}

std::shared_ptr<Device> OutputUnmatchVirtualNode::GetDevice() {
  if (device_mgr_ == nullptr) {
    MBLOG_ERROR << "device_mgr is nullptr ";
    return nullptr;
  }

  auto device = device_mgr_->CreateDevice(device_name_, device_id_);
  if (device == nullptr) {
    MBLOG_ERROR << "device is nullptr."
                << " device_name: " << device_name_
                << " device_id_: " << device_id_;
    return nullptr;
  }
  return device;
}

}  // namespace modelbox
