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

#include "modelbox/external_data_map.h"

#include "modelbox/node.h"
#include "modelbox/session.h"
#include "modelbox/session_context.h"
#include "modelbox/stream.h"

namespace modelbox {

ExternalDataMapImpl::ExternalDataMapImpl(std::shared_ptr<Node> graph_input_node,
                                         std::shared_ptr<Stream> init_stream)
    : init_stream_(init_stream),
      session_(init_stream_->GetSession()),
      session_ctx_(init_stream->GetSession()->GetSessionCtx()) {
  root_buffer_ = std::make_shared<BufferIndexInfo>();
  root_buffer_->SetStream(init_stream);
  root_buffer_->SetIndex(0);
  graph_input_node_ = graph_input_node;
  graph_input_node_device_ = graph_input_node->GetDevice();
  for (auto& ext_port : graph_input_node->GetExternalPorts()) {
    const auto& port_name = ext_port->GetName();
    graph_input_node_ports_[port_name] = ext_port;
    graph_input_ports_cache_[port_name] = std::list<std::shared_ptr<Buffer>>();
    graph_input_ports_stream_[port_name] =
        std::make_shared<Stream>(init_stream_->GetSession());
  }
  graph_output_cache_ = std::make_shared<BlockingQueue<OutputBufferList>>();
}

std::shared_ptr<BufferList> ExternalDataMapImpl::CreateBufferList() {
  if (!graph_input_node_device_) {
    MBLOG_ERROR << "device_ must not be nullptr";
    return nullptr;
  }

  return std::make_shared<BufferList>(graph_input_node_device_);
}

Status ExternalDataMapImpl::SetOutputMeta(const std::string& port_name,
                                          std::shared_ptr<DataMeta> meta) {
  auto item = graph_input_ports_stream_.find(port_name);
  if (item == graph_input_ports_stream_.end()) {
    return {STATUS_INVALID, "Send Port " + port_name + " is not exist"};
  }

  auto& stream = item->second;
  stream->SetStreamMeta(meta);
  return STATUS_OK;
}

Status ExternalDataMapImpl::Send(const std::string& port_name,
                                 std::shared_ptr<BufferList> buffer_list) {
  std::lock_guard<std::recursive_mutex> lock(close_state_lock_);
  if (close_flag_ || init_stream_ == nullptr) {
    return STATUS_STOP;
  }

  auto ret = PushToInputCache(port_name, buffer_list);
  if (!ret) {
    return ret;
  }

  std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>
      matched_port_data;
  size_t matched_data_size = 0;
  PopMachedInput(matched_port_data, matched_data_size);
  return SendMatchData(matched_port_data, matched_data_size);
};

Status ExternalDataMapImpl::PushToInputCache(
    const std::string& port_name, std::shared_ptr<BufferList> buffer_list) {
  auto item = graph_input_ports_cache_.find(port_name);
  if (item == graph_input_ports_cache_.end()) {
    return {STATUS_INVALID, "Send Port " + port_name + " is not exist"};
  }

  auto& port_cache = item->second;
  for (auto& buffer : *buffer_list) {
    port_cache.push_back(buffer->Copy());
  }

  return STATUS_SUCCESS;
}

void ExternalDataMapImpl::PopMachedInput(
    std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>&
        matched_port_data,
    size_t& matched_data_size) {
  if (graph_input_ports_cache_.empty()) {
    return;
  }

  matched_data_size = SIZE_MAX;
  for (auto& port_data_list_iter : graph_input_ports_cache_) {
    auto data_list_size = port_data_list_iter.second.size();
    if (data_list_size < matched_data_size) {
      matched_data_size = data_list_size;
    }
  }

  for (auto& port_data_list_iter : graph_input_ports_cache_) {
    auto& port_name = port_data_list_iter.first;
    auto& port_data_list = port_data_list_iter.second;
    auto& matched_data_list = matched_port_data[port_name];
    auto end_pos = port_data_list.begin();
    std::advance(end_pos, matched_data_size);
    matched_data_list.splice(matched_data_list.begin(), port_data_list,
                             port_data_list.begin(), end_pos);
  }
}

Status ExternalDataMapImpl::SendMatchData(
    const std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>&
        matched_port_data,
    size_t matched_data_size) {
  if (matched_data_size == 0) {
    return STATUS_SUCCESS;
  }

  for (auto& input_port_data_iter : matched_port_data) {
    auto& port_name = input_port_data_iter.first;
    auto& port_data_list = input_port_data_iter.second;
    auto& port_stream = graph_input_ports_stream_[port_name];
    auto& graph_input_port = graph_input_node_ports_[port_name];
    for (auto& port_data : port_data_list) {
      auto& port_buffer_index_info = port_data->index_info_;
      auto port_buffer_index = port_stream->GetBufferCount();
      port_stream->IncreaseBufferCount();
      port_buffer_index_info->SetIndex(port_buffer_index);
      port_buffer_index_info->SetStream(port_stream);
      auto inherit_info = std::make_shared<BufferInheritInfo>();
      inherit_info->SetInheritFrom(root_buffer_);
      inherit_info->SetType(BufferProcessType::EXPAND);
      port_buffer_index_info->SetInheritInfo(inherit_info);
      graph_input_port->Send(port_data);
    }
  }

  auto& port = graph_input_node_ports_.begin()->second;
  port->NotifyPushEvent();
  return STATUS_SUCCESS;
}

Status ExternalDataMapImpl::Recv(OutputBufferList& map_buffer_list,
                                 int32_t timeout) {
  if (graph_output_cache_ == nullptr) {
    return STATUS_NODATA;
  }

  std::vector<OutputBufferList> output_bufferlist_vector;
  auto size = graph_output_cache_->Pop(&output_bufferlist_vector, timeout);
  if (size == 0) {
    std::lock_guard<std::mutex> lock(session_state_lock_);
    if (!session_end_flag_) {
      return STATUS_OK;
    }

    auto selector = selector_.lock();
    if (selector != nullptr) {
      selector->RemoveExternalData(shared_from_this());
    }

    if (last_error_ == nullptr) {
      return STATUS_EOF;
    }

    return STATUS_INVALID;
  }

  for (auto& output_buffer_list : output_bufferlist_vector) {
    if (output_buffer_list.empty()) {
      continue;
    }

    for (auto& port_data_item : output_buffer_list) {
      auto& port_name = port_data_item.first;
      auto& port_data_list = port_data_item.second;
      std::shared_ptr<BufferList> buffer_list;
      auto out_item = map_buffer_list.find(port_name);
      if (out_item == map_buffer_list.end()) {
        buffer_list = std::make_shared<BufferList>();
        map_buffer_list[port_name] = buffer_list;
      } else {
        buffer_list = out_item->second;
      }

      for (auto& buffer : *port_data_list) {
        buffer_list->PushBack(buffer);
      }
    }
  }

  return STATUS_OK;
}

/**
 * @brief close input stream, wait process
 **/
Status ExternalDataMapImpl::Close() {
  std::lock_guard<std::recursive_mutex> lock(close_state_lock_);
  if (close_flag_) {
    return STATUS_OK;
  }

  close_flag_ = true;
  if (init_stream_ == nullptr) {
    return STATUS_OK;
  }

  // add end buffer
  for (auto& input_node_port_item : graph_input_node_ports_) {
    auto& port_name = input_node_port_item.first;
    auto& port = input_node_port_item.second;
    auto& port_stream = graph_input_ports_stream_[port_name];
    auto end_buffer = std::make_shared<Buffer>();
    auto end_index_info = BufferManageView::GetIndexInfo(end_buffer);
    end_index_info->SetStream(port_stream);
    end_index_info->SetIndex(port_stream->GetBufferCount());
    end_index_info->MarkAsEndFlag();
    port_stream->IncreaseBufferCount();
    auto inherit_info = std::make_shared<BufferInheritInfo>();
    inherit_info->SetInheritFrom(root_buffer_);
    inherit_info->SetType(BufferProcessType::EXPAND);
    end_index_info->SetInheritInfo(inherit_info);
    port->Send(end_buffer);
  }

  auto& port = graph_input_node_ports_.begin()->second;
  port->NotifyPushEvent();
  // clear
  init_stream_ = nullptr;
  root_buffer_ = nullptr;
  graph_input_ports_stream_.clear();
  return STATUS_OK;
}

/**
 * @brief stop task immediately
 **/
Status ExternalDataMapImpl::Shutdown() {
  std::lock_guard<std::recursive_mutex> lock(close_state_lock_);
  if (shutdown_flag_) {
    return STATUS_OK;
  }

  shutdown_flag_ = true;
  auto session = session_.lock();
  if (session == nullptr) {
    return STATUS_OK;
  }

  session->Close();
  Close();  // make sure data end has been sent
  return STATUS_OK;
};

std::shared_ptr<SessionContext> ExternalDataMapImpl::GetSessionContext() {
  return session_ctx_.lock();
};

std::shared_ptr<Configuration> ExternalDataMapImpl::GetSessionConfig() {
  auto ctx = session_ctx_.lock();
  if (ctx == nullptr) {
    return nullptr;
  }

  return ctx->GetConfig();
}

void ExternalDataMapImpl::SetLastError(std::shared_ptr<FlowUnitError> error) {
  last_error_ = error;
}

std::shared_ptr<FlowUnitError> ExternalDataMapImpl::GetLastError() {
  return last_error_;
}

void ExternalDataMapImpl::SetSelector(
    std::shared_ptr<ExternalDataSelect> selector) {
  selector_ = selector;
}

bool ExternalDataMapImpl::GetReadyFlag() {
  if (session_end_flag_) {
    return true;
  }

  return !(graph_output_cache_->Empty());
}

void ExternalDataMapImpl::PushGraphOutputBuffer(OutputBufferList& output) {
  auto size = graph_output_cache_->Size();
  if (!graph_output_cache_->Push(output)) {
    MBLOG_ERROR << "graph save output failed";
    return;
  }

  if (size != 0) {
    return;
  }

  auto selector = selector_.lock();
  if (selector == nullptr) {
    return;
  }

  selector->NotifySelect();
}

void ExternalDataMapImpl::SessionEnd(std::shared_ptr<FlowUnitError> error) {
  {
    std::lock_guard<std::mutex> lock(session_state_lock_);
    if (session_end_flag_) {
      return;
    }

    session_end_flag_ = true;
    last_error_ = error;
  }

  auto selector = selector_.lock();
  if (selector != nullptr) {
    selector->NotifySelect();
  }

  graph_output_cache_->Shutdown();
}

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
    const std::shared_ptr<ExternalDataMap>& externl_data) {
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

}  // namespace modelbox