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

#include "modelbox/match_stream.h"

#include "modelbox/buffer_index_info.h"
#include "modelbox/port.h"
#include "modelbox/session.h"

namespace modelbox {
MatchKey* MatchKey::AsKey(BufferIndexInfo* match_at_buffer) {
  return (MatchKey*)match_at_buffer;
}

MatchKey* MatchKey::AsKey(Stream* match_at_stream) {
  return (MatchKey*)match_at_stream;
}

void MatchStreamData::SetStreamMatchKey(MatchKey* match_at) {
  match_at_ = match_at;
}

MatchKey* MatchStreamData::GetStreamMatchKey() { return match_at_; }

void MatchStreamData::SetSession(std::shared_ptr<Session> session) {
  session_ = session;
}

std::shared_ptr<Session> MatchStreamData::GetSession() { return session_; }

void MatchStreamData::SetEvent(std::shared_ptr<FlowUnitInnerEvent>& event) {
  event_ = event;
}

std::shared_ptr<FlowUnitInnerEvent> MatchStreamData::GetEvent() {
  return event_;
}

void MatchStreamData::SetBufferList(std::shared_ptr<PortDataMap> port_buffers) {
  port_to_stream_data_ = port_buffers;
}

std::shared_ptr<PortDataMap> MatchStreamData::GetBufferList() const {
  return port_to_stream_data_;
}

size_t MatchStreamData::GetDataCount() const {
  auto& first_port_stream_data = port_to_stream_data_->begin()->second;
  return first_port_stream_data.size();
}

MatchBufferCache::MatchBufferCache(
    size_t port_count,
    std::unordered_map<std::string, size_t>* stream_count_each_port)
    : port_count_(port_count),
      stream_count_each_port_(stream_count_each_port) {}

Status MatchBufferCache::CacheBuffer(const std::string& port_name,
                                     std::shared_ptr<Buffer>& buffer) {
  // Check state
  auto buffer_index = BufferManageView::GetIndexInfo(buffer);
  if (!buffer_cache_.empty()) {
    // multi port stream length not equal
    if (is_end_flag_ && !buffer_index->IsEndFlag()) {
      MBLOG_ERROR << "port " << port_name
                  << " missmatch, still has data when other port received end";
      return STATUS_FAULT;
    } else if (!is_end_flag_ && buffer_index->IsEndFlag()) {
      MBLOG_ERROR << "port " << port_name
                  << " missmatch, received end when other port still has data";
      return STATUS_FAULT;
    }
  }

  buffer_cache_[port_name] = buffer;
  is_end_flag_ = buffer_index->IsEndFlag();
  auto& buffer_count_at_port = cur_buffer_count_each_port_[port_name];
  ++buffer_count_at_port;

  if (buffer_index->IsPlaceholder()) {
    is_placeholder_ = true;
  }

  // port data already exist, in case if-else combine end flag
  if (buffer_count_at_port > 1) {
    return STATUS_EXIST;
  }

  return STATUS_OK;
}

bool MatchBufferCache::IsMatched() const {
  if (port_count_ != buffer_cache_.size()) {
    return false;
  }

  if (!is_end_flag_) {
    return true;
  }

  if (stream_count_each_port_->empty()) {
    // has no input port, no need to check
    return true;
  }

  // if combine condition node output, will have multi end_flag received
  for (auto& buffer_count_item : cur_buffer_count_each_port_) {
    auto& port_name = buffer_count_item.first;
    auto buffer_count = buffer_count_item.second;
    auto max_buffer_count = (*stream_count_each_port_)[port_name];
    if (buffer_count < max_buffer_count) {
      return false;
    }
  }

  // all end flag received
  return true;
}

bool MatchBufferCache::IsEndFlag() const { return is_end_flag_; }

const std::unordered_map<std::string, std::shared_ptr<Buffer>>&
MatchBufferCache::GetBuffers() const {
  if (!is_placeholder_) {
    return buffer_cache_;
  }

  // match at placeholder, all buffer mark as placeholder
  for (auto& item : buffer_cache_) {
    auto& buffer = item.second;
    auto index_info = BufferManageView::GetIndexInfo(buffer);
    index_info->MarkAsPlaceholder();
  }
  return buffer_cache_;
}

void InPortStreamInfo::ReceiveBuffer(std::shared_ptr<Buffer>& buffer) {}

size_t InPortStreamInfo::GetReceivedBufferCount() { return 0; }

size_t InPortStreamInfo::GetReceivedStreamCount() { return 0; }

bool InPortStreamInfo::ReachEnd() { return false; }

MatchStreamCache::MatchStreamCache(
    const std::string& node_name, size_t port_count,
    std::unordered_map<std::string, size_t>* stream_count_each_port)
    : node_name_(node_name),
      port_count_(port_count),
      stream_count_each_port_(stream_count_each_port) {}

Status MatchStreamCache::CacheBuffer(const std::string& port_name,
                                     std::shared_ptr<Buffer>& buffer) {
  std::shared_ptr<MatchBufferCache> match_buffer;
  auto buffer_index_info = BufferManageView::GetIndexInfo(buffer);
  size_t buffer_index = 0;

  if ((*stream_count_each_port_)[port_name] > 1) {
    // combine condition result, rewrite input index info
    auto buffer_before_condition =
        buffer_index_info->GetInheritInfo()->GetInheritFrom();
    // should not change origin input, need copy buffer
    buffer = buffer->Copy();
    BufferManageView::SetIndexInfo(buffer, buffer_before_condition);
    buffer_index = buffer_before_condition->GetIndex();
  } else {
    buffer_index = buffer_index_info->GetIndex();
  }

  auto item = match_buffers_.find(buffer_index);
  if (item == match_buffers_.end()) {
    match_buffer = std::make_shared<MatchBufferCache>(port_count_,
                                                      stream_count_each_port_);
    match_buffers_[buffer_index] = match_buffer;
  } else {
    match_buffer = item->second;
  }

  auto ret = match_buffer->CacheBuffer(port_name, buffer);
  if (ret == STATUS_FAULT) {
    MBLOG_ERROR << "node " << node_name_ << " match port " << port_name
                << " failed";
    return ret;
  }
  // ret could be exist, success

  if (!match_buffer->IsMatched()) {
    return ret;
  }

  // move to ready cache
  ready_match_buffers_[buffer_index] = match_buffer;
  match_buffers_.erase(buffer_index);
  ++cur_input_count_in_stream_;

  // check stream end
  if (match_buffer->IsEndFlag()) {
    end_flag_received_ = true;
    total_input_count_in_stream_ = buffer_index + 1;
  }

  return ret;
}

std::shared_ptr<PortDataMap> MatchStreamCache::PopReadyMatchBuffers(
    bool in_order, bool gather_all) {
  std::list<std::shared_ptr<MatchBufferCache>> pop_match_buffers;
  if (gather_all && !(end_flag_received_ && ready_match_buffers_.size() ==
                                                total_input_count_in_stream_)) {
    // not all received
    return nullptr;
  }

  for (auto match_buffer_iter = ready_match_buffers_.begin();
       match_buffer_iter != ready_match_buffers_.end();) {
    auto& buffer_index = match_buffer_iter->first;
    auto& match_buffer = match_buffer_iter->second;

    if (in_order && buffer_index != index_in_order_) {
      break;
    }

    pop_match_buffers.push_back(match_buffer);
    match_buffer_iter = ready_match_buffers_.erase(match_buffer_iter);
    ++index_in_order_;
  }

  auto ready_port_buffers = std::make_shared<PortDataMap>();
  size_t match_buffer_count = pop_match_buffers.size();
  for (auto& match_buffer_cache : pop_match_buffers) {
    auto port_buffer_map = match_buffer_cache->GetBuffers();
    for (auto& item : port_buffer_map) {
      auto& port_name = item.first;
      auto& buffer = item.second;
      auto& buffer_list = (*ready_port_buffers)[port_name];
      if (buffer_list.empty()) {
        buffer_list.reserve(match_buffer_count);
      }
      (*ready_port_buffers)[port_name].push_back(buffer);
    }
  }

  return ready_port_buffers;
}

void MatchStreamCache::SetSession(std::shared_ptr<Session> session) {
  session_ = session;
}

std::shared_ptr<Session> MatchStreamCache::GetSession() { return session_; }

bool MatchStreamCache::IsStreamEnd() {
  return total_input_count_in_stream_ != 0 &&
         cur_input_count_in_stream_ == total_input_count_in_stream_;
}

size_t MatchStreamCache::TotalInputCount() {
  return total_input_count_in_stream_;
}

void MatchStreamCache::UpdateInputStreamInfo(const std::string& port_name,
                                             std::shared_ptr<Buffer>& buffer) {
  // will use this info to analysis app problem and performance
  auto item = in_port_stream_info_map_.find(port_name);
  std::shared_ptr<InPortStreamInfo> stream_info;
  if (item == in_port_stream_info_map_.end()) {
    stream_info = std::make_shared<InPortStreamInfo>();
    in_port_stream_info_map_[port_name] = stream_info;
  } else {
    stream_info = item->second;
  }

  stream_info->ReceiveBuffer(buffer);
}

InputMatchStreamManager::InputMatchStreamManager(const std::string& node_name,
                                                 size_t queue_size,
                                                 size_t port_count)
    : node_name_(node_name), queue_size_(queue_size), port_count_(port_count) {}

size_t InputMatchStreamManager::GetInputStreamCount() {
  return match_stream_cache_map_.size();
}

void InputMatchStreamManager::SetInputBufferInOrder(bool is_input_in_order) {
  is_input_in_order_ = is_input_in_order;
}

void InputMatchStreamManager::SetInputStreamGatherAll(bool need_gather_all) {
  need_gather_all_ = need_gather_all;
}

void InputMatchStreamManager::UpdateStreamCountEachPort(
    std::unordered_map<std::string, size_t>&& stream_count_each_port) {
  stream_count_each_port_ = stream_count_each_port;
}

Status InputMatchStreamManager::LoadData(
    std::vector<std::shared_ptr<InPort>>& data_ports) {
  if (port_inherit_backward_level_.empty() &&
      !InitInheritBackwardLevel(data_ports)) {
    // can not process data
    return STATUS_OK;
  }

  for (auto& data_port : data_ports) {
    auto& port_name = data_port->GetName();
    auto read_count = GetReadCount(port_name);
    if (read_count == 0) {
      // too much cache for this port, stop read this port
      continue;
    }

    std::vector<std::shared_ptr<Buffer>> buffer_list;
    data_port->Recv(buffer_list, read_count);
    auto backward_level = port_inherit_backward_level_[port_name];
    for (auto& buffer : buffer_list) {
      auto ret = CacheBuffer(data_port->GetName(), buffer, backward_level);
      if (!ret) {
        return {STATUS_FAULT, "port " + port_name + " match stream failed"};
      }
    }
  }

  return STATUS_OK;
}

Status InputMatchStreamManager::GenMatchStreamData(
    std::list<std::shared_ptr<MatchStreamData>>& match_stream_list) {
  for (auto cache_iter = match_stream_cache_map_.begin();
       cache_iter != match_stream_cache_map_.end();) {
    auto match_key = cache_iter->first;
    auto& match_stream_cache = cache_iter->second;
    auto ready_port_buffers = match_stream_cache->PopReadyMatchBuffers(
        is_input_in_order_, need_gather_all_);

    if (ready_port_buffers != nullptr && !ready_port_buffers->empty()) {
      auto match_stream_data = std::make_shared<MatchStreamData>();
      match_stream_data->SetBufferList(ready_port_buffers);
      match_stream_data->SetStreamMatchKey(match_key);
      match_stream_data->SetSession(match_stream_cache->GetSession());
      match_stream_list.push_back(match_stream_data);

      DecreaseAllPortBufferCount(match_stream_data->GetDataCount());
    }

    if (match_stream_cache->IsStreamEnd()) {
      MBLOG_DEBUG
          << "node " << node_name_ << ", stop input match stream " << match_key
          << ", total input count " << match_stream_cache->TotalInputCount()
          << ", id "
          << match_stream_cache->GetSession()->GetSessionCtx()->GetSessionId();
      cache_iter = match_stream_cache_map_.erase(cache_iter);
      continue;
    }

    ++cache_iter;
  }

  return STATUS_OK;
}

Status InputMatchStreamManager::CacheBuffer(const std::string& port_name,
                                            std::shared_ptr<Buffer>& buffer,
                                            size_t backward_level) {
  auto buffer_index_info = BufferManageView::GetIndexInfo(buffer);
  auto stream = buffer_index_info->GetStream();
  if (stream->GetSession()->IsAbort()) {
    // no need to cache buffer, session is abort
    return STATUS_OK;
  }
  // Match different port
  auto stream_match_key =
      GetInputStreamMatchKey(buffer_index_info, backward_level);
  auto match_stream_cache_item = match_stream_cache_map_.find(stream_match_key);
  std::shared_ptr<MatchStreamCache> match_stream_cache;
  if (match_stream_cache_item == match_stream_cache_map_.end()) {
    match_stream_cache = std::make_shared<MatchStreamCache>(
        node_name_, port_count_, &stream_count_each_port_);
    match_stream_cache->SetSession(
        buffer_index_info->GetStream()->GetSession());
    match_stream_cache_map_[stream_match_key] = match_stream_cache;
    MBLOG_DEBUG << "node " << node_name_ << ", start input match stream "
                << stream_match_key << ", id "
                << stream->GetSession()->GetSessionCtx()->GetSessionId();
  } else {
    match_stream_cache = match_stream_cache_item->second;
  }

  auto ret = match_stream_cache->CacheBuffer(port_name, buffer);
  if (ret == STATUS_SUCCESS) {
    // in case exist and fault, will not record buffer count
    IncreaseOnePortBufferCount(port_name);
  } else if (ret == STATUS_FAULT) {
    return ret;
  }

  return STATUS_SUCCESS;
}

MatchKey* InputMatchStreamManager::GetInputStreamMatchKey(
    std::shared_ptr<BufferIndexInfo> index_info, size_t backward_level) {
  MatchKey* stream_match_key = nullptr;
  // go back to find same level inherit info
  auto cur_buffer_index = index_info;
  std::shared_ptr<BufferInheritInfo> cur_inherit_info =
      index_info->GetInheritInfo();
  for (size_t i = 0; i < backward_level; ++i) {
    cur_buffer_index = cur_inherit_info->GetInheritFrom();
    cur_inherit_info = cur_buffer_index->GetInheritInfo();
  }

  auto inherit_buffer = cur_inherit_info->GetInheritFrom();
  if (cur_inherit_info->GetType() == BufferProcessType::EXPAND) {
    /**
     * 1.match at expand level, one buffer will expand a stream,
     * so child stream match at parent buffer
     * in port1: buffer1, buffer2     out port1: stream1, stream2
     * in port2: buffer1, buffer2 ==> out port2: stream1, stream2
     * in port3: buffer1, buffer2     out port3: stream1, stream2
     **/
    stream_match_key = MatchKey::AsKey(inherit_buffer.get());
  } else if (cur_inherit_info->GetType() ==
             BufferProcessType::CONDITION_START) {
    /**
     * 1.match at if-else level, one stream will devide in to two stream,
     * so child stream match at parent stream
     **/
    stream_match_key = MatchKey::AsKey(inherit_buffer->GetStream().get());
  } else {
    MBLOG_ERROR << "node " << node_name_
                << ", get input stream match key failed, wrong inherit type "
                << (size_t)(cur_inherit_info->GetType());
    return nullptr;
  }

  return stream_match_key;
}

bool InputMatchStreamManager::InitInheritBackwardLevel(
    std::vector<std::shared_ptr<InPort>>& data_ports) {
  size_t min_deepth = SIZE_MAX;
  std::unordered_map<std::string, size_t> port_inherit_deepth_map;
  for (auto& port : data_ports) {
    std::shared_ptr<Buffer> first_buffer;
    auto get_result = port->GetQueue()->Front(&first_buffer);
    if (!get_result) {
      // not all port has data, can not test match releationship
      return false;
    }

    auto buffer_index = BufferManageView::GetIndexInfo(first_buffer);
    auto inherit_info = buffer_index->GetInheritInfo();

    auto deepth = inherit_info->GetDeepth();
    min_deepth = std::min(min_deepth, deepth);
    port_inherit_deepth_map[port->GetName()] = deepth;
  }

  // all match to min deepth
  for (auto& port_deepth_item : port_inherit_deepth_map) {
    auto& port_name = port_deepth_item.first;
    auto& deepth = port_deepth_item.second;
    auto backward_level = deepth - min_deepth;
    port_inherit_backward_level_[port_name] = backward_level;
    MBLOG_INFO << "node " << node_name_ << ", port " << port_name
               << ", inherit backward level " << backward_level;
  }

  return true;
}

void InputMatchStreamManager::IncreaseOnePortBufferCount(
    const std::string& port_name, size_t count) {
  port_cache_count_map_[port_name] += count;
}

void InputMatchStreamManager::DecreaseAllPortBufferCount(size_t count) {
  for (auto& count_item : port_cache_count_map_) {
    count_item.second -= count;
  }
}

size_t InputMatchStreamManager::GetReadCount(const std::string& port_name) {
  auto cur_cache_count = port_cache_count_map_[port_name];
  if (cur_cache_count >= max_cache_count_) {
    MBLOG_WARN << "node " << node_name_ << ", port " << port_name
               << ", cache count " << cur_cache_count << " is great than max "
               << max_cache_count_;
    return 0;
  }

  auto read_count = max_cache_count_ - cur_cache_count;
  if (read_count > queue_size_) {
    read_count = queue_size_;
  }

  // find shortest port
  const std::string* shortest_port_name = nullptr;
  size_t shortest_port_length = SIZE_MAX;
  for (auto& port_cache_count_item : port_cache_count_map_) {
    if (port_cache_count_item.second < shortest_port_length) {
      shortest_port_length = port_cache_count_item.second;
      shortest_port_name = &port_cache_count_item.first;
    }
  }

  // no shortest or current port is shortest, read queue_size
  if (shortest_port_name == nullptr || *shortest_port_name == port_name) {
    return read_count;
  }

  // this port has too much cache, wait shortest port
  if (cur_cache_count + read_count - shortest_port_length > 2 * queue_size_) {
    MBLOG_DEBUG << "node " << node_name_ << ", port " << port_name
                << " cache count " << cur_cache_count + read_count
                << " is great than min port " << *shortest_port_name
                << " cache count " << shortest_port_length;
    return 0;
  }

  return read_count;
}

void InputMatchStreamManager::Clean() {
  for (auto iter = match_stream_cache_map_.begin();
       iter != match_stream_cache_map_.end();) {
    auto& match_cache = iter->second;
    if (match_cache->GetSession()->IsAbort()) {
      // session abort
      iter = match_stream_cache_map_.erase(iter);
      continue;
    }

    ++iter;
  }
}

void OutputMatchStream::SetSession(std::shared_ptr<Session> session) {
  session_ = session;
}

std::shared_ptr<Session> OutputMatchStream::GetSession() { return session_; }

size_t OutputMatchStream::Size() { return port_stream_map_.size(); }

bool OutputMatchStream::Empty() { return port_stream_map_.empty(); }

std::shared_ptr<Stream> OutputMatchStream::GetStream(
    const std::string& port_name) {
  auto item = port_stream_map_.find(port_name);
  if (item == port_stream_map_.end()) {
    return nullptr;
  }

  return item->second;
}

std::shared_ptr<Stream> OutputMatchStream::CreateStream(
    const std::string& port_name) {
  auto stream = std::make_shared<Stream>(session_);
  port_stream_map_[port_name] = stream;
  return stream;
}

OutputMatchStreamManager::OutputMatchStreamManager(
    const std::string& node_name, std::set<std::string>&& output_port_names)
    : node_name_(node_name), output_port_names_(output_port_names) {}

size_t OutputMatchStreamManager::GetOutputStreamCount() {
  return output_stream_map_.size();
}

void OutputMatchStreamManager::SetNeedNewIndex(bool need_new_index) {
  need_new_index_ = need_new_index;
}

Status OutputMatchStreamManager::UpdateStreamInfo(
    const std::unordered_map<std::string,
                             std::vector<std::shared_ptr<modelbox::Buffer>>>&
        stream_data_map,
    const std::unordered_map<std::string, std::shared_ptr<DataMeta>>&
        port_stream_meta,
    std::shared_ptr<Session> session) {
  if (stream_data_map.empty() || stream_data_map.begin()->second.empty()) {
    // no data to process
    return STATUS_OK;
  }
  auto match_key = GetOutputStreamMatchKey(stream_data_map);
  if (match_key == nullptr) {
    MBLOG_ERROR << "node " << node_name_
                << " get output stream match key failed";
    return STATUS_FAULT;
  }
  auto& output_match_stream = output_stream_map_[match_key];
  if (output_match_stream.Empty()) {
    MBLOG_DEBUG << "node " << node_name_ << ", start output match stream "
                << match_key << ", id "
                << session->GetSessionCtx()->GetSessionId();
    output_match_stream.SetSession(session);
    GenerateOutputStream(output_match_stream, stream_data_map, port_stream_meta,
                         session);
  }

  size_t end_stream_count = 0;
  std::stringstream stream_count_stats;
  for (auto& stream_data : stream_data_map) {
    auto& port_name = stream_data.first;
    auto& port_data_list = stream_data.second;
    auto stream = output_match_stream.GetStream(port_name);
    if (stream == nullptr) {
      MBLOG_ERROR << "port [" << port_name
                  << "] in output data is not defined in node";
      return STATUS_FAULT;
    }
    for (auto& port_data : port_data_list) {
      if (port_data == nullptr) {
        // if-else empty output, drop it
        continue;
      }
      auto buffer_index = BufferManageView::GetIndexInfo(port_data);
      buffer_index->SetStream(stream);
      SetIndexInStream(buffer_index, stream);
      if (buffer_index->IsEndFlag()) {
        // output index will not great than end_flag index
        stream->SetMaxBufferCount(buffer_index->GetIndex() + 1);
      }
    }

    if (stream->ReachEnd()) {  // output stream is over
      ++end_stream_count;
      stream_count_stats << ", port " << port_name << ", out ";
      stream_count_stats << stream->GetBufferCount();
    }
  }

  if (end_stream_count == 0) {
    return STATUS_OK;
  }

  if (end_stream_count != stream_data_map.size()) {
    MBLOG_ERROR << "node " << node_name_
                << ", all output port stream should finish togather";
    return STATUS_FAULT;
  }

  output_stream_map_.erase(match_key);
  MBLOG_DEBUG << "node " << node_name_ << ", stop output match stream "
              << match_key << stream_count_stats.str() << ", id "
              << session->GetSessionCtx()->GetSessionId();
  return STATUS_OK;
}

MatchKey* OutputMatchStreamManager::GetOutputStreamMatchKey(
    const std::unordered_map<std::string,
                             std::vector<std::shared_ptr<modelbox::Buffer>>>&
        stream_data_map) {
  std::shared_ptr<Buffer> not_null_output_buffer;
  for (auto port_iter = stream_data_map.begin();
       port_iter != stream_data_map.end(); ++port_iter) {
    auto& port_data_list = port_iter->second;
    not_null_output_buffer = port_data_list.front();
    if (not_null_output_buffer != nullptr) {
      break;
    }
  }

  if (not_null_output_buffer == nullptr) {
    MBLOG_ERROR << "node " << node_name_ << ", all port output buffer is null";
    return nullptr;
  }

  auto output_buffer_index_info =
      BufferManageView::GetIndexInfo(not_null_output_buffer);
  auto output_inherit_info = output_buffer_index_info->GetInheritInfo();
  auto inherit_from_buffer = output_inherit_info->GetInheritFrom();
  if (output_inherit_info->GetType() == BufferProcessType::EXPAND) {
    // output match at expand buffer
    return MatchKey::AsKey(inherit_from_buffer.get());
  }

  // output match at condition stream
  return MatchKey::AsKey(inherit_from_buffer->GetStream().get());
}

void OutputMatchStreamManager::GenerateOutputStream(
    OutputMatchStream& output_match_stream,
    const std::unordered_map<std::string,
                             std::vector<std::shared_ptr<modelbox::Buffer>>>&
        stream_data_map,
    const std::unordered_map<std::string, std::shared_ptr<DataMeta>>&
        port_stream_meta,
    std::shared_ptr<Session> session) {
  // visit input stream, for collapse, will visit expand input
  std::shared_ptr<Buffer> not_null_output_buffer;
  for (auto port_iter = stream_data_map.begin();
       port_iter != stream_data_map.end(); ++port_iter) {
    auto& port_data_list = port_iter->second;
    not_null_output_buffer = port_data_list.front();
    if (not_null_output_buffer != nullptr) {
      break;
    }
  }
  auto out_buffer_index =
      BufferManageView::GetIndexInfo(not_null_output_buffer);
  std::shared_ptr<StreamOrder> stream_order;
  size_t input_buffer_index = 0;
  auto inherit_stream_meta = std::make_shared<DataMeta>();
  auto& input_stream_data_map =
      out_buffer_index->GetProcessInfo()->GetParentBuffers();
  for (auto& in_port_data_item : input_stream_data_map) {
    auto& port_data_list = in_port_data_item.second;
    auto& first_in_buffer = port_data_list.front();
    auto in_port_stream = first_in_buffer->GetStream();
    // combine all input port stream meta
    auto stream_meta = in_port_stream->GetStreamMeta();
    if (stream_meta != nullptr) {
      auto metas = stream_meta->GetMetas();
      for (auto& meta_item : metas) {
        inherit_stream_meta->SetMeta(meta_item.first, meta_item.second);
      }
    }
    // all input stream order is same, only get one
    if (stream_order == nullptr) {
      stream_order = in_port_stream->GetStreamOrder()->Copy();
      input_buffer_index = first_in_buffer->GetIndex();
    }
  }

  // modify stream order
  auto process_info = out_buffer_index->GetProcessInfo();
  if (process_info->GetType() == BufferProcessType::EXPAND) {
    // need record expand at which buffer
    stream_order->Expand(input_buffer_index);
  } else if (process_info->GetType() == BufferProcessType::COLLAPSE) {
    stream_order->Collapse();
  }

  // generate output stream
  for (auto& output_port_name : output_port_names_) {
    if (stream_data_map.find(output_port_name) == stream_data_map.end()) {
      // output port has no data, no need to create output stream
      continue;
    }

    auto new_stream = output_match_stream.CreateStream(output_port_name);
    // write stream meta
    auto new_stream_meta = std::make_shared<DataMeta>(*inherit_stream_meta);
    auto port_stream_meta_item = port_stream_meta.find(output_port_name);
    if (port_stream_meta_item != port_stream_meta.end()) {
      auto& stream_meta = port_stream_meta_item->second;
      for (auto& meta_item : stream_meta->GetMetas()) {
        new_stream_meta->SetMeta(meta_item.first, meta_item.second);
      }
    }
    new_stream->SetStreamMeta(new_stream_meta);
    // write stream order
    new_stream->SetStreamOrder(stream_order);
  }
}

void OutputMatchStreamManager::SetIndexInStream(
    std::shared_ptr<modelbox::BufferIndexInfo> buffer_index,
    std::shared_ptr<modelbox::Stream> stream) {
  if (need_new_index_) {
    buffer_index->SetIndex(stream->GetBufferCount());
  }

  stream->IncreaseBufferCount();
}

void OutputMatchStreamManager::Clean() {
  for (auto iter = output_stream_map_.begin();
       iter != output_stream_map_.end();) {
    auto& output_match_stream = iter->second;
    if (output_match_stream.GetSession()->IsAbort()) {
      iter = output_stream_map_.erase(iter);
      continue;
    }

    ++iter;
  }
}

}  // namespace modelbox