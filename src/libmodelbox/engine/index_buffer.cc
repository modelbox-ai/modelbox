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


#include "modelbox/index_buffer.h"

namespace modelbox {

BufferGroup::BufferGroup()
    : start_flag_(true),
      end_flag_(true),
      order_(1),
      port_id_(0),
      group_port_id_(0),
      session_context_(nullptr) {}

BufferGroup::~BufferGroup() {}

std::shared_ptr<SessionContext> BufferGroup::GetSessionContext() {
  auto root = GetRoot();
  return root->session_context_;
}

std::shared_ptr<BufferGroup> BufferGroup::GetRoot() {
  auto group = shared_from_this();
  while (true) {
    if (group->GetGroup() == nullptr) {
      return group;
    }
    group = group->GetGroup();
  }
}

void BufferGroup::SetSessionContex(
    std::shared_ptr<SessionContext> session_ctx) {
  auto root = GetRoot();
  if (root->session_context_ == nullptr) {
    root->session_context_ = session_ctx;
  }
}

std::shared_ptr<BufferGroup> BufferGroup::GetGroup() { return group_; }

uint32_t BufferGroup::GetOrder() { return order_; }

std::shared_ptr<BufferGroup> BufferGroup::GetOneLevelGroup() {
  auto port_group = this->GetGroup();
  if (port_group == nullptr) {
    return nullptr;
  }
  auto buffer_group = port_group->GetGroup();
  return buffer_group;
}

std::shared_ptr<BufferGroup> BufferGroup::GetStreamLevelGroup() {
  auto one_level_group = this->GetOneLevelGroup();
  if (one_level_group == nullptr) {
    return nullptr;
  }
  auto up_level_group = one_level_group->GetGroup();
  return up_level_group;
}

std::shared_ptr<BufferGroup> BufferGroup::GetGroupLevelGroup() {
  auto stream_level_group = this->GetStreamLevelGroup();
  if (stream_level_group == nullptr) {
    return nullptr;
  }
  auto group_level_group = stream_level_group->GetOneLevelGroup();
  return group_level_group;
}

std::shared_ptr<BufferGroup> BufferGroup::GenerateSameLevelGroup() {
  auto group_ptr = shared_from_this();
  auto port_sub_group = group_ptr->AddSubGroup(true, true);

  if (port_sub_group == nullptr) {
    return nullptr;
  }
  auto buffer_sub_group = port_sub_group->AddSubGroup(true, true);
  return buffer_sub_group;
}

std::shared_ptr<BufferGroup> BufferGroup::AddSubGroup(bool start_flag,
                                                      bool end_flag) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  return AddSubGroup(group_port_id_, start_flag, end_flag);
}

std::shared_ptr<BufferGroup> BufferGroup::AddSubGroup(uint32_t port_id,
                                                      bool start_flag,
                                                      bool end_flag) {
  auto sum_iter = port_sum_map_.find(port_id);
  // the port already exsit and the return nullptr
  if (sum_iter != port_sum_map_.end()) {
    return nullptr;
  }

  auto ordert_iter = port_order_map_.find(port_id);
  if ((ordert_iter != port_order_map_.end()) && (start_flag == true)) {
    return nullptr;
  } else if ((ordert_iter == port_order_map_.end()) && (start_flag == false)) {
    return nullptr;
  }

  auto new_group = std::make_shared<BufferGroup>();
  new_group->port_id_ = port_id;

  new_group->group_ = shared_from_this();
  if (ordert_iter == port_order_map_.end()) {
    port_order_map_[port_id] = new_group->GetOrder();
  } else {
    port_order_map_[port_id] += 1;
  }

  new_group->order_ = port_order_map_[port_id];
  if (end_flag) {
    port_sum_map_[port_id] = port_order_map_[port_id];
    group_port_id_++;
  }

  new_group->start_flag_ = start_flag;
  new_group->end_flag_ = end_flag;

  return new_group;
}

std::shared_ptr<BufferGroup> BufferGroup::InnerAddNewSubGroup(
    uint32_t port_id) {
  // port_order_map_ record the stream total count
  auto sum_iter = port_sum_map_.find(port_id);
  if (sum_iter != port_sum_map_.end()) {
    return nullptr;
  }

  // the differnt stream has differnt port,
  auto new_group = std::make_shared<BufferGroup>();
  new_group->port_id_ = port_id;
  new_group->group_ = shared_from_this();

  // port_order_map_ record the stream current order
  auto ordert_iter = port_order_map_.find(port_id);
  if (ordert_iter == port_order_map_.end()) {
    new_group->start_flag_ = true;
    port_order_map_[port_id] = new_group->GetOrder();
  } else {
    new_group->start_flag_ = false;
    port_order_map_[port_id] += 1;
  }
  new_group->order_ = port_order_map_[port_id];
  new_group->end_flag_ = false;

  // record the last buffer_group,when we close stream.the last buffer_group
  // end_flag will be set true
  last_bg_map_[port_id] = new_group;
  return new_group;
}

std::shared_ptr<BufferGroup> BufferGroup::AddNewSubGroup(uint32_t port_id) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  std::shared_ptr<BufferGroup> group = nullptr;
  if (port_sum_map_.find(port_id) != port_sum_map_.end()) {
    // we deal with the data already finished
    port_start_map_[port_id] = port_sum_map_[port_id] + 1;
    port_sum_map_.erase(port_id);
    group = AddSubGroup(port_id, false, true);
  } else {
    group = InnerAddNewSubGroup(port_id);
  }
  return group;
}

bool BufferGroup::IsFullGroup(uint32_t port_id) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  if (port_sum_map_.find(port_id) != port_sum_map_.end()) {
    return true;
  }
  return false;
}

void BufferGroup::InnerFinishSubGroup(uint32_t port_id) {
  if (port_sum_map_.find(port_id) != port_sum_map_.end()) {
    return;
  }

  port_sum_map_[port_id] = port_order_map_[port_id];
  group_port_id_++;
  if (last_bg_map_.find(port_id) != last_bg_map_.end()) {
    last_bg_map_[port_id]->end_flag_ = true;
    last_bg_map_.erase(port_id);
  }
}

void BufferGroup::FinishSubGroup(uint32_t port_id) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  InnerFinishSubGroup(port_id);
}

std::shared_ptr<BufferGroup> BufferGroup::AddOneMoreSubGroup(uint32_t port_id) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  std::shared_ptr<BufferGroup> group = nullptr;
  if (port_sum_map_.find(port_id) != port_sum_map_.end()) {
    // we deal with the data already finished
    port_start_map_[port_id] = port_sum_map_[port_id] + 1;
    port_sum_map_.erase(port_id);
    group = AddSubGroup(port_id, false, true);
  } else {
    group = InnerAddNewSubGroup(port_id);
    InnerFinishSubGroup(port_id);
  }
  return group;
}

Status BufferGroup::GetSum(uint32_t* sum, int port_id) {
  std::lock_guard<std::mutex> lock(add_mutex_);
  auto it = port_sum_map_.find(port_id);
  if (it == port_sum_map_.end()) {
    return STATUS_NOTFOUND;
  }

  *sum = port_sum_map_[port_id];
  return STATUS_SUCCESS;
}

Status BufferGroup::GetGroupSum(uint32_t* sum) {
  if (group_ == nullptr) {
    return STATUS_NOTFOUND;
  }
  auto status = group_->GetSum(sum, port_id_);
  return status;
}

uint32_t BufferGroup::GetGroupOrder(int port_id) {
  // todo liuchang add lock
  std::lock_guard<std::mutex> lock(add_mutex_);
  auto start_it = port_start_map_.find(port_id);
  if (start_it == port_start_map_.end()) {
    return 1;
  }

  return port_start_map_[port_id];
}

uint32_t BufferGroup::GetPortId() { return port_id_; }

bool BufferGroup::IsStartGroup() { return start_flag_; }

bool BufferGroup::IsEndGroup() { return end_flag_; }

bool BufferGroup::IsRoot() {
  if (group_ == nullptr) {
    return true;
  }
  return false;
}

std::vector<uint32_t> BufferGroup::GetSeqOrder() {
  std::vector<uint32_t> seq;
  auto order_bg = GetOneLevelGroup();
  while (true) {
    order_bg = order_bg->GetOneLevelGroup();
    if (order_bg->IsRoot()) {
      seq.push_back(1);
      break;
    }
    auto order = order_bg->GetOrder();
    seq.push_back(order);
  }
  return seq;
}

void IndexMeta::SetBufferGroup(std::shared_ptr<BufferGroup> bg) {
  group_index_ = bg;
}

std::shared_ptr<BufferGroup> IndexMeta::GetBufferGroup() {
  return group_index_;
}

IndexBufferList::IndexBufferList(
    std::vector<std::shared_ptr<IndexBuffer>> buffer_vector) {
  for (auto buffer : buffer_vector) {
    buffer_list_.push_back(buffer);
  }
}

bool IndexBufferList::BindToRoot() {
  for (uint32_t i = 0; i < GetBufferNum(); i++) {
    auto index_buffer = GetBuffer(i);
    if (index_buffer->GetBufferGroup() != nullptr) {
      return false;
    }
  }

  auto root = std::shared_ptr<BufferGroup>(new BufferGroup());
  // we bind buffer meta nopt to root but a new generate buffer group
  auto buffer_ptr = root->GenerateSameLevelGroup();
  auto port_ptr = buffer_ptr->GetGroup();
  for (uint32_t i = 0; i < GetBufferNum(); i++) {
    bool start_flag = false;
    bool end_flag = false;
    auto index_buffer = GetBuffer(i);

    if (i == 0) {
      start_flag = true;
    }

    if (i == GetBufferNum() - 1) {
      end_flag = true;
    }

    auto new_group_ptr = buffer_ptr->AddSubGroup(start_flag, end_flag);
    if (new_group_ptr == nullptr) {
      return false;
    }

    auto new_node_ptr = new_group_ptr->GenerateSameLevelGroup();
    if (new_node_ptr == nullptr) {
      return false;
    }

    index_buffer->SetBufferGroup(new_node_ptr);
  }
  return true;
}

std::shared_ptr<IndexBuffer> IndexBufferList::GetBuffer(uint32_t order) {
  if (order >= buffer_list_.size()) {
    return nullptr;
  }
  return buffer_list_[order];
}

void IndexBufferList::ExtendBufferList(
    std::shared_ptr<IndexBufferList> buffer_list) {
  if (buffer_list == nullptr) {
    return;
  }

  for (uint32_t i = 0; i < buffer_list->GetBufferNum(); i++) {
    buffer_list_.push_back(buffer_list->GetBuffer(i));
  }
}

std::shared_ptr<FlowUnitError> IndexBufferList::GetDataError() {
  auto key = GetDataErrorIndex();
  auto stream_bg = GetStreamBufferGroup();
  return stream_bg->GetDataError(key);
}

std::tuple<uint32_t, uint32_t> IndexBufferList::GetDataErrorIndex() {
  if (buffer_list_.size() == 0) {
    return std::make_tuple(-1, -1);
  }

  if(buffer_list_[0]->GetBufferGroup() ==nullptr || buffer_list_[0]->GetSameLevelGroup() == nullptr) {
    return std::make_tuple(-1, -1);
  }

  auto port_1 = buffer_list_[0]->GetBufferGroup()->GetGroup()->GetPortId();
  auto port_2 = buffer_list_[0]->GetSameLevelGroup()->GetPortId();
  auto key = std::make_tuple(port_1, port_2);
  return key;
}

std::tuple<uint32_t, uint32_t> IndexBufferList::GetGroupDataErrorIndex() {
  if (buffer_list_.size() == 0) {
    return std::make_tuple(-1, -1);
  }
  auto stream_bg = GetStreamBufferGroup();
  if (stream_bg->GetOneLevelGroup() == nullptr) {
    return std::make_tuple(-1, -1);
  }

  auto port_1 = stream_bg->GetPortId();
  auto port_2 = stream_bg->GetGroup()->GetPortId();
  auto key = std::make_tuple(port_1, port_2);
  return key;
}

void IndexBufferList::SetDataMeta(std::shared_ptr<DataMeta> data_meta) {
  if (buffer_list_.size() == 0) {
    return;
  }

  auto stream_port = buffer_list_[0]->GetSameLevelGroup()->GetPortId();
  auto stream_bg = buffer_list_[0]->GetSameLevelGroup()->GetGroup();
  stream_bg->SetDataMeta(stream_port, data_meta);
}

std::shared_ptr<DataMeta> IndexBufferList::GetDataMeta() {
  if (buffer_list_.size() == 0) {
    return nullptr;
  }

  if(buffer_list_[0]->GetSameLevelGroup() == nullptr) {
    return nullptr;
  }

  auto stream_port = buffer_list_[0]->GetSameLevelGroup()->GetPortId();
  auto stream_bg = buffer_list_[0]->GetSameLevelGroup()->GetGroup();
  return stream_bg->GetDataMeta(stream_port);
}

std::shared_ptr<DataMeta> IndexBufferList::GetGroupDataMeta() {
  if (buffer_list_.size() == 0) {
    return nullptr;
  }
  auto stream_bg = GetStreamBufferGroup();
  if (stream_bg->GetOneLevelGroup() == nullptr) {
    return nullptr;
  }

  auto stream_port = stream_bg->GetGroup()->GetPortId();
  auto stream_group_bg = stream_bg->GetOneLevelGroup();
  return stream_group_bg->GetDataMeta(stream_port);
}

std::shared_ptr<BufferGroup> IndexBufferList::GetStreamBufferGroup() {
  if (buffer_list_.size() == 0) {
    return nullptr;
  }
  return buffer_list_[0]->GetStreamLevelGroup();
}

int32_t IndexBufferList::GetPriority() {
  if (buffer_list_.size() == 0) {
    return 0;
  }
  return buffer_list_[0]->GetPriority();
}

uint32_t IndexBufferList::GetOrder() {
  if (GetStreamBufferGroup() == nullptr) {
    return -1;
  }
  return GetStreamBufferGroup()->GetGroup()->GetOrder() - 1;
}

bool IndexBufferList::IsStartStream() {
  if (GetStreamBufferGroup() == nullptr) {
    return false;
  }
  return GetStreamBufferGroup()->GetGroup()->IsStartGroup();
}

bool IndexBufferList::IsEndStream() {
  if (GetStreamBufferGroup() == nullptr) {
    return false;
  }
  return GetStreamBufferGroup()->GetGroup()->IsEndGroup();
}

uint32_t IndexBufferList::GetBufferNum() { return buffer_list_.size(); }

void IndexBufferList::Clear() { buffer_list_.clear(); }

std::vector<std::shared_ptr<Buffer>> IndexBufferList::GetBufferPtrList() {
  auto bufferlist = std::vector<std::shared_ptr<Buffer>>();
  for (auto index_buffer : buffer_list_) {
    if (!index_buffer->IsPlaceholder()) {
      bufferlist.push_back(index_buffer->GetBufferPtr());
    }
  }
  return bufferlist;
}

bool IndexBufferList::IsExpandBackfillBufferlist() {
  if (buffer_list_.size() != 1) {
    return false;
  }
  auto index_buffer = buffer_list_[0];
  if (!index_buffer->IsPlaceholder()) {
    return false;
  }
  uint32_t sum;
  auto status = index_buffer->GetSameLevelGroup()->GetGroupSum(&sum);
  if (status != STATUS_SUCCESS) {
    return false;
  }
  if (sum != 1) {
    return false;
  }

  return true;
}

std::set<uint32_t> IndexBufferList::GetPlaceholderPos() {
  std::set<uint32_t> pos_set;
  uint32_t pos = 0;
  auto buffer_iter = buffer_list_.begin();
  while (buffer_iter != buffer_list_.end()) {
    auto index_buffer = *buffer_iter;
    if (index_buffer->IsPlaceholder()) {
      pos_set.insert(pos);
    }
    pos++;
    buffer_iter++;
  }
  return pos_set;
}

void IndexBufferList::BackfillBuffer(std::set<uint32_t> id_set,
                                     bool is_placeholder) {
  auto buffer_iter = buffer_list_.begin();
  uint32_t i = 0;
  while (true) {
    if (id_set.find(i) != id_set.end()) {
      auto index_buffer = std::make_shared<IndexBuffer>();
      if (is_placeholder) {
        index_buffer->MarkAsPlaceholder();
      }
      buffer_iter = buffer_list_.insert(buffer_iter, index_buffer);
      i++;
    }

    if (buffer_iter == buffer_list_.end()) {
      break;
    }

    buffer_iter++;
    i++;
  }
}

IndexBuffer::IndexBuffer() : priority_(0), is_placeholder_(false) {}

IndexBuffer::IndexBuffer(IndexBuffer* other) : is_placeholder_(false) {
  if (other != nullptr) {
    buffer_ptr_ = other->buffer_ptr_;
  }
}

IndexBuffer::IndexBuffer(std::shared_ptr<Buffer> other)
    : is_placeholder_(false) {
  buffer_ptr_ = other;
}

IndexBuffer::~IndexBuffer() {}

bool IndexBuffer::BindToRoot() {
  if (GetBufferGroup() != nullptr) {
    return false;
  }

  auto root = std::shared_ptr<BufferGroup>(new BufferGroup());
  // we bind buffer meta nopt to root but a new generate buffer group
  auto buffer_ptr = root->GenerateSameLevelGroup();

  auto port_potr = buffer_ptr->GetGroup();

  auto new_group_ptr = port_potr->AddSubGroup(true, true);
  if (new_group_ptr == nullptr) {
    return false;
  }

  auto new_node_ptr = new_group_ptr->GenerateSameLevelGroup();
  if (new_node_ptr == nullptr) {
    return false;
  }

  SetBufferGroup(new_node_ptr);
  return true;
}

void IndexBuffer::SetDataMeta(std::shared_ptr<DataMeta> data_meta) {
  auto stream_port = GetSameLevelGroup()->GetPortId();
  auto stream_bg = GetSameLevelGroup()->GetGroup();
  stream_bg->SetDataMeta(stream_port, data_meta);
}

std::shared_ptr<DataMeta> IndexBuffer::GetDataMeta() {
  auto stream_port = GetSameLevelGroup()->GetPortId();
  auto stream_bg = GetSameLevelGroup()->GetGroup();
  return stream_bg->GetDataMeta(stream_port);
}

std::shared_ptr<DataMeta> IndexBuffer::GetGroupDataMeta() {
  auto stream_bg = GetBufferGroup()->GetOneLevelGroup()->GetGroup();
  if (stream_bg->GetOneLevelGroup() == nullptr) {
    return nullptr;
  }

  auto stream_group_port = stream_bg->GetGroup()->GetPortId();
  auto stream_group_bg = stream_bg->GetOneLevelGroup();
  return stream_group_bg->GetDataMeta(stream_group_port);
}

bool IndexBuffer::CopyMetaTo(std::shared_ptr<IndexBuffer> other) {
  if (other->GetBufferGroup() != nullptr) {
    return false;
  }

  auto buffer_ptr = GetBufferGroup();
  if (buffer_ptr == nullptr) {
    return false;
  }

  auto group_ptr = buffer_ptr->GetOneLevelGroup();
  if (group_ptr == nullptr) {
    return false;
  }

  auto other_ptr = group_ptr->GenerateSameLevelGroup();
  if (other_ptr == nullptr) {
    return false;
  }

  other->SetBufferGroup(other_ptr);
  other->SetPriority(priority_);
  return true;
}

bool IndexBuffer::BindDownLevelTo(std::shared_ptr<IndexBuffer>& other,
                                  bool start_flag, bool end_flag) {
  if (other->GetBufferGroup() != nullptr) {
    return false;
  }

  auto buffer_ptr = GetBufferGroup();
  if (buffer_ptr == nullptr) {
    return false;
  }

  auto port_ptr = buffer_ptr->GetGroup();
  if (port_ptr == nullptr) {
    return false;
  }

  auto new_group_ptr = port_ptr->AddSubGroup(start_flag, end_flag);
  if (new_group_ptr == nullptr) {
    return false;
  }

  auto new_node_ptr = new_group_ptr->GenerateSameLevelGroup();
  if (new_node_ptr == nullptr) {
    return false;
  }

  other->SetBufferGroup(new_node_ptr);
  return true;
}

bool IndexBuffer::BindUpLevelTo(std::shared_ptr<IndexBuffer>& other) {
  if (other->GetBufferGroup() != nullptr) {
    return false;
  }
  auto level_group = this->GetSameLevelGroup();
  if (level_group == nullptr) {
    return false;
  }

  auto up_level_group = level_group->GetOneLevelGroup();
  if (up_level_group == nullptr) {
    return false;
  }

  auto new_level_group = up_level_group->GenerateSameLevelGroup();
  if (new_level_group == nullptr) {
    return false;
  }

  other->SetBufferGroup(new_level_group);
  return true;
}

int32_t IndexBuffer::GetPriority() { return priority_; }

void IndexBuffer::SetPriority(int32_t priority) { priority_ = priority; }

void IndexBuffer::SetBufferGroup(std::shared_ptr<BufferGroup> bg) {
  index_info_.SetBufferGroup(bg);
}

std::shared_ptr<BufferGroup> IndexBuffer::GetBufferGroup() {
  return index_info_.GetBufferGroup();
}

std::shared_ptr<BufferGroup> IndexBuffer::GetSameLevelGroup() {
  if (index_info_.GetBufferGroup() == nullptr) {
    return nullptr;
  }
  return index_info_.GetBufferGroup()->GetOneLevelGroup();
}

std::shared_ptr<BufferGroup> IndexBuffer::GetStreamLevelGroup() {
  if (index_info_.GetBufferGroup() == nullptr) {
    return nullptr;
  }
  return index_info_.GetBufferGroup()->GetStreamLevelGroup();
}

std::shared_ptr<BufferGroup> IndexBuffer::GetGroupLevelGroup() {
  if (index_info_.GetBufferGroup() == nullptr) {
    return nullptr;
  }
  return index_info_.GetBufferGroup()->GetGroupLevelGroup();
}

std::shared_ptr<IndexBuffer> IndexBuffer::Clone() {
  auto result = std::make_shared<IndexBuffer>(this);
  CopyMetaTo(result);
  result->SetDataMeta(this->GetDataMeta());
  return result;
}
bool IndexBuffer::IsPlaceholder() { return is_placeholder_; }

void IndexBuffer::MarkAsPlaceholder() { is_placeholder_ = true; }

std::vector<uint32_t> IndexBuffer::GetSeqOrder() {
  std::vector<uint32_t> vector;
  if (index_info_.GetBufferGroup() == nullptr) {
    return vector;
  }
  vector = index_info_.GetBufferGroup()->GetSeqOrder();
  return vector;
}

}  // namespace modelbox
