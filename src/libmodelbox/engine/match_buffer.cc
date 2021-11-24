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


#include "modelbox/match_buffer.h"

namespace modelbox {
  
MatchBuffer::MatchBuffer(uint32_t match_num) { match_num_ = match_num; }

MatchBuffer::~MatchBuffer() { }

bool MatchBuffer::IsMatch() { return (match_buffer_.size() == match_num_); }

uint32_t const MatchBuffer::GetOrder() {
  if (!IsMatch()) {
    return -1;
  }

  if (group_ptr_ == nullptr) {
    return -1;
  }
  return group_ptr_->GetOrder();
}

Status MatchBuffer::GetGroupSum(uint32_t* sum) {
  if (!IsMatch()) {
    return STATUS_NODATA;
  }
  if (group_ptr_ == nullptr) {
    return STATUS_NOTFOUND;
  }
  return group_ptr_->GetGroupSum(sum);
}

bool MatchBuffer::SetBuffer(std::string key,
                            std::shared_ptr<IndexBuffer> buffer) {
  auto iter = match_buffer_.find(key);
  if (iter != match_buffer_.end()) {
    return false;
  }
  if (buffer->GetSameLevelGroup() == nullptr) {
    return false;
  }
  if ((group_ptr_ != nullptr) && (group_ptr_ != buffer->GetSameLevelGroup())) {
    return false;
  }
  group_ptr_ = buffer->GetSameLevelGroup();
  match_buffer_[key] = buffer;
  return true;
}

std::shared_ptr<IndexBuffer> MatchBuffer::GetBuffer(std::string key) {
  auto iter = match_buffer_.find(key);
  if (iter == match_buffer_.end()) {
    return nullptr;
  }
  return iter->second;
}

std::shared_ptr<BufferGroup> MatchBuffer::GetBufferGroup() {
  return group_ptr_;
}

}  // namespace modelbox