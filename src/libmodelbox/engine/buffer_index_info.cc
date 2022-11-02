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

#include <utility>

#include "modelbox/buffer_index_info.h"

#include "modelbox/buffer.h"
#include "modelbox/node.h"
#include "modelbox/stream.h"

namespace modelbox {

void BufferInheritInfo::SetType(BufferProcessType type) { type_ = type; }

BufferProcessType BufferInheritInfo::GetType() { return type_; }

void BufferInheritInfo::SetInheritFrom(
    const std::shared_ptr<BufferIndexInfo> &buffer_index) {
  inherit_from_buffer_ = buffer_index;
  auto inherit_info = buffer_index->GetInheritInfo();
  if (inherit_info == nullptr) {
    return;
  }

  inherit_deepth_ = inherit_info->GetDeepth() + 1;
}

std::shared_ptr<BufferIndexInfo> BufferInheritInfo::GetInheritFrom() {
  return inherit_from_buffer_;
}

size_t BufferInheritInfo::GetDeepth() { return inherit_deepth_; }

void BufferProcessInfo::SetParentBuffers(
    const std::string &port_name,
    std::list<std::shared_ptr<BufferIndexInfo>> &&port_buffers) {
  parent_buffers_[port_name] = port_buffers;
}

const std::map<std::string, std::list<std::shared_ptr<BufferIndexInfo>>>
    &BufferProcessInfo::GetParentBuffers() {
  return parent_buffers_;
}

void BufferProcessInfo::SetType(BufferProcessType type) { type_ = type; }

BufferProcessType BufferProcessInfo::GetType() { return type_; }

BufferIndexInfo::BufferIndexInfo() = default;

BufferIndexInfo::~BufferIndexInfo() = default;

void BufferIndexInfo::SetInheritInfo(
    std::shared_ptr<BufferInheritInfo> inherit_info) {
  inherit_info_ = std::move(inherit_info);
}

std::shared_ptr<BufferInheritInfo> BufferIndexInfo::GetInheritInfo() {
  return inherit_info_;
}

void BufferIndexInfo::SetStream(std::shared_ptr<Stream> stream_belong_to) {
  stream_belong_to_ = std::move(stream_belong_to);
}

std::shared_ptr<Stream> BufferIndexInfo::GetStream() {
  return stream_belong_to_;
}

void BufferIndexInfo::SetIndex(size_t index) {
  index_in_current_stream_ = index;
}

size_t BufferIndexInfo::GetIndex() { return index_in_current_stream_; }

bool BufferIndexInfo::IsFirstBufferInStream() {
  return index_in_current_stream_ == 0;
}

void BufferIndexInfo::MarkAsEndFlag() { is_end_flag_ = true; }

bool BufferIndexInfo::IsEndFlag() { return is_end_flag_; }

void BufferIndexInfo::MarkAsPlaceholder() { is_placeholder_ = true; }

bool BufferIndexInfo::IsPlaceholder() { return is_placeholder_; }

void BufferIndexInfo::SetProcessInfo(
    std::shared_ptr<BufferProcessInfo> process_info) {
  process_info_ = std::move(process_info);
}

std::shared_ptr<BufferProcessInfo> BufferIndexInfo::GetProcessInfo() {
  return process_info_;
}

std::shared_ptr<BufferIndexInfo> BufferManageView::GetIndexInfo(
    const std::shared_ptr<Buffer> &buffer) {
  return buffer->index_info_;
}

void BufferManageView::SetIndexInfo(
    const std::shared_ptr<Buffer> &buffer,
    std::shared_ptr<BufferIndexInfo> buffer_index_info) {
  buffer->index_info_ = std::move(buffer_index_info);
}

std::shared_ptr<BufferIndexInfo> BufferManageView::GetFirstParentBuffer(
    const std::shared_ptr<Buffer> &buffer) {
  if (buffer->index_info_ == nullptr) {
    MBLOG_ERROR << "buffer index info is null";
    return nullptr;
  }

  auto process_info = buffer->index_info_->GetProcessInfo();
  if (process_info == nullptr) {
    MBLOG_ERROR << "buffer process info is null";
    return nullptr;
  }

  const auto &parent_buffers = process_info->GetParentBuffers();
  if (parent_buffers.empty()) {
    MBLOG_ERROR << "buffer parent info is empty";
    return nullptr;
  }

  return parent_buffers.begin()->second.front();
}

void BufferManageView::SetPriority(const std::shared_ptr<Buffer> &buffer,
                                   int priority) {
  buffer->SetPriority(priority);
}

int BufferManageView::GetPriority(const std::shared_ptr<Buffer> &buffer) {
  return buffer->GetPriority();
}

void BufferManageView::SetError(const std::shared_ptr<Buffer> &buffer,
              const std::shared_ptr<DataError> &data_error) {
  buffer->data_error_ = data_error;
}

std::shared_ptr<DataError> BufferManageView::GetError(
    const std::shared_ptr<Buffer> &buffer) {
  return buffer->data_error_;
}

}  // namespace modelbox
