
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

#include <modelbox/stream.h>
namespace modelbox {

DataMeta::DataMeta() {}

DataMeta::DataMeta(const DataMeta &other) { private_map_ = other.private_map_; }

DataMeta::~DataMeta() { private_map_.clear(); }

void DataMeta::SetMeta(const std::string &key, std::shared_ptr<void> meta) {
  private_map_[key] = meta;
}

std::shared_ptr<void> DataMeta::GetMeta(const std::string &key) {
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    return nullptr;
  }
  return private_map_[key];
}

std::unordered_map<std::string, std::shared_ptr<void>> DataMeta::GetMetas() {
  return private_map_;
}

StreamOrder2::StreamOrder2() { index_at_each_expand_level_.push_back(0); }

bool StreamOrder2::operator<(const StreamOrder2 &other_stream_order) {
  auto this_index = index_at_each_expand_level_.begin();
  auto other_index = other_stream_order.index_at_each_expand_level_.begin();
  while (true) {
    if (other_index == other_stream_order.index_at_each_expand_level_.end()) {
      // not short than this
      return false;
    }

    if (this_index == index_at_each_expand_level_.end()) {
      // short than other
      return true;
    }

    if (*this_index < *other_index) {
      return true;
    }

    if (*this_index > *other_index) {
      return false;
    }

    // this level is same, compare next level
    ++this_index;
    ++other_index;
  }
}

std::shared_ptr<StreamOrder2> StreamOrder2::Copy() {
  auto stream_order = std::make_shared<StreamOrder2>();
  stream_order->index_at_each_expand_level_ = index_at_each_expand_level_;
  return stream_order;
}

void StreamOrder2::Expand(size_t index_in_this_level) {
  index_at_each_expand_level_.push_back(index_in_this_level);
}

void StreamOrder2::Collapse() { index_at_each_expand_level_.pop_back(); }

Stream::Stream(std::shared_ptr<Session> session) : session_(session) {}

std::shared_ptr<Session> Stream::GetSession() { return session_; }

void Stream::SetMaxBufferCount(size_t max_buffer_count) {
  max_buffer_count_ = max_buffer_count;
}

bool Stream::ReachEnd() {
  if (max_buffer_count_ == 0) {
    return false;
  }

  return max_buffer_count_ <= cur_buffer_count_;
}

size_t Stream::GetBufferCount() { return cur_buffer_count_; }

void Stream::IncreaseBufferCount() { ++cur_buffer_count_; }

void Stream::SetStreamMeta(std::shared_ptr<DataMeta> data_meta) {
  data_meta_ = data_meta;
}

std::shared_ptr<DataMeta> Stream::GetStreamMeta() { return data_meta_; }

std::shared_ptr<StreamOrder2> Stream::GetStreamOrder() { return stream_order_; }

void Stream::SetStreamOrder(std::shared_ptr<StreamOrder2> stream_order) {
  stream_order_ = stream_order;
}

}  // namespace modelbox
