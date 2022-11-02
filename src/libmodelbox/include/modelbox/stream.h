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

#ifndef MODELBOX_STREAM_H_
#define MODELBOX_STREAM_H_

#include <atomic>
#include <memory>
#include <unordered_map>

#include "modelbox/base/status.h"
#include "modelbox/buffer_index_info.h"

namespace modelbox {

class Session;

class DataMeta {
 public:
  DataMeta();

  DataMeta(const DataMeta &other);

  virtual ~DataMeta();

  void SetMeta(const std::string &key, std::shared_ptr<void> meta);

  std::shared_ptr<void> GetMeta(const std::string &key);

  std::unordered_map<std::string, std::shared_ptr<void>> GetMetas();

 private:
  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;
};

/**
 * @brief record stream order for each expand level
 **/
class StreamOrder {
 public:
  StreamOrder();

  bool operator<(const StreamOrder &other_stream_order);

  std::shared_ptr<StreamOrder> Copy();

  void Expand(size_t index_in_this_level);

  void Collapse();

 private:
  std::list<size_t> index_at_each_expand_level_;
};

class Stream {
 public:
  Stream(std::shared_ptr<Session> session);

  virtual ~Stream();

  std::shared_ptr<Session> GetSession();

  void SetMaxBufferCount(size_t max_buffer_count);

  bool ReachEnd();

  size_t GetBufferCount();

  void IncreaseBufferCount();

  void SetStreamMeta(std::shared_ptr<DataMeta> data_meta);

  std::shared_ptr<DataMeta> GetStreamMeta();

  std::shared_ptr<StreamOrder> GetStreamOrder();

  void SetStreamOrder(std::shared_ptr<StreamOrder> stream_order);

 private:
  std::shared_ptr<Session> session_;
  std::atomic_size_t cur_buffer_count_{0};
  size_t max_buffer_count_{0};
  std::shared_ptr<DataMeta> data_meta_;

  std::shared_ptr<StreamOrder> stream_order_ = std::make_shared<StreamOrder>();
};

class StreamPtrOrderCmp {
 public:
  bool operator()(const std::shared_ptr<Stream> &s1,
                  const std::shared_ptr<Stream> &s2) const {
    return *(s1->GetStreamOrder()) < *(s2->GetStreamOrder());
  }
};

}  // namespace modelbox
#endif