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

#ifndef MODELBOX_MATCH_STREAM_H_
#define MODELBOX_MATCH_STREAM_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/inner_event.h"

namespace modelbox {

class InPort;

using PortDataMap =
    std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>;

class MatchKey {
 public:
  static MatchKey* AsKey(BufferIndexInfo* match_at_buffer);

  static MatchKey* AsKey(Stream* match_at_stream);
};

class MatchStreamData {
 public:
  MatchStreamData();
  virtual ~MatchStreamData();

  void SetStreamMatchKey(MatchKey* match_at);

  MatchKey* GetStreamMatchKey();

  void SetSession(std::shared_ptr<Session> session);

  std::shared_ptr<Session> GetSession();

  void SetEvent(std::shared_ptr<FlowUnitInnerEvent>& event);

  std::shared_ptr<FlowUnitInnerEvent> GetEvent();

  void SetBufferList(std::shared_ptr<PortDataMap> port_buffers);

  std::shared_ptr<PortDataMap> GetBufferList() const;

  size_t GetDataCount() const;

 private:
  MatchKey* match_at_{nullptr};
  std::shared_ptr<Session> session_;
  std::shared_ptr<FlowUnitInnerEvent> event_;
  std::shared_ptr<PortDataMap> port_to_stream_data_;
};

class MatchBufferCache {
 public:
  MatchBufferCache(
      size_t port_count,
      std::unordered_map<std::string, size_t>* stream_count_each_port);
  virtual ~MatchBufferCache();

  Status CacheBuffer(const std::string& port_name,
                     std::shared_ptr<Buffer>& buffer);

  bool IsMatched() const;

  bool IsEndFlag() const;

  const std::unordered_map<std::string, std::shared_ptr<Buffer>>& GetBuffers()
      const;

 private:
  size_t port_count_;
  std::unordered_map<std::string, size_t> end_flag_count_;
  bool is_end_flag_{false};
  bool is_placeholder_{false};
  std::unordered_map<std::string, std::shared_ptr<Buffer>> buffer_cache_;
  std::unordered_map<std::string, size_t> cur_buffer_count_each_port_;
  std::unordered_map<std::string, size_t>* stream_count_each_port_;
};

/**
 * Manage stream info for each input port
 * will use this info to analysis app problem and performance
 **/
class InPortStreamInfo {
 public:
  void ReceiveBuffer(std::shared_ptr<Buffer>& buffer);

  size_t GetReceivedBufferCount();

  size_t GetReceivedStreamCount();

  bool ReachEnd();
};

class MatchStreamCache {
 public:
  MatchStreamCache(
      std::string node_name, size_t port_count,
      std::unordered_map<std::string, size_t>* stream_count_each_port);

  virtual ~MatchStreamCache();

  Status CacheBuffer(const std::string& port_name,
                     std::shared_ptr<Buffer>& buffer);

  std::shared_ptr<PortDataMap> PopReadyMatchBuffers(bool in_order,
                                                    bool gather_all);

  void SetSession(std::shared_ptr<Session> session);

  std::shared_ptr<Session> GetSession();

  bool IsStreamEnd();

  size_t TotalInputCount();

 private:
  void UpdateInputStreamInfo(const std::string& port_name,
                             std::shared_ptr<Buffer>& buffer);

  std::string node_name_;
  std::map<size_t, std::shared_ptr<MatchBufferCache>>
      match_buffers_;  // ordered by buffer index
  std::map<size_t, std::shared_ptr<MatchBufferCache>>
      ready_match_buffers_;  // ordered by buffer index, all port buffer
                             // received

  size_t port_count_;
  std::unordered_map<std::string, size_t>* stream_count_each_port_;

  std::unordered_map<std::string, std::shared_ptr<InPortStreamInfo>>
      in_port_stream_info_map_;

  size_t index_in_order_{0};
  bool end_flag_received_{false};
  size_t total_input_count_in_stream_{0};
  size_t cur_input_count_in_stream_{0};

  std::shared_ptr<Session> session_;
};

class InputMatchStreamManager {
 public:
  InputMatchStreamManager(std::string node_name, size_t queue_size,
                          size_t port_count);

  virtual ~InputMatchStreamManager();

  size_t GetInputStreamCount();

  void SetInputBufferInOrder(bool is_input_in_order);

  void SetInputStreamGatherAll(bool need_gather_all);

  void UpdateStreamCountEachPort(
      std::unordered_map<std::string, size_t>&& stream_count_each_port);

  Status LoadData(std::vector<std::shared_ptr<InPort>>& data_ports,
                  const std::function<bool(std::shared_ptr<Buffer>)>&
                      drop_filter = nullptr);

  Status GenMatchStreamData(
      std::list<std::shared_ptr<MatchStreamData>>& match_stream_list);

  void Clean();

 private:
  Status CacheBuffer(const std::string& port_name,
                     std::shared_ptr<Buffer>& buffer, size_t backward_level);

  void IncreaseOnePortBufferCount(const std::string& port_name,
                                  size_t count = 1);

  void DecreaseAllPortBufferCount(size_t count = 1);

  size_t GetReadCount(const std::string& port_name);

  MatchKey* GetInputStreamMatchKey(
      const std::shared_ptr<BufferIndexInfo>& index_info,
      size_t backward_level);

  bool InitInheritBackwardLevel(
      std::vector<std::shared_ptr<InPort>>& data_ports);

  std::string node_name_;
  size_t queue_size_{0};
  size_t port_count_{0};
  std::unordered_map<std::string, size_t> stream_count_each_port_;

  bool need_gather_all_{false};
  bool is_input_in_order_{false};
  std::unordered_map<MatchKey*, std::shared_ptr<MatchStreamCache>>
      match_stream_cache_map_;
  std::unordered_map<std::string, size_t> port_inherit_backward_level_;

  const size_t max_cache_count_{16384};
  std::unordered_map<std::string, size_t> port_cache_count_map_;
};

class OutputMatchStream {
 public:
  void SetSession(std::shared_ptr<Session> session);

  std::shared_ptr<Session> GetSession();

  size_t Size();

  bool Empty();

  std::shared_ptr<Stream> GetStream(const std::string& port_name);

  std::shared_ptr<Stream> CreateStream(const std::string& port_name);

 private:
  std::unordered_map<std::string, std::shared_ptr<Stream>> port_stream_map_;

  std::shared_ptr<Session> session_;
};

class OutputMatchStreamManager {
 public:
  OutputMatchStreamManager(std::string node_name,
                           std::set<std::string>&& output_port_names);

  virtual ~OutputMatchStreamManager();

  size_t GetOutputStreamCount();

  void SetNeedNewIndex(bool need_new_index);

  Status UpdateStreamInfo(
      const std::unordered_map<
          std::string, std::vector<std::shared_ptr<Buffer>>>& stream_data_map,
      const std::unordered_map<std::string, std::shared_ptr<DataMeta>>&
          port_stream_meta,
      const std::shared_ptr<Session>& session);

  void Clean();

 private:
  MatchKey* GetOutputStreamMatchKey(
      const std::unordered_map<
          std::string, std::vector<std::shared_ptr<Buffer>>>& stream_data_map);

  void GenerateOutputStream(
      OutputMatchStream& output_match_stream,
      const std::unordered_map<
          std::string, std::vector<std::shared_ptr<Buffer>>>& stream_data_map,
      const std::unordered_map<std::string, std::shared_ptr<DataMeta>>&
          port_stream_meta,
      const std::shared_ptr<Session>& session);

  void SetIndexInStream(const std::shared_ptr<BufferIndexInfo>& buffer_index,
                        const std::shared_ptr<Stream>& stream);

  std::string node_name_;
  std::set<std::string> output_port_names_;
  bool need_new_index_{false};

  std::unordered_map<MatchKey*, OutputMatchStream> output_stream_map_;
};
}  // namespace modelbox

#endif  // MODELBOX_MATCH_STREAM_H_