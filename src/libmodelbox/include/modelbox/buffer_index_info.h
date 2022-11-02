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

#ifndef MODELBOX_BUFFER_INDEX_INFO_H_
#define MODELBOX_BUFFER_INDEX_INFO_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace modelbox {

class Stream;
class BufferIndexInfo;
class Node;
class Buffer;
class DataError;

/**
 * @brief define all process for buffer
 **/
enum class BufferProcessType : size_t {
  EXPAND,
  CONDITION_START,
  COLLAPSE,
  ORIGIN
};

/**
 * @brief during buffer process, some operation is key operation
 * we need direct record these operation
 * key operation: expand, conditino_start
 * for each key operation, we make new inherit info, so we could trace back
 * there key operation
 **/
class BufferInheritInfo {
 public:
  void SetType(BufferProcessType type);

  BufferProcessType GetType();

  void SetInheritFrom(const std::shared_ptr<BufferIndexInfo> &buffer_index);

  std::shared_ptr<BufferIndexInfo> GetInheritFrom();

  size_t GetDeepth();

 private:
  BufferProcessType type_{BufferProcessType::EXPAND};
  std::shared_ptr<BufferIndexInfo> inherit_from_buffer_;
  size_t inherit_deepth_{0};
};

/**
 * @brief record info for each process at buffer
 * easy to trace through buffer
 **/
class BufferProcessInfo {
 public:
  void SetParentBuffers(
      const std::string &port_name,
      std::list<std::shared_ptr<BufferIndexInfo>> &&port_buffers);

  const std::map<std::string, std::list<std::shared_ptr<BufferIndexInfo>>>
      &GetParentBuffers();

  void SetType(BufferProcessType type);

  BufferProcessType GetType();

 private:
  std::map<std::string, std::list<std::shared_ptr<BufferIndexInfo>>>
      parent_buffers_;
  BufferProcessType type_{BufferProcessType::ORIGIN};
};

/**
 * @brief record index info in stream for each buffer
 **/
class BufferIndexInfo {
 public:
  BufferIndexInfo();

  virtual ~BufferIndexInfo();

  void SetInheritInfo(std::shared_ptr<BufferInheritInfo> inherit_info);

  std::shared_ptr<BufferInheritInfo> GetInheritInfo();

  void SetStream(std::shared_ptr<Stream> stream_belong_to);

  std::shared_ptr<Stream> GetStream();

  void SetIndex(size_t index);

  size_t GetIndex();

  bool IsFirstBufferInStream();

  /**
   * @brief mark end for stream
   **/
  void MarkAsEndFlag();

  bool IsEndFlag();

  /**
   * @brief in case: user drop one buffer, we need keep index
   **/
  void MarkAsPlaceholder();

  bool IsPlaceholder();

  void SetProcessInfo(std::shared_ptr<BufferProcessInfo> process_info);

  std::shared_ptr<BufferProcessInfo> GetProcessInfo();

 private:
  std::shared_ptr<Stream> stream_belong_to_;
  size_t index_in_current_stream_{0};
  std::shared_ptr<BufferInheritInfo> inherit_info_;
  std::shared_ptr<BufferProcessInfo>
      process_info_;  // record how to generate this buffer

  bool is_end_flag_{false};
  bool is_placeholder_{false};
};

/**
 * @brief To access manage info in buffer
 **/
class BufferManageView {
 public:
  static std::shared_ptr<BufferIndexInfo> GetIndexInfo(
      const std::shared_ptr<Buffer> &buffer);

  static void SetIndexInfo(const std::shared_ptr<Buffer> &buffer,
                           std::shared_ptr<BufferIndexInfo> buffer_index_info);

  /**
   * @brief buffer generate from input, we call input buffer as parent
   **/
  static std::shared_ptr<BufferIndexInfo> GetFirstParentBuffer(
      const std::shared_ptr<Buffer> &buffer);

  static void SetPriority(const std::shared_ptr<Buffer> &buffer, int priority);

  static int GetPriority(const std::shared_ptr<Buffer> &buffer);

  static void SetError(const std::shared_ptr<Buffer> &buffer,
                       const std::shared_ptr<DataError> &data_error);

  static std::shared_ptr<DataError> GetError(
      const std::shared_ptr<Buffer> &buffer);

  /**
   * @brief record the direct input where this buffer comes from
   **/
  template <typename T>
  static void GenProcessInfo(
      const std::unordered_map<std::string, T> &parent_data,
      size_t data_count_per_port,
      const std::function<std::shared_ptr<Buffer>(const T &container,
                                                  size_t idx)> &get_buffer_at,
      std::vector<std::shared_ptr<BufferProcessInfo>> &process_info_list,
      bool all_in_one_process = false);
};

template <typename T>
void BufferManageView::GenProcessInfo(
    const std::unordered_map<std::string, T> &parent_data,
    size_t data_count_per_port,
    const std::function<std::shared_ptr<Buffer>(const T &container, size_t idx)>
        &get_buffer_at,
    std::vector<std::shared_ptr<BufferProcessInfo>> &process_info_list,
    bool all_in_one_process) {
  if (all_in_one_process) {
    process_info_list.push_back(std::make_shared<BufferProcessInfo>());
  } else {
    process_info_list.reserve(data_count_per_port);
    for (size_t i = 0; i < data_count_per_port; ++i) {
      process_info_list.push_back(std::make_shared<BufferProcessInfo>());
    }
  }

  for (auto &port_data_item : parent_data) {
    auto &port_name = port_data_item.first;
    auto &data_list = port_data_item.second;
    auto index_info_list =
        std::make_shared<std::list<std::shared_ptr<BufferIndexInfo>>>();
    for (size_t i = 0; i < data_count_per_port; ++i) {
      auto buffer = get_buffer_at(data_list, i);
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      index_info_list->push_back(index_info);
      if (!all_in_one_process) {
        auto inherit_info = process_info_list[i];
        inherit_info->SetParentBuffers(port_name, std::move(*index_info_list));
        index_info_list->clear();
      }
    }
    if (all_in_one_process) {
      auto inherit_info = process_info_list.front();
      inherit_info->SetParentBuffers(port_name, std::move(*index_info_list));
    }
  }
}

}  // namespace modelbox

#endif  // MODELBOX_BUFFER_INDEX_INFO_H_