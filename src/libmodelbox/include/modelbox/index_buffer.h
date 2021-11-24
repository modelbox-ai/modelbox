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


#ifndef MODELBOX_INDEX_BUFFER_H_
#define MODELBOX_INDEX_BUFFER_H_

#include "modelbox/buffer_list.h"
#include "modelbox/stream.h"

namespace modelbox {

// The same level stream should share one context，and this will bind to a
// BufferGroup

class FlowUnit;

class BufferGroup : public std::enable_shared_from_this<BufferGroup> {
 public:
  /**
   * @brief Construct a new BufferGroup object
   *
   */
  BufferGroup();

  /**
   * @brief Destructor
   *
   */
  virtual ~BufferGroup();

  /**
   * @brief Get the Group object,should be hidden
   *
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> GetGroup();

  /**
   * @brief Get the SubGroup Order,belong  to stream
   *
   * @return uint32_t the order
   */

  uint32_t GetOrder();

  /**
   * @brief Get the Group Sum object,belong  to stream
   * @param sum the group sum
   * @return Status {status} if success return STATUS_SUCCESS if There is none
   * return STATUS_NOTFOUND
   */
  Status GetGroupSum(uint32_t* sum);

  /**
   * @brief Get the Group Order object ,belong  to stream
   *
   * @param port_id
   * @return uint32_t
   */
  uint32_t GetGroupOrder(int port_id);

  /**
   * @brief Get the SubGroup belong to which port,belong  to stream
   *
   * @return uint32_t
   */

  uint32_t GetPortId();

  /**
   * @brief ,belong  to stream
   *
   * @param port_id
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> AddNewSubGroup(uint32_t port_id);

  /**
   * @brief ,belong  to stream
   *
   * @param port_id
   * @return true
   * @return false
   */
  bool IsFullGroup(uint32_t port_id);

  /**
   * @brief ,belong  to stream
   *
   * @param port_id
   */
  void FinishSubGroup(uint32_t port_id);

  /**
   * @brief ,belong  to stream
   *
   * @param port_id
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> AddOneMoreSubGroup(uint32_t port_id);

  /**
   * @brief Add a subgroup to group.It should be thread safe,belong  to stream
   *
   * @param start_flag if the subgroup is the start
   * @param end_flag if the subgroup is the end
   * @return std::shared_ptr<BufferGroup> return the subgroup if not success
   * return nullptr
   */
  std::shared_ptr<BufferGroup> AddSubGroup(bool start_flag, bool end_flag);

  /**
   * @brief Generate a same level BufferGroup,belong  to match
   *
   * @return std::shared_ptr<BufferGroup>  return the subgroup if not success
   * return nullptr
   */

  std::shared_ptr<BufferGroup> GenerateSameLevelGroup();

  /**
   * @brief Get the same level Group ptr ,belong  to match
   *
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> GetOneLevelGroup();

  /**
   * @brief Get the Up Level Group ptr,belong  to match
   *
   * @return std::shared_ptr<BufferGroup>
   */

  std::shared_ptr<BufferGroup> GetStreamLevelGroup();

  /**
   * @brief Get the Up Level Group ptr,belong  to stream
   *
   * @return std::shared_ptr<BufferGroup>
   */

  std::shared_ptr<BufferGroup> GetGroupLevelGroup();

  /**
   * @brief If the group is the start group,belong  to stream
   *
   * @return true
   * @return false
   */
  bool IsStartGroup();

  /**
   * @brief If the group is the end group,belong  to stream
   *
   * @return true
   * @return false
   */
  bool IsEndGroup();

  std::shared_ptr<DataMeta> GetDataMeta(uint32_t port) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (data_meta_map_.find(port) == data_meta_map_.end()) {
      return nullptr;
    }
    return data_meta_map_.find(port)->second;
  }

  void SetDataMeta(uint32_t port, std::shared_ptr<DataMeta> data_meta) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (data_meta_map_.find(port) == data_meta_map_.end()) {
      data_meta_map_.emplace(port, data_meta);
    }
  }

  std::shared_ptr<FlowUnitError> GetDataError(
      std::tuple<uint32_t, uint32_t> port) {
    std::lock_guard<std::mutex> lock(exception_mutex_);
    if (exception_map_.find(port) == exception_map_.end()) {
      return nullptr;
    }
    return exception_map_.find(port)->second;
  }

  void SetDataError(std::tuple<uint32_t, uint32_t> port,
                    std::shared_ptr<FlowUnitError> data_exception) {
    std::lock_guard<std::mutex> lock(exception_mutex_);
    if (exception_map_.find(port) == exception_map_.end()) {
      exception_map_.emplace(port, data_exception);
    }
  }

  std::shared_ptr<SessionContext> GetSessionContext();

  void SetSessionContex(std::shared_ptr<SessionContext> session_ctx);

  bool IsRoot();

  std::vector<uint32_t> GetSeqOrder();

 private:
  Status GetSum(uint32_t* sum,int port_id);
  std::shared_ptr<BufferGroup> InnerAddNewSubGroup(uint32_t port_id);
  void InnerFinishSubGroup(uint32_t port_id);

  std::shared_ptr<BufferGroup> GetRoot();
  /**
   * @brief Add a subgroup to group.It should be thread safe
   *
   * @param port_id if the subgroup is the start
   * @param start_flag if the subgroup is the start
   * @param end_flag if the subgroup is the end
   * @return std::shared_ptr<BufferGroup> return the subgroup if not success
   * return nullptr
   */
  std::shared_ptr<BufferGroup> AddSubGroup(uint32_t port_id, bool start_flag,
                                           bool end_flag);
  std::mutex add_mutex_;

  std::mutex meta_mutex_;

  std::mutex exception_mutex_;
  // flag mean the subgroup is the start one
  // belong to stream
  bool start_flag_;

  // flag mean the subgroup is the end one
  // belong to stream
  bool end_flag_;

  // the subgroup order
  // belong to stream
  uint32_t order_;

  // the subgroup belong the which port
  // belong to stream
  uint32_t port_id_;

  // port_id auto generate for subgroup
  // belong to stream
  uint32_t group_port_id_;

  // the point to group
  // common
  std::shared_ptr<BufferGroup> group_;

  // belong to stream
  std::unordered_map<uint32_t, uint32_t> port_start_map_;

  // record the subgroup's current order,if there's one new subgroup add to this
  // group port add 1 to the value
  // belong to stream
  std::unordered_map<uint32_t, uint32_t> port_order_map_;

  // record the subgroup's total sum,if the port is in port_sum_map_ it can
  // not add any subgroup to this port
  // belong to stream
  std::unordered_map<uint32_t, uint32_t> port_sum_map_;

  // belong to stream
  std::unordered_map<uint32_t, std::shared_ptr<BufferGroup>> last_bg_map_;

  // belong to stream
  std::unordered_map<uint32_t, std::shared_ptr<DataMeta>> data_meta_map_;

  // belong to stream
  std::map<std::tuple<uint32_t, uint32_t>, std::shared_ptr<FlowUnitError>>
      exception_map_;

  std::shared_ptr<SessionContext> session_context_;
};

class IndexMeta {
 public:
  IndexMeta(){};
  virtual ~IndexMeta(){};
  /**
   * @brief Set the BufferGroup
   *
   * @param bg the BufferGroup
   */
  void SetBufferGroup(std::shared_ptr<BufferGroup> bg);

  /**
   * @brief Get the BufferGroup
   *
   * @return std::shared_ptr<BufferGroup>  return nullptr if there is none
   */
  std::shared_ptr<BufferGroup> GetBufferGroup();

 private:
  std::shared_ptr<BufferGroup> group_index_;
};

/**
 * @brief The buffer list to save
 *
 */

class IndexBuffer {
 public:
  /**
   * @brief Construct a new IndexBuffer
   *
   */
  IndexBuffer();

  /**
   * @brief Construct a new IndexBuffer from other IndexBuffer
   *
   */
  IndexBuffer(IndexBuffer* other);

  /**
   * @brief Construct a new IndexBuffer without meta from other Buffer
   *
   * @param other
   */
  IndexBuffer(std::shared_ptr<Buffer> other);

  /**
   * @brief Destructor
   *
   */
  virtual ~IndexBuffer();

  /**
   * @brief Generate the root group and bind the IndexBuffer to the group
   *
   * @return true success to bind to the root
   * @return false failed to bind to the root
   */
  bool BindToRoot();

  /**
   * @brief Copy the buffer meta to other buffer,they are in the same level
   *
   * @param other the new  same level buffer
   * @return true succes to copy meta
   * @return false fail to copy meta
   */
  bool CopyMetaTo(std::shared_ptr<IndexBuffer> other);

  /**
   * @brief Generate a new down level group meta for this buffer,need to delete
   *
   * @param other the target buffer
   * @param start_flag whether the  down level  buffer is the start
   * @param end_flag whether the down level buffer is the end
   * @return true succes to generate the meta
   * @return false fail to generate the meta
   */

  bool BindDownLevelTo(std::shared_ptr<IndexBuffer>& other, bool start_flag,
                       bool end_flag);

  /**
   * @brief Generate a new up level group meta for this buffer,need to delete
   *
   * @param other the target buffer
   * @return true succes to generate the meta
   * @return false fail to generate the meta
   */
  bool BindUpLevelTo(std::shared_ptr<IndexBuffer>& other);

  /**
   * @brief Get the Same Level BufferGroup
   *
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> GetSameLevelGroup();

  /**
   * @brief Get the Up Level BufferGroup,belong to stream
   *
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> GetStreamLevelGroup();

  /**
   * @brief Clone this IndexBuffer to a new IndexBuffer,belong to stream
   *
   * @return std::shared_ptr<IndexBuffer> new IndexBuffer
   */
  std::shared_ptr<IndexBuffer> Clone();

  /**
   * @brief Get the IndexBuffer Priority
   *
   * @return int32_t
   */
  int32_t GetPriority();

  /**
   * @brief Set the IndexBuffer Priority
   *
   * @param priority
   */
  void SetPriority(int32_t priority);

  /**
   * @brief Get the BufferGroup,belong to stream
   *
   * @return std::shared_ptr<BufferGroup>  return nullptr if there is none
   */
  std::shared_ptr<BufferGroup> GetBufferGroup();

  /**
   * @brief Get the Buffer Poinr object,belong to stream
   *
   * @return std::shared_ptr<Buffer>
   */
  inline std::shared_ptr<Buffer> GetBufferPtr() { return buffer_ptr_; }

  /**
   * @brief Set the BufferGroup,belong to stream
   *
   * @param bg the BufferGroup
   */
  void SetBufferGroup(std::shared_ptr<BufferGroup> bg);

  /**
   * @brief Set the Data Meta object,belong to stream
   *
   * @param data_meta
   */
  void SetDataMeta(std::shared_ptr<DataMeta> data_meta);

  /**
   * @brief Get the Data Meta object,belong to stream
   *
   * @return std::shared_ptr<DataMeta>
   */
  std::shared_ptr<DataMeta> GetDataMeta();

  /**
   * @brief Get the Group Data Meta object,belong to stream
   *
   * @return std::shared_ptr<DataMeta>
   */
  std::shared_ptr<DataMeta> GetGroupDataMeta();

  bool IsPlaceholder();

  void MarkAsPlaceholder();

  /**
   * @brief Get the Seq Order object,belong to stream
   *
   * @return std::vector<uint32_t>
   */
  std::vector<uint32_t> GetSeqOrder();

 private:
  std::shared_ptr<BufferGroup> GetGroupLevelGroup();
  // the index buffer priority
  int32_t priority_{0};

  friend class IndexBufferList;

  IndexMeta index_info_;

  // A shared point to buffer
  std::shared_ptr<Buffer> buffer_ptr_;

  bool is_placeholder_{false};
};

using DataList = std::vector<std::shared_ptr<BufferList>>;
// IndexBufferList is a buffer list which all buffer belong to one stream
class IndexBufferList {
 public:
  /**
   * @brief Construct a new IndexBufferList
   */
  IndexBufferList(){};

  /**
   * @brief Construct a new IndexBufferList
   * @param buffer_vector A IndexBuffer Vector
   */
  IndexBufferList(std::vector<std::shared_ptr<IndexBuffer>> buffer_vector);

  /**
   * @brief destructor
   */
  virtual ~IndexBufferList(){};

  /**
   * @brief Get the Index Buffer object
   *
   * @param order the Buffer order in the BufferList
   * @return std::shared_ptr<IndexMeta> if order out of range return nullptr
   */
  std::shared_ptr<IndexBuffer> GetBuffer(uint32_t order);

  void ExtendBufferList(std::shared_ptr<IndexBufferList> buffer_list);

  /**
   * @brief Get the Data Meta object,belong to stream
   *
   * @return std::shared_ptr<DataMeta>
   */
  std::shared_ptr<DataMeta> GetDataMeta();

  /**
   * @brief Get the Group Data Meta object,belong to stream
   *
   * @return std::shared_ptr<DataMeta>
   */
  std::shared_ptr<DataMeta> GetGroupDataMeta();

  /**
   * @brief Set the Data Meta object,belong to stream
   *
   * @param data_meta
   */
  void SetDataMeta(std::shared_ptr<DataMeta> data_meta);

  /**
   * @brief Get the Data Error Index object,belong to stream
   *
   * @return std::tuple<uint32_t, uint32_t>
   */
  std::tuple<uint32_t, uint32_t> GetDataErrorIndex();

  /**
   * @brief Get the Data Error object,belong to stream
   *
   * @return std::shared_ptr<FlowUnitError>
   */
  std::shared_ptr<FlowUnitError> GetDataError();

  /**
   * @brief Get the Group Data Error Index object,belong to stream
   *
   * @return std::tuple<uint32_t, uint32_t>
   */
  std::tuple<uint32_t, uint32_t> GetGroupDataErrorIndex();

  std::shared_ptr<BufferGroup> GetStreamBufferGroup();

  int32_t GetPriority();

  /**
   * @brief Get the Order object,belong to stream
   *
   * @return uint32_t
   */
  uint32_t GetOrder();

  /**
   * @brief ,belong to stream
   *
   * @return true
   * @return false
   */
  bool IsStartStream();

  /**
   * @brief ,belong to stream
   *
   * @return true
   * @return false
   */
  bool IsEndStream();

  /**
   * @brief Get the Index Buffer Size
   *
   * @return uint32_t the Index Buffer Size
   */
  uint32_t GetBufferNum();

  void Clear();

  /**
   * @brief Get the Bufferlist object
   *
   * @return std::shared_ptr<BufferList>
   */
  std::vector<std::shared_ptr<Buffer>> GetBufferPtrList();

  bool BindToRoot();

  std::set<uint32_t> GetPlaceholderPos();

  void BackfillBuffer(std::set<uint32_t> id_set, bool is_placeholder = true);

  bool IsExpandBackfillBufferlist();

  /**
   * @brief  push back the buffer
   *
   * @param buffer
   */
  inline void PushBack(std::shared_ptr<IndexBuffer> buffer) {
    buffer_list_.push_back(buffer);
  }

  inline std::vector<std::shared_ptr<IndexBuffer>> GetIndexBufferVector() {
    return buffer_list_;
  }

 private:
  std::shared_ptr<FlowUnitError> error_;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_list_;
};

}  // namespace modelbox

#endif  // MODELBOX_BUFFER_H_
