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

#ifndef MODELBOX_BUFFER_LIST_H_
#define MODELBOX_BUFFER_LIST_H_

#include <modelbox/base/device.h>
#include <modelbox/base/log.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer.h>
#include <modelbox/buffer_type.h>
#include <modelbox/tensor.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <unordered_map>

namespace modelbox {

class BufferList;
using BufferListMap =
    std::unordered_map<std::string, std::shared_ptr<BufferList>>;

/**
 * @brief Buffer list
 */
class BufferList {
 public:
  /**
   * @brief Buffer list
   */
  BufferList();

  /**
   * @brief Buffer list with device
   * @param device pointer to device
   * @param device_mem_flags Flags to crete device mem
   */
  BufferList(const std::shared_ptr<Device>& device,
             uint32_t device_mem_flags = 0);

  /**
   * @brief Buffer list from buffer
   * @param buffer pointer to buffer
   */
  BufferList(const std::shared_ptr<Buffer>& buffer);

  /**
   * @brief Buffer list from vector of buffers
   * @param buffer_vector buffer vector
   */
  BufferList(const std::vector<std::shared_ptr<Buffer>>& buffer_vector);

  /**
   * @brief BufferList from other BufferList
   * @param other BufferList
   */
  BufferList(const BufferList& other);

  virtual ~BufferList();

  /**
   * @brief Builder buffer, create memory
   * @param data_size_list buffer size list
   * @param contiguous all buffer in single mem area
   */
  virtual Status Build(const std::vector<size_t>& data_size_list,
                       bool contiguous = true);

  /**
   * @brief Builder buffer from host memory
   * @param data_size_list buffer size list
   * @param data memory pointer to host.
   * @param data_size host memory size.
   * @param func host memory delete or free function.
   */
  virtual Status BuildFromHost(const std::vector<size_t>& data_size_list,
                               void* data, size_t data_size,
                               const DeleteFunction& func = nullptr);

  /**
   * @brief Get bufferlist size
   * @return bufffer list size
   */
  virtual size_t Size() const;

  /**
   * @brief Get bufferlist memory bytes number
   * @return buffer list memory bytes number
   */
  virtual size_t GetBytes() const;

  /**
   * @brief Buffer iterator begin
   * @return iterator
   */
  virtual std::vector<std::shared_ptr<Buffer>>::iterator begin();

  /**
   * @brief Buffer iterator begin
   * @return iterator
   */
  virtual std::vector<std::shared_ptr<Buffer>>::const_iterator begin() const;

  /**
   * @brief Buffer iterator end
   * @return iterator
   */
  virtual std::vector<std::shared_ptr<Buffer>>::iterator end();

  /**
   * @brief Buffer iterator end
   * @return iterator
   */
  virtual std::vector<std::shared_ptr<Buffer>>::const_iterator end() const;

  /**
   * @brief Get buffer at pos
   * @param pos position of buffer
   * @return pointer to buffer
   */
  virtual std::shared_ptr<Buffer>& operator[](size_t pos);

  /**
   * @brief Get buffer at pos
   * @param pos position of buffer
   * @return pointer to buffer
   */
  virtual const std::shared_ptr<Buffer>& operator[](size_t pos) const;

  /**
   * @brief Get buffer at pos
   * @param idx position of buffer
   * @return pointer to buffer
   */
  virtual std::shared_ptr<Buffer> At(size_t idx);

  /**
   * @brief Get buffer at pos
   * @param idx position of buffer
   * @return pointer to buffer
   */
  virtual std::shared_ptr<Buffer> At(size_t idx) const;

  /**
   * @brief Push new buffer to buffer list
   * @param buf pointer to buffer
   */
  virtual void PushBack(const std::shared_ptr<Buffer>& buf);

  /**
   * @brief Assign buffer list
   * @param buffer_list buffer list to assign
   */
  virtual void Assign(const std::vector<std::shared_ptr<Buffer>>& buffer_list);

  /**
   * @brief Swap buffer list
   * @param buffer_list buffer list to swap
   */
  virtual void Swap(std::vector<std::shared_ptr<Buffer>>& buffer_list);

  /**
   * @brief Get mutable buffer data pointer
   * @param idx buffer index
   * @return buffer data pointer
   */
  virtual void* MutableBufferData(size_t idx);

  /**
   * @brief Get unmutable buffer data pointer
   * @param idx buffer index
   * @return buffer data pointer
   */
  virtual const void* ConstBufferData(size_t idx) const;

  /**
   * @brief Get mutable buffer data pointer from begining
   * @return buffer data pointer from begining
   */
  virtual void* MutableData();

  /**
   * @brief Get unmutable buffer data pointer from begining
   * @return buffer data pointer from begining
   */
  virtual const void* ConstData() const;

  /**
   * @brief Set meta to all buffers
   * @param key meta key
   * @param value meta value
   */
  template <typename T>
  void Set(const std::string& key, T&& value) {
    for (const auto& buffer : buffer_list_) {
      buffer->Set(key, value);
    }
  }

  /**
   * @brief Copy meta from another buffer list
   * @param bufferlist another buffer list
   * @param is_override override exists meta.
   */
  virtual Status CopyMeta(const std::shared_ptr<BufferList>& bufferlist,
                          bool is_override = false);

  /**
   * @brief Get device memory pointer of buffer list
   * @return pointer to device memory
   */
  std::shared_ptr<DeviceMemory> GetDeviceMemory();

  /**
   * @brief Get device memory pointer of buffer list
   * @return pointer to device memory
   */
  std::vector<std::shared_ptr<DeviceMemory>> GetAllBufferDeviceMemory();

  /**
   * @brief Get device of buffer list
   * @return pointer to device
   */
  std::shared_ptr<Device> GetDevice();

  /**
   * @brief Make all buffer memory contiguous
   * @return make result
   */
  Status MakeContiguous();

  /**
   * @brief Copy a vector of buffer into a buffer list
   * @param buffer_vector the vector of buffer
   */
  void Copy(const std::vector<std::shared_ptr<Buffer>>& buffer_vector);
  /**
   * @brief Reset buffer list
   * @return reset result
   */
  Status Reset();

  /**
   * @brief Move all buffer to one device
   * if buffer_list has device, then move this device
   * if buffer_list has no device, move to the device of first buffer
   * @return Move result
   */
  Status MoveAllBufferToTargetDevice();

  /**
   * @brief Set error
   * @param error_code buffer error code
   * @param error_msg buffer error message
   * @return buffer reference
   */
  void SetError(const std::string& error_code, const std::string& error_msg);

  /**
   * @brief push current device data to buffer list.
   * support host data to cpu
   * @param device_data data in current flowunit device
   * @param data_size size of data
   * @param func to manage device_data, avoid extra copy
   * @return result
   */
  Status EmplaceBack(void* device_data, size_t data_size,
                     const DeleteFunction& func = nullptr);

  /**
   * @brief push current device data to buffer list.
   * support host data to cpu
   * @param device_data data in current flowunit device
   * @param data_size size of data
   * @return result
   */
  Status EmplaceBack(const std::shared_ptr<void>& device_data,
                     size_t data_size);

  /**
   * @brief push host data to device buffer list.
   * not recommend for host data to cpu
   * @param host_data host data
   * @param data_size size of data
   * @return result
   */
  Status EmplaceBackFromHost(void* host_data, size_t data_size);

  /**
   * @brief get front buffer in bufferlist
   * @return null if no front buffer
   */
  std::shared_ptr<Buffer> Front();

  /**
   * @brief get last buffer in bufferlist
   * @return null if no back buffer
   */
  std::shared_ptr<Buffer> Back();

  /**
   * @brief test whether buffer list supports mem contiguous
   * @return support or not
   **/
  bool SupportMemContiguous();

 private:
  friend class FlowUnitExecData;
  friend class FlowUnitGroup;
  friend class TensorList;

  void SetNoContiguous();
  bool IsContiguous() const;
  Status SetMutable(bool is_mutable);
  Status CopyToNewBufferList(std::shared_ptr<DeviceMemory>& dev_mem);
  Status GenerateDeviceMemory(
      const std::vector<std::shared_ptr<DeviceMemory>>& buffer_dev_mems);
  bool is_contiguous_{false};
  std::shared_ptr<DeviceMemory> dev_mem_;
  uint32_t dev_mem_flags_{0};
  std::vector<std::shared_ptr<Buffer>> buffer_list_;

  Status BuildContiguous(const std::shared_ptr<Device>& device,
                         const std::vector<size_t>& data_size_list);
  Status BuildSeparate(const std::shared_ptr<Device>& device,
                       const std::vector<size_t>& data_size_list);
};

}  // namespace modelbox

#endif  // MODELBOX_BUFFER_LIST_H_
