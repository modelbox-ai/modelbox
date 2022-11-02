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

#ifndef MODELBOX_DEVICE_MEMORY_H_
#define MODELBOX_DEVICE_MEMORY_H_

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "modelbox/base/status.h"

namespace modelbox {

/**
 * @brief Test mem aligned
 * @param addr Target mem to test
 * @param align Mem need align to
 */
inline bool IsMemAligned(uintptr_t addr, uintptr_t align) {
  return addr % align == 0;
}

enum class DeviceMemoryCopyKind { FromHost, ToHost, SameDeviceType };

class Device;
class DeviceMemoryManager;

/**
 * Simple device memory manage, cloud share one raw memory block
 * Opertation: resize, copy, slice
 * |   <- Raw memory block ->  |
 *  offset -> | <- size -> |
 *            | <- capacity -> |
 */
class DeviceMemory : public std::enable_shared_from_this<DeviceMemory> {
  friend class Device;
  friend class CpuMemory;

 public:
  DeviceMemory(const DeviceMemory &deviceMemory) = delete;
  DeviceMemory &operator=(const DeviceMemory &deviceMemory) = delete;
  DeviceMemory(const DeviceMemory &&deviceMemory) = delete;
  DeviceMemory &operator=(const DeviceMemory &&deviceMemory) = delete;

  virtual ~DeviceMemory();

  /**
   * @brief Get memory pointer to access data, memory must be mutable
   * @return Memory pointer if mutable or nullptr
   */
  template <typename T>
  std::shared_ptr<T> GetPtr() {
    if (!is_content_mutable_) {
      return nullptr;
    }

    std::shared_ptr<T> data_ptr(
        (T *)((uint8_t *)device_mem_ptr_.get() + offset_), [](T *ptr) {});
    return data_ptr;
  };

  /**
   * @brief Get const memory pointer to read data
   * @return Const memory pointer
   */
  template <typename T>
  std::shared_ptr<const T> GetConstPtr() const {
    std::shared_ptr<T> data_ptr(
        (T *)((uint8_t *)device_mem_ptr_.get() + offset_), [](T *ptr) {});
    return data_ptr;
  }

  /**
   * @brief Mutable if memory content can be modified
   * @return Mutable
   */
  bool IsContentMutable() const;

  /**
   * @brief Mutable if memory content can be modified
   * @param content_mutable Content mutable
   */
  Status SetContentMutable(bool content_mutable);

  /**
   * @brief Get memory size, 0 if null
   * @return memory size
   */
  size_t GetSize() const;

  /**
   * @brief Get memory capacity, 0 if null
   * @return memory capacity
   */
  size_t GetCapacity() const;

  /**
   * @brief Get memory id
   * @return Memory id
   */
  std::string GetMemoryID() const;

  /**
   * @brief Get device that memory located
   * @return Device
   */
  std::shared_ptr<Device> GetDevice() const;

  /**
   * @brief Get device memory flag.
   * @return device memory flag
   */
  uint32_t GetMemFlags() const;

  /**
   * @brief Check this memory belong to host
   * @return Host or not
   */
  bool IsHost() const;

  /**
   * @brief Check memory on same device
   * @param dev_mem other device memory
   * @return same or not
   */
  bool IsSameDevice(const std::shared_ptr<DeviceMemory> &dev_mem);

  /**
   * @brief Check memory is continguous in same mem block strictly
   *  |-- mem1 --|-- mem2 --|-- mem3 --|
   *  |---------    mem block ---------|
   * @param mem_list list of mem to judge
   * @param with_order
   *  true: mem order in list should be same with order in mem block
   *  false: mem order in list can be different with order in mem block
   * @return continguous or not
   */
  static bool IsContiguous(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
      bool with_order = true);

  /**
   * @brief Combine mem to one mem block
   * @param mem_list to combine.
   * @param target_device target device.
   * @param target_mem_flags Flags to create device memory
   * @return Mem block
   */
  static std::shared_ptr<DeviceMemory> Combine(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
      const std::shared_ptr<Device> &target_device = nullptr,
      uint32_t target_mem_flags = 0);

  /**
   * @brief Count mem total size
   * @param mem_list to count
   * @param total_size mem size returned
   * @return Status
   */
  static Status CountMemSize(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
      size_t &total_size);

  /**
   * @brief Check memory out of bound
   * @return Result of verify, 0 is ok
   */
  virtual Status Verify() const;

  /**
   * @brief Resize memory, but will not exceed capacity
   * @param new_size New memory size
   * @return Status
   */
  Status Resize(size_t new_size);

  /**
   * @brief Realloc memory block
   * @param new_capacity New memory size
   * @return Status
   */
  Status Realloc(size_t new_capacity);

  /**
   * @brief Read data from other device memory
   * @param src_memory Memory read from
   * @param src_offset Offset in the memory read from
   * @param src_size Size in the memory read from
   * @param dest_offset Offset in memory write to
   * @return Status
   */
  virtual Status ReadFrom(const std::shared_ptr<const DeviceMemory> &src_memory,
                          size_t src_offset, size_t src_size,
                          size_t dest_offset = 0);

  /**
   * @brief Write data to other device memory
   * @param dest_memory Memory write to
   * @param src_offset Offset in the memory read from
   * @param src_size Size in the memory read from
   * @param dest_offset Offset in memory write to
   * @return Status
   */
  Status WriteTo(const std::shared_ptr<DeviceMemory> &dest_memory,
                 size_t src_offset, size_t src_size,
                 size_t dest_offset = 0) const;
  /**
   * @brief If capacity of this is enough, only data copy happend.
   * otherwise, new memory is allocated
   * @return Memory with append data
   */
  std::shared_ptr<DeviceMemory> Append(
      const std::shared_ptr<DeviceMemory> &dev_mem);

  /**
   * @brief If capacity of this is enough, only data copy happend.
   * otherwise, new memory is allocated
   * @return Memory with append data
   */
  std::shared_ptr<DeviceMemory> Append(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list);

  /**
   * @brief A new device memory point to the part of this mem, same mem block in
   * low level
   * @return New device memory point to this mem
   */
  std::shared_ptr<DeviceMemory> Cut(size_t offset, size_t size);

  /**
   * @brief A new device memory with new mem block
   *  Data in param will not copy
   * @return New device memory with data you want
   */
  std::shared_ptr<DeviceMemory> Delete(size_t offset, size_t size);

  /**
   * @brief A new device memory with new mem block
   *  Data in param will not be copied
   * @return New device memory with data you want
   */
  std::shared_ptr<DeviceMemory> Delete(size_t offset, size_t size,
                                       size_t capacity);

  /**
   * @brief A new device memory with new mem block
   *  Data in param will be copied
   * @return New device memory with data you want
   */
  std::shared_ptr<DeviceMemory> Copy(size_t offset, size_t size);

  /**
   * @brief A new device memory with new mem block
   *  Data in param will be copied
   * @return New device memory with data you want
   */
  std::shared_ptr<DeviceMemory> Copy(size_t offset, size_t size,
                                     size_t capacity);
  /**
   * @brief A new device memory with full data
   * @param is_copy means device memory will share one mem block or not
   * @return A new device memory with data
   */
  std::shared_ptr<DeviceMemory> Clone(bool is_copy = false);

  /* memory protect magic */
  static const uint64_t MEM_MAGIC_CODE;

 protected:
  bool is_host_mem_{false};
  std::shared_ptr<Device> device_;
  std::shared_ptr<DeviceMemoryManager> mem_mgr_;
  std::shared_ptr<void> device_mem_ptr_;
  size_t offset_{0};
  size_t size_{0};
  size_t capacity_{0};
  std::string memory_id_;
  bool is_content_mutable_{true};
  uint32_t mem_flags_{0};

  /**
   * @brief Construct a device memory with physical mem ptr, called by device
   * @param device Memory belong to
   * @param mem_mgr device manager
   * @param device_mem_ptr shared_ptr Memory pointer
   * @param size Memory size
   * @param is_host_mem is host memory, default is false.
   */
  DeviceMemory(const std::shared_ptr<Device> &device,
               const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
               const std::shared_ptr<void> &device_mem_ptr, size_t size,
               bool is_host_mem = false);

  void SetMemFlags(uint32_t mem_flags);

  /**
   * @brief Check param for readFrom function
   * @param src_memory source device memory
   * @param src_offset memory offset
   * @param src_size memory size
   * @param dest_offset dest memory offset
   * @return Is param ok
   */
  bool CheckReadFromParam(const std::shared_ptr<const DeviceMemory> &src_memory,
                          size_t src_offset, size_t src_size,
                          size_t dest_offset);

  virtual Status CopyExtraMetaTo(std::shared_ptr<DeviceMemory> &device_mem);

  virtual Status CombineExtraMeta(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list);

 private:
  void UpdateMemID(void *device_mem_ptr);

  /**
   * @brief Check param for Realloc function
   * @param new_size New size for device memory
   * @return Is param ok
   */
  bool CheckReallocParam(size_t new_capacity);

  /**
   * @brief We need host to transfer data in different type devices
   * @param src_memory Source memory
   * @param src_offset Source memory offset
   * @param src_size Source memory size
   * @param dest_offset Destination memory offset
   */
  Status TransferInHost(const std::shared_ptr<const DeviceMemory> &src_memory,
                        size_t src_offset, size_t src_size, size_t dest_offset);
  /**
   * @brief Transfer data by specified device
   * @param src_memory Source memory
   * @param src_offset Source memory offset
   * @param src_size Source memory size
   * @param dest_offset Destination memory offset
   */
  Status TransferInDevice(const std::shared_ptr<const DeviceMemory> &src_memory,
                          size_t src_offset, size_t src_size,
                          size_t dest_offset);

  std::shared_ptr<DeviceMemory> PrepareAppendMem(size_t append_size);

  Status AppendData(const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
                    std::shared_ptr<DeviceMemory> &target_device_mem);

  Status MemAcquire(const std::shared_ptr<void> &mem_ptr, size_t size);

  static std::shared_ptr<DeviceMemory> CombineContinuous(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
      size_t total_size, const std::shared_ptr<Device> &target_device);

  static std::shared_ptr<DeviceMemory> CombineFragment(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
      size_t total_size, std::shared_ptr<Device> target_device,
      uint32_t target_mem_flags);
};

/**
 * @brief device memory manager
 */
class DeviceMemoryManager
    : public std::enable_shared_from_this<DeviceMemoryManager> {
 public:
  DeviceMemoryManager(std::string device_id);

  virtual ~DeviceMemoryManager();

  /**
   * @brief Set allocatable memory limit
   * @param mem_quota quota memory size
   */
  void SetMemQuota(size_t mem_quota);

  /**
   * @brief Get allocatable memory limit
   * @return Memory limit
   */
  size_t GetMemQuota() const;

  /**
   * @brief Get allocated memory size
   * @return Allocated memory size
   */
  size_t GetAllocatedMemSize() const;

  /**
   * @brief Preservce memory alloc
   * @param size Memory size to allocate
   * @return Is param ok
   */
  bool PreserveMem(size_t size);

  /**
   * @brief Restore preserved size
   * @param size Memory size to restore
   */
  void RestoreMem(size_t size);

  virtual std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) = 0;
  /**
   * @brief Implement by specified device, alloc memory
   * @param size Memory size to allocate
   * @param mem_flags Flags to create device memory
   * @return Device memory in shared ptr
   */
  virtual std::shared_ptr<void> AllocSharedPtr(size_t size, uint32_t mem_flags);

  /**
   * @brief Implement by specified device, alloc memory
   * @param size Memory size to allocate
   * @param mem_flags Flags to create device memory
   * @return Device memory in shared ptr
   */
  virtual void *Malloc(size_t size, uint32_t mem_flags) = 0;

  /**
   * @brief Implement by specified device, free memory
   * @param mem_ptr Memory to free
   * @param mem_flags Flags of device memory
   */
  virtual void Free(void *mem_ptr, uint32_t mem_flags) = 0;

  /**
   * @brief Write host data to device by raw pointer
   * @param host_data Host data to read
   * @param host_size Host data size
   * @param device_buffer Device buffer to write
   * @param device_size Device buffer size
   * @return Status
   */
  virtual Status Write(const void *host_data, size_t host_size,
                       void *device_buffer, size_t device_size);

  /**
   * @brief Read device data to host by raw pointer
   * @param device_data Device data to read
   * @param device_size Device data size
   * @param host_buffer Host buffer to write
   * @param host_size Host buffer size
   * @return Status
   */
  virtual Status Read(const void *device_data, size_t device_size,
                      void *host_buffer, size_t host_size);

  /**
   * @brief Implement by specified device, copy data from src to dest
   * @param dest dest buffer to write
   * @param dest_size dest buffer size
   * @param src_buffer src buffer to read
   * @param src_size read data size
   * @param kind data copy kind
   * @return Status
   */
  virtual Status Copy(void *dest, size_t dest_size, const void *src_buffer,
                      size_t src_size, DeviceMemoryCopyKind kind) = 0;

  /**
   * @brief Implement by specified device, copy memory between current device
   *and host
   * @param dest_memory Destination memory
   * @param dest_offset Destination offset
   * @param src_memory Source memory
   * @param src_offset Source offset
   * @param src_size Source memory size
   * @param copy_kind copy mode
   * @return Status
   */
  virtual Status DeviceMemoryCopy(
      const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
      const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
      size_t src_size,
      DeviceMemoryCopyKind copy_kind = DeviceMemoryCopyKind::FromHost) = 0;

  /**
   * @brief Implement by specified device, get device memory info
   * @param free Free memory
   * @param total Total memory
   * @return Status
   */
  virtual Status GetDeviceMemUsage(size_t *free, size_t *total) const = 0;

 protected:
  std::string device_id_;
  size_t mem_quota_{0};
  size_t mem_allocated_{0};
  std::mutex allocated_size_lock_;
};

class DeviceMemoryLog {
 public:
  DeviceMemoryLog(std::string memory_id, std::string user_id,
                  std::string device_id, size_t size);

  virtual ~DeviceMemoryLog();

  std::string memory_id_;
  std::string user_id_;
  std::string device_id_;
  size_t size_{0};
};

class DeviceMemoryTrace {
 public:
  virtual ~DeviceMemoryTrace();
  /**
   * @brief Trace memory allocation
   * @param memory_id Memory id
   * @param user_id Memory request by
   * @param device_id Memory belong to
   * @param size Memory size
   */
  void TraceMemoryAlloc(const std::string &memory_id,
                        const std::string &user_id,
                        const std::string &device_id, size_t size);

  /**
   * @brief Trace memory free
   * @param memory_id Memory to free
   */
  void TraceMemoryFree(const std::string &memory_id);

  /**
   * @brief Get memory log
   * @param memory_id Memory id
   * @return Memory log
   */
  std::shared_ptr<DeviceMemoryLog> GetMemoryLog(const std::string &memory_id);

 private:
  // store trace logs of all device memory
  std::map<std::string, std::shared_ptr<DeviceMemoryLog>> memory_logs_;
  std::mutex memory_logs_lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_MEMORY_H_