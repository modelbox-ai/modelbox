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

#ifndef MODELBOX_CPU_MEMORY_H_
#define MODELBOX_CPU_MEMORY_H_

#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/timer.h>

extern modelbox::Timer *GetTimer();

namespace modelbox {

class CpuMemory;
class CpuMemoryManager;

class CpuMemory : public DeviceMemory {
 public:
  /**
   * @brief Construct a host memory with physical mem ptr, called by cpu device
   * @param device Memory belong to
   * @param mem_mgr Device Memory manager
   * @param device_mem_ptr Memory pointer
   * @param size Memory size
   */
  CpuMemory(const std::shared_ptr<Device> &device,
            const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
            const std::shared_ptr<void> &device_mem_ptr, size_t size);

  ~CpuMemory() override;
  /**
   * @brief Read data from other device memory
   * @param src_memory Memory read from
   * @param src_offset Offset in the memory read from
   * @param src_size Size in the memory read from
   * @param dest_offset Offset in memory write to
   * @return Status
   */
  Status ReadFrom(const std::shared_ptr<const DeviceMemory> &src_memory,
                  size_t src_offset, size_t src_size,
                  size_t dest_offset = 0) override;

  /**
   * @brief Check memory out of bound; Make checksum
   * @return Result of verify, 0 is ok
   */
  Status Verify() const override;
};

class CpuMemoryPool : public MemoryPoolBase {
 public:
  CpuMemoryPool();

  ~CpuMemoryPool() override;

  Status Init();

  void *MemAlloc(size_t size) override;

  void MemFree(void *ptr) override;

  virtual void OnTimer();

 private:
  std::shared_ptr<TimerTask> flush_timer_;
};

class CpuMemoryManager : public DeviceMemoryManager {
 public:
  CpuMemoryManager(const std::string &device_id);

  ~CpuMemoryManager() override;

  /**
   * @brief Init memory manager
   * @return init result
   */
  Status Init();

  /**
   * @brief Create a specified memory container
   * @param device pointer to device
   * @param mem_ptr shared pointer to memory
   * @param size memory size
   * @return Empty memory container
   */
  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) override;

  /**
   * @brief Implement by specified device, alloc memory
   * @param size Memory size to allocate
   * @param mem_flags Flags to create device memory
   * @return Device memory in shared ptr
   */
  std::shared_ptr<void> AllocSharedPtr(size_t size,
                                       uint32_t mem_flags = 0) override;

  /**
   * @brief Implement by specified device, alloc memory
   * @param size Memory size to allocate
   * @param mem_flags Flags to create device memory
   * @return Device memory.
   */
  void *Malloc(size_t size, uint32_t mem_flags = 0) override;

  /**
   * @brief Implement by specified device, free memory
   * @param mem_ptr Memory to free
   * @param mem_flags Flags of device memory
   */
  void Free(void *mem_ptr, uint32_t mem_flags = 0) override;

  /**
   * @brief Implement by specified device, copy data from src to dest
   * @param dest dest buffer to write
   * @param dest_size dest buffer size
   * @param src_buffer src buffer to read
   * @param src_size read data size
   * @param kind data copy kind
   * @return Status
   */
  Status Copy(void *dest, size_t dest_size, const void *src_buffer,
              size_t src_size, DeviceMemoryCopyKind kind) override;

  /**
   * @brief Copy memory between current device and host
   * @param dest_memory Destination memory
   * @param dest_offset Destination memory offset
   * @param src_memory Source memory
   * @param src_offset Source offset
   * @param src_size Source memory size
   * @param copy_kind memory copy mode
   * @return Status
   */
  Status DeviceMemoryCopy(
      const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
      const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
      size_t src_size,
      DeviceMemoryCopyKind copy_kind = DeviceMemoryCopyKind::FromHost) override;

  /**
   * @brief Get device memory info
   * @return Status
   */
  Status GetDeviceMemUsage(size_t *free, size_t *total) const override;

 private:
  std::shared_ptr<CpuMemoryPool> mem_pool_;
};

}  // namespace modelbox
#endif  // MODELBOX_CPU_MEMORY_H_