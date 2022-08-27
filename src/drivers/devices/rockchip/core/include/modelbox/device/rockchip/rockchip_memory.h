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

#ifndef MODELBOX_ROCKCHIP_MEMORY_H_
#define MODELBOX_ROCKCHIP_MEMORY_H_

#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <queue>
#include <thread>
#include <unordered_map>

#include "rk_mpi.h"
#include "rk_type.h"

namespace modelbox {

Timer *GetTimer();

class RockChipMemory : public DeviceMemory {
 public:
  RockChipMemory(const std::shared_ptr<Device> &device,
                 const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                 void *device_mem_ptr, size_t size);

  RockChipMemory(const std::shared_ptr<Device> &device,
                 const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                 const std::shared_ptr<void> &device_mem_ptr, size_t size);

  ~RockChipMemory() override = default;
};

class RockChipMemoryManager;
class RockChipMemoryPool : public MemoryPoolBase {
 public:
  RockChipMemoryPool(RockChipMemoryManager *mem_manager);
  ~RockChipMemoryPool() override;
  Status Init();
  void *MemAlloc(size_t size) override;
  void MemFree(void *ptr) override;
  virtual void OnTimer();

  size_t CalSlabSize(size_t object_size) override;

 private:
  RockChipMemoryManager *mem_manager_;
  std::shared_ptr<TimerTask> flush_timer_;
};

class RockChipMemoryManager : public DeviceMemoryManager {
 public:
  RockChipMemoryManager(const std::string &device_id);
  ~RockChipMemoryManager() override;

  Status Init();

  /* *
   * @brief Create a rockchip memory container
   * @param device pointer to device
   * @param mem_ptr shared pointer to memory
   * @param size memory size
   * @return Empty memory container
   */
  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) override;

  /* *
   * @brief Implement by rockchip device, alloc memory
   * @param size Memory size to allocate.
   * @return Device memory.
   */
  void *Malloc(size_t size, uint32_t mem_flags) override;

  /* *
   * @brief Implement by rockchip device, alloc memory
   * @param size Memory size to allocate
   * @return Device memory in shared ptr
   *   */
  std::shared_ptr<void> AllocSharedPtr(size_t size,
                                       uint32_t mem_flags) override;

  /**
   * @brief Implement by rockchip device, copy data from src to dest
   * @param dest dest buffer to write
   * @param dest_size dest buffer size
   * @param src_buffer src buffer to read
   * @param src_size read data size
   * @param kind data copy kind
   * @return Status
   */
  Status Copy(void *dest, size_t dest_size, const void *src_buffer,
              size_t src_size, DeviceMemoryCopyKind kind) override;
  /* *
   * @brief Copy memory between rockchip device and host
   * @param dest_memory Destination memory
   * @param dest_offset Destination memory offset
   * @param src_memory Source memory
   * @param src_offset Source offset
   * @param src_size Source memory size
   * @param copy_kind Memory copy mode
   * @return Status
   */
  Status DeviceMemoryCopy(
      const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
      const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
      size_t src_size,
      DeviceMemoryCopyKind copy_kind = DeviceMemoryCopyKind::FromHost) override;

  /* *
   * @brief Get device memory info
   * @return Status
   */
  Status GetDeviceMemUsage(size_t *free, size_t *total) const override;

  /* *
   * @brief Implement by rockchip device, free memory
   * @param mem_ptr Memory to free
   */
  void Free(void *mem_ptr, uint32_t mem_flags) override;

 private:
  RockChipMemoryPool mem_pool_;
  MppBufferGroup buf_grp_ = nullptr;
  std::mutex malloc_mtx_;
};

}  // namespace modelbox

#endif  // MODELBOX_ROCKCHIP_MEMORY_H_
