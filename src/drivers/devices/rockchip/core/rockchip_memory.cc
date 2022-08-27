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

#include "modelbox/device/rockchip/rockchip_memory.h"

#include <securec.h>

#include "modelbox/device/rockchip/device_rockchip.h"

// -- only linux: get free memory
#include <linux/kernel.h>
#include <linux/unistd.h>
#include <sys/sysinfo.h>

namespace modelbox {

RockChipMemoryPool::RockChipMemoryPool(RockChipMemoryManager *mem_manager) {
  mem_manager_ = mem_manager;
}

Status RockChipMemoryPool::Init() {
  auto status = InitSlabCache();
  if (!status) {
    return {status, "init mempool failed."};
  }
  auto timer = std::make_shared<TimerTask>();
  timer->Callback(&RockChipMemoryPool::OnTimer, this);
  flush_timer_ = timer;

  // flush slab every 10s
  GetTimer()->Schedule(flush_timer_, 1000, 10000);
  return STATUS_OK;
}

RockChipMemoryPool::~RockChipMemoryPool() {
  if (flush_timer_) {
    flush_timer_->Stop();
    flush_timer_ = nullptr;
  }
}

void RockChipMemoryPool::OnTimer() {
  // TODO support config shrink time.
}

void *RockChipMemoryPool::MemAlloc(size_t size) {
  return mem_manager_->Malloc(size, 0);
}

void RockChipMemoryPool::MemFree(void *ptr) { mem_manager_->Free(ptr, 0); }

size_t RockChipMemoryPool::CalSlabSize(size_t object_size) {
  return object_size;
}

RockChipMemory::RockChipMemory(
    const std::shared_ptr<Device> &device,
    const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
    const std::shared_ptr<void> &device_mem_ptr, size_t size)
    : DeviceMemory(device, mem_mgr, device_mem_ptr, size, false) {}

RockChipMemoryManager::RockChipMemoryManager(const std::string &device_id)
    : DeviceMemoryManager(device_id), mem_pool_(this) {}

RockChipMemoryManager::~RockChipMemoryManager() {
  if (buf_grp_ != nullptr) {
    mpp_buffer_group_put(buf_grp_);
  }
  mem_pool_.DestroySlabCache();
}

Status RockChipMemoryManager::Init() {
  auto ret = mpp_buffer_group_get_internal(&buf_grp_, MPP_BUFFER_TYPE_DRM);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to get buffer group, MemoryManager init");
    MBLOG_ERROR << msg << ret;
    return {STATUS_FAULT, msg};
  }

  return mem_pool_.Init();
}

std::shared_ptr<DeviceMemory> RockChipMemoryManager::MakeDeviceMemory(
    const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
    size_t size) {
  return std::make_shared<RockChipMemory>(device, shared_from_this(), mem_ptr,
                                          size);
}

void *RockChipMemoryManager::Malloc(size_t size, uint32_t mem_flags) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(malloc_mtx_);
  MppBuffer buf = nullptr;
  auto ret = mpp_buffer_get(buf_grp_, &buf, size);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "Malloc mpp buffer fail, size = " << size;
    return nullptr;
  }

  return (void *)buf;
}

std::shared_ptr<void> RockChipMemoryManager::AllocSharedPtr(
    size_t size, uint32_t mem_flags) {
  return mem_pool_.AllocSharedPtr(size);
}

void RockChipMemoryManager::Free(void *mem_ptr, uint32_t mem_flags) {
  std::lock_guard<std::mutex> lock(malloc_mtx_);
  if (mem_ptr != nullptr) {
    mpp_buffer_put(mem_ptr);
  }
}

Status RockChipMemoryManager::Copy(void *dest, size_t dest_size,
                                   const void *src_buffer, size_t src_size,
                                   DeviceMemoryCopyKind kind) {
  if (dest == nullptr || src_buffer == nullptr) {
    MBLOG_ERROR << "RockChip copy src " << src_buffer << " to dest " << dest
                << "failed";
    return STATUS_INVALID;
  }

  if (dest_size < src_size) {
    MBLOG_ERROR << "RockChip memcpy failed, dest size[" << dest_size
                << "] < src size[" << src_size << "]";
    return STATUS_RANGE;
  }

  void *cp_dest = (void *)dest;
  void *cp_src = (void *)src_buffer;

  if (kind == DeviceMemoryCopyKind::FromHost ||
      kind == DeviceMemoryCopyKind::SameDeviceType) {
    cp_dest = mpp_buffer_get_ptr((MppBuffer)dest);
  } else if (kind == DeviceMemoryCopyKind::ToHost ||
             kind == DeviceMemoryCopyKind::SameDeviceType) {
    cp_src = mpp_buffer_get_ptr((MppBuffer)src_buffer);
  }

  int ret = memcpy_s(cp_dest, dest_size, cp_src, src_size);
  if (ret != EOK) {
    MBLOG_ERROR << "RockChip Copy memcpy failed";
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status RockChipMemoryManager::GetDeviceMemUsage(size_t *free,
                                                size_t *total) const {
  struct sysinfo s_rkinfo;
  // todo npu memory share cpu
  auto ret = sysinfo(&s_rkinfo);
  if (ret == 0) {
    if (free != nullptr) {
      *free = s_rkinfo.freeram;
    }

    if (total != nullptr) {
      *total = s_rkinfo.totalram;
    }

    return STATUS_SUCCESS;
  }

  auto msg = "failed to sysinfo ret = " + std::to_string(ret);
  MBLOG_ERROR << msg;

  return {STATUS_FAULT, msg};
}

Status RockChipMemoryManager::DeviceMemoryCopy(
    const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, DeviceMemoryCopyKind copy_kind) {
  auto src_device = src_memory->GetDevice();
  auto dest_device = dest_memory->GetDevice();
  if (copy_kind == DeviceMemoryCopyKind::SameDeviceType &&
      src_device != dest_device) {
    return STATUS_NOTSUPPORT;
  }

  uint8_t *dest_ptr = nullptr;
  if (dest_memory->IsHost()) {
    dest_ptr = dest_memory->GetPtr<uint8_t>().get();
  } else {
    MppBuffer dest_devp = dest_memory->GetPtr<MppBufHdl>().get();
    if (dest_devp) {
      dest_ptr = (uint8_t *)(mpp_buffer_get_ptr(dest_devp));
    }
  }

  const uint8_t *src_ptr = nullptr;
  if (src_memory->IsHost()) {
    src_ptr = src_memory->GetConstPtr<uint8_t>().get();
  } else {
    const MppBufHdl *src_devp = src_memory->GetConstPtr<MppBufHdl>().get();
    if (src_devp) {
      src_ptr = (const uint8_t *)(mpp_buffer_get_ptr((MppBuffer)src_devp));
    }
  }

  if (memcpy_s(dest_ptr + dest_offset, src_size, src_ptr + src_offset,
               src_size) != EOK) {
    MBLOG_ERROR << "DeviceMemoryCopy memcpy_s fail ";
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

}  // namespace modelbox
