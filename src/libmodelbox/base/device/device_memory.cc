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

#include "modelbox/base/device_memory.h"

#include <utility>

#include "modelbox/base/device.h"
#include "modelbox/base/log.h"
#include "modelbox/base/slab.h"

namespace modelbox {

const uint64_t DeviceMemory::MEM_MAGIC_CODE = 0x446d4d654d6f5279;

DeviceMemory::DeviceMemory(const std::shared_ptr<Device> &device,
                           const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                           const std::shared_ptr<void> &device_mem_ptr,
                           size_t size, bool is_host_mem)
    : is_host_mem_(is_host_mem),
      device_(device),
      mem_mgr_(mem_mgr),

      size_(size),
      capacity_(size) {
  UpdateMemID(device_mem_ptr.get());
  if (device_mem_ptr != nullptr) {
    auto id = memory_id_;
    device_mem_ptr_.reset(device_mem_ptr.get(), [device, mem_mgr, id, size,
                                                 device_mem_ptr](void *ptr) {
      auto trace = device->GetMemoryTrace();
      if (trace != nullptr) {
        trace->TraceMemoryFree(id);
        mem_mgr->RestoreMem(size);
      }
    });
  }
}

void DeviceMemory::UpdateMemID(void *device_mem_ptr) {
  memory_id_ = std::to_string((uintptr_t)device_mem_ptr);
}

Status DeviceMemory::SetContentMutable(bool content_mutable) {
  is_content_mutable_ = content_mutable;
  // TODO: Protect memory in device
  return STATUS_SUCCESS;
}

size_t DeviceMemory::GetSize() const { return size_; };

size_t DeviceMemory::GetCapacity() const { return capacity_; };

std::string DeviceMemory::GetMemoryID() const { return memory_id_; }

std::shared_ptr<Device> DeviceMemory::GetDevice() const { return device_; }

uint32_t DeviceMemory::GetMemFlags() const { return mem_flags_; }

bool DeviceMemory::IsHost() const { return is_host_mem_; }

bool DeviceMemory::IsSameDevice(const std::shared_ptr<DeviceMemory> &dev_mem) {
  return dev_mem ? (device_ == dev_mem->device_ &&
                    mem_flags_ == dev_mem->mem_flags_)
                 : false;
}

bool DeviceMemory::IsContiguous(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    bool with_order) {
  if (mem_list.size() <= 1) {
    return true;
  }

  std::vector<std::shared_ptr<DeviceMemory>> order_mem_list(mem_list);
  if (!with_order) {
    std::sort(order_mem_list.begin(), order_mem_list.end(),
              [](const std::shared_ptr<DeviceMemory> &mem1,
                 const std::shared_ptr<DeviceMemory> &mem2) {
                return mem1->GetConstPtr<void>() < mem2->GetConstPtr<void>();
              });
  }

  auto device_mem_ptr = order_mem_list[0]->device_mem_ptr_;
  size_t offset = order_mem_list[0]->offset_;
  for (auto &mem : order_mem_list) {
    if (device_mem_ptr != mem->device_mem_ptr_) {
      return false;
    }

    if (offset != mem->offset_) {
      return false;
    }

    offset += mem->size_;
  }

  return true;
}

std::shared_ptr<DeviceMemory> DeviceMemory::Combine(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    const std::shared_ptr<Device> &target_device, uint32_t target_mem_flags) {
  if (mem_list.empty()) {
    MBLOG_ERROR << "Combine mem list is empty";
    return nullptr;
  }

  size_t total_size = 0;
  auto ret = CountMemSize(mem_list, total_size);
  if (ret != STATUS_SUCCESS) {
    return nullptr;
  }

  auto contiguous =
      ((target_device && target_device != mem_list[0]->GetDevice()) ||
       (target_mem_flags != mem_list[0]->mem_flags_))
          ? false
          : IsContiguous(mem_list, true);
  if (contiguous) {
    return CombineContinuous(mem_list, total_size, target_device);
  }

  return CombineFragment(mem_list, total_size, target_device, target_mem_flags);
}

std::shared_ptr<DeviceMemory> DeviceMemory::CombineContinuous(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    size_t total_size, const std::shared_ptr<Device> &target_device) {
  auto first_mem_ptr = std::min_element(
      mem_list.begin(), mem_list.end(),
      [](const std::shared_ptr<DeviceMemory> &mem1,
         const std::shared_ptr<DeviceMemory> &mem2) {
        return mem1->GetConstPtr<void>() < mem2->GetConstPtr<void>();
      });
  const auto &mem = *first_mem_ptr;
  auto device = mem->GetDevice();
  auto continuous_mem = device->MemAlloc(0);
  continuous_mem->offset_ = mem->offset_;
  continuous_mem->size_ = total_size;
  continuous_mem->capacity_ = mem->capacity_;
  continuous_mem->device_mem_ptr_ = mem->device_mem_ptr_;
  continuous_mem->memory_id_ = mem->memory_id_;
  continuous_mem->mem_flags_ = mem->mem_flags_;
  auto ret = continuous_mem->CombineExtraMeta(mem_list);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Combine extra meta failed";
    return nullptr;
  }

  if (target_device == nullptr || target_device == mem->device_) {
    return continuous_mem;
  }

  auto new_device_mem = target_device->MemAlloc(total_size);
  if (new_device_mem == nullptr) {
    MBLOG_ERROR << "Mem alloc failed, size " << total_size;
    return nullptr;
  }

  new_device_mem->ReadFrom(continuous_mem, 0, total_size);
  return new_device_mem;
}

std::shared_ptr<DeviceMemory> DeviceMemory::CombineFragment(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    size_t total_size, std::shared_ptr<Device> target_device,
    uint32_t target_mem_flags) {
  if (target_device == nullptr) {
    target_device = mem_list[0]->GetDevice();
  }

  auto new_mem = target_device->MemAlloc(total_size, target_mem_flags);
  if (new_mem == nullptr) {
    MBLOG_ERROR << "Mem alloc failed, size " << total_size;
    return nullptr;
  }

  size_t dest_offset = 0;
  for (const auto &mem : mem_list) {
    if (mem->GetSize() == 0) {
      continue;
    }

    auto ret = new_mem->ReadFrom(mem, 0, mem->GetSize(), dest_offset);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Combine read data failed, " << ret;
      return nullptr;
    }

    dest_offset += mem->GetSize();
  }

  return new_mem;
}

Status DeviceMemory::CountMemSize(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    size_t &total_size) {
  total_size = 0;
  for (const auto &mem : mem_list) {
    if (SIZE_MAX - total_size < mem->GetSize()) {
      MBLOG_ERROR << "Mem size > SIZE_MAX";
      return STATUS_FAULT;
    }

    total_size += mem->GetSize();
  }

  return STATUS_SUCCESS;
}

Status DeviceMemory::Verify() const { return STATUS_SUCCESS; }

Status DeviceMemory::Resize(size_t new_size) {
  if (new_size > capacity_) {
    MBLOG_ERROR << "New size " << new_size << " > capacity " << capacity_;
    return STATUS_RANGE;
  }

  size_ = new_size;
  return STATUS_SUCCESS;
}

Status DeviceMemory::Realloc(size_t new_capacity) {
  if (new_capacity < capacity_) {
    return STATUS_SUCCESS;
  }

  if (!CheckReallocParam(new_capacity)) {
    MBLOG_ERROR << "Check realloc param failed";
    return STATUS_INVALID;
  }

  auto new_device_memory = device_->MemAlloc(size_, new_capacity, mem_flags_);
  if (nullptr == new_device_memory) {
    MBLOG_ERROR << "Device malloc failed";
    return STATUS_FAULT;
  }

  if (size_ > 0) {
    auto ret = new_device_memory->ReadFrom(shared_from_this(), 0, size_);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Copy old data to new buffer failed, copy size " << size_;
      return ret;
    }
  }

  device_mem_ptr_ = new_device_memory->device_mem_ptr_;
  offset_ = 0;
  size_ = new_device_memory->size_;
  capacity_ = new_device_memory->capacity_;
  memory_id_ = new_device_memory->memory_id_;
  auto this_mem = shared_from_this();
  new_device_memory->CopyExtraMetaTo(this_mem);
  return STATUS_SUCCESS;
}

bool DeviceMemory::CheckReallocParam(size_t new_capacity) {
  if (0 == new_capacity) {
    MBLOG_ERROR << "Realloc mem to zero failed";
    return false;
  }

  return true;
}

Status DeviceMemory::ReadFrom(
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, size_t dest_offset) {
  if (src_memory->device_mem_ptr_ == device_mem_ptr_) {
    MBLOG_ERROR << "Memory read from same mem block is not supported";
    return STATUS_INVALID;
  }

  if (!CheckReadFromParam(src_memory, src_offset, src_size, dest_offset)) {
    MBLOG_ERROR << "Check read param failed";
    return STATUS_INVALID;
  }

  auto ret = TransferInDevice(src_memory, src_offset, src_size, dest_offset);
  if (ret == STATUS_NOTSUPPORT) {
    // Try to transfer data in host
    ret = TransferInHost(src_memory, src_offset, src_size, dest_offset);
  }

  return ret;
}

Status DeviceMemory::WriteTo(const std::shared_ptr<DeviceMemory> &dest_memory,
                             size_t src_offset, size_t src_size,
                             size_t dest_offset) const {
  return dest_memory->ReadFrom(shared_from_this(), src_offset, src_size,
                               dest_offset);
}

bool DeviceMemory::CheckReadFromParam(
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, size_t dest_offset) {
  if (!IsContentMutable()) {
    MBLOG_ERROR << "Target memory content is not mutable";
    return false;
  }

  auto src_max_size = src_memory->GetSize();
  if (src_offset >= src_max_size) {
    MBLOG_ERROR << "src_offset " << src_offset << " >= src_memory size"
                << src_max_size;
    return false;
  }

  if (0 == src_size) {
    MBLOG_ERROR << "src_size is zero";
    return false;
  }

  auto src_data_size = src_max_size - src_offset;
  if (src_data_size < src_size) {
    MBLOG_ERROR << "src_size " << src_size << " + src_offset " << src_offset
                << " > src_memory size" << src_max_size;
    return false;
  }

  if (dest_offset >= this->size_) {
    MBLOG_ERROR << "dest_offset " << dest_offset << " >= dest_memory size"
                << this->size_;
    return false;
  }

  auto dest_data_size = this->size_ - dest_offset;
  if (dest_data_size < src_size) {
    MBLOG_ERROR << "src_size " << src_size << " + dest_offset " << dest_offset
                << " > dest_memory size " << this->size_;
    return false;
  }

  return true;
}

Status DeviceMemory::TransferInHost(
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, size_t dest_offset) {
  // TODO: consider 4k cache
  std::shared_ptr<uint8_t> host_cache(new (std::nothrow) uint8_t[src_size],
                                      [](const uint8_t *ptr) { delete[] ptr; });
  if (host_cache == nullptr) {
    MBLOG_ERROR << "No memory for host cache";
    return STATUS_NOMEM;
  }

  auto src_mem_mgr = src_memory->mem_mgr_;
  auto src_dev = src_memory->GetDevice();
  auto ret =
      src_mem_mgr->Read(src_memory->GetConstPtr<uint8_t>().get() + src_offset,
                        src_size, host_cache.get(), src_size);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Read data from device " << src_dev->GetType() << ":"
                << src_dev->GetDeviceID() << "to host failed, size "
                << src_size;
    return STATUS_FAULT;
  }

  ret = mem_mgr_->Write(host_cache.get(), src_size,
                        GetPtr<uint8_t>().get() + dest_offset, src_size);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Write data to device " << device_->GetType() << ":"
                << device_->GetDeviceID() << "from host failed, size "
                << src_size;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status DeviceMemory::TransferInDevice(
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, size_t dest_offset) {
  DeviceMemoryCopyKind copy_kind;
  if (src_memory->device_->GetType() == device_->GetType()) {
    copy_kind = DeviceMemoryCopyKind::SameDeviceType;
  } else if (src_memory->IsHost()) {
    copy_kind = DeviceMemoryCopyKind::FromHost;
  } else {
    // Different type device transfer is not support by device
    return STATUS_NOTSUPPORT;
  }

  return mem_mgr_->DeviceMemoryCopy(shared_from_this(), dest_offset, src_memory,
                                    src_offset, src_size, copy_kind);
}

std::shared_ptr<DeviceMemory> DeviceMemory::Append(
    const std::shared_ptr<DeviceMemory> &dev_mem) {
  return Append(std::vector<std::shared_ptr<DeviceMemory>>{dev_mem});
}

std::shared_ptr<DeviceMemory> DeviceMemory::Append(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list) {
  if (mem_list.empty()) {
    MBLOG_ERROR << "Append mem list is empty";
    return nullptr;
  }

  size_t append_size;
  auto ret = CountMemSize(mem_list, append_size);
  if (ret != STATUS_SUCCESS) {
    return nullptr;
  }

  if (SIZE_MAX - append_size < size_) {
    MBLOG_ERROR << "Total mem size > SIZE_MAX";
    return nullptr;
  }

  auto new_device_mem = PrepareAppendMem(append_size);
  if (new_device_mem == nullptr) {
    return nullptr;
  }

  ret = AppendData(mem_list, new_device_mem);
  if (ret != STATUS_SUCCESS) {
    return nullptr;
  }

  return new_device_mem;
}

std::shared_ptr<DeviceMemory> DeviceMemory::PrepareAppendMem(
    size_t append_size) {
  std::shared_ptr<DeviceMemory> new_device_mem;
  if (capacity_ - size_ < append_size) {
    new_device_mem = device_->MemAlloc(size_ + append_size, mem_flags_);
    if (new_device_mem == nullptr) {
      MBLOG_ERROR << "Alloc mem failed, size " << size_ + append_size;
      return nullptr;
    }

    if (size_ > 0) {
      auto ret = new_device_mem->ReadFrom(shared_from_this(), 0, size_);
      if (ret != STATUS_SUCCESS) {
        MBLOG_ERROR << "Append read data failed";
        return nullptr;
      }
    }
  } else {
    new_device_mem = device_->MemAlloc(0);
    new_device_mem->device_mem_ptr_ = device_mem_ptr_;
    new_device_mem->offset_ = offset_;
    new_device_mem->size_ = size_ + append_size;
    new_device_mem->capacity_ = capacity_;
    new_device_mem->memory_id_ = memory_id_;
    new_device_mem->mem_flags_ = mem_flags_;
  }

  return new_device_mem;
}

Status DeviceMemory::AppendData(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list,
    std::shared_ptr<DeviceMemory> &target_device_mem) {
  size_t offset = size_;
  for (const auto &mem : mem_list) {
    if (mem->GetSize() == 0) {
      continue;
    }

    auto ret = target_device_mem->ReadFrom(mem, 0, mem->GetSize(), offset);
    offset += mem->GetSize();
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Append mem data failed";
      return STATUS_FAULT;
    }
  }

  return STATUS_SUCCESS;
}

std::shared_ptr<DeviceMemory> DeviceMemory::Cut(size_t offset, size_t size) {
  if (offset + size > capacity_) {
    MBLOG_ERROR << "cut failed, offset[" << offset << "] + size[" << size
                << "] > capacity[" << size << "]";
    return nullptr;
  }

  auto new_device_mem = device_->MemAlloc(0);
  new_device_mem->device_mem_ptr_ = device_mem_ptr_;
  new_device_mem->offset_ = offset_ + offset;
  new_device_mem->size_ = size;
  new_device_mem->capacity_ = capacity_ - offset;
  new_device_mem->memory_id_ = memory_id_;
  new_device_mem->mem_flags_ = mem_flags_;
  CopyExtraMetaTo(new_device_mem);
  return new_device_mem;
}

std::shared_ptr<DeviceMemory> DeviceMemory::Delete(size_t offset, size_t size) {
  if (size >= capacity_) {
    MBLOG_ERROR << "Delete size " << size << " >= capacity " << capacity_;
    return nullptr;
  }

  return Delete(offset, size, capacity_ - size);
}

std::shared_ptr<DeviceMemory> DeviceMemory::Delete(size_t offset, size_t size,
                                                   size_t capacity) {
  if (offset >= capacity_) {
    MBLOG_ERROR << "Delete offset " << offset << " >= capacity " << capacity_;
    return nullptr;
  }

  if (size == 0) {
    MBLOG_ERROR << "Delete size is zero";
    return nullptr;
  }

  auto data_offset = offset + size;
  if (data_offset > capacity_) {
    MBLOG_ERROR << "Delete offset " << offset << " + size " << size
                << " > capacity " << capacity_;
    return nullptr;
  }

  auto content_size = capacity_ - size;
  if (content_size == 0) {
    MBLOG_ERROR << "Delete size " << size << " == capacity " << capacity_;
    return nullptr;
  }

  auto new_device_mem = device_->MemAlloc(capacity, mem_flags_);
  if (offset > 0) {
    auto ret = new_device_mem->ReadFrom(shared_from_this(), 0, offset);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Delete, read from first part [0," << offset
                  << ",0] failed";
      return nullptr;
    }
  }

  if (data_offset < capacity_) {
    auto ret = new_device_mem->ReadFrom(shared_from_this(), offset + size,
                                        capacity_ - data_offset, offset);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Delete, read from second part [" << offset + size << ","
                  << capacity_ - data_offset << "," << offset << "] failed";
      return nullptr;
    }
  }

  new_device_mem->size_ = capacity_ - size;
  return new_device_mem;
}

std::shared_ptr<DeviceMemory> DeviceMemory::Copy(size_t offset, size_t size) {
  return Copy(offset, size, size);
}

std::shared_ptr<DeviceMemory> DeviceMemory::Copy(size_t offset, size_t size,
                                                 size_t capacity) {
  if (offset >= capacity_) {
    MBLOG_ERROR << "Copy offset " << offset << " >= capacity " << capacity_;
    return nullptr;
  }

  if (size == 0) {
    MBLOG_ERROR << "Copy size is zero";
    return nullptr;
  }

  if (offset + size > capacity_) {
    MBLOG_ERROR << "Copy offset " << offset << " + size " << size
                << " > capacity " << capacity_;
    return nullptr;
  }

  auto new_device_mem = device_->MemAlloc(capacity, mem_flags_);
  if (new_device_mem == nullptr) {
    MBLOG_ERROR << "Mem alloc failed, size " << capacity;
    return nullptr;
  }

  auto ret = new_device_mem->ReadFrom(shared_from_this(), offset, size);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Copy data failed";
    return nullptr;
  }

  new_device_mem->size_ = size;
  return new_device_mem;
}

std::shared_ptr<DeviceMemory> DeviceMemory::Clone(bool is_copy) {
  std::shared_ptr<DeviceMemory> new_device_memory;
  if (is_copy) {
    new_device_memory = device_->MemAlloc(size_, capacity_, mem_flags_);
    auto ret = new_device_memory->ReadFrom(shared_from_this(), 0, size_);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Clone data failed";
      return nullptr;
    }
  } else {
    new_device_memory = device_->MemAlloc(0);
    new_device_memory->device_mem_ptr_ = device_mem_ptr_;
    new_device_memory->offset_ = offset_;
    new_device_memory->size_ = size_;
    new_device_memory->capacity_ = capacity_;
    new_device_memory->memory_id_ = memory_id_;
    new_device_memory->is_content_mutable_ = is_content_mutable_;
    new_device_memory->mem_flags_ = mem_flags_;
    CopyExtraMetaTo(new_device_memory);
  }

  return new_device_memory;
}

Status DeviceMemory::MemAcquire(const std::shared_ptr<void> &mem_ptr,
                                size_t size) {
  if (mem_ptr == nullptr) {
    MBLOG_ERROR << "Mem acquire mem_ptr is nullptr";
    return STATUS_INVALID;
  }

  if (size == 0) {
    MBLOG_ERROR << "Mem acquire size is 0";
    return STATUS_INVALID;
  }

  device_mem_ptr_ = mem_ptr;
  offset_ = 0;
  size_ = size;
  capacity_ = size;
  UpdateMemID(device_mem_ptr_.get());
  return STATUS_SUCCESS;
}

bool DeviceMemoryManager::PreserveMem(size_t size) {
  std::lock_guard<std::mutex> lock_gurad(allocated_size_lock_);
  auto mem_availalbe = mem_quota_ - mem_allocated_;
  if (size > mem_availalbe) {
    MBLOG_ERROR << "Alloc size " << size << " > avaiable mem " << mem_availalbe;
    return false;
  }

  mem_allocated_ += size;
  return true;
}

void DeviceMemoryManager::RestoreMem(size_t size) {
  std::lock_guard<std::mutex> lock_guard(allocated_size_lock_);
  mem_allocated_ -= size;
}

std::shared_ptr<void> DeviceMemoryManager::AllocSharedPtr(size_t size,
                                                          uint32_t mem_flags) {
  void *ptr = Malloc(size, mem_flags);
  if (ptr == nullptr) {
    return nullptr;
  }

  std::shared_ptr<void> ret(
      ptr, [this, mem_flags](void *ptr) { this->Free(ptr, mem_flags); });
  return ret;
}

Status DeviceMemoryManager::Write(const void *host_data, size_t host_size,
                                  void *device_buffer, size_t device_size) {
  return Copy(device_buffer, device_size, host_data, host_size,
              DeviceMemoryCopyKind::FromHost);
}

Status DeviceMemoryManager::Read(const void *device_data, size_t device_size,
                                 void *host_buffer, size_t host_size) {
  return Copy(host_buffer, host_size, device_data, device_size,
              DeviceMemoryCopyKind::ToHost);
}

DeviceMemoryLog::DeviceMemoryLog(std::string memory_id, std::string user_id,
                                 std::string device_id, size_t size)
    : memory_id_(std::move(memory_id)),
      user_id_(std::move(user_id)),
      device_id_(std::move(device_id)),
      size_(size) {}

DeviceMemoryLog::~DeviceMemoryLog() = default;

DeviceMemoryTrace::~DeviceMemoryTrace() = default;

void DeviceMemoryTrace::TraceMemoryAlloc(const std::string &memory_id,
                                         const std::string &user_id,
                                         const std::string &device_id,
                                         size_t size) {
  std::lock_guard<std::mutex> lock(memory_logs_lock_);
  auto mem_log =
      std::make_shared<DeviceMemoryLog>(memory_id, user_id, device_id, size);
  memory_logs_[memory_id] = mem_log;
}

void DeviceMemoryTrace::TraceMemoryFree(const std::string &memory_id) {
  std::lock_guard<std::mutex> lock(memory_logs_lock_);
  auto item = memory_logs_.find(memory_id);
  if (item != memory_logs_.end()) {
    memory_logs_.erase(item);
  }
}

std::shared_ptr<DeviceMemoryLog> DeviceMemoryTrace::GetMemoryLog(
    const std::string &memory_id) {
  std::lock_guard<std::mutex> lock(memory_logs_lock_);
  auto item = memory_logs_.find(memory_id);
  if (item == memory_logs_.end()) {
    return nullptr;
  }
  return item->second;
}

}  // namespace modelbox