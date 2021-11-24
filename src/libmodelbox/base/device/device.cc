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

#include "modelbox/base/device.h"

#include <stdio.h>

#include "modelbox/base/log.h"

namespace modelbox {
Device::Device(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : memory_trace_(std::make_shared<DeviceMemoryTrace>()),
      memory_manager_(mem_mgr) {
  executor_ = std::make_shared<Executor>();
}

Device::Device(size_t thread_count,
               const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : memory_trace_(std::make_shared<DeviceMemoryTrace>()),
      memory_manager_(mem_mgr) {
  if (0 == thread_count) {
    executor_ = nullptr;
  } else {
    executor_ = std::make_shared<Executor>(thread_count);
  }
}

bool Device::NeedResourceNice() { return false; }

std::list<std::future<Status>> Device::DeviceExecuteAsync(
    const DevExecuteCallBack &fun, int32_t priority, size_t count,
    bool resource_nice) {
  if (0 == count) {
    return {};
  }

  if (NeedResourceNice() && resource_nice) {
    return DeviceExecuteAsyncNice(fun, priority, count);
  }

  return DeviceExecuteAsyncRude(fun, priority, count);
}

std::list<std::future<Status>> Device::DeviceExecuteAsyncRude(
    const DevExecuteCallBack &fun, int32_t priority, size_t count) {
  std::list<std::future<Status>> future_status_list;
  for (size_t i = 0; i < count; ++i) {
    auto future_status = executor_->Run(fun, priority, i);
    future_status_list.push_back(std::move(future_status));
  }

  return future_status_list;
}

std::list<std::future<Status>> Device::DeviceExecuteAsyncNice(
    const DevExecuteCallBack &fun, int32_t priority, size_t count) {
  auto serial_process = [fun, count]() {
    Status final_result;
    for (size_t i = 0; i < count; ++i) {
      auto ret = fun(i);
      if (final_result == STATUS_OK || final_result == STATUS_CONTINUE) {
        final_result = ret;
      }
    }

    return final_result;
  };

  auto future_status = executor_->Run(serial_process, priority);
  std::list<std::future<Status>> status_list;
  status_list.push_back(std::move(future_status));
  return status_list;
}

Status Device::Init() {
  size_t total;
  auto ret = GetMemInfo(nullptr, &total);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Get device " << GetDeviceID() << " mem info failed";
    return {ret, "device init failed."};
  }

  SetMemQuota(total);
  return STATUS_SUCCESS;
}

std::shared_ptr<DeviceMemory> Device::MemAlloc(size_t size, uint32_t mem_flags,
                                               const std::string &user_id) {
  return MemAlloc(size, size, mem_flags, user_id);
}

std::shared_ptr<DeviceMemory> Device::MemAlloc(size_t size, size_t capacity,
                                               uint32_t mem_flags,
                                               const std::string &user_id) {
  // TODO: Get user_id from thread module
  if (size > capacity) {
    StatusError = {STATUS_RANGE, "Mem capacity must >= size"};
    MBLOG_ERROR << StatusError.Errormsg();
    return nullptr;
  }

  if (size == 0) {
    auto mem =
        memory_manager_->MakeDeviceMemory(shared_from_this(), nullptr, 0);
    mem->SetMemFlags(mem_flags);
    return mem;
  }

  if (!memory_manager_->PreserveMem(capacity)) {
    return nullptr;
  }

  auto device_mem_shared_ptr =
      memory_manager_->AllocSharedPtr(capacity, mem_flags);
  if (device_mem_shared_ptr == nullptr) {
    return nullptr;
  }

  auto device_mem = memory_manager_->MakeDeviceMemory(
      shared_from_this(), device_mem_shared_ptr, capacity);
  if (device_mem == nullptr) {
    return nullptr;
  }

  device_mem->SetMemFlags(mem_flags);
  device_mem->Resize(size);
  memory_trace_->TraceMemoryAlloc(device_mem->GetMemoryID(), user_id,
                                  GetDeviceID(), capacity);

  return device_mem;
}

std::shared_ptr<DeviceMemory> Device::MemAcquire(void *mem_ptr, size_t size,
                                                 const DeleteFunction &deleter,
                                                 uint32_t mem_flags) {
  std::shared_ptr<void> mem_shared_ptr(mem_ptr, deleter);
  return MemAcquire(mem_shared_ptr, size, mem_flags);
}

std::shared_ptr<DeviceMemory> Device::MemAcquire(std::shared_ptr<void> mem_ptr,
                                                 size_t size,
                                                 uint32_t mem_flags) {
  auto dev_mem = MemAlloc(0, mem_flags);
  auto ret = dev_mem->MemAcquire(mem_ptr, size);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Mem acquire (" << mem_ptr << " , " << size << ") failed";
    return nullptr;
  }

  return dev_mem;
}

std::shared_ptr<DeviceMemory> Device::MemWrite(const void *host_data,
                                               size_t host_size,
                                               const std::string &user_id) {
  if (0 == host_size || nullptr == host_data) {
    MBLOG_ERROR << "Mem write failed, src size is zero or host data null";
    StatusError = {STATUS_INVALID, "invalid argument"};
    return nullptr;
  }

  auto device_mem = MemAlloc(host_size, 0, user_id);
  if (nullptr == device_mem) {
    MBLOG_ERROR << "Malloc failed, size " << host_size;
    return nullptr;
  }

  auto device_buffer = device_mem->GetPtr<void>();
  auto ret = memory_manager_->Write(host_data, host_size, device_buffer.get(),
                                    device_mem->GetSize());
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Write host data to device memory failed";
    return nullptr;
  }

  return device_mem;
}

Status Device::GetMemInfo(size_t *free, size_t *total) const {
  return memory_manager_->GetDeviceMemUsage(free, total);
}

std::shared_ptr<DeviceMemoryTrace> Device::GetMemoryTrace() const {
  return memory_trace_;
}

std::shared_ptr<DeviceMemory> Device::MemClone(
    std::shared_ptr<DeviceMemory> src_memory, const std::string &user_id) {
  if (!src_memory->IsContentMutable() &&
      src_memory->GetDevice().get() == this) {
    return src_memory;
  }

  auto dest_memroy =
      MemAlloc(src_memory->GetSize(), src_memory->mem_flags_, user_id);
  if (dest_memroy == nullptr) {
    MBLOG_ERROR << "MemAlloc failed in clone";
    return nullptr;
  }

  auto ret = dest_memroy->ReadFrom(src_memory, 0, src_memory->GetSize());
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Read data from source failed";
    return nullptr;
  }

  return dest_memroy;
}

void Device::SetDeviceDesc(std::shared_ptr<DeviceDesc> device_desc) {
  device_desc_ = device_desc;
}

std::shared_ptr<DeviceDesc> Device::GetDeviceDesc() { return device_desc_; }

std::shared_ptr<DeviceManager> Device::GetDeviceManager() {
  return device_mgr_.lock();
}

void Device::SetDeviceManager(std::shared_ptr<DeviceManager> device_mgr) {
  device_mgr_ = device_mgr;
}

void DeviceDesc::SetDeviceId(const std::string &device_id) {
  device_id_ = device_id;
}

void DeviceDesc::SetDeviceType(const std::string &device_type) {
  device_type_ = device_type;
}

void DeviceDesc::SetDeviceMemory(const std::string &device_memory) {
  device_memory_ = device_memory;
}

void DeviceDesc::SetDeviceVersion(const std::string &device_version) {
  device_version_ = device_version;
}

void DeviceDesc::SetDeviceDesc(const std::string &device_description) {
  device_description_ = device_description;
}

std::string DeviceDesc::GetDeviceId() { return device_id_; }

std::string DeviceDesc::GetDeviceType() { return device_type_; }

std::string DeviceDesc::GetDeviceMemory() { return device_memory_; }

std::string DeviceDesc::GetDeviceVersion() { return device_version_; }

std::string DeviceDesc::GetDeviceDesc() { return device_description_; }

}  // namespace modelbox
