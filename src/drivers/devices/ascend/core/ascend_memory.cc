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

#include "modelbox/device/ascend/ascend_memory.h"

#include <dsmi_common_interface.h>

namespace modelbox {

std::map<uint32_t, std::string> g_ascend_flags_name{
    {ASCEND_MEM_DVPP, "ASCEND_MEM_DVPP"},
    {ASCEND_MEM_NORMAL, "ASCEND_MEM_NORMAL"}};

/**
 * @brief Call be ascend stream.
 **/
void AscendReleaseMemoryTask(void *mem_list_ptr) {
  auto *list =
      (std::vector<std::shared_ptr<const DeviceMemory>> *)(mem_list_ptr);
  list->clear();
  delete list;
}

void AscendReleaseMemoryAsync(void *mem_list_ptr) {
  // Should not operate the mem and stream in this callback
  auto *timer = GetTimer();
  auto task = std::make_shared<TimerTask>();
  task->Callback(AscendReleaseMemoryTask, mem_list_ptr);
  task->SetName("AscendMemReleaseTask");
  timer->Schedule(task, 0, 0, true);
}

AscendStream::AscendStream(int32_t device_id, uint64_t callback_tid)
    : device_id_(device_id), callback_thread_id_(callback_tid) {}

AscendStream::~AscendStream() { Deinit(); }

Status AscendStream::Sync() const {
  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind ascend device " << device_id_ << " failed, acl ret "
                << ret;
    return STATUS_FAULT;
  }

  ret = aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Ascend stream sync failed, device id " << device_id_
                << ",acl ret " << ret;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status AscendStream::Bind(
    std::vector<std::shared_ptr<const DeviceMemory>> mem_list) const {
  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind ascend device " << device_id_ << " failed, acl ret "
                << ret;
    return STATUS_FAULT;
  }

  auto *mem_list_ptr =
      new (std::nothrow) std::vector<std::shared_ptr<const DeviceMemory>>();
  if (mem_list_ptr == nullptr) {
    MBLOG_ERROR << "New std::vector<>() failed";
    return STATUS_FAULT;
  }

  mem_list_ptr->assign(mem_list.begin(), mem_list.end());
  ret =
      aclrtLaunchCallback(AscendReleaseMemoryAsync, (void *)mem_list_ptr,
                          aclrtCallbackBlockType::ACL_CALLBACK_BLOCK, stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtLaunchCallback failed, acl ret " << ret;
    delete mem_list_ptr;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status AscendStream::Init() {
  if (init_flag_) {
    return STATUS_SUCCESS;
  }

  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind ascend device " << device_id_ << " failed, acl ret "
                << ret;
    return STATUS_FAULT;
  }

  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Create ascend stream failed, acl ret " << ret;
    return STATUS_FAULT;
  }

  ret = aclrtSubscribeReport(callback_thread_id_, stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSubscribeReport failed, acl ret " << ret;
    aclrtDestroyStream(stream_);
    return STATUS_FAULT;
  }

  init_flag_ = true;
  return STATUS_SUCCESS;
}

void AscendStream::Deinit() {
  if (!init_flag_) {
    return;
  }

  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSetDevice failed, acl ret " << ret;
  }

  ret = aclrtUnSubscribeReport(callback_thread_id_, stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtUnSubscribeReport failed, acl ret " << ret;
  }

  ret = aclrtDestroyStream(stream_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtDestroyStream failed, acl ret " << ret;
  }

  init_flag_ = false;
}

AscendStreamPool::AscendStreamPool(const std::string &device_id) {
  device_id_ = atoi(device_id.c_str());
  stream_callback_thread_ =
      std::make_shared<std::thread>(&AscendStreamPool::StreamCallBack, this);
  std::stringstream ss;
  ss << stream_callback_thread_->get_id();
  callback_thread_id_ = std::stoull(ss.str());
  stream_shrink_timer_.SetName("AscendStreamShrinkTimer");
  stream_shrink_timer_.Start();
  // TODO: Activate timer to shrink stream pool
}

void AscendStreamPool::StreamCallBack() {
  while (!is_exit_) {
    auto ret = aclrtProcessReport(100);
    if (ret == ACL_ERROR_RT_THREAD_SUBSCRIBE) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } else if (ret == ACL_ERROR_RT_REPORT_TIMEOUT) {
      // This is ok
    } else if (ret != ACL_SUCCESS) {
      MBLOG_ERROR << "aclrtProcessReport return err " << ret;
      return;
    }
  }
}

void AscendStreamPool::Shrink() {
  std::list<AscendStream *> stream_to_del;
  {
    std::lock_guard<std::mutex> lock(stream_list_lock_);
    MBLOG_INFO << "AscendStreamPool before shrink, total stream:"
               << allocate_count_ << ", idel stream:" << stream_list_.size();
    auto keep = allocate_count_ / 5;  // Reserve 20% idel
    if (keep >= stream_list_.size()) {
      // utilization >= 80%, no need to free
      return;
    }

    auto del = stream_list_.size() - keep;
    auto end_pos = stream_list_.begin();
    std::advance(end_pos, del);
    stream_to_del.splice(stream_to_del.begin(), stream_list_,
                         stream_list_.begin(), end_pos);
    allocate_count_ -= del;
    MBLOG_INFO << "AscendStreamPool after shrink, total stream:"
               << allocate_count_ << ", idel stream:" << stream_list_.size();
  }

  for (auto *stream : stream_to_del) {
    stream->Sync();
    delete stream;
  }
}

AscendStreamPool::~AscendStreamPool() {
  is_exit_ = true;
  if (stream_callback_thread_ != nullptr) {
    MBLOG_INFO << "Wait for ascend stream callback exit in ~AscendStreamPool";
    stream_callback_thread_->join();
    MBLOG_INFO << "Ascend stream callback exit ok";
  }

  stream_shrink_timer_.Stop();
  for (auto *stream : stream_list_) {
    delete stream;
  }
}

std::shared_ptr<AscendStream> AscendStreamPool::Alloc() {
  std::shared_ptr<AscendStream> stream;
  std::weak_ptr<AscendStreamPool> pool_ref = shared_from_this();
  auto free_func = [pool_ref](AscendStream *stream_ptr) {
    auto pool = pool_ref.lock();
    if (pool == nullptr) {
      delete stream_ptr;
      return;
    }

    pool->Free(stream_ptr);
  };

  {
    std::lock_guard<std::mutex> lock(stream_list_lock_);
    if (!stream_list_.empty()) {
      auto *stream_ptr = stream_list_.front();
      stream_list_.pop_front();
      stream.reset(stream_ptr, free_func);
      return stream;
    }
  }

  auto *stream_ptr = new AscendStream(device_id_, callback_thread_id_);
  auto ret = stream_ptr->Init();
  if (ret != STATUS_SUCCESS) {
    delete stream_ptr;
    return nullptr;
  }

  allocate_count_++;
  stream.reset(stream_ptr, free_func);
  return stream;
}

Status AscendStreamPool::Free(AscendStream *&stream) {
  if (stream == nullptr) {
    return STATUS_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(stream_list_lock_);
  stream_list_.push_back(stream);

  return STATUS_SUCCESS;
}

AscendMemoryPool::AscendMemoryPool(AscendMemoryManager *mem_manager) {
  mem_manager_ = mem_manager;
}

Status AscendMemoryPool::Init() {
  auto status = InitSlabCache();
  if (!status) {
    return {status, "init mempool failed."};
  }

  auto timer = std::make_shared<TimerTask>();
  timer->Callback(&AscendMemoryPool::OnTimer, this);
  flush_timer_ = timer;

  // flush slab every 10s
  GetTimer()->Schedule(flush_timer_, 1000, 10000);
  return STATUS_OK;
}

AscendMemoryPool::~AscendMemoryPool() {
  if (flush_timer_) {
    flush_timer_->Stop();
    flush_timer_ = nullptr;
  }
}

void AscendMemoryPool::OnTimer() {
  // TODO support config shrink time.
}

void *AscendMemoryPool::MemAlloc(size_t size) {
  return mem_manager_->Malloc(size, ASCEND_MEM_NORMAL);
}

void AscendMemoryPool::MemFree(void *ptr) {
  mem_manager_->Free(ptr, ASCEND_MEM_NORMAL);
}

AscendMemory::AscendMemory(const std::shared_ptr<Device> &device,
                           const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                           const std::shared_ptr<void> &device_mem_ptr,
                           size_t size)
    : DeviceMemory(device, mem_mgr, device_mem_ptr, size) {}

AscendMemory::~AscendMemory() = default;

Status AscendMemory::BindStream(
    const std::shared_ptr<AscendStream> &stream_ptr) {
  if (ascend_stream_ptr_ != nullptr) {
    if (ascend_stream_ptr_ == stream_ptr) {
      return STATUS_SUCCESS;
    }
    // Change stream to another is not allowed
    return {STATUS_BUSY, "Memory has bind to a stream"};
  }

  Status ret = STATUS_SUCCESS;
  if (stream_ptr != nullptr) {
    if (stream_ptr->IsInDevice(device_->GetDeviceID())) {
      ascend_stream_ptr_ = stream_ptr;
      return STATUS_SUCCESS;
    }
    // We need create a new stream when cross gpu device, so bind failed in fact
    ret = STATUS_BUSY;
  }

  auto ascend_mem_mgr = std::static_pointer_cast<AscendMemoryManager>(mem_mgr_);
  ascend_stream_ptr_ = ascend_mem_mgr->AllocStream();
  return ret;
}

Status AscendMemory::DetachStream() {
  if (ascend_stream_ptr_ == nullptr) {
    return STATUS_SUCCESS;
  }

  auto ret = ascend_stream_ptr_->Sync();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  ascend_stream_ptr_.reset();
  return STATUS_SUCCESS;
}

Status AscendMemory::CopyExtraMetaTo(
    std::shared_ptr<DeviceMemory> &device_mem) {
  if (device_mem->GetDevice() != device_) {
    return STATUS_SUCCESS;
  }

  auto target = std::static_pointer_cast<AscendMemory>(device_mem);
  target->ascend_stream_ptr_ = ascend_stream_ptr_;
  return STATUS_SUCCESS;
}

Status AscendMemory::CombineExtraMeta(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list) {
  for (const auto &mem : mem_list) {
    auto ascend_mem = std::dynamic_pointer_cast<AscendMemory>(mem);
    if (ascend_stream_ptr_ == nullptr) {
      // If this has no stream, use the first stream we found
      ascend_stream_ptr_ = ascend_mem->ascend_stream_ptr_;
    } else {
      // If this has valid stream now, other stream should be synchronized
      auto other_ascend_stream_ptr = ascend_mem->ascend_stream_ptr_;
      if (other_ascend_stream_ptr == nullptr) {
        continue;
      }

      if (ascend_stream_ptr_ == other_ascend_stream_ptr) {
        continue;
      }

      auto ret = other_ascend_stream_ptr->Sync();
      if (ret != STATUS_SUCCESS) {
        MBLOG_ERROR << "Sync ascend stream failed when combine ascend memory";
        return STATUS_FAULT;
      }
    }
  }

  return STATUS_SUCCESS;
}

AscendMemoryManager::AscendMemoryManager(const std::string &device_id)
    : DeviceMemoryManager(device_id),
      mem_pool_(this),
      mem_copy_kind_map_{{DeviceMemoryCopyKind::FromHost,
                          aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE},
                         {DeviceMemoryCopyKind::SameDeviceType,
                          aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE},
                         {DeviceMemoryCopyKind::ToHost,
                          aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST}} {
  stream_pool_ = std::make_shared<AscendStreamPool>(device_id);
  npu_id_ = atoi(device_id.c_str());
}

AscendMemoryManager::~AscendMemoryManager() = default;

Status AscendMemoryManager::Init() { return STATUS_OK; }

std::shared_ptr<DeviceMemory> AscendMemoryManager::MakeDeviceMemory(
    const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
    size_t size) {
  return std::make_shared<AscendMemory>(device, shared_from_this(), mem_ptr,
                                        size);
}

void *AscendMemoryManager::Malloc(size_t size, uint32_t mem_flags) {
  auto ret = aclrtSetDevice(npu_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind device " << npu_id_ << " failed, acl ret " << ret;
    return nullptr;
  }

  void *npu_mem_ptr = nullptr;
  switch (mem_flags) {
    case ASCEND_MEM_DVPP:
      ret = acldvppMalloc(&npu_mem_ptr, size);
      break;

    case ASCEND_MEM_NORMAL:
      ret = aclrtMalloc(&npu_mem_ptr, size,
                        aclrtMemMallocPolicy::ACL_MEM_MALLOC_NORMAL_ONLY);
      break;

    default:
      MBLOG_ERROR << "Not support mem alloc flags " << mem_flags;
      return nullptr;
  }

  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Malloc failed, size " << size << ", acl ret " << ret
                << ", flags " << g_ascend_flags_name[mem_flags];
    return nullptr;
  }

  return npu_mem_ptr;
}

void AscendMemoryManager::Free(void *mem_ptr, uint32_t mem_flags) {
  auto ret = aclrtSetDevice(npu_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind device " << npu_id_ << " failed, acl ret " << ret;
  }

  switch (mem_flags) {
    case ASCEND_MEM_DVPP:
      ret = acldvppFree(mem_ptr);
      break;

    case ASCEND_MEM_NORMAL:
      ret = aclrtFree(mem_ptr);
      break;

    default:
      MBLOG_ERROR << "Not support mem free flags, flags " << mem_flags;
      return;
  }

  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Free on ascend " << npu_id_ << " failed, acl ret " << ret
                << ", flags " << g_ascend_flags_name[mem_flags];
    return;
  }
}

Status AscendMemoryManager::Copy(void *dest, size_t dest_size,
                                 const void *src_buffer, size_t src_size,
                                 DeviceMemoryCopyKind kind) {
  if (dest == nullptr || src_buffer == nullptr) {
    MBLOG_ERROR << "Ascend copy src " << src_buffer << " to dest " << dest
                << "failed";
    return STATUS_INVALID;
  }

  if (dest_size < src_size) {
    MBLOG_ERROR << "Ascend memcpy failed, dest size[" << dest_size
                << "] < src size[" << src_size << "]";
    return STATUS_RANGE;
  }

  auto ret = aclrtSetDevice(npu_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Bind device " << npu_id_ << " failed, acl ret " << ret;
    return STATUS_FAULT;
  }

  aclrtMemcpyKind ascend_copy_kind;
  GetAscendMemcpyKind(kind, ascend_copy_kind);
  ret = aclrtMemcpy(dest, dest_size, src_buffer, src_size, ascend_copy_kind);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Asend memcpy failed, ret " << ret << ", src size "
                << src_size << ", ascend cpy kind " << ascend_copy_kind;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status AscendMemoryManager::GetDeviceMemUsage(size_t *free,
                                              size_t *total) const {
  dsmi_memory_info_stru mem_info;
  auto ret = dsmi_get_memory_info(npu_id_, &mem_info);
  if (ret != 0) {
    MBLOG_ERROR << "Get npu " << npu_id_ << " mem info failed, dsmi ret "
                << ret;
    return STATUS_FAULT;
  }

  const size_t mb = 1024 * 1024;
  size_t total_in_byte = mem_info.memory_size * mb;
  if (free != nullptr) {
    *free = total_in_byte * (100 - mem_info.utiliza) / 100;
  }

  if (total != nullptr) {
    *total = total_in_byte;
  }

  return STATUS_SUCCESS;
}

Status AscendMemoryManager::DeviceMemoryCopy(
    const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, DeviceMemoryCopyKind copy_kind) {
  auto src_device = src_memory->GetDevice();
  auto dest_device = dest_memory->GetDevice();
  if (copy_kind == DeviceMemoryCopyKind::SameDeviceType &&
      src_device != dest_device) {
    return STATUS_NOTSUPPORT;
  }

  aclrtMemcpyKind ascend_copy_kind;
  GetAscendMemcpyKind(copy_kind, ascend_copy_kind);
  std::shared_ptr<AscendStream> ascend_stream_ptr;
  auto ret = SetupAscendStream(src_memory, dest_memory, ascend_stream_ptr);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Set up ascend stream failed, stream is null";
  }

  aclrtStream ascend_stream =
      ascend_stream_ptr == nullptr ? nullptr : ascend_stream_ptr->Get();
  auto *dest_ptr = dest_memory->GetPtr<uint8_t>().get() + dest_offset;
  const auto *src_ptr = src_memory->GetConstPtr<uint8_t>().get() + src_offset;
  if (!CheckCopyAsync(src_ptr, dest_ptr) && ascend_stream_ptr != nullptr) {
    ascend_stream_ptr->Sync();
    ascend_stream = nullptr;
  }

  aclrtSetDevice(npu_id_);
  aclError acl_ret = ACL_SUCCESS;
  if (ascend_stream != nullptr) {
    acl_ret = aclrtMemcpyAsync(dest_ptr, src_size, src_ptr, src_size,
                               ascend_copy_kind, ascend_stream);
  } else {
    acl_ret =
        aclrtMemcpy(dest_ptr, src_size, src_ptr, src_size, ascend_copy_kind);
  }

  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtMemcpyAsync failed, acl ret " << acl_ret
                << ", src_size:" << src_size << ",kind:" << ascend_copy_kind
                << ",stream:" << ascend_stream;
    return STATUS_FAULT;
  }

  if (ascend_stream != nullptr) {
    if (dest_memory->IsHost()) {
      ascend_stream_ptr->Sync();
    } else {
      // When async operation complete, the reference of memory will be
      // released
      ascend_stream_ptr->Bind({src_memory, dest_memory});
    }
  }

  return STATUS_SUCCESS;
}

void AscendMemoryManager::GetAscendMemcpyKind(
    DeviceMemoryCopyKind copy_kind, aclrtMemcpyKind &ascend_copy_kind) {
  ascend_copy_kind = mem_copy_kind_map_[copy_kind];
}

bool AscendMemoryManager::CheckCopyAsync(const void *src_addr,
                                         const void *dest_addr) {
  if (IsMemAligned((uintptr_t)src_addr, ASCEND_ASYNC_ALIGN) &&
      IsMemAligned((uintptr_t)dest_addr, ASCEND_ASYNC_ALIGN)) {
    return true;
  }

  return false;
}

Status AscendMemoryManager::SetupAscendStream(
    const std::shared_ptr<const DeviceMemory> &src_memory,
    const std::shared_ptr<DeviceMemory> &dest_memory,
    std::shared_ptr<AscendStream> &ascend_stream_ptr) {
  if (src_memory->IsHost()) {
    ascend_stream_ptr = nullptr;
  } else {
    ascend_stream_ptr = std::static_pointer_cast<const AscendMemory>(src_memory)
                            ->GetBindStream();
  }

  if (!dest_memory->IsHost()) {
    auto dest_ascend_memory =
        std::dynamic_pointer_cast<AscendMemory>(dest_memory);
    auto ret = dest_ascend_memory->BindStream(ascend_stream_ptr);
    if (ret == STATUS_BUSY && ascend_stream_ptr != nullptr) {
      // Case: Two memory has different stream, we choose to sync source
      ascend_stream_ptr->Sync();
    }

    ascend_stream_ptr = dest_ascend_memory->GetBindStream();
  }

  return STATUS_SUCCESS;
}
}  // namespace modelbox
