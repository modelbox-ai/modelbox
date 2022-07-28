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

#include "modelbox/device/cuda/cuda_memory.h"

#include "modelbox/base/collector.h"
namespace modelbox {
/**
 * @brief Call be cuda stream.
 *   Will release mem reference used before.
 *   We need a new thread due to cuda api might be called.
 **/
void CudaReleaseMemoryAsync(void *mem_list_ptr) {
  auto list =
      (std::vector<std::shared_ptr<const DeviceMemory>> *)(mem_list_ptr);
  list->clear();
  delete list;
}

CudaStream::CudaStream(cudaStream_t stream, int32_t device_id)
    : stream_(stream), device_id_(device_id) {}

Status CudaStream::Sync() const {
  auto cuda_ret = cudaSetDevice(device_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Bind cuda device " << device_id_ << " failed, cuda ret "
                << cuda_ret;
    return STATUS_FAULT;
  }

  cuda_ret = cudaStreamSynchronize(stream_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Cuda stream synchronize failed, gpu " << device_id_
                << " cuda ret " << cuda_ret;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status CudaStream::Bind(
    std::vector<std::shared_ptr<const DeviceMemory>> mem_list) const {
  auto cuda_ret = cudaSetDevice(device_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Bind cuda device " << device_id_ << " failed, cuda ret "
                << cuda_ret;
    return STATUS_FAULT;
  }

  auto mem_list_ptr = new std::vector<std::shared_ptr<const DeviceMemory>>();
  mem_list_ptr->assign(mem_list.begin(), mem_list.end());
  cuda_ret =
      cudaLaunchHostFunc(stream_, CudaReleaseMemoryAsync, (void *)mem_list_ptr);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "cudaLaunchHostFunc failed, cuda ret " << cuda_ret;
    delete mem_list_ptr;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

CudaStreamPool::CudaStreamPool(const std::string &device_id) {
  device_id_ = atoi(device_id.c_str());
  is_running_ = true;
  release_stream_thread_ =
      std::make_shared<std::thread>(&CudaStreamPool::ReleaseStreamWorker, this);
}

CudaStreamPool::~CudaStreamPool() {
  release_stream_queue_.Shutdown();
  is_running_ = false;
  if (release_stream_thread_ != nullptr) {
    MBLOG_INFO << "Join release stream thread start";
    release_stream_thread_->join();
    MBLOG_INFO << "Release stream thread stop";
  }
}

void CudaStreamPool::ReleaseStreamWorker() {
  while (is_running_ || !release_stream_queue_.Empty()) {
    cudaStream_t stream = nullptr;
    auto ret = release_stream_queue_.Pop(&stream, 100);
    if (!ret || stream == nullptr) {
      continue;
    }

    auto cuda_ret = cudaStreamDestroy(stream);
    if (cudaSuccess != cuda_ret) {
      MBLOG_ERROR << "Destroy cuda stream failed, cuda ret " << cuda_ret;
      continue;
    }

    allocate_count_--;
  }
}

std::shared_ptr<CudaStream> CudaStreamPool::Alloc() {
  auto cuda_ret = cudaSetDevice(device_id_);
  if (cudaSuccess != cuda_ret) {
    MBLOG_ERROR << "Bind cuda device " << device_id_ << " failed, cuda ret "
                << cuda_ret;
    return nullptr;
  }

  cudaStream_t stream;
  cuda_ret = cudaStreamCreate(&stream);
  if (cudaSuccess != cuda_ret) {
    MBLOG_ERROR << "Create cuda stream failed, cuda ret " << cuda_ret;
    return nullptr;
  }

  allocate_count_++;
  std::shared_ptr<CudaStream> stream_ptr(new CudaStream(stream, device_id_),
                                         [&](const CudaStream *stream_ptr) {
                                           Free(stream_ptr);
                                           delete stream_ptr;
                                         });
  return stream_ptr;
}

Status CudaStreamPool::Free(const CudaStream *stream) {
  if (stream == nullptr) {
    return STATUS_SUCCESS;
  }

  release_stream_queue_.Push(stream->Get());
  return STATUS_SUCCESS;
}

CudaMemory::CudaMemory(const std::shared_ptr<Device> &device,
                       const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                       std::shared_ptr<void> device_mem_ptr, size_t size)
    : DeviceMemory(device, mem_mgr, device_mem_ptr, size, false) {}

CudaMemory::~CudaMemory() {}

Status CudaMemory::BindStream(
    const std::shared_ptr<CudaStream> &stream_ptr) {
  if (cuda_stream_ptr_ != nullptr) {
    if (cuda_stream_ptr_ == stream_ptr) {
      return STATUS_SUCCESS;
    }
    // Change stream to another is not allowed
    return {STATUS_BUSY, "Memory has been bound to a stream"};
  }

  Status ret = STATUS_SUCCESS;
  if (stream_ptr != nullptr) {
    if (stream_ptr->IsInDevice(device_->GetDeviceID())) {
      cuda_stream_ptr_ = stream_ptr;
      return STATUS_SUCCESS;
    }
    // We need create a new stream when cross gpu device, so bind failed in fact
    ret = STATUS_BUSY;
  }

  auto cuda_mem_mgr = std::static_pointer_cast<CudaMemoryManager>(mem_mgr_);
  cuda_stream_ptr_ = cuda_mem_mgr->AllocStream();
  return ret;
}

Status CudaMemory::DetachStream() {
  if (cuda_stream_ptr_ == nullptr) {
    return STATUS_SUCCESS;
  }

  auto ret = cuda_stream_ptr_->Sync();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  cuda_stream_ptr_.reset();
  return STATUS_SUCCESS;
}

Status CudaMemory::CopyExtraMetaTo(std::shared_ptr<DeviceMemory> &device_mem) {
  if (device_mem->GetDevice() != device_) {
    return STATUS_SUCCESS;
  }

  auto target = std::static_pointer_cast<CudaMemory>(device_mem);
  target->cuda_stream_ptr_ = cuda_stream_ptr_;
  return STATUS_SUCCESS;
}

Status CudaMemory::CombineExtraMeta(
    const std::vector<std::shared_ptr<DeviceMemory>> &mem_list) {
  for (auto mem : mem_list) {
    auto cuda_mem = std::dynamic_pointer_cast<CudaMemory>(mem);
    if (cuda_stream_ptr_ == nullptr) {
      cuda_stream_ptr_ = cuda_mem->cuda_stream_ptr_;
    } else {
      auto other_cuda_stream_ptr = cuda_mem->cuda_stream_ptr_;
      if (other_cuda_stream_ptr == nullptr) {
        continue;
      }

      if (cuda_stream_ptr_ == other_cuda_stream_ptr) {
        continue;
      }

      auto ret = other_cuda_stream_ptr->Sync();
      if (ret != STATUS_SUCCESS) {
        MBLOG_ERROR << "Sync cuda stream failed when combine cuda memory";
        return STATUS_FAULT;
      }
    }
  }

  return STATUS_SUCCESS;
}

CudaMemoryPool::CudaMemoryPool(const std::string &device_id) {
  gpu_id_ = atoi(device_id.c_str());
}

Status CudaMemoryPool::Init() {
  auto status = InitSlabCache();
  if (!status) {
    return {status, "init mempool failed."};
  }

  auto timer = std::make_shared<TimerTask>();
  timer->Callback(&CudaMemoryPool::OnTimer, this);
  flush_timer_ = timer;

  // flush slab every 10s
  GetTimer()->Schedule(flush_timer_, 1000, 10000);
  return STATUS_OK;
}

CudaMemoryPool::~CudaMemoryPool() {
  if (flush_timer_) {
    flush_timer_->Stop();
    flush_timer_ = nullptr;
  }
}

void CudaMemoryPool::OnTimer() {
  // TODO support config shrink time.
}

void *CudaMemoryPool::MemAlloc(size_t size) {
  auto cuda_ret = cudaSetDevice(gpu_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Bind device " << gpu_id_ << " failed, cuda ret "
                << cuda_ret;
    return nullptr;
  }

  void *cuda_mem_ptr = nullptr;
  cuda_ret = cudaMalloc(&cuda_mem_ptr, size);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Cuda malloc failed, size " << size << ", err code "
                << cuda_ret;
    return nullptr;
  }

  return cuda_mem_ptr;
}

void CudaMemoryPool::MemFree(void *ptr) {
  auto free_func = [](int32_t gpu_id, void *mem_ptr, bool with_log) {
    cudaError_t cuda_ret = cudaSuccess;
    DeferCond { return cuda_ret != cudaSuccess && with_log; };
    DeferCondAdd {
      MBLOG_ERROR << "Free mem on gpu " << gpu_id << " failed, cuda ret "
                  << cuda_ret;
    };

    cuda_ret = cudaSetDevice(gpu_id);
    if (cuda_ret != cudaSuccess) {
      return cuda_ret;
    }

    cuda_ret = cudaFree(mem_ptr);
    if (cuda_ret != cudaSuccess) {
      return cuda_ret;
    }

    return cuda_ret;
  };

  auto timer = GetTimer();
  auto with_log = (timer == nullptr);
  auto ret = free_func(gpu_id_, ptr, with_log);
  if (ret == cudaSuccess || timer == nullptr) {
    return;
  }

  auto free_task = std::make_shared<TimerTask>(free_func, gpu_id_, ptr, true);
  free_task->SetName("cudaMemFreeTask");
  timer->Schedule(free_task, 0, 0, true);
}

CudaMemoryManager::CudaMemoryManager(const std::string &device_id)
    : DeviceMemoryManager(device_id),
      stream_pool_(device_id),
      mem_pool_(std::make_shared<CudaMemoryPool>(device_id)),
      mem_copy_kind_map_{{DeviceMemoryCopyKind::FromHost,
                          cudaMemcpyKind::cudaMemcpyHostToDevice},
                         {DeviceMemoryCopyKind::SameDeviceType,
                          cudaMemcpyKind::cudaMemcpyDeviceToDevice},
                         {DeviceMemoryCopyKind::ToHost,
                          cudaMemcpyKind::cudaMemcpyDeviceToHost}} {
  try {
    gpu_id_ = std::stoi(device_id);
  } catch (const std::exception &e) {
    MBLOG_WARN << "Convert device id to int failed, id " << device_id
               << ", err " << e.what() << "; use device 0 as default";
  }

  std::string name = "cuda_" + std::to_string(gpu_id_);
  mem_pool_->RegisterCollector(name);
}

CudaMemoryManager::~CudaMemoryManager() {
  mem_pool_->DestroySlabCache();
  std::string name = "cuda_" + std::to_string(gpu_id_);
  mem_pool_->UnregisterCollector(name);
}

Status CudaMemoryManager::Init() { return mem_pool_->Init(); }

std::shared_ptr<DeviceMemory> CudaMemoryManager::MakeDeviceMemory(
    const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
    size_t size) {
  return std::make_shared<CudaMemory>(device, shared_from_this(), mem_ptr,
                                      size);
}

std::shared_ptr<void> CudaMemoryManager::AllocSharedPtr(size_t size,
                                                        uint32_t mem_flags) {
  return mem_pool_->AllocSharedPtr(size);
}

void *CudaMemoryManager::Malloc(size_t size, uint32_t mem_flags) {
  return mem_pool_->MemAlloc(size);
};

void CudaMemoryManager::Free(void *mem_ptr, uint32_t mem_flags) {
  mem_pool_->MemFree(mem_ptr);
}

Status CudaMemoryManager::Copy(void *dest, size_t dest_size,
                               const void *src_buffer, size_t src_size,
                               DeviceMemoryCopyKind kind) {
  if (dest == nullptr || src_buffer == nullptr) {
    MBLOG_ERROR << "Cuda copy src " << src_buffer << " to dest " << dest
                << "failed";
    return STATUS_INVALID;
  }

  if (dest_size < src_size) {
    MBLOG_ERROR << "Cuda memcpy failed, dest size < src size";
    return STATUS_RANGE;
  }

  auto cuda_ret = cudaSetDevice(gpu_id_);
  if (cudaSuccess != cuda_ret) {
    MBLOG_ERROR << "Bind device " << gpu_id_ << " failed, cuda ret "
                << cuda_ret;
    return STATUS_FAULT;
  }

  cudaMemcpyKind cuda_copy_kind;
  GetCudaMemcpyKind(kind, cuda_copy_kind);
  cuda_ret = cudaMemcpy(dest, src_buffer, src_size, cuda_copy_kind);
  if (cudaSuccess != cuda_ret) {
    MBLOG_ERROR << "Cuda memcpy failed, ret " << cuda_ret << ", src size "
                << src_size << ", cuda cpy kind " << cuda_copy_kind;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status CudaMemoryManager::GetDeviceMemUsage(size_t *free, size_t *total) const {
  auto cuda_ret = cudaSetDevice(gpu_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Bind gpu device " << device_id_ << " failed, cuda ret "
                << cuda_ret;
    return STATUS_FAULT;
  }

  size_t t_free;
  size_t t_total;
  cuda_ret = cudaMemGetInfo(&t_free, &t_total);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Get gpu " << device_id_ << " mem info failed, cuda ret"
                << cuda_ret;
    return STATUS_FAULT;
  }

  if (free != nullptr) {
    *free = t_free;
  }

  if (total != nullptr) {
    *total = t_total;
  }

  return STATUS_SUCCESS;
}

Status CudaMemoryManager::DeviceMemoryCopy(
    const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, DeviceMemoryCopyKind copy_kind) {
  cudaMemcpyKind cuda_copy_kind;
  GetCudaMemcpyKind(copy_kind, cuda_copy_kind);
  std::shared_ptr<CudaStream> cuda_stream_ptr;
  auto ret = SetupCudaStream(src_memory, dest_memory, cuda_stream_ptr);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Set up cuda stream failed, using sync mem copy";
  }

  cudaStream_t cuda_stream =
      cuda_stream_ptr == nullptr ? nullptr : cuda_stream_ptr->Get();
  auto dest_device = dest_memory->GetDevice();
  auto src_device = src_memory->GetDevice();
  auto dest_ptr = dest_memory->GetPtr<uint8_t>().get() + dest_offset;
  auto src_ptr = src_memory->GetConstPtr<uint8_t>().get() + src_offset;
  ret = CudaMemcpyAsync(dest_ptr, src_ptr, src_size, dest_device, src_device,
                        cuda_copy_kind, cuda_stream);
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  if (cuda_stream_ptr != nullptr) {
    if (dest_memory->IsHost()) {
      cuda_stream_ptr->Sync();
    } else {
      // When async operation complete, the reference of memory will be
      // released
      cuda_stream_ptr->Bind({src_memory, dest_memory});
    }
  }

  return STATUS_SUCCESS;
}

Status CudaMemoryManager::CudaMemcpyAsync(
    uint8_t *dest_ptr, const uint8_t *src_ptr, size_t src_size,
    std::shared_ptr<Device> dest_device, std::shared_ptr<Device> src_device,
    cudaMemcpyKind cuda_copy_kind, cudaStream_t cuda_stream) {
  cudaError_t cuda_ret;
  auto dest_dev_id = atoi(dest_device->GetDeviceID().c_str());
  auto src_dev_id = atoi(src_device->GetDeviceID().c_str());
  if (cuda_copy_kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice &&
      dest_device != src_device) {
    TryEnablePeerAccess(src_dev_id, dest_dev_id);
    cudaSetDevice(dest_dev_id);
    cuda_ret = cudaMemcpyPeerAsync(dest_ptr, dest_dev_id, src_ptr, src_dev_id,
                                   src_size, cuda_stream);
    if (cudaSuccess != cuda_ret) {
      MBLOG_ERROR << "cudaMemcpyAsync between gpu " << src_dev_id << " and gpu "
                  << dest_dev_id << " failed, try transfer in host, cuda ret "
                  << cuda_ret << ", size " << src_size << ", copy kind "
                  << cuda_copy_kind << ", stream " << cuda_stream
                  << ", src_ptr " << (void *)src_ptr << ", dest_ptr "
                  << (void *)dest_ptr;
      return STATUS_NOTSUPPORT;
    }
  } else {
    auto gpu_id = dest_dev_id;
    if (cuda_copy_kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
      gpu_id = src_dev_id;
    }

    cudaSetDevice(gpu_id);
    cuda_ret = cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cuda_copy_kind,
                               cuda_stream);
    if (cudaSuccess != cuda_ret) {
      MBLOG_ERROR << "cudaMemcpyAsync failed, err code " << cuda_ret
                  << ", size " << src_size << ", copy kind " << cuda_copy_kind
                  << ", stream " << cuda_stream << ", src " << (void *)src_ptr
                  << ", dest_ptr " << (void *)dest_ptr << ", gpu_id " << gpu_id;
      return STATUS_FAULT;
    }
  }

  return STATUS_SUCCESS;
}

void CudaMemoryManager::GetCudaMemcpyKind(DeviceMemoryCopyKind copy_kind,
                                          cudaMemcpyKind &cuda_copy_kind) {
  cuda_copy_kind = mem_copy_kind_map_[copy_kind];
}

void CudaMemoryManager::TryEnablePeerAccess(int32_t src_gpu_id,
                                            int32_t dest_gpu_id) {
  cudaSetDevice(src_gpu_id);
  cudaDeviceEnablePeerAccess(dest_gpu_id, 0);
  cudaSetDevice(dest_gpu_id);
  cudaDeviceEnablePeerAccess(src_gpu_id, 0);
}

Status CudaMemoryManager::SetupCudaStream(
    std::shared_ptr<const DeviceMemory> src_memory,
    std::shared_ptr<DeviceMemory> dest_memory,
    std::shared_ptr<CudaStream> &cuda_stream_ptr) {
  if (src_memory->IsHost()) {
    cuda_stream_ptr = nullptr;
  } else {
    cuda_stream_ptr =
        std::static_pointer_cast<const CudaMemory>(src_memory)->GetBindStream();
  }

  if (!dest_memory->IsHost()) {
    auto dest_cuda_memory = std::dynamic_pointer_cast<CudaMemory>(dest_memory);
    auto ret = dest_cuda_memory->BindStream(cuda_stream_ptr);
    if (ret == STATUS_BUSY && cuda_stream_ptr != nullptr) {
      // Case: Two memory has different stream, we choose to sync source
      cuda_stream_ptr->Sync();
    }

    cuda_stream_ptr = dest_cuda_memory->GetBindStream();
  }

  return STATUS_SUCCESS;
}
}  // namespace modelbox