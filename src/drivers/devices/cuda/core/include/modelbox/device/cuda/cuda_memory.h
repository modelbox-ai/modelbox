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

#ifndef MODELBOX_CUDA_MEMORY_H_
#define MODELBOX_CUDA_MEMORY_H_

#include <cuda_runtime.h>
#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/slab.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <atomic>

extern modelbox::Timer *GetTimer();

namespace modelbox {
void CudaReleaseMemoryAsync(void *mem_list_ptr);

class CudaMemory;
class CudaMemoryManager;

class CudaStream {
  friend class CudaMemory;

 public:
  CudaStream(const CudaStream &stream) = delete;
  CudaStream(const CudaStream &&stream) = delete;
  CudaStream &operator=(const CudaStream &stream) = delete;
  CudaStream &operator=(const CudaStream &&stream) = delete;

  CudaStream(cudaStream_t stream, int32_t device_id);

  virtual ~CudaStream() = default;

  inline bool IsInDevice(const std::string &device_id) const {
    auto device_id_num = atoi(device_id.c_str());
    return IsInDevice(device_id_num);
  }

  inline bool IsInDevice(int32_t device_id) const {
    return device_id == device_id_;
  }

  inline cudaStream_t Get() const { return stream_; }

  Status Sync() const;

  Status Bind(std::vector<std::shared_ptr<const DeviceMemory>> mem_list) const;

 private:
  cudaStream_t stream_;
  int32_t device_id_;
};

class CudaStreamPool {
 public:
  /**
   * @brief Cuda stream pool
   * @param device_id cuda device id
   */
  CudaStreamPool(const std::string &device_id);

  virtual ~CudaStreamPool();
  /**
   * @brief Allocate cuda stream associated with device
   * @return Cuda stream or nullptr
   */
  std::shared_ptr<CudaStream> Alloc();

  /**
   * @brief Release cuda stream
   * @param stream Cuda stream to free
   */
  Status Free(const CudaStream *stream);

  /**
   * @brief Get allocated stream count
   * @return Allocated stream count
   */
  inline size_t GetAllocatedStreamCount() const {
    return allocate_count_.load();
  }

  /**
   * @brief Release streawm worker
   */
  void ReleaseStreamWorker();

 private:
  std::atomic<size_t> allocate_count_{0};
  int32_t device_id_;

  std::shared_ptr<std::thread> release_stream_thread_;
  std::atomic_bool is_running_{false};
  BlockingQueue<cudaStream_t> release_stream_queue_;
};

class CudaMemory : public DeviceMemory {
  friend class CudaStream;

 public:
  /**
   * @brief Cuda memory
   * @param device pointer to device
   * @param mem_mgr pointer to memory manager
   * @param device_mem_ptr cuda device memory pointer
   * @param size device memory size
   */
  CudaMemory(const std::shared_ptr<Device> &device,
             const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
             const std::shared_ptr<void> &device_mem_ptr, size_t size);

  ~CudaMemory() override;
  /**
   * @brief Get bind cuda stream
   * @return Cuda stream
   */
  inline std::shared_ptr<CudaStream> GetBindStream() const {
    return cuda_stream_ptr_;
  }

  /**
   * @brief Bind cuda stream
   * @param stream_ptr Cuda stream
   *        if null
   *          new stream will be created
   *        else
   *          set stream when return success
   *          has one different stream when return busy
   */
  Status BindStream(const std::shared_ptr<CudaStream> &stream_ptr = nullptr);

  /**
   * @brief Detach cuda stream
   * @return detach result
   */
  Status DetachStream();

 protected:
  Status CopyExtraMetaTo(std::shared_ptr<DeviceMemory> &device_mem) override;

  Status CombineExtraMeta(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list) override;

 private:
  std::shared_ptr<CudaStream> cuda_stream_ptr_;
};

class CudaMemoryPool : public MemoryPoolBase {
 public:
  CudaMemoryPool(const std::string &device_id);

  ~CudaMemoryPool() override;

  Status Init();

  void *MemAlloc(size_t size) override;

  void MemFree(void *ptr) override;

  virtual void OnTimer();

 private:
  int32_t gpu_id_{0};
  std::shared_ptr<TimerTask> flush_timer_;
};

class CudaMemoryManager : public DeviceMemoryManager {
 public:
  /**
   * @brief Cude memory manager
   * @param device_id device id
   */
  CudaMemoryManager(const std::string &device_id);

  ~CudaMemoryManager() override;

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
   * @param copy_kind copy memory mode
   * @return copy result
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

  /**
   * @brief Alloc a new cuda stream
   * @return pointer to cuda stream
   */
  inline std::shared_ptr<CudaStream> AllocStream() {
    return stream_pool_.Alloc();
  };

 private:
  /**
   * @brief Get matched cudaMemcpyKind
   * @param copy_kind Device memory copy kind
   * @param cuda_copy_kind Matched cudaMemcpyKind
   */
  void GetCudaMemcpyKind(DeviceMemoryCopyKind copy_kind,
                         cudaMemcpyKind &cuda_copy_kind);

  /**
   * @brief Prepare cuda stream according to copy kind
   * @param src_memory Source device memory in copy operation
   * @param dest_memory Destination device memory in copy operation
   * @param cuda_stream Cuda stream to use in cuda copy api
   * @return Status
   */
  Status SetupCudaStream(const std::shared_ptr<const DeviceMemory> &src_memory,
                         const std::shared_ptr<DeviceMemory> &dest_memory,
                         std::shared_ptr<CudaStream> &cuda_stream_ptr);

  void TryEnablePeerAccess(int32_t src_gpu_id, int32_t dest_gpu_id);

  Status CudaMemcpyAsync(uint8_t *dest_ptr, const uint8_t *src_ptr,
                         size_t src_size,
                         const std::shared_ptr<Device> &dest_device,
                         const std::shared_ptr<Device> &src_device,
                         cudaMemcpyKind cuda_copy_kind,
                         cudaStream_t cuda_stream);

  CudaStreamPool stream_pool_;
  std::shared_ptr<CudaMemoryPool> mem_pool_;
  std::map<DeviceMemoryCopyKind, cudaMemcpyKind> mem_copy_kind_map_;
  int32_t gpu_id_{0};
};

}  // namespace modelbox

#endif  // MODELBOX_CUDA_MEMORY_H_