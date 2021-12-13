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

#ifndef MODELBOX_ASCEND_MEMORY_H_
#define MODELBOX_ASCEND_MEMORY_H_

#include <acl/acl.h>
#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <list>
#include <thread>

#ifndef ENABLE_DVPP_INTERFACE
#define ENABLE_DVPP_INTERFACE
#endif

#include <acl/ops/acl_dvpp.h>

extern modelbox::Timer *GetTimer();

namespace modelbox {
constexpr uint32_t ASCEND_MEM_NORMAL = 0;
constexpr uint32_t ASCEND_MEM_DVPP = 1;

constexpr uintptr_t ASCEND_ASYNC_ALIGN =
    64;  // Precondition for ascend async copy

void AscendReleaseMemoryAsync(void *mem_list_ptr);

class AscendMemory;
class AscendStreamPool;
class AscendMemoryManager;

class AscendStream {
  friend class AscendMemory;
  friend class AscendStreamPool;

 public:
  AscendStream(const AscendStream &stream) = delete;
  AscendStream(const AscendStream &&stream) = delete;
  AscendStream &operator=(const AscendStream &stream) = delete;
  AscendStream &operator=(const AscendStream &&stream) = delete;
  virtual ~AscendStream();
  AscendStream(int32_t device_id, uint64_t callback_tid);

  inline bool IsInDevice(const std::string &device_id) const {
    auto device_id_num = atoi(device_id.c_str());
    return IsInDevice(device_id_num);
  }

  inline bool IsInDevice(int32_t device_id) const {
    return device_id == device_id_;
  }

  inline aclrtStream Get() const { return stream_; }

  Status Sync() const;

  Status Bind(std::vector<std::shared_ptr<const DeviceMemory>> mem_list) const;

 protected:
  Status Init();

  void Deinit();

 private:
  aclrtStream stream_;
  int32_t device_id_{0};
  uint64_t callback_thread_id_{0};
  std::atomic_bool init_flag_{false};
};

class AscendStreamPool : public std::enable_shared_from_this<AscendStreamPool> {
 public:
  AscendStreamPool(const std::string &device_id);

  virtual ~AscendStreamPool();

  /**
   * @brief Allocate cuda stream associated with device
   * @return Cuda stream or nullptr
   */
  std::shared_ptr<AscendStream> Alloc();

  /**
   * @brief Release cuda stream
   * @param stream Cuda stream to free
   */
  Status Free(AscendStream *&stream);

  /**
   * @brief Get allocated stream count
   * @return Allocated stream count
   */
  inline size_t GetAllocatedStreamCount() const {
    return allocate_count_.load();
  }

  void StreamCallBack();

  void Shrink();

 private:
  std::atomic<size_t> allocate_count_{0};
  int32_t device_id_{0};
  std::shared_ptr<std::thread> stream_callback_thread_;
  uint64_t callback_thread_id_{0};
  bool is_exit_{false};

  std::mutex stream_list_lock_;
  std::list<AscendStream *> stream_list_;
  Timer stream_shrink_timer_;
  const uint64_t shrink_interval_ms_{10 * 60 * 1000};  // 10 min
};

class AscendMemory : public DeviceMemory {
 public:
  AscendMemory(const std::shared_ptr<Device> &device,
               const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
               std::shared_ptr<void> device_mem_ptr, size_t size);

  virtual ~AscendMemory();

  /**
   * @brief Get bind ascend stream
   * @return Ascend stream
   */
  inline std::shared_ptr<AscendStream> GetBindStream() const {
    return ascend_stream_ptr_;
  }

  /**
   * @brief Bind ascend stream
   * @param stream_ptr Ascend stream
   *        if null
   *          new stream will be created
   *        else
   *          set stream when return success
   *          has one different stream when return busy
   */
  Status BindStream(const std::shared_ptr<AscendStream> &stream_ptr = nullptr);

  Status DetachStream();

 protected:
  Status CopyExtraMetaTo(std::shared_ptr<DeviceMemory> &device_mem) override;

  Status CombineExtraMeta(
      const std::vector<std::shared_ptr<DeviceMemory>> &mem_list) override;

 private:
  std::shared_ptr<AscendStream> ascend_stream_ptr_;
};

class AscendMemoryPool : public MemoryPoolBase {
 public:
  AscendMemoryPool(AscendMemoryManager *mem_manager);

  virtual ~AscendMemoryPool();

  Status Init();

  virtual void *MemAlloc(size_t size);

  virtual void MemFree(void *ptr);

  virtual void OnTimer();

 private:
  AscendMemoryManager *mem_manager_;
  std::shared_ptr<TimerTask> flush_timer_;
};

class AscendMemoryManager : public DeviceMemoryManager {
 public:
  AscendMemoryManager(const std::string &device_id);
  virtual ~AscendMemoryManager();

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
   * @param size Memory size to allocate.
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
   * @param copy_kind Memory copy mode
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

  inline std::shared_ptr<AscendStream> AllocStream() {
    return stream_pool_->Alloc();
  };

 private:
  void GetAscendMemcpyKind(DeviceMemoryCopyKind copy_kind,
                           aclrtMemcpyKind &ascend_copy_kind);

  Status SetupAscendStream(std::shared_ptr<const DeviceMemory> src_memory,
                           std::shared_ptr<DeviceMemory> dest_memory,
                           std::shared_ptr<AscendStream> &ascend_stream_ptr);

  bool CheckCopyAsync(const void *src_addr, const void *dest_addr);

  std::shared_ptr<AscendStreamPool> stream_pool_;
  AscendMemoryPool mem_pool_;
  std::map<DeviceMemoryCopyKind, aclrtMemcpyKind> mem_copy_kind_map_;
  int32_t npu_id_{0};
};

}  // namespace modelbox

#endif  // MODELBOX_CUDA_MEMORY_H_
