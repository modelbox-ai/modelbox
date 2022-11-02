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

#ifndef MODELBOX_DEVICE_H_
#define MODELBOX_DEVICE_H_

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "modelbox/base/device_memory.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/executor.h"
#include "modelbox/base/status.h"

namespace modelbox {

constexpr const char *DRIVER_CLASS_DEVICE = "DRIVER-DEVICE";
constexpr const int MAX_CIRCLE_LIST_SIZE = 100;
class DeviceManager;

using DeleteFunction = std::function<void(void *)>;

/**
 * @brief circle list used to save trace log
 */
class CircleQueue {
 public:
  CircleQueue();
  virtual ~CircleQueue();

  void EnQueue(const std::string &data);
  std::string &DeQueue();
  std::string &GetQueue();
  bool Empty();
  bool Full();

 private:
  std::string data_[MAX_CIRCLE_LIST_SIZE];
  int front_;
  int rear_;
};

class DeviceDesc {
 public:
  DeviceDesc();
  virtual ~DeviceDesc();

  virtual std::string GetDeviceId();
  virtual std::string GetDeviceType();
  virtual std::string GetDeviceMemory();
  virtual std::string GetDeviceVersion();
  virtual std::string GetDeviceDesc();

  void SetDeviceId(const std::string &device_id);
  void SetDeviceType(const std::string &device_type);
  void SetDeviceMemory(const std::string &device_memory);
  void SetDeviceVersion(const std::string &device_version);
  void SetDeviceDesc(const std::string &device_desc);

 protected:
  std::string device_id_;
  std::string device_type_;
  std::string device_memory_;
  std::string device_version_;
  std::string device_description_;
};

using DevExecuteCallBack = std::function<Status(size_t idx)>;

class Device : public std::enable_shared_from_this<Device> {
 public:
  Device();
  Device(std::shared_ptr<DeviceMemoryManager> mem_mgr);
  Device(size_t thread_count, std::shared_ptr<DeviceMemoryManager> mem_mgr);
  virtual ~Device();

  virtual std::string GetDeviceID() const;

  virtual std::string GetType() const;

  /**
   * @brief when make mem contiguous, need test whether the device supports
   * @return whether specify device supports mem contiguous
   **/
  virtual bool SupportMemContiguous() const;

  void SetDeviceDesc(std::shared_ptr<DeviceDesc> device_desc);

  std::shared_ptr<DeviceDesc> GetDeviceDesc();

  Status Init();

  virtual Status DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                               size_t count);

  virtual std::list<std::future<Status>> DeviceExecuteAsync(
      const DevExecuteCallBack &fun, int32_t priority, size_t count,
      bool resource_nice);

  std::shared_ptr<Executor> GetDeviceExecutor();

  /**
   * @brief Set allocatable memory limit
   * @param mem_quota quota memory size
   **/
  void SetMemQuota(size_t mem_quota);

  /**
   * @brief Get allocatable memory limit
   * @return Memory limit
   **/
  size_t GetMemQuota() const;

  /**
   * @brief Get allocated memory size
   * @return Memory allocated
   **/
  size_t GetAllocatedMemSize() const;

  /**
   * @brief Malloc device memory, memory size = 0 is ok
   * @param size Memory size
   * @param mem_flags Flags to create device memory
   * @param user_id user id
   * @return Device memory
   **/
  std::shared_ptr<DeviceMemory> MemAlloc(size_t size, uint32_t mem_flags = 0,
                                         const std::string &user_id = "");

  /**
   * @brief Malloc device memory, memory size = 0 is ok
   * @param size Memory size
   * @param capacity Memory physic size
   * @param mem_flags memory flags
   * @param user_id user id
   * @return Device memory
   **/
  std::shared_ptr<DeviceMemory> MemAlloc(size_t size, size_t capacity,
                                         uint32_t mem_flags,
                                         const std::string &user_id = "");

  /**
   * @brief Manage exist device mem
   * @param mem_ptr Exist mem
   * @param size Memory size
   * @param deleter memory delete function
   * @param mem_flags memory flags
   * @return Device memory
   **/
  std::shared_ptr<DeviceMemory> MemAcquire(void *mem_ptr, size_t size,
                                           const DeleteFunction &deleter,
                                           uint32_t mem_flags = 0);

  /**
   * @brief Manage exist device mem
   * @param mem_ptr Exist mem
   * @param size Memory size
   * @param mem_flags memory flags
   * @return Device memory
   **/
  std::shared_ptr<DeviceMemory> MemAcquire(void *mem_ptr, size_t size,
                                           uint32_t mem_flags = 0);

  /**
   * @brief Manage exist device mem
   * @param mem_ptr Exist mem
   * @param size Memory size
   * @param mem_flags Flags of device mem
   * @return Device memory
   **/
  std::shared_ptr<DeviceMemory> MemAcquire(const std::shared_ptr<void> &mem_ptr,
                                           size_t size, uint32_t mem_flags = 0);

  /**
   * @brief Write host data to device, and create a new device memory, host_data
   * is collected by os interface
   * @param host_data Host data to write
   * @param host_size Host data size
   * @param user_id User id
   **/
  std::shared_ptr<DeviceMemory> MemWrite(const void *host_data,
                                         size_t host_size,
                                         const std::string &user_id = "");

  /**
   * @brief Clone source device memory to this device
   * if source device memory is readonly, and in this device, same pointer will
   * return. Otherwise, will make a copy.
   * @param src_memory Memory to clone
   * @param user_id user id
   * @return A clone memory
   **/
  std::shared_ptr<DeviceMemory> MemClone(
      std::shared_ptr<DeviceMemory> src_memory,
      const std::string &user_id = "");

  /**
   * @brief Get device memory info
   * @return Status
   */
  Status GetMemInfo(size_t *free, size_t *total) const;

  /**
   * @brief Get memory usage trace
   * @return Memory usage trace
   **/
  std::shared_ptr<DeviceMemoryTrace> GetMemoryTrace() const;

  /**
   * @brief Get device manager
   * @return device manager
   **/
  std::shared_ptr<DeviceManager> GetDeviceManager();

  /**
   * @brief Set device manager
   * @return void
   **/
  friend class DeviceManager;

 protected:
  std::shared_ptr<Executor> executor_;
  void SetDeviceManager(const std::shared_ptr<DeviceManager> &device_mgr);

  std::list<std::future<Status>> DeviceExecuteAsyncRude(
      const DevExecuteCallBack &fun, int32_t priority, size_t count);

  std::list<std::future<Status>> DeviceExecuteAsyncNice(
      const DevExecuteCallBack &fun, int32_t priority, size_t count);

  virtual bool NeedResourceNice();

 private:
  std::shared_ptr<DeviceMemoryTrace> memory_trace_;
  std::shared_ptr<DeviceMemoryManager> memory_manager_;
  std::shared_ptr<DeviceDesc> device_desc_ = std::make_shared<DeviceDesc>();
  std::weak_ptr<DeviceManager> device_mgr_;
};

class DeviceFactory : public DriverFactory {
 public:
  DeviceFactory();
  ~DeviceFactory() override;

  virtual std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe();

  virtual std::shared_ptr<Device> CreateDevice(const std::string &device_id);

  virtual std::string GetDeviceFactoryType();

  virtual std::vector<std::string> GetDeviceList();

 private:
};

class DeviceManager : public std::enable_shared_from_this<DeviceManager> {
 public:
  DeviceManager();
  virtual ~DeviceManager();

  static std::shared_ptr<DeviceManager> GetInstance();
  Status Initialize(const std::shared_ptr<Drivers> &driver,
                    const std::shared_ptr<Configuration> &config);

  virtual std::vector<std::string> GetDevicesTypes();

  virtual std::vector<std::string> GetDevicesIdList(
      const std::string &device_type);

  std::shared_ptr<Device> CreateDevice(const std::string &device_type,
                                       const std::string &device_id);

  std::shared_ptr<Device> GetDevice(const std::string &device_type,
                                    const std::string &device_id);

  Status Register(const std::shared_ptr<DeviceFactory> &factory);

  /**
   * @brief Return host device
   * @return Host device
   **/
  std::shared_ptr<Device> GetHostDevice();
  void Clear();

  /**
   * GetDeviceFactoryList(), GetDeviceDescList(), GetDeviceList()
   * only for test
   */
  const std::map<std::string, std::shared_ptr<DeviceFactory>>
      &GetDeviceFactoryList();
  const std::map<std::string,
                 std::map<std::string, std::shared_ptr<DeviceDesc>>>
      &GetDeviceDescList();
  const std::map<std::string, std::map<std::string, std::shared_ptr<Device>>>
      &GetDeviceList();

  Status DeviceProbe();
  Status InitDeviceFactory(const std::shared_ptr<Drivers> &driver);
  std::shared_ptr<Drivers> GetDrivers();

 private:
  Status CheckDeviceManagerInit();
  void SetDrivers(std::shared_ptr<Drivers> drivers);
  std::map<std::string, std::shared_ptr<DeviceFactory>> device_factory_;
  std::map<std::string, std::map<std::string, std::shared_ptr<DeviceDesc>>>
      device_desc_list_;
  std::map<std::string, std::map<std::string, std::shared_ptr<Device>>>
      device_list_;
  std::shared_ptr<Drivers> drivers_;
};

}  // namespace modelbox
#endif  // MODELBOX_DEVICE_H_
