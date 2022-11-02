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


#ifndef MODELBOX_OS_H_
#define MODELBOX_OS_H_

#include <memory>
#include <thread>
#include <vector>

#include "modelbox/base/status.h"

namespace modelbox {

/**
 * @brief OS process API
 */
class OSProcess {
 public:
  OSProcess();
  virtual ~OSProcess();

  /**
   * @brief Get thread number of specific process
   * @param pid process id
   * @return thread number
   */
  virtual int32_t GetThreadsNumber(uint32_t pid) = 0;

  /**
   * @brief Get memory usage of specific process
   * @param pid process id
   * @return memory size
   */
  virtual uint32_t GetMemorySize(uint32_t pid) = 0;

  /**
   * @brief Get memory rss usage of specific process
   * @param pid process id
   * @return memory size
   */
  virtual uint32_t GetMemoryRSS(uint32_t pid) = 0;

  /**
   * @brief Get shared memory usage of specific process
   * @param pid process id
   * @return memory size
   */
  virtual uint32_t GetMemorySHR(uint32_t pid) = 0;

  /**
   * @brief Get current process id
   * @return process id
   */
  virtual uint32_t GetPid() = 0;

  /**
   * @brief Get Process parent pid
   * @return process id
   */
  virtual uint32_t GetPPid() = 0;

  virtual std::vector<uint32_t> GetProcessTime(uint32_t pid) = 0;

  virtual std::vector<uint32_t> GetTotalTime(uint32_t pid) = 0;
};

/**
 * @brief OS thread API
 */
class OSThread {
 public:
  OSThread();
  virtual ~OSThread();

  /**
   * @brief Get current thread id
   * @return thread id
   */
  virtual std::thread::id GetTid() = 0;

  /**
   * @brief Set current thread name
   * @param name thread name.
   * @return whether success
   */
  virtual Status SetName(const std::string &name) = 0;

  /**
   * @brief Set current thread priority
   * @param thread thread handler
   * @param priority priority
   */
  virtual Status SetThreadPriority(const std::thread::id &thread,
                                   int32_t priority) = 0;

  /**
   * @brief Set current thread logical affinity
   * @param thread thread handler
   * @param l_cpus logical cpu list
   * @return whether success
   */
  virtual Status SetThreadLogicalCPUAffinity(
      const std::thread::id &thread, const std::vector<int16_t> &l_cpus) = 0;

  /**
   * @brief Set current thread physical affinity
   * @param thread thread handler
   * @param p_cpus physical cpu list
   * @return whether success
   */
  virtual Status SetThreadPhysicalCPUAffinity(
      const std::thread::id &thread, const std::vector<int16_t> &p_cpus) = 0;

  /**
   * @brief Get current thread priority
   * @param thread thread handler
   * @return thread priority
   */
  virtual int32_t GetThreadPriority(const std::thread::id &thread) = 0;
};

/**
 * @brief OS information API
 */
class OSInfo {
 public:
  OSInfo();
  virtual ~OSInfo();

  /**
   * @brief Get system memory usage
   * @param free free memory
   * @param total total memory
   * @return whether success
   */
  virtual Status GetMemoryUsage(size_t *free, size_t *total) = 0;

  /**
   * @brief Get system cpu run time
   * @return cpu run time list
   */
  virtual std::vector<uint32_t> GetCpuRunTime() = 0;

  /**
   * @brief Get system physical cpu number
   * @return physical cpu number
   */
  virtual int32_t GetPhysicalCpuNumbers() = 0;

  /**
   * @brief Get system logical cpu number
   * @return logical cpu number
   */
  virtual int32_t GetLogicalCpuNumbers() = 0;

  /**
   * @brief Get system id
   * @return system id in string
   */
  virtual std::string GetSystemID() = 0;

  /**
   * @brief Get network interface mac address
   * @param nic network interface name, default is first.
   * @return mac address
   */
  virtual std::string GetMacAddress(const std::string &nic = "") = 0;

  /// Process API
  std::shared_ptr<OSProcess> Process;

  /// Thread API
  std::shared_ptr<OSThread> Thread;
};

/// OS API
extern OSInfo *os;

}  // namespace modelbox
#endif