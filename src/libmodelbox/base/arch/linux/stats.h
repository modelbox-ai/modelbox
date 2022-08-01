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


#include <iostream>
#include "modelbox/base/os.h"

namespace modelbox {

class LinuxOSProcess : public OSProcess {
 public:
  LinuxOSProcess();
  ~LinuxOSProcess() override;
  int32_t GetThreadsNumber(uint32_t pid) override;
  uint32_t GetMemorySize(uint32_t pid) override;
  uint32_t GetMemoryRSS(uint32_t pid) override;
  uint32_t GetMemorySHR(uint32_t pid) override;
  uint32_t GetPid() override;
  uint32_t GetPPid() override;

  std::vector<uint32_t> GetProcessTime(uint32_t pid) override;
  std::vector<uint32_t> GetTotalTime(uint32_t pid) override;
};

class LinuxOSThread : public OSThread {
 public:
  LinuxOSThread();
  ~LinuxOSThread() override;

  std::thread::id GetTid() override;
  Status SetName(const std::string &name) override;
  Status SetThreadPriority(const std::thread::id &thread,
                           int32_t priority) override;
  Status SetThreadLogicalCPUAffinity(
      const std::thread::id &thread,
      const std::vector<int16_t> &l_cpus) override;
  Status SetThreadPhysicalCPUAffinity(
      const std::thread::id &thread,
      const std::vector<int16_t> &p_cpus) override;
  int32_t GetThreadPriority(const std::thread::id &thread) override;
};

class LinuxOSInfo : public OSInfo {
 public:
  LinuxOSInfo();
  ~LinuxOSInfo() override;

  Status GetMemoryUsage(size_t *free, size_t *total) override;

  std::vector<uint32_t> GetCpuRunTime() override;

  int32_t GetPhysicalCpuNumbers() override;
  int32_t GetLogicalCpuNumbers() override;

  std::string GetSystemID() override;

  std::string GetMacAddress(const std::string &nic = "") override;

  static LinuxOSInfo &GetInstance();
};

}  // namespace modelbox