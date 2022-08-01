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

#include "stats.h"

#include <net/if.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/base/status.h"
#include "securec.h"

namespace modelbox {

OSInfo *os = &LinuxOSInfo::GetInstance();

// OSProcess
LinuxOSProcess::LinuxOSProcess() = default;

LinuxOSProcess::~LinuxOSProcess() = default;

uint32_t LinuxOSProcess::GetPid() { return getpid(); }

int32_t LinuxOSProcess::GetThreadsNumber(uint32_t pid) { return 0; }

uint32_t LinuxOSProcess::GetMemorySize(uint32_t pid) { return 0; }

uint32_t LinuxOSProcess::GetMemorySHR(uint32_t pid) { return 0; }

uint32_t LinuxOSProcess::GetMemoryRSS(uint32_t pid) { return 0; }

std::vector<uint32_t> LinuxOSProcess::GetProcessTime(uint32_t pid) {
  std::vector<uint32_t> ss{0, 0};
  return ss;
}

std::vector<uint32_t> LinuxOSProcess::GetTotalTime(uint32_t pid) {
  std::vector<uint32_t> ss{0, 0};
  return ss;
}

// OSThread
LinuxOSThread::LinuxOSThread() = default;
LinuxOSThread::~LinuxOSThread() = default;

uint32_t LinuxOSProcess::GetPPid() { return 0; };

std::thread::id LinuxOSThread::GetTid() { return std::this_thread::get_id(); };

Status LinuxOSThread::SetName(const std::string &name) {
  prctl(PR_SET_NAME, name.c_str(), 0, 0, 0);
  return STATUS_OK;
}

Status LinuxOSThread::SetThreadPriority(const std::thread::id &thread,
                                        int32_t priority) {
  return STATUS_OK;
};

Status LinuxOSThread::SetThreadLogicalCPUAffinity(
    const std::thread::id &thread, const std::vector<int16_t> &l_cpus) {
  return STATUS_OK;
};

Status LinuxOSThread::SetThreadPhysicalCPUAffinity(
    const std::thread::id &thread, const std::vector<int16_t> &p_cpus) {
  return STATUS_OK;
};

int32_t LinuxOSThread::GetThreadPriority(const std::thread::id &thread) {
  return 0;
};

// OSInfo
LinuxOSInfo::LinuxOSInfo() {
  Process = std::make_shared<LinuxOSProcess>();
  Thread = std::make_shared<LinuxOSThread>();
};

LinuxOSInfo::~LinuxOSInfo() = default;

LinuxOSInfo &LinuxOSInfo::GetInstance() {
  static LinuxOSInfo os;
  return os;
};

Status LinuxOSInfo::GetMemoryUsage(size_t *free, size_t *total) {
  struct sysinfo si;
  auto ret = sysinfo(&si);
  if (ret != 0) {
    MBLOG_ERROR << "Get sys mem info failed, ret " << ret;
    return STATUS_FAULT;
  }

  if (free != nullptr) {
    *free = si.freeram;
  }

  if (total != nullptr) {
    *total = si.totalram;
  }

  return STATUS_SUCCESS;
}

std::vector<uint32_t> LinuxOSInfo::GetCpuRunTime() {
  std::vector<uint32_t> ss{0, 0};
  return ss;
};

int32_t LinuxOSInfo::GetPhysicalCpuNumbers() { return 1; };

int32_t LinuxOSInfo::GetLogicalCpuNumbers() {
  return sysconf(_SC_NPROCESSORS_ONLN);
};

std::string LinuxOSInfo::GetSystemID() {
  std::string result;
  std::ifstream file("/etc/machine-id");
  if (!file.fail()) {
    getline(file, result);
    if (result.length() > 0) {
      return result;
    }
    file.close();
  }

  struct utsname buf;
  if (uname(&buf) != 0) {
    StatusError = {STATUS_FAULT, StrError(errno)};
    return "";
  }

  result = buf.machine;
  result += buf.nodename;
  return result;
}

std::string LinuxOSInfo::GetMacAddress(const std::string &nic) {
  std::string mac;
  struct ifreq ifr;
  struct ifconf ifc;
  char buf[1024];

  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock == -1) {
    modelbox::StatusError = {modelbox::STATUS_FAULT, "create socket failed."};
    return mac;
  };
  Defer { close(sock); };

  ifc.ifc_len = sizeof(buf);
  ifc.ifc_buf = buf;
  if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) {
    modelbox::StatusError = {modelbox::STATUS_FAULT, "get socket info failed."};
    return mac;
  };

  struct ifreq *it = ifc.ifc_req;
  const struct ifreq *const end = it + (ifc.ifc_len / sizeof(struct ifreq));

  modelbox::StatusError = {modelbox::STATUS_NOTFOUND, "not found nic"};
  for (; it != end; ++it) {
    strncpy_s(ifr.ifr_name, IFNAMSIZ, it->ifr_name, IFNAMSIZ);
    if (ioctl(sock, SIOCGIFFLAGS, &ifr) != 0) {
      continue;
    }

    if (ifr.ifr_flags & IFF_LOOPBACK) {
      continue;
    }

    if (nic.length() > 0 && nic == it->ifr_name) {
      continue;
    }

    if (ioctl(sock, SIOCGIFHWADDR, &ifr) != 0) {
      continue;
    }

    char tmp[64];
    int len = snprintf_s(
        tmp, sizeof(tmp), sizeof(tmp), "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",
        (uint8_t)ifr.ifr_hwaddr.sa_data[0], (uint8_t)ifr.ifr_hwaddr.sa_data[1],
        (uint8_t)ifr.ifr_hwaddr.sa_data[2], (uint8_t)ifr.ifr_hwaddr.sa_data[3],
        (uint8_t)ifr.ifr_hwaddr.sa_data[4], (uint8_t)ifr.ifr_hwaddr.sa_data[5]);
    if (len < 0) {
      continue;
    }
    mac = tmp;
    break;
  }

  return mac;
}

}  // namespace modelbox