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

#include "modelbox/common/utils.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/capability.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <mutex>

#include "modelbox/base/utils.h"
#include "securec.h"

namespace modelbox {

#define TMP_BUFF_LEN_32 32

static int kPidFileFd = -1;

extern "C" int capget(struct __user_cap_header_struct *header,
                      struct __user_cap_data_struct *cap);
extern "C" int capset(struct __user_cap_header_struct *header,
                      struct __user_cap_data_struct *cap);

std::once_flag root_dir_flag;

const std::string &modelbox_root_dir() {
  static std::string rootdir;

  std::call_once(root_dir_flag, []() {
    char buff[PATH_MAX] = {0};
    int len;

    len = readlink("/proc/self/exe", buff, sizeof(buff) - 1);
    if (len < 0) {
      rootdir = "";
      return;
    }

    buff[len] = {0};
    rootdir = modelbox::GetDirName(buff);
    rootdir = rootdir + "../../../../";
    rootdir = PathCanonicalize(rootdir);
    if (rootdir == "/") {
      rootdir = "";
    }
  });

  return rootdir;
}

std::string modelbox_full_path(const std::string &path) {
  std::string fullpath = path;
  modelbox::StringReplaceAll(fullpath, MODELBOX_ROOT_VAR, modelbox_root_dir());
  return fullpath;
}

int modelbox_create_pid(const char *pid_file) {
  int fd = -1;
  unsigned int flags;
  char buff[TMP_BUFF_LEN_32];
  int ret;

  if (pid_file == nullptr) {
    return -1;
  }

  /*  create pid file, and lock this file */
  fd = open(pid_file, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "create pid file failed, path %s, %s\n", pid_file,
            StrError(errno).c_str());
    return -1;
  }

  flags = fcntl(fd, F_GETFD);
  if (flags < 0) {
    fprintf(stderr, "Could not get flags for PID file %s, %s\n", pid_file,
            StrError(errno).c_str());
    goto errout;
  }

  flags |= FD_CLOEXEC;
  if (fcntl(fd, F_SETFD, flags) == -1) {
    fprintf(stderr, "Could not set flags for PID file %s, %s\n", pid_file,
            StrError(errno).c_str());
    goto errout;
  }

  if (lockf(fd, F_TLOCK, 0) < 0) {
    fprintf(stderr, "Server is already running.\n");
    goto errout;
  }

  ret = ftruncate(fd, 0);
  UNUSED_VAR(ret);

  ret =
      snprintf_s(buff, TMP_BUFF_LEN_32, TMP_BUFF_LEN_32 - 1, "%d\n", getpid());
  if (ret < 0 || ret == TMP_BUFF_LEN_32) {
    fprintf(stderr, "format pid failed.\n");
    goto errout;
  }

  if (write(fd, buff, strnlen(buff, TMP_BUFF_LEN_32)) < 0) {
    fprintf(stderr, "write pid to file failed, %s.\n", StrError(errno).c_str());
    goto errout;
  }

  if (kPidFileFd > 0) {
    close(kPidFileFd);
    kPidFileFd = -1;
  }

  kPidFileFd = fd;

  return 0;
errout:
  if (fd > 0) {
    close(fd);
  }
  return -1;
}

int modelbox_sig_register(const int sig_list[], int sig_num,
                          void (*action)(int, siginfo_t *, void *)) {
  int i = 0;
  struct sigaction sig_act;

  for (i = 0; i < sig_num; i++) {
    memset_s(&sig_act, sizeof(sig_act), 0, sizeof(sig_act));
    sig_act.sa_sigaction = action;
    sig_act.sa_flags = SA_SIGINFO;

    if (sigaction(sig_list[i], &sig_act, nullptr) < 0) {
      fprintf(stderr, "Register signal %d failed.", sig_list[i]);
    }
  }

  return 0;
}

#if defined(__aarch64__)
enum {
  REG_R0 = 0,
  REG_R1,
  REG_R2,
  REG_R3,
  REG_R4,
  REG_R5,
  REG_R6,
  REG_R7,
  REG_R8,
  REG_R9,
  REG_R10,
  REG_R11,
  REG_R12,
  REG_R13,
  REG_R14,
  REG_R15,
  REG_R16,
  REG_R17,
  REG_R18,
  REG_R19,
  REG_R20,
  REG_R21,
  REG_R22,
  REG_R23,
  REG_R24,
  REG_R25,
  REG_R26,
  REG_R27,
  REG_R28,
  REG_R29,
  REG_R30
};
#endif

int modelbox_cpu_register_data(char *buf, int buf_size, ucontext_t *ucontext) {
  greg_t *gregs = nullptr;
  if (buf == nullptr || buf_size <= 0 || ucontext == nullptr) {
    return -1;
  }

  int len = -1;
#if defined(__aarch64__)
  gregs = (greg_t *)&(ucontext->uc_mcontext.regs);
  len = snprintf_s(
      buf, buf_size, buf_size - 1,
      "[R0]=0x%.16lx\t\t[R1]=0x%.16lx\t\t[R2]=0x%.16lx\t\t[R3]=0x%.16lx\n"
      "[R4]=0x%.16lx\t\t[R5]=0x%.16lx\t\t[R6]=0x%.16lx\t\t[R7]=0x%.16lx\n"
      "[R8]=0x%.16lx\t\t[R9]=0x%.16lx\t\t[R10]=0x%.16lx\t[R11]=0x%.16lx\n"
      "[R12]=0x%.16lx\t\t[R13]=0x%.16lx\t[R14]=0x%.16lx\t[R15]=0x%.16lx\n"
      "[R16]=0x%.16lx\t\t[R17]=0x%.16lx\t[R18]=0x%.16lx\t[R19]=0x%.16lx\n"
      "[R20]=0x%.16lx\t\t[R21]=0x%.16lx\t[R22]=0x%.16lx\t[R23]=0x%.16lx\n"
      "[R24]=0x%.16lx\t\t[R25]=0x%.16lx\t[R26]=0x%.16lx\t[R27]=0x%.16lx\n"
      "[R28]=0x%.16lx\t\t[R29]=0x%.16lx\t[R30]=0x%.16lx\n"
      "sp: 0x%.16llx\t\tpc: 0x%.16llx\t\tpstate: 0x%.16llx\tfault_address: "
      "0x%.16llx\n",
      *(gregs + REG_R1), *(gregs + REG_R2), *(gregs + REG_R3),
      *(gregs + REG_R4), *(gregs + REG_R5), *(gregs + REG_R6),
      *(gregs + REG_R7), *(gregs + REG_R8), *(gregs + REG_R9),
      *(gregs + REG_R10), *(gregs + REG_R11), *(gregs + REG_R12),
      *(gregs + REG_R13), *(gregs + REG_R14), *(gregs + REG_R15),
      *(gregs + REG_R16), *(gregs + REG_R17), *(gregs + REG_R18),
      *(gregs + REG_R19), *(gregs + REG_R20), *(gregs + REG_R21),
      *(gregs + REG_R22), *(gregs + REG_R23), *(gregs + REG_R24),
      *(gregs + REG_R25), *(gregs + REG_R26), *(gregs + REG_R27),
      *(gregs + REG_R28), *(gregs + REG_R29), *(gregs + REG_R30),
      ucontext->uc_mcontext.sp, ucontext->uc_mcontext.pc,
      ucontext->uc_mcontext.pstate, ucontext->uc_mcontext.fault_address);
#elif defined(__x86_64__)
  gregs = (greg_t *)&(ucontext->uc_mcontext.gregs);
  len = snprintf_s(
      buf, buf_size, buf_size - 1,
      "[R8]=0x%.16lx\t\t[R9]=0x%.16lx\t\t[R10]=0x%.16lx\t[R11]=0x%.16lx\n"
      "[R12]=0x%.16lx\t[R13]=0x%.16lx\t[R14]=0x%.16lx\t[R15]=0x%.16lx\n"
      "[RDI]=0x%.16lx\t[RSI]=0x%.16lx\t[RBP]=0x%.16lx\t[RBX]=0x%.16lx\n"
      "[RDX]=0x%.16lx\t[RAX]=0x%.16lx\t[RCX]=0x%.16lx\t[RSP]=0x%.16lx\n"
      "[RIP]=0x%.16lx\t[RFLAGS]=0x%.16lx\n",
      *(gregs + REG_R8), *(gregs + REG_R9), *(gregs + REG_R10),
      *(gregs + REG_R12), *(gregs + REG_R13), *(gregs + REG_R14),
      *(gregs + REG_R15), *(gregs + REG_RDI), *(gregs + REG_RSI),
      *(gregs + REG_RBP), *(gregs + REG_RBX), *(gregs + REG_RDX),
      *(gregs + REG_RAX), *(gregs + REG_RCX), *(gregs + REG_RSP),
      *(gregs + REG_RIP), *(gregs + REG_EFL));
#endif
  if (len < 0 || len >= buf_size) {
    return -1;
  }

  return 0;
}

Status GetUidGid(const std::string &user, uid_t &uid, gid_t &gid) {
  struct passwd *result = nullptr;
  struct passwd pwd;
  std::vector<char> buff;
  ssize_t bufsize = 0;
  int ret = -1;

  if (user == "") {
    return {STATUS_INVALID, "user is empty"};
  }

  bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (bufsize == -1) {
    bufsize = 1024 * 16;
  }

  buff.reserve(bufsize);
  ret = getpwnam_r(user.c_str(), &pwd, buff.data(), bufsize, &result);
  if (ret != 0) {
    return {STATUS_FAULT, "get user " + user + " failed: " + StrError(errno)};
  }

  if (result == nullptr) {
    return {STATUS_NOTFOUND, "user " + user + " not found"};
  }

  uid = result->pw_uid;
  gid = result->pw_gid;

  return STATUS_OK;
}

Status ChownToUser(const std::string &user, const std::string &path) {
  uid_t uid = 0;
  gid_t gid = 0;
  int unused __attribute__((unused)) = 0;

  auto ret = GetUidGid(user, uid, gid);
  if (ret != STATUS_OK) {
    return ret;
  }

  if (chown(path.c_str(), uid, gid) != 0) {
    return {STATUS_INVALID, "chown " + path + " failed: " + StrError(errno)};
  }

  return STATUS_OK;
}
#include <sys/stat.h>
#include <sys/types.h>
Status RunAsUser(const std::string &user) {
  struct __user_cap_data_struct cap;
  struct __user_cap_header_struct header;
#ifdef _LINUX_CAPABILITY_VERSION_3
  header.version = _LINUX_CAPABILITY_VERSION_3;
#else
  header.version = _LINUX_CAPABILITY_VERSION;
#endif
  header.pid = 0;
  uid_t uid = 0;
  gid_t gid = 0;
  int unused __attribute__((unused)) = 0;

  auto ret = GetUidGid(user, uid, gid);
  if (ret != STATUS_OK) {
    return ret;
  }

  if (getuid() == uid) {
    return STATUS_OK;
  }

  if (capget(&header, &cap) < 0) {
    return {STATUS_INVALID, "capget failed: " + StrError(errno)};
  }

  prctl(PR_SET_KEEPCAPS, 1, 0, 0, 0);
  Defer { prctl(PR_SET_KEEPCAPS, 0, 0, 0, 0); };
  cap.effective = 1 << CAP_NET_ADMIN;
  cap.permitted = 1 << CAP_NET_ADMIN;
  cap.inheritable = 0;
  unused = setgid(gid);
  unused = setuid(uid);
  if (capset(&header, &cap) < 0) {
    if (errno == EPERM) {
      return {STATUS_PERMIT, "capset failed: " + StrError(errno)};
    }
  }

  if (getuid() != uid) {
    return {STATUS_INVALID, "change user failed. " + StrError(errno)};
  }

  return STATUS_OK;
}

Status SplitIPPort(const std::string &host, std::string &ip,
                   std::string &port) {
  auto pos = host.find_last_of(':');

  if (pos == std::string::npos) {
    const auto *msg = "invalid ip address, please try ip:port";
    return {STATUS_INVALID, msg};
  }

  port = host.substr(pos + 1, host.length());
  int n_port = atol(port.c_str());
  if (n_port <= 0 || n_port > 65535) {
    const auto *msg = "invalid port";
    return {STATUS_INVALID, msg};
  }

  ip = host.substr(0, pos);
  /* process ipv6 format */
  pos = ip.find_first_of('[');
  if (pos != std::string::npos) {
    ip = ip.substr(pos + 1, ip.length());
  }

  pos = ip.find_first_of(']');
  if (pos != std::string::npos) {
    ip = ip.substr(0, pos);
  }

  if (ip == "") {
    ip = "0.0.0.0";
  };

  return STATUS_OK;
}

}  // namespace modelbox