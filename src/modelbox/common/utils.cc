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
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "modelbox/base/utils.h"
#include "securec.h"

namespace modelbox {

#define TMP_BUFF_LEN_32 32

static int kPidFileFd = -1;

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
            strerror(errno));
    return -1;
  }

  flags = fcntl(fd, F_GETFD);
  if (flags < 0) {
    fprintf(stderr, "Could not get flags for PID file %s, %s\n", pid_file,
            strerror(errno));
    goto errout;
  }

  flags |= FD_CLOEXEC;
  if (fcntl(fd, F_SETFD, flags) == -1) {
    fprintf(stderr, "Could not set flags for PID file %s, %s\n", pid_file,
            strerror(errno));
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
    fprintf(stderr, "write pid to file failed, %s.\n", strerror(errno));
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
    fd = -1;
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

    if (sigaction(sig_list[i], &sig_act, NULL) < 0) {
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

}  // namespace modelbox