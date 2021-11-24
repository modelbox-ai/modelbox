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
#include "modelbox/base/utils.h"
#include "securec.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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
    sig_act.sa_flags = SA_SIGINFO | SA_RESTART;

    if (sigaction(sig_list[i], &sig_act, NULL) < 0) {
      fprintf(stderr, "Register signal %d failed.", sig_list[i]);
    }
  }

  return 0;
}

}  // namespace modelbox