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

#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef BUF_LEN_32
#define BUF_LEN_32 32
#endif

int create_pid(char *pid_file) {
  int fd = 0;
  int flags;
  char buff[BUF_LEN_32];

  fd = open(pid_file, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "create pid file failed, %s\n", strerror(errno));
    return -1;
  }

  flags = fcntl(fd, F_GETFD);
  if (flags < 0) {
    fprintf(stderr, "Could not get flags for PID file %s\n", pid_file);
    goto errout;
  }

  flags |= FD_CLOEXEC;
  if (fcntl(fd, F_SETFD, flags) == -1) {
    fprintf(stderr, "Could not set flags for PID file %s\n", pid_file);
    goto errout;
  }

  if (lockf(fd, F_TLOCK, 0) < 0) {
    fprintf(stderr, "Server is already running.\n");
    goto errout;
  }

  snprintf(buff, BUF_LEN_32, "%d\n", getpid());

  if (write(fd, buff, strnlen(buff, BUF_LEN_32)) < 0) {
    fprintf(stderr, "write pid to file failed, %s.\n", strerror(errno));
    goto errout;
  }

  return fd;
errout:
  if (fd > 0) {
    close(fd);
  }
  return -1;
}

pid_t get_pid_from_pidfile(char *pid_file, int *is_locked) {
  int fd = 0;
  char buff[BUF_LEN_32];
  int locked = 0;
  pid_t pid;

  fd = open(pid_file, O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "open pid file failed, %s\n", strerror(errno));
    return -1;
  }

  if (lockf(fd, F_TLOCK, 0) < 0) {
    locked = 1;
  }

  snprintf(buff, BUF_LEN_32, "%d\n", getpid());

  buff[0] = 0;
  if (read(fd, buff, BUF_LEN_32) < 0) {
    fprintf(stderr, "read pid from file failed, %s.\n", strerror(errno));
    goto errout;
  }

  pid = atoi(buff);

  if (is_locked) {
    *is_locked = locked;
  }

  close(fd);
  return pid;
errout:
  if (fd > 0) {
    close(fd);
  }
  return -1;
}

unsigned long get_tick(void) {
  struct timespec ts;
  unsigned theTick = 0U;
  clock_gettime(CLOCK_REALTIME, &ts);
  theTick = ts.tv_nsec / 1000000;
  theTick += ts.tv_sec * 1000;
  return theTick;
}
