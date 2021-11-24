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


#include "util.h"

#include <dirent.h>
#include <libgen.h>
#include <sys/time.h>
#include <time.h>

#include "common.h"

unsigned long get_tick_count(void) {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void close_all_fd(void) {
  char path_name[PATH_MAX];
  DIR *dir = NULL;
  struct dirent *ent;
  int dir_fd = -1;

  snprintf(path_name, sizeof(path_name), "/proc/self/fd/");
  dir = opendir(path_name);
  if (dir == NULL) {
    fprintf(stderr, "open directory failed, %s\n", strerror(errno));
    goto errout;
  }

  dir_fd = dirfd(dir);

  while ((ent = readdir(dir)) != NULL) {
    int fd = atoi(ent->d_name);
    if (fd < 0 || dir_fd == fd) {
      continue;
    }
    switch (fd) {
      case STDIN_FILENO:
      case STDOUT_FILENO:
      case STDERR_FILENO:
        continue;
        break;
      default:
        break;
    }

    close(fd);
  }

  closedir(dir);

  return;
errout:
  if (dir) {
    closedir(dir);
  }
  return;
}

int get_prog_path(char *path, int max_len) {
  /* 读取进程的二进制路径 */
  int len = readlink("/proc/self/exe", path, max_len - 1);
  if (len < 0) {
    return -1;
  }

  path[len] = 0;

  dirname(path);

  return 0;
}
