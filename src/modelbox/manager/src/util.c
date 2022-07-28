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

const char *get_modelbox_root_path(void) {
  static char root_path[PATH_MAX] = {0};
  static int is_init = false;

  if (is_init) {
    return root_path;
  }

  is_init = 1;
  char prog_path[PATH_MAX] = {0};
  char path_tmp[PATH_MAX];

  if (get_prog_path(prog_path, PATH_MAX - 1) != 0) {
    return root_path;
  }

  snprintf_s(path_tmp, PATH_MAX - 1, PATH_MAX - 1, "%s/../../../", prog_path);
  if (realpath(path_tmp, root_path) == NULL) {
    return root_path;
  }

  if (root_path[0] == '/' && root_path[1] == '\0') {
    root_path[0] = '\0';
  }

  return root_path;
}

static char *string_replace(const char *in, char *out, int out_max,
                            const char *from, const char *to) {
  char *needle;
  size_t from_len = strnlen(from, PATH_MAX);
  size_t to_len = strnlen(to, PATH_MAX);
  size_t resoffset = 0;
  int ret = 0;

  while ((needle = strstr(in, from)) && out_max - resoffset > 0) {
    ret = memcpy_s(out + resoffset, out_max - resoffset, in, needle - in);
    if (ret != 0) {
      return NULL;
    }
    resoffset += needle - in;

    in = needle + from_len;
    ret = strncpy_s(out + resoffset, out_max - resoffset, to, to_len);
    if (ret != 0) {
      return NULL;
    }

    resoffset += to_len;
  }

  if (out_max - resoffset <= 0) {
    return NULL;
  }

  ret = strncpy_s(out + resoffset, out_max - resoffset, in, out_max);
  if (ret != 0) {
    return NULL;
  }

  return out;
}

const char *get_modelbox_full_path(const char *path) {
  const char *root_path = get_modelbox_root_path();
  static char full_path[PATH_MAX * 4] = {0};
  full_path[0] = '\0';
  if (string_replace(path, full_path, PATH_MAX * 4, "${MODELBOX_ROOT}",
                     root_path) == NULL) {
    return NULL;
  }

  return full_path;
}
