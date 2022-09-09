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


#include "log.h"

#include <execinfo.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "tlog.h"

static manager_log_callback log_func = NULL;

int manager_log_ext(MANAGER_LOG_LEVEL level, const char *file, int line,
                   const char *func, void *userptr, const char *format, ...) {
  int len = 0;
  va_list ap;

  if (log_func == NULL) {
    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
    printf("\n");
    return 0;
  }

  va_start(ap, format);
  len = log_func(level, file, line, func, userptr, format, ap);
  va_end(ap);

  return len;
}

int manager_log_vext(MANAGER_LOG_LEVEL level, const char *file, int line,
                    const char *func, void *userptr, const char *format,
                    va_list ap) {
  int len = 0;
  if (log_func == NULL) {
    return 0;
  }

  len = log_func(level, file, line, func, userptr, format, ap);

  return len;
}

void manager_log_callback_reg(manager_log_callback callback) {
  log_func = callback;
}

void manager_backtrace(MANAGER_LOG_LEVEL loglevel, const char *format, ...) {
  int j, nptrs;
#define SIZE 100
  void *buffer[100];
  char stack_buffer[4096];
  char *buff = stack_buffer;
  char **strings;
  int total_len = 0;
  int len = 0;
  va_list ap;

  nptrs = backtrace(buffer, SIZE);

  /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
            would produce similar output to the following: */

  strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    return;
  }

  va_start(ap, format);
  len =
      vsnprintf(buff + total_len, sizeof(stack_buffer) - total_len, format, ap);
  total_len += len;
  if (*(buff + len) != '\n') {
    len = snprintf(buff + total_len, sizeof(stack_buffer) - total_len, "\n");
    total_len += len;
  }
  va_end(ap);
  for (j = 0; j < nptrs; j++) {
    len = snprintf(buff + total_len, sizeof(stack_buffer) - total_len,
                   "    @ %s\n", strings[j]);
    if (len >= sizeof(stack_buffer) - total_len) {
      break;
    }
    total_len += len;
  }

  manager_log_ext(loglevel, BASE_FILE_NAME, __LINE__, __func__, 0, "%s",
                 stack_buffer);

  free(strings);
}