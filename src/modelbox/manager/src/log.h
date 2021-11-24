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


#ifndef MANAGER_LOG_H
#define MANAGER_LOG_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

#ifndef MANAGER_LOG_HAS_LEVEL
#define MANAGER_LOG_HAS_LEVEL
typedef enum {
  MANAGER_LOG_DBG = 0,
  MANAGER_LOG_INFO = 1,
  MANAGER_LOG_NOTE = 2,
  MANAGER_LOG_WARN = 3,
  MANAGER_LOG_ERR = 4,
  MANAGER_LOG_FATAL = 5,
  MANAGER_LOG_END = 6
} MANAGER_LOG_LEVEL;
#endif

#ifndef BASE_FILE_NAME
#define BASE_FILE_NAME __FILE__
#endif
#define manager_log(level, format, ...)                                    \
  manager_log_ext(level, BASE_FILE_NAME, __LINE__, __func__, NULL, format, \
                  ##__VA_ARGS__)

extern int manager_log_ext(MANAGER_LOG_LEVEL level, const char *file, int line,
                           const char *func, void *userptr, const char *format,
                           ...) __attribute__((format(printf, 6, 7)));

extern int manager_log_vext(MANAGER_LOG_LEVEL level, const char *file, int line,
                            const char *func, void *userptr, const char *format,
                            va_list ap);

typedef int (*manager_log_callback)(MANAGER_LOG_LEVEL level, const char *file,
                                    int line, const char *func, void *userptr,
                                    const char *format, va_list ap);

extern void manager_log_callback_reg(manager_log_callback callback);

void manager_backtrace(MANAGER_LOG_LEVEL loglevel, const char *format, ...);

#ifdef __cplusplus
}
#endif /*__cplusplus */
#endif
