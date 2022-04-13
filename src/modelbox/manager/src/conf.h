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


#ifndef MANAGER_CONF_H
#define MANAGER_CONF_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

#include <getopt.h>
#include <limits.h>
#include <unistd.h>

#include "log.h"

#define CONF_LOG_LEVEL "loglevel"
#define CONF_LOG_NUM "lognum"
#define CONF_LOG_SIZE "logsize"
#define CONF_LOG_FILE "logfile"

struct CONF_PARSE_INT {
  int *value;
  int min;
  int max;
};
extern int conf_parse_string(void *item, int argc, char *argv[]);

struct CONF_PARSE_STRING {
  char *value;
  int max;
};
extern int conf_parse_int(void *item, int argc, char *argv[]);

struct CONF_PARSE_SIZE {
  size_t *value;
  size_t min;
  size_t max;
};
extern int conf_parse_size(void *item, int argc, char *argv[]);

extern MANAGER_LOG_LEVEL conf_log_level;
extern int conf_log_num;
extern size_t conf_log_size;
extern char conf_log_file[PATH_MAX];

typedef int (*parse_func)(void *item, int argc, char *argv[]);
struct config_map {
  const char *config_name;
  parse_func parse_func;
  void *item;
};

extern int load_conf(struct config_map config_map[], const char *conf_file);

#ifdef __cplusplus
}
#endif  /*__cplusplus */
#endif  // !MANAGER_CONF_H
