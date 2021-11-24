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


#include "conf.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONF_LINE_MAX 512
#define DEFAULT_LOG_NUM 48
#define DEFAULT_LOG_SIZE (1024 * 1024 * 64)
#define BUF_LEN_64 64
#define MAX_ARGV_NUM 128

MANAGER_LOG_LEVEL conf_log_level = MANAGER_LOG_INFO;
int conf_log_num = DEFAULT_LOG_NUM;
size_t conf_log_size = DEFAULT_LOG_SIZE;

int conf_parse_int(void *item, int argc, char *argv[]) {
  if (argc < 0) {
    return -1;
  }

  char *num = argv[1];
  struct CONF_PARSE_INT *conf_int = item;
  int num_int = atoi(num);

  if (num_int > conf_int->max) {
    *(conf_int->value) = conf_int->max;
  } else if (num_int < conf_int->min) {
    *(conf_int->value) = conf_int->min;
  } else {
    *(conf_int->value) = num_int;
  }

  return 0;
}

int conf_parse_string(void *item, int argc, char *argv[]) {
  if (argc < 0) {
    return -1;
  }

  char *str = argv[1];
  struct CONF_PARSE_STRING *conf_string = item;

  strncpy(conf_string->value, str, conf_string->max);

  return 0;
}

int conf_parse_size(void *item, int argc, char *argv[]) {
  size_t size_num;
  struct CONF_PARSE_SIZE *conf_size = item;

  if (argc < 0) {
    return -1;
  }

  char *size = argv[1];

  if (strstr(size, "k") || strstr(size, "K")) {
    size_num = atoi(size) * 1024;
  } else if (strstr(size, "m") || strstr(size, "M")) {
    size_num = atoi(size) * 1024 * 1024;
  } else if (strstr(size, "g") || strstr(size, "G")) {
    size_num = atoi(size) * 1024 * 1024 * 1024;
  } else {
    size_num = atoi(size);
  }

  if (size_num > conf_size->max) {
    *(conf_size->value) = conf_size->max;
  } else if (size_num < conf_size->min) {
    *(conf_size->value) = conf_size->min;
  } else {
    *(conf_size->value) = size_num;
  }

  return 0;
}

int conf_parse_loglevel(void *item, int argc, char *argv[]) {
  if (argc < 0) {
    return -1;
  }

  char *log_level = argv[1];

  if ((strncmp("DEBUG", log_level, sizeof("DEBUG")) == 0) ||
      (strncmp("debug", log_level, sizeof("debug")) == 0)) {
    conf_log_level = MANAGER_LOG_DBG;
  } else if ((strncmp("INFO", log_level, sizeof("INFO")) == 0) ||
             (strncmp("info", log_level, sizeof("info")) == 0)) {
    conf_log_level = MANAGER_LOG_INFO;
  } else if ((strncmp("NOTICE", log_level, sizeof("NOTICE")) == 0) ||
             (strncmp("notice", log_level, sizeof("notice")) == 0)) {
    conf_log_level = MANAGER_LOG_NOTE;
  } else if ((strncmp("WARN", log_level, sizeof("WARN")) == 0) ||
             (strncmp("warn", log_level, sizeof("warn")) == 0)) {
    conf_log_level = MANAGER_LOG_WARN;
  } else if ((strncmp("ERROR", log_level, sizeof("ERROR")) == 0) ||
             (strncmp("error", log_level, sizeof("error")) == 0)) {
    conf_log_level = MANAGER_LOG_ERR;
  } else if ((strncmp("FATAL", log_level, sizeof("FATAL")) == 0) ||
             (strncmp("fatal", log_level, sizeof("fatal")) == 0)) {
    conf_log_level = MANAGER_LOG_FATAL;
  } else {
    conf_log_level = MANAGER_LOG_INFO;
  }

  return 0;
}

static int _parse_args(char *key, char *value, int max_argv, int *argc,
                       char *argv[]) {
  int num = 0;
  int quotation = 0;
  int is_in_args = 0;
  char end_char = ' ';
  int escape = 0;

  if (value == NULL) {
    return -1;
  }

  argv[0] = key;
  num++;

  do {
    /* 如果字符是转义，则跳过转义字符 */
    if (escape) {
      escape = 0;
      value++;
      continue;
    }
    /* 如果是转义字符，则设置转义标志 */
    if (*value == '\\') {
      if (escape == 0) {
        escape = 1;

        /* 将后续字符前移一位 */
        char *tmp = value + 1;
        for (; *tmp; tmp++) {
          *(tmp - 1) = *tmp;
        }
        *(tmp - 1) = 0;
        continue;
      }
    }
    /* 如果是引号，则引号中的数据解释为参数 */
    if (*value == '"' || *value == '\'') {
      if (quotation == 0) {
        quotation = 1;
        /* 设置解析结束标志字符 */
        end_char = *value;
        value++;
        continue;
      }
    }

    /* 如果遇到结束字符 */
    if (*value == end_char) {
      /* 如果未处理任何参数，则跳过 */
      if (is_in_args == 0) {
        if (quotation == 0) {
          value++;
          continue;
        }

        /*如果是""号，则设置值为'\0'*/
        argv[num] = value;
      }

      /* 将对应最后标志位置0 */
      *value = '\0';
      /* 还原标志字符为空格 */
      end_char = ' ';
      /* 参数解析结束，参数个数加1 */
      is_in_args = 0;
      num++;
      if (num >= max_argv - 1) {
        break;
      }
      quotation = 0;
      value++;
      continue;
    }

    /* 参数开始，初始化argv */
    if (is_in_args == 0) {
      argv[num] = value;
      is_in_args = 1;
    }

    value++;
  } while (*value != '\0');

  /* 处理最后一个参数 */
  if (is_in_args == 1) {
    num++;
  }

  if (num <= 1) {
    return -1;
  }

  *argc = num;
  argv[num] = 0;

  return 0;
}

/* getopts函数不支持不同args重入，getopt_data.__nextchar是上次的结果，
 * 此函数将上次处理结果还原，通过这种方法重置getopts
 */
static void _reset_getopts(void) {
  static struct option options[] = {{"-", 0, 0, 0}, {0, 0, 0, 0}};
  static int argc = 2;
  static char *argv[2];

  argv[0] = "reset";
  argv[1] = "";

  optind = 1;
  opterr = 0;
  getopt_long_only(argc, argv, "", options, NULL);
}

static int parse_conf(struct config_map config_map[], char *key, char *value) {
  int i = 0;
  int argc = 0;
  char *argv[MAX_ARGV_NUM];
  int old_optind;
  int old_opterr;
  int ret = 0;

  for (i = 0;
       config_map[i].parse_func != NULL && config_map[i].config_name != NULL;
       i++) {
    if (strncmp(config_map[i].config_name, key, CONF_LINE_MAX) != 0) {
      continue;
    }

    argc = 0;
    memset(argv, 0, sizeof(argv));
    if (_parse_args(key, value, MAX_ARGV_NUM, &argc, argv) != 0) {
      manager_log(MANAGER_LOG_ERR, "parse config line failed.");
      return -1;
    }

    /* 重置getopts，并调用处理函数 */
    old_optind = optind;
    old_opterr = opterr;
    _reset_getopts();
    optind = 1;
    opterr = 1;
    ret = config_map[i].parse_func(config_map[i].item, argc, argv);
    _reset_getopts();
    optind = old_optind;
    opterr = old_opterr;

    if (ret != 0) {
      return -1;
    }

    return 0;
  }

  return 0;
}

static int load_conf_from_file(struct config_map config_map[],
                               const char *conf_file) {
  char file_line[CONF_LINE_MAX];
  char conf_key[BUF_LEN_64];
  char conf_value[CONF_LINE_MAX];
  int filed_num = 0;
  int line_no = 0;
  FILE *fp;

  if (conf_file == NULL) {
    manager_log(MANAGER_LOG_ERR, "conf file is invalid.");
    return -1;
  }

  fp = fopen(conf_file, "r");
  if (fp == NULL) {
    manager_log(MANAGER_LOG_ERR, "open %s failed, %s", conf_file,
               strerror(errno));
    return -1;
  }

  while (fgets(file_line, sizeof(file_line), fp) != NULL) {
    line_no++;
    filed_num = sscanf(file_line, "%63s %1023[^\n]s", conf_key, conf_value);
    if (filed_num <= 0) {
      continue;
    }

    if (conf_key[0] == '#') {
      continue;
    }

    if (filed_num != 2) {
      goto errout;
    }

    if (parse_conf(config_map, conf_key, conf_value) != 0) {
      goto errout;
    }
  }

  fclose(fp);

  return 0;

errout:
  if (fp) {
    if (line_no > 0) {
      manager_log(MANAGER_LOG_ERR, "invalid config at line %s:%d %s", conf_file,
                 line_no, file_line);
    }
    fclose(fp);
  }
  return -1;
}

static struct config_map common_config_map[] = {
    {CONF_LOG_LEVEL, conf_parse_loglevel, NULL},
    {CONF_LOG_NUM, conf_parse_int,
     .item = &(
         struct CONF_PARSE_INT){.value = &conf_log_num, .min = 1, .max = 512}},
    {CONF_LOG_SIZE, conf_parse_size,
     .item = &(struct CONF_PARSE_SIZE){.value = &conf_log_size,
                                       .min = 1024 * 1024,
                                       .max = 1024 * 1024 * 1024}},
    {NULL, NULL, NULL},
};

int load_conf(struct config_map config_map[], const char *conf_file) {
  if (conf_file == NULL) {
    manager_log(MANAGER_LOG_ERR, "conf file is null");
    return -1;
  }

  if (config_map == NULL) {
    manager_log(MANAGER_LOG_END, "config_map is null");
    return -1;
  }

  /* 加载公共部分配置 */
  if (load_conf_from_file(common_config_map, conf_file) != 0) {
    manager_log(MANAGER_LOG_END, "load common config failed");
    return -1;
  }

  return load_conf_from_file(config_map, conf_file);
}
