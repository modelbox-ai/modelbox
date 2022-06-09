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

#include "manager_conf.h"
#include "util.h"
#include <string.h>

#include "conf.h"
#include "log.h"

static char g_conf_file[PATH_MAX];
char pid_file_path[PATH_MAX];
char key_file_path[PATH_MAX];
int conf_watchdog_timeout = DEFAULT_WATCHDOG_TIMEOUT;
int conf_force_kill_time = DEFAULT_FORCE_KILL_TIMEOUT;

struct conf_app conf_apps[CONF_MAX_APPS];
int conf_apps_num;

int _manager_is_app_exists(char *name) {
  int i = 0;

  for (i = 0; i < CONF_MAX_APPS; i++) {
    struct conf_app *conf_app = &conf_apps[i];
    if (conf_app->name[0] == 0) {
      continue;
    }

    if (strncmp(conf_app->name, name, APP_NAME_LEN) == 0) {
      return 0;
    }
  }

  return -1;
}

int manager_load_app(void *item, int argc, char *argv[]) {
  static struct option options[] = {{"name", 1, 0, 'n'},
                                    {"pidfile", 1, 0, 'p'},
                                    {"check-alive", 0, 0, 'k'},
                                    {0, 0, 0, 0}};

  int cmdtype;
  struct conf_app *conf_app;
  int end_opt = 0;
  int i = 0;

  if (conf_apps_num >= CONF_MAX_APPS) {
    manager_log(MANAGER_LOG_ERR, "apps configuration is full.");
    return 0;
  }

  conf_app = conf_apps + conf_apps_num;
  memset(conf_app, 0, sizeof(*conf_app));

  while ((cmdtype = getopt_long_only(argc, argv, "", options, NULL)) != -1 && end_opt == 0) {
    switch (cmdtype) {
      case 'n':
        if (_manager_is_app_exists(optarg) == 0) {
          manager_log(MANAGER_LOG_ERR, "app %s exists.", optarg);
          return -1;
        }
        strncpy(conf_app->name, optarg, APP_NAME_LEN);
        break;
      case 'k':
        conf_app->check_alive = 1;
        break;
      case 'p':
        strncpy(conf_app->pidfile, get_modelbox_full_path(optarg), PATH_MAX);
        break;
      default:
        break;
    }
  }

  for (i = optind; i < argc; i++) {
    strncat(conf_app->cmd, " ", PATH_MAX);
    strncat(conf_app->cmd, get_modelbox_full_path(argv[i]), PATH_MAX);
  }

  if (strlen(conf_app->name) <= 0 || strlen(conf_app->cmd) <= 0) {
    manager_log(MANAGER_LOG_ERR, "load app failed, name %s, cmd %s.",
                conf_app->name, conf_app->cmd);
    return -1;
  }

  conf_apps_num++;
  return 0;
}

static struct config_map conf_parse_map[] = {
    {CONF_WATCHDOG_TIMEOUT, conf_parse_int,
     .item =
         &(struct CONF_PARSE_INT){
             .value = &conf_watchdog_timeout, .min = 3, .max = 60 * 5}},
    {CONF_FORCE_KILLTIME, conf_parse_int,
     .item =
         &(struct CONF_PARSE_INT){
             .value = &conf_force_kill_time, .min = 3, .max = 60 * 3}},
    {CONF_APP, manager_load_app, 0},
    {CONF_KEY_FILE, conf_parse_string,
     .item =
         &(struct CONF_PARSE_STRING){.value = key_file_path, .max = PATH_MAX}},
    {NULL, NULL, NULL},
};

int manager_reload_conf(void) {
  manager_log(MANAGER_LOG_INFO, "reload configuration file %s", g_conf_file);

  conf_apps_num = 0;
  memset(conf_apps, 0, sizeof(conf_apps));

  return load_conf(conf_parse_map, g_conf_file);
}

int manager_load_conf(const char *conf_file) {
  if (conf_file == NULL) {
    manager_log(MANAGER_LOG_ERR, "conf file is null");
    return -1;
  }

  strncpy(g_conf_file, get_modelbox_full_path(conf_file), sizeof(g_conf_file));

  return load_conf(conf_parse_map, g_conf_file);
}
