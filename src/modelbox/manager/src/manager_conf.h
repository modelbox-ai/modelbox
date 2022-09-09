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

#ifndef MODELBOX_MANAGER_CONF_H
#define MODELBOX_MANAGER_CONF_H

#include "conf.h"
#include "manager_common.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

#define DEFAULT_WATCHDOG_TIMEOUT 90
#define DEFAULT_HEARTBEAT_INTERVAL 30
#define DEFAULT_FORCE_KILL_TIMEOUT 6
#define CONF_MAX_APPS 256

#define CONF_FILE_NAME "manager.conf"
#define CONF_FILE_PATH "/etc"

#define CONF_WATCHDOG_TIMEOUT "watchdog-timeout"
#define CONF_FORCE_KILLTIME "force-kill-time"
#define CONF_KEY_FILE "key-file"
#define CONF_LOCK_PAGE "lock-page"

#define CONF_APP "app"

extern char pid_file_path[PATH_MAX];
extern char key_file_path[PATH_MAX];

extern int conf_watchdog_timeout;

extern int conf_force_kill_time;

extern int conf_lockpage;

struct conf_app {
  char name[APP_NAME_LEN];
  char pidfile[PATH_MAX];
  char cmd[PATH_MAX];
  char killcmd[PATH_MAX];
  int check_alive;
  int check_alive_time;
  int heartbeat_interval;
};

extern struct conf_app conf_apps[CONF_MAX_APPS];

extern int conf_apps_num;

extern int manager_reload_conf(void);

extern int manager_load_conf(const char *conf_file);

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif  // !MODELBOX_MANAGER_CONF_H