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

#include "modelbox/manager/manager_monitor_client.h"

#include "manager_common.h"
#include "manager_conf.h"
#include "manager_monitor.h"

char g_key_file[PATH_MAX];

struct app_struct {
  char name[APP_NAME_LEN];
  pid_t pid;
  time_t time;
};

struct app_struct app_info = {.name[0] = 0};
static key_t g_msgkey = -1;
static int g_msgid = -1;

extern void app_log_reg(manager_log_callback callback);

static void _app_monitor_reset_msgkey(void) {
  g_msgkey = -1;
  g_msgid = -1;
}

int app_monitor_init(const char *name) {
  if (name == NULL) {
    return -1;
  }

  strncpy(app_info.name, name, APP_NAME_LEN);
  app_info.pid = getpid();
  _app_monitor_reset_msgkey();
  if (strlen(g_key_file) <= 0) {
    snprintf(g_key_file, PATH_MAX, "%s/%s.key", MANAGER_PID_PATH, MANAGER_NAME);
  }

  return 0;
}

int app_monitor_keyfile(char *file) {
  if (file == NULL) {
    return -1;
  }

  strncpy(g_key_file, file, PATH_MAX);
  _app_monitor_reset_msgkey();

  return 0;
}

int app_attach_msg_queue(void) {
  const char *key_file = NULL;
#ifdef BUILD_TEST
  key_file = "/proc/self/exe";
#else
  key_file = g_key_file;
#endif

  if (g_msgkey <= 0) {
    if (key_file[0] == 0) {
      return -1;
    }

    g_msgkey = ftok(key_file, 1);
    if (g_msgkey < 0) {
      if (errno == ENOENT) {
        return 0;
      }
      return 0;
    }
  }

  if (g_msgid < 0) {
    g_msgid = msgget(g_msgkey, 0600);
    if (g_msgid < 0) {
      return -1;
    }
  }

  return 0;
}

int app_monitor_heartbeat(void) {
  struct heartbeat_msg msgs[2];
  struct heartbeat_msg *msg = &msgs[0];
  int ret;
  time_t now;

  if (app_info.name[0] == 0) {
    return 0;
  }

  time(&now);
  if (now == app_info.time) {
    return 0;
  }

  app_info.time = now;

  if (app_attach_msg_queue() != 0) {
    return -1;
  }

  if (g_msgid < 0) {
    return -1;
  }

  memset(&msgs, 0, sizeof(msgs));
  msg->pid = app_info.pid;
  msg->mtype = HEARTBEAT_MSG;
  strncpy(msg->name, app_info.name, APP_NAME_LEN);
  msg->time = app_info.time;

  ret = msgsnd(g_msgid, msg, sizeof(*msg), IPC_NOWAIT);
  if (ret != 0) {
    if (errno == EIDRM) {
      _app_monitor_reset_msgkey();
    }
  }

  return ret;
}