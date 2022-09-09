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

#include "manager_monitor.h"

#include <errno.h>
#include <sched.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "common.h"
#include "hashtable.h"
#include "list.h"
#include "manager_conf.h"
#include "modelbox/manager/manager_monitor_client.h"
#include "util.h"

#define MANAGER_MON_MAX_APP_NUM 256
#define SHM_MODE 0600
#define APP_MAX_ARGS 128

static key_t g_msgkey = 0;
static int g_msgid = 0;
static int g_manager_restarting = 0;

#define APP_MAP_BITS 9

typedef enum APP_STATUS {
  APP_STOP,
  APP_NOT_RUNNING,
  APP_PENDING,
  APP_RUNNING,
  APP_DEAD,
  APP_EXITING,
  APP_FORCE_KILL,
} APP_STATUS;

/* app control block */
struct app_monitor {
  char name[APP_NAME_LEN];
  char cmdline[PATH_MAX];
  char killcmd[PATH_MAX];
  char pid_file[PATH_MAX];

  struct hlist_node map;
  pid_t pid;
  pid_t kill_pid;
  time_t last_alive;
  time_t dead_time;
  int check_alive;
  int check_alive_time;
  int heartbeat_interval;
  APP_STATUS state;
};

struct app_mon {
  MANAGER_DECLARE_HASHTABLE(app_map, APP_MAP_BITS);
  MANAGER_DECLARE_HASHTABLE(app_map_pid, APP_MAP_BITS);
};

static struct app_mon app_mon;

#ifdef BUILD_TEST
struct Test_App test_app;
#endif

/* find app control block by name*/
struct app_monitor *_app_find_app_byid(const char *name) {
  struct app_monitor *app;

  unsigned int key = hash_string(name);
  hash_for_each_possible(app_mon.app_map, app, map, key) {
    if (strncmp(app->name, name, APP_NAME_LEN) == 0) {
      return app;
    }
  }

  return NULL;
}

/* find app control block by pid*/
struct app_monitor *_app_find_app_bypid(pid_t pid) {
  struct app_monitor *app;

  if (pid <= 0) {
    return NULL;
  }

  unsigned int key = hash_long(pid, APP_MAP_BITS);
  hash_for_each_possible(app_mon.app_map_pid, app, map, key) {
    if (app->pid == pid) {
      return app;
    }
  }

  return NULL;
}

/* whether pid exits */
int _app_pid_exists(pid_t pid) {
  if (pid <= 0) {
    return -1;
  }

  if (kill(pid, 0) != 0) {
    return -1;
  }

  return 0;
}

/* whether app exits */
int _app_exists(struct app_monitor *app) {
  if (_app_pid_exists(app->pid) != 0) {
    return -1;
  }

  return 0;
}

void _app_do_execve(char *cmdline) {
  char *argv[APP_MAX_ARGS];
  int argc = 0;
  int i = 0;

  argv[argc] = cmdline;
  for (i = 0; i < PATH_MAX - 2 && argc < APP_MAX_ARGS - 1; i++) {
    if (cmdline[i] != '\0') {
      continue;
    }

    argc++;
    argv[argc] = cmdline + i + 1;
    if (cmdline[i + 1] == '\0') {
      break;
    }
  }

  argv[argc] = 0;

  setpgrp();

  execvp(argv[0], argv);
  manager_log(MANAGER_LOG_ERR, "execvp %s failed: %s\n", argv[0],
              strerror(errno));
}

/* run shell command */
void _app_start_exec(struct app_monitor *app) {
  if (app->check_alive) {
    char keyfileenv[PATH_MAX];
    char appnameenv[PATH_MAX];
    char keepalivetime[64];
    char intervaltime[64];

    snprintf_s(keyfileenv, sizeof(keyfileenv), sizeof(keyfileenv),
               "MANAGER_MONITOR_KEYFILE=%s", key_file_path);
    snprintf_s(appnameenv, sizeof(appnameenv), sizeof(appnameenv),
               "MANAGER_MONITOR_NAME=%s", app->name);
    snprintf_s(keepalivetime, sizeof(keepalivetime), sizeof(keepalivetime),
               "MANAGER_MONITOR_KEEPALIVE_TIME=%d", app->check_alive_time);
    snprintf_s(intervaltime, sizeof(intervaltime), sizeof(intervaltime),
               "MANAGER_MONITOR_HEARTBEAT_INTERVAL=%d",
               app->heartbeat_interval);

    putenv(keyfileenv);
    putenv(appnameenv);
    putenv(keepalivetime);
    putenv(intervaltime);

    if (get_modelbox_root_path()[0] != '\0') {
      char modelbox_root[PATH_MAX];
      snprintf_s(modelbox_root, sizeof(modelbox_root), sizeof(modelbox_root),
                 "MODELBOX_ROOT=%s", get_modelbox_root_path());
      putenv(modelbox_root);
    }
  }

  _app_do_execve(app->cmdline);
}

int app_getpid_from_pidfile(struct app_monitor *app) {
  pid_t pid = 0;
  int islocked = 0;
  if (app->pid_file[0] == 0) {
    return -1;
  }

  pid = get_pid_from_pidfile(app->pid_file, &islocked);

  if (pid <= 0 || islocked == 0) {
    return -1;
  }

  if (app->pid <= 0) {
    manager_log(MANAGER_LOG_INFO, "app %s attach monitoring, pid %d", app->name,
                pid);
  } else {
    manager_log(MANAGER_LOG_INFO,
                "app %s attach monitoring new pid %d, old pid %d", app->name,
                pid, app->pid);
  }

  app->state = APP_RUNNING;
  app->pid = pid;
  time(&app->last_alive);

  return 0;
}

int app_getpid_by_ps(struct app_monitor *app) {
  FILE *fp = NULL;
  char cmdline[PATH_MAX * 2];
  char pid_buff[PATH_MAX];
  int pid = -1;

  /* if use pid file, skip*/
  if (app->pid_file[0] != 0) {
    return -1;
  }

  pid_buff[0] = 0;
  snprintf(cmdline, PATH_MAX * 2 - 1, "pgrep -x -f \"%s\"",
           strcmds(app->cmdline, PATH_MAX));
  fp = popen(cmdline, "re");
  if (fp == NULL) {
    manager_log(MANAGER_LOG_ERR, "run command %s failed",
                strcmds(app->cmdline, PATH_MAX));
    goto errout;
  }

  if (fread(pid_buff, 1, sizeof(pid_buff), fp) > 0) {
    pid = atoi(pid_buff);
    if (kill(pid, 0) != 0) {
      pid = -1;
    }
  }

  if (pclose(fp)) {
    fp = NULL;
    goto errout;
  }

  if (pid <= 0) {
    goto errout;
  }

  if (app->pid <= 0) {
    manager_log(MANAGER_LOG_INFO,
                "app %s attach monitoring by finding process, pid %d",
                app->name, pid);
  }

  app->state = APP_RUNNING;
  app->pid = pid;

  return 0;

errout:
  if (fp) {
    pclose(fp);
    fp = NULL;
  }
  return -1;
}

int _app_start(struct app_monitor *app) {
  int pid = -1;
  int unused __attribute__((unused));

  if (app->pid > 0) {
    return 0;
  }

#ifdef BUILD_TEST
  /* for test */
  pid = fork();
  if (pid < 0) {
    manager_log(MANAGER_LOG_ERR, "start process failed, %s", strerror(errno));
    return -1;
  } else if (pid == 0) {
    char keyfileenv[PATH_MAX];
    char appnameenv[PATH_MAX];
    char keepalivetime[64];
    char intervaltime[64];

    snprintf_s(keyfileenv, sizeof(keyfileenv), sizeof(keyfileenv),
               "MANAGER_MONITOR_KEYFILE=%s", key_file_path);
    snprintf_s(appnameenv, sizeof(appnameenv), sizeof(appnameenv),
               "MANAGER_MONITOR_NAME=%s", app->name);
    snprintf_s(keepalivetime, sizeof(keepalivetime), sizeof(keepalivetime),
               "MANAGER_MONITOR_KEEPALIVE_TIME=%d", app->check_alive_time);
    snprintf_s(intervaltime, sizeof(intervaltime), sizeof(intervaltime),
               "MANAGER_MONITOR_HEARTBEAT_INTERVAL=%d",
               app->heartbeat_interval);

    putenv(keyfileenv);
    putenv(appnameenv);
    putenv(keepalivetime);
    putenv(intervaltime);

    if (get_modelbox_root_path()[0] != '\0') {
      char modelbox_root[PATH_MAX];
      snprintf_s(modelbox_root, sizeof(modelbox_root), sizeof(modelbox_root),
                 "MODELBOX_ROOT=%s", get_modelbox_root_path());
      putenv(modelbox_root);
    }

    close_all_fd();
    app_test(app);
    _exit(1);
  }
#else
  pid = vfork();
  if (pid < 0) {
    manager_log(MANAGER_LOG_ERR, "app %s start failed, %s", app->name,
                strerror(errno));
    return -1;
  } else if (pid == 0) {
    /* close all fd */
    close_all_fd();
    /* start process */
    _app_start_exec(app);
    _exit(1);
  }
#endif

  if (app->pid_file[0]) {
    app->state = APP_PENDING;
  } else {
    /* update state */
    app->state = APP_RUNNING;
  }

  app->pid = pid;
  app->dead_time = 0;
  time(&app->last_alive);

  manager_log(MANAGER_LOG_INFO, "app %s start success, pid %d ", app->name,
              app->pid);

  return 0;
}

int _app_run_killcmd(struct app_monitor *app) {
  manager_log(MANAGER_LOG_INFO, "run kill-cmd %s",
              strcmds(app->killcmd, sizeof(app->killcmd)));
  pid_t pid = vfork();
  if (pid < 0) {
    manager_log(MANAGER_LOG_ERR, "app %s kill-cmd failed, %s", app->name,
                strerror(errno));
    return -1;
  } else if (pid == 0) {
    char childpid[PATH_MAX];
    if (get_modelbox_root_path()[0] != '\0') {
      char modelbox_root[PATH_MAX];
      snprintf_s(modelbox_root, sizeof(modelbox_root), sizeof(modelbox_root),
                 "MODELBOX_ROOT=%s", get_modelbox_root_path());
      putenv(modelbox_root);
    }

    snprintf_s(childpid, sizeof(childpid), sizeof(childpid), "APP_PID=%d",
               app->pid);
    putenv(childpid);

    /* close all fd */
    close_all_fd();
    /* start process */
    _app_do_execve(app->killcmd);
    _exit(1);
  }

  app->kill_pid = pid;

  return 0;
}

int _app_stop(struct app_monitor *app, int gracefull) {
  int status;
  int kill_mode = SIGKILL;

  /* process is not running when pid < 0 */
  if (app->pid <= 0) {
    manager_log(MANAGER_LOG_DBG, "app %s not started.", app->name);
    return 0;
  }

  /* check whether process is running */
  if (kill(app->pid, 0) != 0) {
    if (errno == ESRCH) {
      manager_log(MANAGER_LOG_INFO, "app %s not exist, pid %d", app->name,
                  app->pid);
      goto clearout;
    }
  }

  if (gracefull) {
    kill_mode = SIGTERM;
  } else {
    kill_mode = SIGKILL;
  }

  /* send SEGV to process before kill
   * after MANAGER_APP_DEAD_EXIT_TIME second, force kill process。
   */
  if (app->state == APP_DEAD) {
    app->state = APP_EXITING;
    if (app->killcmd[0] != 0 && app->kill_pid <= 0) {
      _app_run_killcmd(app);
      goto out;
    }
    kill_mode = SIGSEGV;
    time(&app->dead_time);
  } else if (app->state == APP_EXITING) {
    time_t now;
    time(&now);
    if (now < (app->dead_time + conf_force_kill_time)) {
      goto out;
    }
    app->state = APP_FORCE_KILL;
  }

  /* force kill process */
  manager_log(MANAGER_LOG_ERR, "app %s send signal %d, pid %d", app->name,
              kill_mode, app->pid);
  if (killpg(app->pid, kill_mode) != 0) {
    if (errno == ESRCH) {
      goto clearout;
    }
    manager_log(MANAGER_LOG_WARN, "app %s kill(%d) failed, pid %d, %s",
                app->name, kill_mode, app->pid, strerror(errno));
    return 0;
  }

  usleep(1000);
out:
  /* wait pid */
  if (waitpid(app->pid, &status, WNOHANG) > 0) {
    /* clear app control block */
    manager_log(MANAGER_LOG_INFO, "app %s stop success, pid %d ", app->name,
                app->pid);
  clearout:
    app->pid = -1;
    app->last_alive = 0;
    app->state = APP_NOT_RUNNING;
    if (app->kill_pid > 0) {
      killpg(app->kill_pid, SIGKILL);
      manager_log(MANAGER_LOG_INFO, "app %s stop kill command: %s, pid %d.",
                  app->name, app->killcmd, app->kill_pid);
      app->kill_pid = -1;
    }
    return 0;
  }

  if (app->state == APP_EXITING) {
    return 0;
  }

  return -1;
}

int app_start(struct app_start_info *start_info) {
  struct app_monitor *app = NULL;
  unsigned int key;

  if (start_info == NULL) {
    manager_log(MANAGER_LOG_ERR, "parameter is invalid");
    return -1;
  }

  if (start_info->name == NULL || start_info->name[0] == '\0') {
    manager_log(MANAGER_LOG_ERR, "app name is invalid");
    return -1;
  }

  if (start_info->cmdline == NULL || start_info->cmdline[0] == '\0' ||
      start_info->cmd_max_len > MAX_CMDLINE_LEN - 2 ||
      start_info->cmd_max_len <= 0) {
    manager_log(MANAGER_LOG_ERR, "app cmdline is invalid");
    return -1;
  }

  app = _app_find_app_byid(start_info->name);
  if (app) {
    manager_log(MANAGER_LOG_WARN, "app %s exists", start_info->name);
    return -1;
  }

  app = malloc(sizeof(struct app_monitor));
  if (app == NULL) {
    manager_log(MANAGER_LOG_ERR, "malloc for app_monitor failed.");
    goto errout;
  }
  memset_s(app, sizeof(struct app_monitor), 0, sizeof(struct app_monitor));

  strncpy(app->name, start_info->name, APP_NAME_LEN - 1);
  copycmds(app->cmdline, sizeof(app->cmdline), start_info->cmdline,
           start_info->cmd_max_len);
  if (start_info->killcmd && start_info->killcmd_max_len > 0 &&
      start_info->killcmd_max_len <= MAX_CMDLINE_LEN - 2) {
    copycmds(app->killcmd, sizeof(app->killcmd), start_info->killcmd,
             start_info->killcmd_max_len);
  } else {
    app->killcmd[0] = '\0';
  }

  if (start_info->pidfile) {
    strncpy(app->pid_file, start_info->pidfile, PATH_MAX - 1);
  } else {
    app->pid_file[0] = 0;
  }

  app->pid = -1;
  app->state = APP_NOT_RUNNING;
  app->last_alive = 0;
  app->dead_time = 0;
  app->check_alive = (start_info->check_alive) ? 1 : 0;
  app->check_alive_time = start_info->keepalive_time;
  app->heartbeat_interval = start_info->heartbeat_interval;

  if (g_manager_restarting) {
    /* if process exists, attach process*/
    if (app_getpid_from_pidfile(app) != 0 && app->pid_file[0] != 0) {
      time(&app->last_alive);
      app->last_alive -= app->check_alive_time + 5;
      app->state = APP_PENDING;
      manager_log(MANAGER_LOG_INFO, "start app %s, pending ", start_info->name);
    } else if (app_getpid_by_ps(app) != 0 && app->pid_file[0] == 0) {
      manager_log(MANAGER_LOG_INFO,
                  "try start app %s, may cause duplicate process ",
                  start_info->name);
    }
  } else {
    manager_log(MANAGER_LOG_INFO, "start app %s", start_info->name);
  }

  key = hash_string(start_info->name);
  hash_add(app_mon.app_map, &app->map, key);

  return 0;

errout:
  if (app) {
    free(app);
  }
  return -1;
}

int app_stop(const char *name, int gracefull) {
  /* find process and stop */
  struct app_monitor *app = _app_find_app_byid(name);
  if (app == NULL) {
    manager_log(MANAGER_LOG_ERR, "app %s is not found.", name);
    return -1;
  }

  manager_log(MANAGER_LOG_INFO, "stop app %s, pid %d", app->name, app->pid);

  /* stop process*/
  if (gracefull == 0) {
    app->state = APP_FORCE_KILL;
  }

  if (_app_stop(app, gracefull) != 0) {
    manager_log(MANAGER_LOG_WARN, "stop app failed.");
  }

  hash_del(&app->map);
  free(app);

  return 0;
}

/* is process alive? */
int app_alive(const char *name) {
  struct app_monitor *app = _app_find_app_byid(name);
  if (app == NULL) {
    return -1;
  }

  if (app->pid > 0) {
    return 0;
  }

  /* process is running? */
  if (kill(app->pid, 0) == 0) {
    return 0;
  }

  return -1;
}

/* 获取进程pid */
int app_getpid(const char *name) {
  struct app_monitor *app = _app_find_app_byid(name);
  if (app == NULL) {
    return -1;
  }

  return app->pid;
}

int _app_stopall(void) {
  struct app_monitor *app;
  struct hlist_node *tmp;
  int bucket;

  /* 循环停止所有进程 */
  manager_log(MANAGER_LOG_INFO, "stop all apps.");
  hash_for_each_safe(app_mon.app_map, bucket, tmp, app, map) {
    app_stop(app->name, 1);
  }

  return 0;
}

/* 重启进程 */
int _app_restart(struct app_monitor *app) {
  /* 强制停止进程 */
  if (_app_stop(app, 0) != 0) {
    manager_log(MANAGER_LOG_ERR, "app stop failed.");
  }

  /* 重启进程 */
  if (_app_start(app) != 0) {
    manager_log(MANAGER_LOG_ERR, "app start failed.");
    return -1;
  }

  return 0;
}

int _app_waitchild(void) {
  int status;
  int ret = 0;
  do {
    ret = waitpid(-1, &status, WNOHANG);
  } while (ret > 0);

  return 0;
}

/* process heartbeat message*/
int _app_heartbeat_process(struct heartbeat_msg *msg) {
  struct app_monitor *app = NULL;

  /* find app control block by name */
  app = _app_find_app_byid(msg->name);
  if (app == NULL) {
    manager_log(MANAGER_LOG_WARN, "app not found, name %s", msg->name);
    return 0;
  }

  if (app->state == APP_PENDING) {
    if (app->pid <= 0) {
      /* 如果master初始化过程，则更新pid信息 */
      app->pid = msg->pid;
      app->last_alive = msg->time;
      app->state = APP_RUNNING;
      manager_log(MANAGER_LOG_INFO, "re-monitoring app %s, pid %d", app->name,
                  app->pid);
      return 0;
    } else {
      /* process running as daemon */
      if (app_getpid_from_pidfile(app) == 0) {
        manager_log(MANAGER_LOG_INFO, "app %s, run as child deamon, new pid %d",
                    app->name, app->pid);
        return 0;
      }
    }
  }

  /* if pid not match, output err message*/
  if (app->pid != msg->pid) {
    if (_app_exists(app) != 0) {
      manager_log(MANAGER_LOG_ERR, "app %s, pid is not match %d:%d", app->name,
                  app->pid, msg->pid);
      return -1;
    }
    /* if pid not exists, update infomation*/
    manager_log(MANAGER_LOG_ERR,
                "app %s pid is not exists, and not match %d:%d", app->name,
                app->pid, msg->pid);
    app->pid = msg->pid;
  }

  /* if heart message stalls, skip */
  if (app->last_alive > msg->time) {
    manager_log(MANAGER_LOG_DBG, "msg is stall.");
    return -1;
  }

  /* update alive time */
  app->last_alive = msg->time;

  return 0;
}

/* create message queue */
int _app_mon_create_msg(void) {
  unsigned int ipc_mode = 0600;
  const char *key_file;

#ifdef BUILD_TEST
  key_file = "/proc/self/exe";
#else
  key_file = key_file_path;
#endif
  g_msgkey = ftok(key_file, 1);
  if (g_msgkey < 0) {
    manager_log(MANAGER_LOG_ERR, "get key failed.");
    goto errout;
  }

  g_msgid = msgget(g_msgkey, ipc_mode | IPC_EXCL | IPC_CREAT);
  if (g_msgid < 0) {
    if (errno != EEXIST) {
      manager_log(MANAGER_LOG_ERR, "create msg key failed, %s",
                  strerror(errno));
      goto errout;
    }

    g_msgid = msgget(g_msgkey, ipc_mode);
    if (g_msgid < 0) {
      manager_log(MANAGER_LOG_ERR, "attatch msg key failed, %s",
                  strerror(errno));
      goto errout;
    }

    g_manager_restarting = 1;
  }

  manager_log(MANAGER_LOG_INFO, "key file %s, key is 0x%x, id is %u", key_file,
              g_msgkey, g_msgid);

  return 0;
errout:
  if (g_msgid > 0) {
    msgctl(g_msgid, IPC_RMID, 0);
    g_msgid = -1;
  }
  return -1;
}

int _recv_heartbeat(void) {
  struct heartbeat_msg msg;
  int ret;

  for (;;) {
    /* recv message  */
    ret = msgrcv(g_msgid, &msg, sizeof(msg), HEARTBEAT_MSG, IPC_NOWAIT);
    if (ret != sizeof(msg)) {
      if (errno == ENOMSG) {
        break;
      }

      /* if queue not exists, recreate message queue */
      if (errno == EIDRM || errno == EINVAL) {
        manager_log(MANAGER_LOG_ERR, "key %u not exists, recreate.", g_msgid);
        _app_mon_create_msg();
      }

      manager_log(MANAGER_LOG_INFO, "recv msg failed, len = %d:%ld, %s", ret,
                  sizeof(msg), strerror(errno));
      break;
    }

    manager_log(MANAGER_LOG_DBG,
                "heartbeat msg, name = %s, time = %lu, pid = %d", msg.name,
                msg.time, msg.pid);

    /* process */
    ret = _app_heartbeat_process(&msg);
    if (ret != 0) {
      manager_log(MANAGER_LOG_DBG, "process heart beat message failed");
    }
  }

  return 0;
}

int _app_alive(struct app_monitor *app) {
  time_t now;
  int time_out = app->check_alive_time;

  if (app->pid > 0) {
    if (kill(app->pid, 0) != 0 && app->state != APP_PENDING) {
      manager_log(MANAGER_LOG_ERR, "app %s exited, pid %d", app->name,
                  app->pid);
      app->state = APP_NOT_RUNNING;
      return -1;
    }
  } else {
    return -1;
  }

  time(&now);

  if (app->state == APP_PENDING) {
    time_out = 10;
  }

  /* if no check, return */
  if (app->check_alive == 0 && app->state == APP_RUNNING) {
    return 0;
  }

  if (app->last_alive > (now - time_out)) {
    return 0;
  }

  char buffer[64];
  struct tm tm_last;
  localtime_r(&app->last_alive, &tm_last);
  strftime(buffer, sizeof(buffer), "%x %X", &tm_last);
  app->state = APP_DEAD;
  app->dead_time = now;
  manager_log(MANAGER_LOG_ERR, "app %s dead, last %s(%lu:%lu) pid %d.",
              app->name, buffer, app->last_alive, now, app->pid);
  return -1;
}

/* check process running status */
int _app_state_check(struct app_monitor *app) {
  if (app->state != APP_RUNNING && app->state != APP_PENDING) {
    return -1;
  }

  /* if pending, skip */
  if (app->state == APP_PENDING) {
    if (app_getpid_from_pidfile(app) == 0) {
      return 0;
    }
  }

  if (_app_alive(app) == 0) {
    return 0;
  }

  return -1;
}

int _app_timeout_check(void) {
  struct app_monitor *app;
  int bucket = 0;

  hash_for_each(app_mon.app_map, bucket, app, map) {
    if (_app_state_check(app) == 0) {
      continue;
    }

    if (_app_restart(app) != 0) {
      manager_log(MANAGER_LOG_ERR, "restart app %s failed", app->name);
    }
  }

  return 0;
}

int app_monitor(void) {
  /* recv heartbeat message */
  _recv_heartbeat();

  /* check timeout */
  _app_timeout_check();

  /* wait child process */
  _app_waitchild();

  if (unlikely(g_manager_restarting)) {
    g_manager_restarting = 0;
  }

  return 0;
}

/* delete message queue */
int _app_mon_destroy_msg(void) {
  if (g_msgid > 0) {
    msgctl(g_msgid, IPC_RMID, 0);
    g_msgid = -1;
  }

  return 0;
}

int manager_monitor_init(void) {
  hash_init(app_mon.app_map);
  hash_init(app_mon.app_map_pid);
  g_manager_restarting = 0;

  if (_app_mon_create_msg() != 0) {
    manager_log(MANAGER_LOG_ERR, "create monitor msg failed.");
    return -1;
  }

  if (g_manager_restarting) {
    manager_log(MANAGER_LOG_ERR, "master restart, try to recive all messages.");
  }

  return 0;
}

int manager_monitor_exit(void) {
  _app_stopall();

  _app_mon_destroy_msg();

  return 0;
}

#ifdef BUILD_TEST
/* test child process */
void app_test(struct app_monitor *app) {
  int count = 0;
  int pidfile = -1;

  signal(SIGINT, SIG_DFL);
  signal(SIGTERM, SIG_DFL);

  if (app->pid_file[0] != 0) {
    pidfile = create_pid(app->pid_file);
  }

  if (app_monitor_init(app->name, NULL) != 0) {
    goto out;
  }

  while (true) {
    /* report heart beat */
    if (app_monitor_heartbeat() != 0) {
      printf("send heartbeat message faild.");
      break;
    }

    /* callback function */
    if (test_app.run) {
      if (test_app.run(&test_app, count, app->name) != 0) {
        break;
      }
    } else if (count > 10) {
      break;
    }

    count++;
    sleep(1);
  }

out:
  if (pidfile > 0) {
    close(pidfile);
    pidfile = 0;
  }
  printf("app %s exit, pid %d", app->name, getpid());
  return;
}

void app_free_memory(void) {
  struct app_monitor *app;
  struct hlist_node *tmp;
  int bucket;

  /* stop all process */
  manager_log(MANAGER_LOG_INFO, "force free all apps.");
  hash_for_each_safe(app_mon.app_map, bucket, tmp, app, map) {
    hash_del(&app->map);
    free(app);
  }

  return;
}

#endif