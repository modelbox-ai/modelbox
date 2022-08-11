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

#include "manager.h"

#include <libgen.h>
#include <sys/mman.h>

#include "common.h"
#include "manager_conf.h"
#include "manager_monitor.h"
#include "securec.h"
#include "tlog.h"
#include "util.h"

static int g_reload_config;
static int pid_fd;
static int key_fd;
extern char *program_invocation_name;
extern char *program_invocation_short_name;
static int exit_signal;
static int g_run_server = 0;
static int g_is_verbose = 0;

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};

static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);

static void manager_showhelp(void) {
  /* clang-format off */
    char help[] = ""
        "Usage: manager [OPTION]...\n"
        "Start manager server.\n"
        "  -c            configuration file.\n"
        "  -f            run forground.\n"
        "  -p            pid file.\n"
        "  -v            output log to screen.\n"
        "  -h            show this help message.\n"
        "\n";
    printf("%s", help);
  /* clang-format on */
}

static void manager_sig_handler(int volatile sig_no, siginfo_t *sig_info,
                                void *volatile ptr) {
  switch (sig_no) {
    case SIGINT:
    case SIGTERM:
      exit_signal = sig_no;
      exit(1);
      break;
    case SIGQUIT:
      return;
      break;
    case SIGSEGV:
    case SIGPIPE:
    case SIGFPE:
    case SIGABRT:
    case SIGBUS:
    case SIGILL:
      sleep(1);
      break;
    default:
      break;
  }

  _exit(1);
}

static int manager_sig_register(void) {
  int i = 0;
  struct sigaction sig_act;

  for (i = 0; i < g_sig_num; i++) {
    memset_s(&sig_act, sizeof(sig_act), 0, sizeof(sig_act));
    sig_act.sa_sigaction = manager_sig_handler;
    sig_act.sa_flags = SA_SIGINFO | SA_RESTART;

    if (sigaction(g_sig_list[i], &sig_act, NULL) < 0) {
      fprintf(stderr, "Register signal %d failed.", g_sig_list[i]);
    }
  }

  return 0;
}

int manager_add_apps(void) {
  int i = 0;
  for (i = 0; i < conf_apps_num; i++) {
    manager_log(MANAGER_LOG_INFO, "add %s, %s, %s, check alive %d\n", conf_apps[i].name,
                conf_apps[i].cmd, conf_apps[i].pidfile, conf_apps[i].check_alive);
    if (app_start(conf_apps[i].name, conf_apps[i].cmd, PATH_MAX, conf_apps[i].pidfile,
                  conf_apps[i].check_alive, conf_apps[i].check_alive_time,
                  conf_apps[i].heartbeat_interval) != 0) {
      manager_log(MANAGER_LOG_ERR, "add app %s failed.", conf_apps[i].name);
      return -1;
    }
  }

  return 0;
}

void manager_reload_apps(struct conf_app oldapps[CONF_MAX_APPS]) {
  /* stop app */
  int i, j;
  manager_log(MANAGER_LOG_INFO, "start reload apps.");

  for (i = 0; i < CONF_MAX_APPS; i++) {
    for (j = 0; j < CONF_MAX_APPS; j++) {
      if (memcmp(&oldapps[i], &conf_apps[j], sizeof(struct conf_app)) == 0) {
        break;
      }
    }

    if (j == CONF_MAX_APPS) {
      /* stop app, when config not exists */
      if (app_stop(oldapps[i].name, 1) == 0) {
        manager_log(MANAGER_LOG_INFO, "stop app %s success.", oldapps[i].name);
      } else {
        manager_log(MANAGER_LOG_ERR, "stop app %s failed.", oldapps[i].name);
      }

      memset_s(&oldapps[i], sizeof(struct conf_app), 0,
               sizeof(struct conf_app));
    }
  }

  /* start new apps */
  for (i = 0; i < CONF_MAX_APPS; i++) {
    for (j = 0; j < CONF_MAX_APPS; j++) {
      if (memcmp(&conf_apps[i], &oldapps[j], sizeof(struct conf_app)) == 0) {
        break;
      }
    }

    if (j == CONF_MAX_APPS) {
      if (app_start(conf_apps[i].name, conf_apps[i].cmd, PATH_MAX, conf_apps[i].pidfile,
                    conf_apps[i].check_alive, conf_apps[i].check_alive_time,
                    conf_apps[i].heartbeat_interval) == 0) {
        manager_log(MANAGER_LOG_INFO, "start app %s success.",
                    conf_apps[i].name);
      } else {
        manager_log(MANAGER_LOG_ERR, "start app %s failed.", conf_apps[i].name);
      }
    }
  }
}

int manager_reload(void) {
  struct conf_app oldapps[CONF_MAX_APPS];
  memcpy(oldapps, conf_apps, sizeof(oldapps));

  manager_log(MANAGER_LOG_INFO, "start reload configuration.");
  if (manager_reload_conf() != 0) {
    manager_log(MANAGER_LOG_ERR, "reload configuration failed.\n");
    return -1;
  }

  tlog_setlevel(conf_log_level);

  manager_reload_apps(oldapps);

  return 0;
}

int manager_init_server(void) {
  /* init monitor*/
  if (manager_monitor_init() != 0) {
    manager_log(MANAGER_LOG_ERR, "init monitor failed.\n");
    return -1;
  }

  /* load apps */
  if (manager_add_apps() != 0) {
    manager_log(MANAGER_LOG_ERR, "add apps failed.");
    return -1;
  }

  if (mlockall(MCL_FUTURE) != 0) {
    manager_log(MANAGER_LOG_WARN, "lock memory failed.");
  }
  return 0;
}

int manager_run(void) {
  unsigned long now = {0};
#ifdef BUILD_TEST
  int sleep = 10;
#else
  int sleep = 500;
#endif
  int sleep_time = 0;
  unsigned long expect_time = 0;

  g_run_server = 1;

  sleep_time = sleep;
  now = get_tick_count() - sleep;
  expect_time = now + sleep;
  while (g_run_server) {
    now = get_tick_count();
    if (now >= expect_time) {
      sleep_time = sleep - (now - expect_time);
      if (sleep_time < 0) {
        sleep_time = 0;
      }
      expect_time += sleep;
    }

    if (g_reload_config) {
      manager_reload();
      g_reload_config = 0;
    }

    usleep(sleep_time * 1000);
    app_monitor();
  }

  manager_log(MANAGER_LOG_INFO, "master run stopped.");
  return 0;
}

void manager_stop(void) { g_run_server = 0; }

void manager_exit(void) {
  manager_stop();

  /* stop monitor */
  manager_monitor_exit();
}

static void manager_onexit(void) {
  if (exit_signal > 0) {
    manager_log(MANAGER_LOG_INFO, "process exit with signal %d", exit_signal);
  }

  manager_exit();

  tlog_exit();
}

static void manager_sighup(int signo) { g_reload_config = 1; }

int manager_tlog(MANAGER_LOG_LEVEL level, const char *file, int line,
                 const char *func, void *userptr, const char *format,
                 va_list ap) {
  return tlog_vext((tlog_level)(level), file, line, func, userptr, format, ap);
}

static void _manager_default_conf_file(char *path, int max_len) {
  char current_path[1024] = {0};
  if (get_prog_path(current_path, sizeof(current_path) - 1) != 0) {
    path[0] = 0;
    return;
  }

  snprintf(path, max_len, "%s/../%s/%s", current_path, CONF_FILE_PATH,
           CONF_FILE_NAME);
  if (access(path, R_OK) == 0) {
    return;
  }

  snprintf(path, max_len, "%s/%s", CONF_FILE_PATH, CONF_FILE_NAME);
  if (access(path, R_OK) == 0) {
    return;
  }

  path[0] = 0;

  return;
}

int manager_init(const char *conf_file, char *name) {
  char log_file[PATH_MAX];
  char default_conf_file[PATH_MAX] = {0};
  char piddir[PATH_MAX];

  if (manager_sig_register()) {
    fprintf(stderr, "register signal failed\n");
    return -1;
  }

  mkdir(MANAGER_PID_PATH, 0750);

  manager_log_callback_reg(manager_tlog);

  if (strnlen(pid_file_path, sizeof(pid_file_path)) <= 0) {
    if (name) {
      snprintf_s(pid_file_path, sizeof(pid_file_path), sizeof(pid_file_path),
                 "%s/%s.pid", MANAGER_PID_PATH, name);
    } else {
      snprintf_s(pid_file_path, sizeof(pid_file_path), sizeof(pid_file_path),
                 "%s/%s.pid", MANAGER_PID_PATH, MANAGER_NAME);
    }
  }

  strncpy_s(piddir, sizeof(piddir), pid_file_path, sizeof(pid_file_path));
  dirname(piddir);

  /* create key */
  if (strnlen(key_file_path, sizeof(key_file_path)) <= 0) {
    if (name) {
      snprintf_s(key_file_path, sizeof(key_file_path), sizeof(key_file_path),
                 "%s/%s.key", piddir, name);
    } else {
      snprintf_s(key_file_path, sizeof(key_file_path), sizeof(key_file_path),
                 "%s/%s.key", piddir, MANAGER_NAME);
    }
  }

  pid_fd = create_pid(pid_file_path);
  if (pid_fd <= 0) {
    fprintf(stderr, "create pid failed, failed\n");
    return -1;
  }

  key_fd = create_pid(key_file_path);
  if (key_fd <= 0) {
    fprintf(stderr, "create key file failed, failed\n");
    return -1;
  }

  /* generate log */
  snprintf(conf_log_file, sizeof(log_file), "%s/%s.log", MANAGER_LOG_PATH,
           MANAGER_NAME);

  if (conf_file == NULL) {
    _manager_default_conf_file(default_conf_file, PATH_MAX);
    if (default_conf_file[0]) {
      conf_file = default_conf_file;
    }
  }

  if (manager_load_conf(conf_file) != 0) {
    fprintf(stderr, "load master config file %s failed.\n", conf_file);
    return -1;
  }

  if (tlog_init(get_modelbox_full_path(conf_log_file), conf_log_size,
                conf_log_num, 0, 0) != 0) {
    fprintf(stderr, "init master log failed.\n");
    return -1;
  }

  tlog_setlogscreen(g_is_verbose);
  tlog_setlevel(conf_log_level);

  manager_log(MANAGER_LOG_INFO, "%s starting... (Build : %s %s)",
              program_invocation_short_name, __DATE__, __TIME__);

  if (manager_init_server() != 0) {
    manager_log(MANAGER_LOG_ERR, "init master server failed.");
    return -1;
  }

  return 0;
}

#ifdef BUILD_TEST
int manager_main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
  int is_forground = 0;
  int opt;
  char conf_file[PATH_MAX] = {0};
  char *name = NULL;

  while ((opt = getopt(argc, argv, "fvhc:n:p:k:")) != -1) {
    switch (opt) {
      case 'c':
        strncpy(conf_file, get_modelbox_full_path(optarg),
                sizeof(conf_file) - 1);
        break;
      case 'n':
        name = optarg;
        break;
      case 'p':
        strncpy(pid_file_path, get_modelbox_full_path(optarg),
                sizeof(pid_file_path) - 1);
        break;
      case 'k':
        strncpy(key_file_path, get_modelbox_full_path(optarg),
                sizeof(key_file_path) - 1);
        break;
      case 'f':
        is_forground = 1;
        break;
      case 'v':
        g_is_verbose = 1;
        break;
      case 'h':
        manager_showhelp();
        return 1;
    }
  }

  if (is_forground == 0) {
    if (daemon(0, 0) < 0) {
      fprintf(stderr, "run daemon process failed, %s\n", strerror(errno));
      return 1;
    }
  }

  atexit(manager_onexit);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGHUP, manager_sighup);
  g_reload_config = 0;

  if (manager_init(conf_file, name) != 0) {
    fprintf(stderr, "master init failed.\n");
    return 1;
  }

  if (manager_run() != 0) {
    return 1;
  }

  return 0;
}

#ifdef BUILD_TEST

void manager_force_exit(void) {
  manager_stop();
  usleep(100000);
  app_free_memory();
}

#endif
