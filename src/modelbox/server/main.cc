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

#include <getopt.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <memory>
#include <thread>

#include "config.h"
#include "modelbox/base/utils.h"
#include "modelbox/common/command.h"
#include "modelbox/common/log.h"
#include "modelbox/common/utils.h"
#include "modelbox/manager/manager_monitor_client.h"
#include "modelbox/server/timer.h"
#include "modelbox/server/utils.h"
#include "securec.h"
#include "server.h"

#define MODELBOX_SERVER_LOG_PATH "/var/log/modelbox/modelbox.log"
#define MODELBOX_SERVER_PID_FILE "/var/run/modelbox.pid"
#define MODELBOX_MAX_INIT_TIME (60 * 12)

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};

static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);
static bool kVerbose = false;
static bool kForground = false;

static void showhelp() {
  /* clang-format off */
    char help[] = ""
        "Usage: modelbox [OPTION]...\n"
        "Start modelbox server.\n"
        "  -c            configuration file.\n"
        "  -f            run forground.\n"
        "  -p            pid file.\n"
        "  -u [user]     run as user.\n"
        "  -V            output log to screen.\n"
        "  -v            show server version.\n"
        "  -h            show this help message.\n"
        "\n";

    printf("%s", help);
  /* clang-format on */
}

void modelbox_stop() { modelbox::kServerTimer->Stop(); }

static void modelbox_sig_handler(int volatile sig_no, siginfo_t *sig_info,
                                 void *volatile ptr) {
  switch (sig_no) {
    case SIGINT:
    case SIGTERM:
      modelbox_stop();
      return;
      break;
    case SIGQUIT:
      return;
      break;
    case SIGSEGV:
    case SIGPIPE:
    case SIGFPE:
    case SIGABRT:
    case SIGBUS:
    case SIGILL: {
      char buf[4096];
      MBLOG_ERROR << "Segment fault"
                  << ", Signal: " << sig_no << ", Addr: " << sig_info->si_addr
                  << ", Code: " << sig_info->si_code << ", Caused by: ";
      if (modelbox::modelbox_cpu_register_data(buf, sizeof(buf),
                                               (ucontext_t *)ptr) == 0) {
        MBLOG_ERROR << "CPU Register Info:\n" << buf;
      }
      MBLOG_STACKTRACE(modelbox::LOG_FATAL);
      sleep(1);
    } break;
    default:
      break;
  }

  _exit(1);
}

static int modelbox_reg_signal() {
  if (modelbox::modelbox_sig_register(g_sig_list, g_sig_num,
                                      modelbox_sig_handler) != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  return 0;
}

int modelbox_init_log(std::vector<std::string> &path_list) {
  std::shared_ptr<modelbox::ModelboxServerLogger> logger =
      std::make_shared<modelbox::ModelboxServerLogger>();

  auto log_size = modelbox::GetBytesFromReadable(
      modelbox::kConfig->GetString("log.size", "64MB"));
  auto log_num = modelbox::kConfig->GetUint32("log.num", 64);
  auto log_path =
      modelbox::kConfig->GetString("log.path", MODELBOX_SERVER_LOG_PATH);
  auto log_screen = modelbox::kConfig->GetBool("log.screen", false);
  auto log_level = modelbox::kConfig->GetString("log.level", "INFO");
  if (log_screen) {
    kVerbose = true;
  }

  log_path = modelbox::modelbox_full_path(log_path);

  if (logger->Init(log_path, log_size, log_num, kVerbose) == false) {
    fprintf(stderr, "init logger failed.\n");
    return 1;
  }

  ModelBoxLogger.SetLogger(logger);
  logger->SetLogLevel(modelbox::LogLevelStrToLevel(log_level));

  modelbox::CreateDirectory(modelbox::GetDirName(log_path), 0750);
  path_list.push_back(modelbox::GetDirName(log_path));
  path_list.push_back(log_path);

  return 0;
}

int modelbox_init(std::vector<std::string> &path_list) {
  if (modelbox_reg_signal() != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  if (modelbox_init_log(path_list)) {
    return 1;
  }

  /* if in standalone mode */
  if (modelbox::modelbox_root_dir().length() > 0) {
    std::string default_scanpath =
        modelbox::modelbox_full_path(std::string(modelbox::MODELBOX_ROOT_VAR) +
                                     MODELBOX_DEFAULT_DRIVER_PATH);
    modelbox::Drivers::SetDefaultScanPath(default_scanpath);

    std::string default_driver_info_path =
        modelbox::modelbox_full_path(std::string(modelbox::MODELBOX_ROOT_VAR) +
                                     "/var/run/modelbox-driver-info");
    modelbox::Drivers::SetDefaultInfoPath(default_driver_info_path);
  }

  return 0;
}

void modelbox_hung_check(const std::shared_ptr<modelbox::Server> &server) {
  int is_status_ok = 1;
  auto root = modelbox::Statistics::GetGlobalItem();

  auto flowitem = root->GetItem("flow");
  if (flowitem == nullptr) {
    app_monitor_heartbeat();
    return;
  }

  auto flownames = flowitem->GetItemNames();
  for (auto const &name : flownames) {
    auto schedule_item = flowitem->GetItem(name + ".scheduler.status");
    if (schedule_item == nullptr) {
      continue;
    }
    std::string schedule_status;
    schedule_item->GetValue(schedule_status);
    if (schedule_status != "blocking") {
      continue;
    }
    MBLOG_WARN << "flow " << name << " is blocking";

    is_status_ok = 0;
  }

  if (server->Check() != modelbox::STATUS_OK) {
    is_status_ok = 0;
  }

  if (is_status_ok == 0) {
    return;
  }

  app_monitor_heartbeat();
}

int modelbox_run(const std::shared_ptr<modelbox::Server> &server) {
  int retval = 0;
  auto server_init_timer = std::make_shared<modelbox::TimerTask>([]() {
    MBLOG_INFO << "server init timeout, you may change the init timeout "
                  "value by setting the init_timeout in modelbox.conf";
    modelbox::kServerTimer->Stop();
    modelbox::Abort("server init timeout");
  });

  auto future =
      std::async(std::launch::async, [server, &retval, &server_init_timer]() {
        modelbox::Status ret;
        Defer {
          if (!ret) {
            modelbox::kServerTimer->Stop();
            retval = 1;
          }
        };

        ret = server->Init();
        if (!ret) {
          MBLOG_ERROR << "server init failed !";
          return 1;
        }

        ret = server->Start();
        if (!ret) {
          MBLOG_ERROR << "server start failed !";
          return 1;
        }

        server_init_timer->Stop();
        server_init_timer = nullptr;

        return 0;
      });

  if (app_monitor_init(nullptr, nullptr) == 0) {
    MBLOG_INFO << "start manager heartbeat";
    std::shared_ptr<modelbox::TimerTask> heart_beattask =
        std::make_shared<modelbox::TimerTask>();
    heart_beattask->Callback(modelbox_hung_check, server);
    modelbox::kServerTimer->Schedule(
        heart_beattask, 0, 1000 * app_monitor_heartbeat_interval(), true);
  }

  auto init_timeout = modelbox::kConfig->GetUint32("server.init_timeout",
                                                   MODELBOX_MAX_INIT_TIME);
  modelbox::kServerTimer->Schedule(server_init_timer, 1000 * init_timeout, 0,
                                   false);

  // run timer loop.
  modelbox::kServerTimer->Run();

  future.get();
  server->Stop();
  return retval;
}

static void onexit() {}

int modelbox_change_user(const std::string &user_from_cmd,
                         const std::vector<std::string> &path_list) {
  /* run as user */
  auto user = modelbox::kConfig->GetString("server.user", "");
  if (user_from_cmd.length() > 0) {
    user = user_from_cmd;
  }

  if (user.length() == 0) {
    return 0;
  }

  for (const auto &path : path_list) {
    modelbox::ChownToUser(user, path);
  }

  auto ret = modelbox::RunAsUser(user);
  if (ret != modelbox::STATUS_OK) {
    fprintf(stderr, "run as user %s failed, %s\n", user.c_str(),
            ret.WrapErrormsgs().c_str());
    return 1;
  }

  MBLOG_INFO << "run as user: " << user;
  return 0;
}

#ifdef BUILD_TEST
int modelbox_server_main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
  std::string pidfile = MODELBOX_SERVER_PID_FILE;
  int cmdtype = 0;
  std::vector<std::string> path_list;
  std::string user_from_cmd;

  MODELBOX_COMMAND_GETOPT_SHORT_BEGIN(cmdtype, "hc:Vvfp:n:k:Ku:", nullptr)
  switch (cmdtype) {
    case 'p':
      pidfile = modelbox::modelbox_full_path(optarg);
      break;
    case 'V':
      kVerbose = true;
      break;
    case 'f':
      kForground = true;
      break;
    case 'h':
      showhelp();
      return 1;
    case 'c':
      modelbox::kConfigPath = modelbox::modelbox_full_path(optarg);
      break;
    case 'v':
      printf("modelbox-server %s\n", modelbox::GetModelBoxVersion());
      return 0;
    case 'u':
      user_from_cmd = optarg;
      break;
    default:
      printf("Try %s -h for more information.\n", argv[0]);
      return 1;
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  if (modelbox::LoadConfig(modelbox::kConfigPath) == false) {
    fprintf(stderr, "can not load configuration : %s \n",
            modelbox::kConfigPath.c_str());
    return 1;
  }

  auto log_screen = modelbox::kConfig->GetBool("log.screen", kVerbose);
  if (kForground == false && kVerbose == false) {
    if (daemon(0, log_screen) < 0) {
      fprintf(stderr, "run daemon process failed, %s\n",
              modelbox::StrError(errno).c_str());
      return 1;
    }
  }

  Defer { onexit(); };
  /* 忽略SIGPIPE，避免发送缓冲区慢导致的进程退出 */
  signal(SIGPIPE, SIG_IGN);

  if (pidfile != "-") {
    modelbox::CreateDirectory(modelbox::GetDirName(pidfile), 0755);
    path_list.push_back(modelbox::GetDirName(pidfile));
    if (modelbox::modelbox_create_pid(pidfile.c_str()) != 0) {
      fprintf(stderr, "create pid file failed.\n");
      return 1;
    }
    path_list.push_back(pidfile);
  }

  if (modelbox_init(path_list) != 0) {
    fprintf(stderr, "init failed.\n");
    return 1;
  }

  if (modelbox_change_user(user_from_cmd, path_list) != 0) {
    fprintf(stderr, "change user failed.\n");
    return 1;
  }

  MBLOG_INFO << "modelbox config path : " << modelbox::kConfigPath;
  auto server = std::make_shared<modelbox::Server>(modelbox::kConfig);
  modelbox::kServerTimer->Start();

  if (modelbox_run(server) != 0) {
    return 1;
  }

  MBLOG_INFO << "exit modelbox process";
  return 0;
}
