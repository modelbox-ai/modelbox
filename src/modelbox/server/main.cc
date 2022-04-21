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

using namespace modelbox;

#define MODELBOX_SERVER_LOG_PATH "/var/log/modelbox/modelbox.log"
#define MODELBOX_SERVER_PID_FILE "/var/run/modelbox.pid"

extern char *program_invocation_name;
extern char *program_invocation_short_name;

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};

static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);
static bool kVerbose = false;
static bool kForground = false;

enum MODELBOX_SERVER_ARG {
  MODELBOX_SERVER_ARG_CHECKPORT,
  MODELBOX_SERVER_ARG_GETCONF,
};

static int option_flag = 0;
static struct option options[] = {
    /* internal command for develop mode */
    {"check-port", 1, &option_flag, MODELBOX_SERVER_ARG_CHECKPORT},
    {"get-conf-value", 1, &option_flag, MODELBOX_SERVER_ARG_GETCONF},
    {0, 0, 0, 0},
};

static void showhelp(void) {
  /* clang-format off */
    char help[] = ""
        "Usage: modelbox [OPTION]...\n"
        "Start modelbox server.\n"
        "  -c            configuration file.\n"
        "  -f            run forground.\n"
        "  -p            pid file.\n"
        "  -V            output log to screen.\n"
        "  -v            show server version.\n"
        "  -n            service name for manager.\n"
        "  -k            keep alive time for manager.\n"
        "  -K            keep alive key file.\n"
        "  -h            show this help message.\n"
        "\n";

    printf("%s", help);
  /* clang-format on */
}

void modelbox_stop() { kServerTimer->Stop(); }

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

static int modelbox_reg_signal(void) {
  if (modelbox_sig_register(g_sig_list, g_sig_num, modelbox_sig_handler) != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  return 0;
}

int modelbox_init_log(void) {
  std::shared_ptr<ModelboxServerLogger> logger =
      std::make_shared<ModelboxServerLogger>();

  auto log_size =
      modelbox::GetBytesFromReadable(kConfig->GetString("log.size", "64MB"));
  auto log_num = kConfig->GetUint32("log.num", 64);
  auto log_path = kConfig->GetString("log.path", MODELBOX_SERVER_LOG_PATH);
  auto log_screen = kConfig->GetBool("log.screen", false);
  auto log_level = kConfig->GetString("log.level", "INFO");
  if (log_screen) {
    kVerbose = true;
  }

  if (logger->Init(log_path, log_size, log_num, kVerbose) == false) {
    fprintf(stderr, "init logger failed.\n");
    return 1;
  }

  ModelBoxLogger.SetLogger(logger);
  logger->SetLogLevel(modelbox::LogLevelStrToLevel(log_level));

  return 0;
}

int modelbox_init(void) {
  if (modelbox_reg_signal() != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  if (LoadConfig(kConfigPath) == false) {
    fprintf(stderr, "can not load configuration : %s \n", kConfigPath.c_str());
    return 1;
  }

  if (modelbox_init_log()) {
    return 1;
  }

  return 0;
}

int modelbox_run(std::shared_ptr<Server> server, const std::string &keep_name,
                 int keep_time, const char *keyfile) {
  auto ret = server->Init();
  if (!ret) {
    MBLOG_ERROR << "server init failed !";
    return 1;
  }

  ret = server->Start();
  if (!ret) {
    MBLOG_ERROR << "server start failed !";
    return 1;
  }

  std::shared_ptr<TimerTask> heart_beattask = std::make_shared<TimerTask>();
  heart_beattask->Callback([&]() { app_monitor_heartbeat(); });

  auto future = std::async(
      std::launch::async, [heart_beattask, keep_name, keep_time, keyfile]() {
        if (keep_time > 0 && keep_name.length() > 0) {
          if (app_monitor_init(keep_name.c_str(), keyfile) != 0) {
            MBLOG_ERROR << "init app monitor failed.";
            return;
          }

          sleep(1);
          kServerTimer->Schedule(heart_beattask, 0, 1000 * keep_time, true);
        }
      });

  // run timer loop.
  kServerTimer->Run();

  server->Stop();
  return 0;
}

int GetConfig(const std::string key) {
  if (LoadConfig(kConfigPath) == false) {
    fprintf(stderr, "can not load configuration : %s \n", kConfigPath.c_str());
    return 1;
  }

  auto values = kConfig->GetStrings(key);
  if (values.size() <= 0) {
    fprintf(stderr, "Not found key %s\n", key.c_str());
    return 1;
  }

  for (auto value : values) {
    std::cout << value << std::endl;
  }

  return 0;
}

int CheckPort(const std::string host) {
  struct addrinfo hints;
  struct addrinfo *result = NULL;

  memset_s(&hints, sizeof(hints), 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;

  std::string ip;
  std::string port;

  auto ret_val = SplitIPPort(host, ip, port);
  if (!ret_val) {
    std::cerr << ret_val.Errormsg() << std::endl;
    return 1;
  }

  auto ret = getaddrinfo(ip.c_str(), port.c_str(), &hints, &result);
  if (ret != 0) {
    std::cerr << "check port failed, " << gai_strerror(ret) << std::endl;
    return 1;
  }

  Defer { freeaddrinfo(result); };

  int sock = socket(result->ai_family, SOCK_STREAM, 0);
  if (sock < 0) {
    std::cerr << "create socket failed\n";
    return 1;
  }
  Defer { close(sock); };

  if (bind(sock, result->ai_addr, result->ai_addrlen) != 0) {
    if (errno == EADDRINUSE) {
      /* in use */
      return 2;
    } else if (errno == EACCES) {
      /* no permission */
      return 3;
    }

    std::cerr << "check failed, errno is " << errno << "\n";
    return 1;
  }

  return 0;
}

static void onexit(void) {}

#ifdef BUILD_TEST
int modelbox_server_main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
  std::string pidfile = MODELBOX_SERVER_PID_FILE;
  std::string keep_name = "";
  int keep_time = 5;
  int cmdtype = 0;
  std::string key_file;
  std::string get_conf_key;

  MODELBOX_COMMAND_GETOPT_SHORT_BEGIN(cmdtype, "hc:Vvfp:n:k:K", options)
  switch (cmdtype) {
    case 0: {
      switch (option_flag) {
        case MODELBOX_SERVER_ARG_CHECKPORT:
          return CheckPort(optarg);
        case MODELBOX_SERVER_ARG_GETCONF:
          get_conf_key = optarg;
          break;
        default:
          printf("Try %s -h for more information.\n", argv[0]);
          return 1;
          break;
      }
      case 'p':
        pidfile = optarg;
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
        kConfigPath = optarg;
        break;
      case 'n':
        keep_name = optarg;
        break;
      case 'k':
        keep_time = atoi(optarg);
        break;
      case 'K':
        key_file = optarg;
        break;
      case 'v':
        printf("modelbox-server %s\n", modelbox::GetModelBoxVersion());
        return 0;
      default:
        printf("Try %s -h for more information.\n", argv[0]);
        return 1;
        break;
    }
  }
  MODELBOX_COMMAND_GETOPT_END()

  if (get_conf_key.length()) {
    return GetConfig(get_conf_key);
  }

  if (kForground == false) {
    if (daemon(0, 0) < 0) {
      fprintf(stderr, "run daemon process failed, %s\n",
              modelbox::StrError(errno).c_str());
      return 1;
    }
  }

  Defer { onexit(); };
  /* 忽略SIGPIPE，避免发送缓冲区慢导致的进程退出 */
  signal(SIGPIPE, SIG_IGN);

  if (modelbox_create_pid(pidfile.c_str()) != 0) {
    fprintf(stderr, "create pid file failed.\n");
    return 1;
  }

  if (modelbox_init() != 0) {
    fprintf(stderr, "init failed.\n");
    return 1;
  }

  MBLOG_INFO << "modelbox config path : " << kConfigPath;
  auto server = std::make_shared<Server>(kConfig);
  kServerTimer->Start();

  if (modelbox_run(server, keep_name, keep_time, key_file.c_str()) != 0) {
    return 1;
  }

  MBLOG_INFO << "exit modelbox process";
  return 0;
}
