/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#include "modelbox/base/config.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/popen.h"
#include "modelbox/base/utils.h"
#include "modelbox/common/command.h"
#include "modelbox/common/utils.h"
#include "serving.h"

using namespace modelbox;

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};

static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);

enum MODELBOX_SERVING_ARG {
  MODELBOX_SERVING_ARG_MODEL_NAME,
  MODELBOX_SERVING_ARG_MODEL_PATH,
  MODELBOX_SERVING_ARG_DAEMON,
  MODELBOX_SERVING_ARG_PORT,
  MODELBOX_SERVING_ARG_HELP,
};

static struct option options[] = {
    {"model-name", 1, 0, MODELBOX_SERVING_ARG_MODEL_NAME},
    {"model-path", 1, 0, MODELBOX_SERVING_ARG_MODEL_PATH},
    {"daemon", 0, 0, MODELBOX_SERVING_ARG_DAEMON},
    {"port", 1, 0, MODELBOX_SERVING_ARG_PORT},
    {"h", 0, 0, MODELBOX_SERVING_ARG_HELP},
    {0, 0, 0, 0},
};

static void showhelp(void) {
  /* clang-format off */
    char help[] = ""
        "Usage: modelbox-serving [OPTION]...\n"
        "Start modelbox-serving.\n"
        "  -model-name            model-name.\n"
        "  -model-path            model-path.\n"
        "  -daemon                run by daemon.\n"
        "  -port                  rest api port.\n"
        "\n";

    printf("%s", help);
  /* clang-format on */
}

static void modelbox_sig_handler(int volatile sig_no, siginfo_t *sig_info,
                                 void *volatile ptr) {
  switch (sig_no) {
    case SIGINT:
    case SIGTERM:
      exit(1);
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

int modelbox_serving_init(void) {
  if (modelbox_reg_signal() != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  if (modelbox_root_dir().length() > 0) {
    std::string default_scanpath = modelbox_full_path(
        std::string(MODELBOX_ROOT_VAR) + MODELBOX_DEFAULT_DRIVER_PATH);
    modelbox::Drivers::SetDefaultScanPath(default_scanpath);

    std::string default_driver_info_path = modelbox_full_path(
        std::string(MODELBOX_ROOT_VAR) + "/var/run/modelbox-driver-info");
    modelbox::Drivers::SetDefaultInfoPath(default_driver_info_path);
  }

  return 0;
}

int modelbox_serving_generate_template(const std::string &model_name,
                                       const std::string &model_path,
                                       int port) {
  fprintf(stdout, "modelbox config path : %s\n", model_path.c_str());
  auto serving = std::make_shared<ModelServing>();
  auto status = serving->GenerateTemplate(model_name, model_path, port);
  if (status != modelbox::STATUS_OK) {
    return 1;
  }

  return 0;
}

int modelbox_run() {
  auto p = std::make_shared<modelbox::Popen>();
  std::string modelbox_bin{"/etc/init.d/modelbox"};
  if (modelbox_root_dir().length() > 0) {
    modelbox_bin =
        modelbox_full_path(std::string(MODELBOX_ROOT_VAR) + modelbox_bin);
  }

  std::vector<std::string> cmd{modelbox_bin, "restart"};
  auto status = p->Open(cmd, -1);
  if (status != modelbox::STATUS_OK) {
    fprintf(stderr, "execute modelbox cmd failed");
    return 1;
  }

  int ret = p->Close();
  if (ret != 0) {
    fprintf(stderr, "popen close failed failed.");
    return 1;
  }

  return 0;
}

static void onexit(void) {}

#ifdef BUILD_TEST
int modelbox_serving_main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
  std::string model_name = "";
  std::string model_path = "";
  bool kDaemon{false};
  int port = 9110;
  int cmdtype = 0;

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, options)
  switch (cmdtype) {
    case MODELBOX_SERVING_ARG_HELP:
      showhelp();
      return 1;
    case MODELBOX_SERVING_ARG_MODEL_NAME:
      printf("model-name %s \n", optarg);
      model_name = optarg;
      break;
    case MODELBOX_SERVING_ARG_MODEL_PATH:
      printf("model-path %s \n", optarg);
      model_path = optarg;
      break;
    case MODELBOX_SERVING_ARG_PORT:
      try {
        port = std::stoi(optarg);
      } catch (const std::string &exception) {
        fprintf(stderr, "rest port %s failed, use default port %d", optarg,
                port);
        return 1;
      }
      break;
    case MODELBOX_SERVING_ARG_DAEMON:
      kDaemon = true;
      break;
    default:
      printf("Try %s -h for more information.\n", argv[0]);
      return 1;
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  if (argc <= 1) {
    showhelp();
    return 1;
  }

  if (kDaemon == true) {
    if (daemon(0, 0) < 0) {
      fprintf(stderr, "run daemon process failed, %s\n",
              modelbox::StrError(errno).c_str());
      return 1;
    }
  }

  Defer { onexit(); };
  /* 忽略SIGPIPE，避免发送缓冲区慢导致的进程退出 */
  signal(SIGPIPE, SIG_IGN);

  if (modelbox_serving_init() != 0) {
    fprintf(stderr, "init failed.\n");
    return 1;
  }

  if (modelbox_serving_generate_template(model_name, model_path, port) != 0) {
    fprintf(stderr, "generate model serving failed.\n");
    return 1;
  }

  if (modelbox_run() != 0) {
    return 1;
  }

  fprintf(stdout, "exit modelbox process\n");
  return 0;
}
