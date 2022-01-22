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
#include <modelbox/base/utils.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <memory>
#include <thread>

#include "driver.h"
#include "external_command.h"
#include "key.h"
#include "log.h"
#include "modelbox/common/log.h"
#include "modelbox/common/utils.h"

using namespace modelbox;

#define TMP_BUFF_LEN_32 32
#define MODELBOX_TOOL_LOG_PATH "/var/log/modelbox/modelbox-tool.log"

extern char *program_invocation_name;
extern char *program_invocation_short_name;

static int g_sig_list[] = {
    SIGIO,   SIGPWR,    SIGSTKFLT, SIGPROF, SIGINT,  SIGTERM,
    SIGBUS,  SIGVTALRM, SIGTRAP,   SIGXCPU, SIGXFSZ, SIGILL,
    SIGABRT, SIGFPE,    SIGSEGV,   SIGQUIT, SIGSYS,
};

static int g_sig_num = sizeof(g_sig_list) / sizeof(g_sig_list[0]);
static bool kVerbose = false;
std::string kLogLevel = "ERROR";
std::string kLogFile;
std::shared_ptr<ModelboxServerLogger> modelbox::kServerLogger;

enum MODELBOX_TOOL_COMMAND {
  MODELBOX_TOOL_COMMAND_KEY,
  MODELBOX_TOOL_COMMAND_DRIVER,
  MODELBOX_TOOL_COMMAND_VERBOSE,
  MODELBOX_TOOL_COMMAND_LOG_LEVEL,
  MODELBOX_TOOL_COMMAND_LOG_PATH,
  MODELBOX_TOOL_COMMAND_HELP,
  MODELBOX_TOOL_SHOW_VERSION,
};

static struct option options[] = {
    {"verbose", 0, 0, MODELBOX_TOOL_COMMAND_VERBOSE},
    {"log-level", 1, 0, MODELBOX_TOOL_COMMAND_LOG_LEVEL},
    {"log-path", 1, 0, MODELBOX_TOOL_COMMAND_LOG_PATH},
    {"h", 0, 0, MODELBOX_TOOL_COMMAND_HELP},
    {"v", 0, 0, MODELBOX_TOOL_SHOW_VERSION},
    {0, 0, 0, 0},
};

static void showhelp(void) {
  /* clang-format off */
    char help[] = ""
        "Usage: modelbox-tool [OPTION]...\n"
        "modelbox tool main options: \n"
        "  -verbose      output log to screen.\n"
        "  -log-level    log level: DEBUG, INFO, NOTICE, WARN, ERROR, FATAL.\n"
        "  -log-path     log file: default : " MODELBOX_TOOL_LOG_PATH "\n"
        "  -h            show this help message.\n"
        "  -v            show modelbox tool version.\n"
        "\n"
        "show command help:\n"
        "  modelbox-tool help [cmd]    show help message for specific command\n"
        "\n";

    printf("%s", help);
  /* clang-format on */
  printf("modelbox-tool commands list:\n");
  printf("Usage: modelbox-tool [cmd] [OPTION]...\n");
  auto all_cmds = modelbox::ToolCommandList::Instance()->GetAllCommands();
  for (const auto cmd : all_cmds) {
    printf("  %-10.10s \t\t%s\n", cmd->GetCommandName().c_str(),
           cmd->GetCommandDesc().c_str());
  }
}
static void modelbox_tool_sig_handler(int volatile sig_no, siginfo_t *sig_info,
                                      void *volatile ptr) {
  switch (sig_no) {
    case SIGINT:
    case SIGTERM:
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

static int modelbox_tool_init_bbox(void) {
  if (modelbox_sig_register(g_sig_list, g_sig_num, modelbox_tool_sig_handler) !=
      0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  return 0;
}

int modelbox_tool_init_log(void) {
  kServerLogger = std::make_shared<ModelboxServerLogger>();
  if (kServerLogger->Init(kLogFile, 1024 * 1024, 32, kVerbose) == false) {
    fprintf(stderr, "init logger failed.\n");
    return 1;
  }

  ModelBoxLogger.SetLogger(kServerLogger);
  auto log_level = modelbox::LogLevelStrToLevel(kLogLevel);
  kServerLogger->SetLogLevel(log_level);

  return 0;
}

int modelbox_tool_init(void) {
  if (modelbox_tool_init_bbox() != 0) {
    fprintf(stderr, "register signal failed.\n");
    return 1;
  }

  if (modelbox_tool_init_log()) {
    return 1;
  }

  return 0;
}

int modelbox_tool_run(int argc, char *argv[]) {
  if (argc <= 0) {
    printf("Try -h for more information.\n");
    return -1;
  }

  const char *action = argv[0];
  auto cmd = modelbox::ToolCommandList::Instance()->GetCommand(action);
  if (cmd == nullptr) {
    printf("command %s not exist, try -h for more information.\n", action);
    return -1;
  }

  return cmd->Run(argc, argv);
}

void modelbox_tool_stop() {}

static void onexit(void) {}

#ifdef BUILD_TEST
int modelbox_tool_main(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{
  kLogFile = MODELBOX_TOOL_LOG_PATH;
  int cmdtype = 0;

  modelbox::ExternalCommandLoader::Load();
  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, options)
  switch (cmdtype) {
    case MODELBOX_TOOL_COMMAND_VERBOSE:
      kVerbose = true;
      break;
    case MODELBOX_TOOL_COMMAND_LOG_LEVEL:
      kLogLevel = optarg;
      break;
    case MODELBOX_TOOL_COMMAND_LOG_PATH:
      kLogFile = optarg;
      break;
    case MODELBOX_TOOL_COMMAND_HELP:
      showhelp();
      return 0;
      break;
    case MODELBOX_TOOL_SHOW_VERSION:
      printf("modelbox-tool %s\n", modelbox::GetModelBoxVersion());
      return 0;
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  if (argc <= 1) {
    showhelp();
    return 1;
  }

  Defer { onexit(); };
  signal(SIGPIPE, SIG_IGN);

  if (modelbox_tool_init() != 0) {
    fprintf(stderr, "init failed.\n");
    return 1;
  }

  int argc_sub = argc - optind;
  char **argv_sub = argv + optind;
  optind = 1;
  if (modelbox_tool_run(argc_sub, argv_sub) != 0) {
    return 1;
  }

  return 0;
}
