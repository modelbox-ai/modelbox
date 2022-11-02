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

#ifndef MODELBOX_COMMON_TOOL_COMMANDS_H
#define MODELBOX_COMMON_TOOL_COMMANDS_H

#include <getopt.h>

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "modelbox/common/utils.h"

namespace modelbox {

extern void ToolCommandGetOptReset();
extern std::recursive_mutex ToolCommandGetOptLock;

#define MODELBOX_TOOL_STRCAT_(a, b) a##b
#define MODELBOX_TOOL_STRCAT(a, b) MODELBOX_TOOL_STRCAT_(a, b)
#define MODELBOX_TOOL_ADD_COMMAND(new_func) \
  ::modelbox::ToolCommandList::Instance()->AddCommand(new_func);
#define MODELBOX_TOOL_RMV_COMMAND(name) \
  ::modelbox::ToolCommandList::Instance()->RmvCommand(name);
#define MODELBOX_TOOL_CLEAR_COMMAND() \
  ::modelbox::ToolCommandList::Instance()->Reset();

#define REG_MODELBOX_TOOL_COMMAND(class)                                     \
  static std::string cmd_name_##class;                                       \
  static auto __attribute__((unused))                                        \
  MODELBOX_TOOL_STRCAT(__auto_reg__, __LINE__) = []() {                      \
    auto new_func = []() -> std::shared_ptr<::modelbox::ToolCommand> {       \
      auto cmd = std::make_shared<class>();                                  \
      return cmd;                                                            \
    };                                                                       \
    auto cmd = new_func();                                                   \
    cmd_name_##class = cmd->GetCommandName();                                \
    ::modelbox::ToolCommandList::Instance()->AddCommand(new_func);           \
    return 0;                                                                \
  }();                                                                       \
  DeferExt() {                                                               \
    /* Remove command from command list.*/                                   \
    if (cmd_name_##class.length() > 0) {                                     \
      ::modelbox::ToolCommandList::Instance()->RmvCommand(cmd_name_##class); \
    }                                                                        \
  };

#define MODELBOX_COMMAND_SUB_ARGC argc_sub
#define MODELBOX_COMMAND_SUB_ARGV argv_sub
#define MODELBOX_COMMAND_SUB_UNLOCK() get_opt_lock.unlock()

/**
 * @brief Lock globally in the macro to avoid concurrent access to the getopt
 * function. You can use the MODELBOX_COMMAND_SUB_UNLOCK() function to unlock,
 * but after unlocking, you need to return the function immediately
 */
#define MODELBOX_COMMAND_GETOPT_SHORT_BEGIN(cmdtype, short_options, options) \
  optind = 1;                                                                \
  int option_index = 0;                                                      \
  if (argc <= 0 || argv == nullptr) {                                        \
    return -1;                                                               \
  }                                                                          \
                                                                             \
  std::unique_lock<std::recursive_mutex> get_opt_lock(                       \
      ::modelbox::ToolCommandGetOptLock);                                    \
  ::modelbox::ToolCommandGetOptReset();                                      \
  while (((cmdtype) = getopt_long_only(argc, argv, short_options, options,   \
                                       &option_index)) != EOF) {             \
    int MODELBOX_COMMAND_SUB_ARGC = argc - optind + 1;                       \
    char **MODELBOX_COMMAND_SUB_ARGV = argv + optind - 1;                    \
    { auto &unused __attribute__((unused)) = MODELBOX_COMMAND_SUB_ARGC; }    \
    { auto &unused __attribute__((unused)) = MODELBOX_COMMAND_SUB_ARGV; }

/**
 * @brief Lock globally in the macro to avoid concurrent access to the getopt
 * function. You can use the MODELBOX_COMMAND_SUB_UNLOCK() function to unlock,
 * but after unlocking, you need to return the function immediately
 */
#define MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, options) \
  MODELBOX_COMMAND_GETOPT_SHORT_BEGIN(cmdtype, "", options)

#define MODELBOX_COMMAND_GETOPT_END() \
  }                                   \
  get_opt_lock.unlock();

class StdOutStream : public OutStream {
 public:
  StdOutStream();
  virtual ~StdOutStream();

 protected:
  void ProcessStream(OStream *st) override;
};

class StdErrStream : public OutStream {
 public:
  StdErrStream();
  virtual ~StdErrStream();

 protected:
  void ProcessStream(OStream *st);
};

class ToolCommand {
 protected:
#define TOOL_COUT *out_cout_->Stream()
#define TOOL_CERR *out_cerr_->Stream()
  std::shared_ptr<OutStream> out_cout_ = std::make_shared<StdOutStream>();
  std::shared_ptr<OutStream> out_cerr_ = std::make_shared<StdErrStream>();

 public:
  ToolCommand();
  virtual ~ToolCommand();

  void SetUp(std::shared_ptr<OutStream> cout, std::shared_ptr<OutStream> cerr);

  virtual int Run(int argc, char *argv[]) = 0;

  virtual std::string GetHelp() = 0;

  virtual std::string GetCommandName() = 0;

  virtual std::string GetCommandDesc() = 0;
};

using ToolCommandCreate = std::function<std::shared_ptr<ToolCommand>()>;
class ToolCommandList {
  ToolCommandList();
  virtual ~ToolCommandList();

 public:
  static ToolCommandList *Instance();
  void AddCommand(const ToolCommandCreate &new_func);

  void RmvCommand(const std::string &name);

  void Reset();

  std::shared_ptr<ToolCommand> GetCommand(const std::string &name);

  std::vector<std::shared_ptr<ToolCommand>> GetAllCommands();

  void ShowHelp();

  void ShowHelp(const std::string &name);

  std::map<std::string, ToolCommandCreate> commands_;
};

}  // namespace modelbox

#endif
