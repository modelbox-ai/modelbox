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


#include "modelbox/common/command.h"

namespace modelbox {

std::recursive_mutex ToolCommandGetOptLock;

ToolCommandList::ToolCommandList() {}

ToolCommandList::~ToolCommandList() {}

ToolCommandList *ToolCommandList::Instance() {
  static bool env_set = false;
  if (!env_set) {
    setenv("POSIXLY_CORRECT", "1", 1);
    env_set = true;
  }
  static ToolCommandList list;
  return &list;
}

void ToolCommandList::AddCommand(ToolCommandCreate new_func) {
  auto cmd = new_func();
  auto name = cmd->GetCommandName();
  auto itr = commands_.find(name);
  if (itr != commands_.end()) {
    commands_.erase(itr);
  }

  commands_[name] = new_func;
}

void ToolCommandList::RmvCommand(const std::string &name) {
  auto itr = commands_.find(name);
  if (itr != commands_.end()) {
    commands_.erase(itr);
  }
}

void ToolCommandList::Reset() {
  commands_.clear();
}

std::shared_ptr<ToolCommand> ToolCommandList::GetCommand(
    const std::string &name) {
  auto itr = commands_.find(name);
  if (itr == commands_.end()) {
    return nullptr;
  }

  return commands_[name]();
}

std::vector<std::shared_ptr<ToolCommand>> ToolCommandList::GetAllCommands() {
  std::vector<std::shared_ptr<ToolCommand>> cmds;
  for (const auto &itr : commands_) {
    cmds.push_back(itr.second());
  }

  return cmds;
}

void ToolCommandGetOptReset(void) {
  static struct option long_options[] = {{"-", 0, 0, 0}, {0, 0, 0, 0}};
  int argc = 2;
  char const *argv[] = {"reset", "", nullptr};

  optind = 0;
  opterr = 0;
  optopt = 0;
  getopt_long(argc, const_cast<char **>(argv), "", long_options, NULL);
  optind = 0;
  opterr = 0;
  optopt = 0;
}

}  // namespace modelbox
