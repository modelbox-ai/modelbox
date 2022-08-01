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


#include "help.h"

#include <modelbox/modelbox.h>
#include <modelbox/flow.h>
#include <getopt.h>
#include <stdio.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace modelbox {

REG_MODELBOX_TOOL_COMMAND(ToolCommandHelp)

ToolCommandHelp::ToolCommandHelp() = default;
ToolCommandHelp::~ToolCommandHelp() = default;

std::string ToolCommandHelp::GetHelp() {
  char help[] =
      "help [cmd]    Display help for command [cmd]"
      "\n";
  return help;
}

int ToolCommandHelp::Run(int argc, char *argv[]) {
  if (argc <= 1) {
    fprintf(stderr, "please input command for help, try modelbox-tool cmd [cmd]\n");
    return -1;
  }

  auto cmd = modelbox::ToolCommandList::Instance()->GetCommand(argv[1]);
  if (cmd == nullptr) {
    fprintf(stderr, "Command %s not found.\n", argv[1]);
    return -1;
  }

  std::cout << "modelbox-tool " << cmd->GetCommandName() << " [OPTION]..."
            << std::endl;
  std::cout << cmd->GetHelp();

  return 0;
}

}  // namespace modelbox
