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

#include "example.h"
#include <modelbox/common/command.h>

class ToolCommandExample : public modelbox::ToolCommand {
 public:
  ToolCommandExample() {};
  virtual ~ToolCommandExample() {};

  int Run(int argc, char *argv[]) {
    TOOL_COUT << "Example command output message." << std::endl;
    TOOL_CERR << "Example command stderror message." << std::endl;
    return 0;
  }
  std::string GetHelp() {
    return "Example Help.";
  }

  std::string GetCommandName() { return "example"; };
  std::string GetCommandDesc() { return "control server log"; };
};

REG_MODELBOX_TOOL_COMMAND(ToolCommandExample)

std::shared_ptr<modelbox::Plugin> CreatePlugin() {
  MBLOG_INFO << "Example create success.";
  return std::make_shared<ExamplePlugin>();
}

bool ExamplePlugin::Init(std::shared_ptr<modelbox::Configuration> config) {
  MBLOG_INFO << "Example plugin Init.";
  return true;
}

bool ExamplePlugin::Start() {
  MBLOG_INFO << "Example plugin Start.";
  return true;
}

bool ExamplePlugin::Stop() {
  MBLOG_INFO << "Example plugin Stop.";
  return true;
}
