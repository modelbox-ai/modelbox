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


#include "mock_tool.h"

#include "../config.h"
#include "modelbox/common/command.h"
#include "test_config.h"

namespace modelbox {

MockTool::MockTool() {}

MockTool::~MockTool() {}

int MockTool::Run(const std::string &cmd) {
  auto cmds = modelbox::StringSplit(cmd, ' ');
  int argc = cmds.size();
  char *argv[cmds.size()];
  for (size_t i = 0; i < cmds.size(); i++) {
    argv[i] = (char *)cmds[i].data();
  }
  return Run(argc, argv);
}

int MockTool::Run(int argc, char *argv[]) {
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

void MockTool::SetDefaultConfig(std::shared_ptr<Configuration> config) {
  std::vector<std::string> plugin_path;
  config->SetProperty("control.enable", "true");
  config->SetProperty("control.listen",
                      std::string(TEST_DATA_DIR) + "/modelbox.sock");
}

Status MockTool::Init(std::shared_ptr<Configuration> config) {
  if (config == nullptr) {
    ConfigurationBuilder builder;
    config = builder.Build();
  }

  SetDefaultConfig(config);

  return STATUS_OK;
}

}  // namespace modelbox
