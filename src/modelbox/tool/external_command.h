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


#ifndef MODELBOX_TOOL_EXTERNAL_COMMAND_H
#define MODELBOX_TOOL_EXTERNAL_COMMAND_H

#include <modelbox/base/status.h>

#include "modelbox/common/command.h"

namespace modelbox {

class ExternalCommandKey : public ToolCommand {
 public:
  ExternalCommandKey();
  virtual ~ExternalCommandKey();

  int Run(int argc, char *argv[]);
  std::string GetHelp();

  Status SetExecuteCmd(const std::string &cmd);
  void SetCommandName(const std::string &name);
  void SetCommandDesc(const std::string &desc);
  void SetHelpCmd(const std::string &help_cmd);
  void SetTimeout(int timeout);
  std::string GetCommandName();
  std::string GetCommandDesc();

 private:
  std::string cmd_;
  std::string name_;
  std::string desc_;
  std::string help_cmd_;
  int timeout_{-1};
};

constexpr const char *EXTERNAL_TOOLS_PATH = "${MODELBOX_ROOT}/usr/local/share/modelbox/tools";

class ExternalCommandLoader {
 public:
  static Status Load(const std::string &path = EXTERNAL_TOOLS_PATH);

 private:
  static Status LoadCmds(const std::string &cmd_json_file);
};

}  // namespace modelbox
#endif