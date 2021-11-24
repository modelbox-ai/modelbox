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


#ifndef MODELBOX_TOOL_FLOW_H
#define MODELBOX_TOOL_FLOW_H

#include "modelbox/common/command.h"
namespace modelbox {

constexpr const char *FLOW_DESC = "Run flow, convert config file format";

class ToolCommandFlow : public ToolCommand {
 public:
  ToolCommandFlow();
  virtual ~ToolCommandFlow();

  int Run(int argc, char *argv[]);
  std::string GetHelp();

  std::string GetCommandName() { return "flow"; };
  std::string GetCommandDesc() { return FLOW_DESC; };

 protected:
  int RunFlow(const std::string &file);
  int RunConfConvertCommand(int argc, char *argv[]);
};

}  // namespace modelbox
#endif