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


#ifndef MODELBOX_TOOL_KEY_H
#define MODELBOX_TOOL_KEY_H

#include <modelbox/base/status.h>

#include "modelbox/common/command.h"

namespace modelbox {

constexpr const char *KEY_DESC = "Key encrypt";

class ToolCommandKey : public ToolCommand {
 public:
  ToolCommandKey();
  virtual ~ToolCommandKey();

  int Run(int argc, char *argv[]);
  std::string GetHelp();

  std::string GetCommandName() { return "key"; };
  std::string GetCommandDesc() { return KEY_DESC;};

 private:
  int RunPassCommand(int argc, char *argv[], std::string &fname);
  Status EnKey(bool sysrelated, std::string *rootkey, std::string *enpass, std::string &fname);
  Status ReadPassword(std::string *pass);
};

}  // namespace modelbox
#endif