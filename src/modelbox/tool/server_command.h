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


#ifndef MODELBOX_TOOL_SERVER_H
#define MODELBOX_TOOL_SERVER_H

#include <modelbox/base/status.h>
#include <modelbox/common/control_msg.h>

#include "modelbox/common/command.h"

namespace modelbox {

constexpr const char *SERVER_DESC = "Server commands";
constexpr const char *DEFAULT_MODELBOX_CONF = "${MODELBOX_ROOT}/usr/local/etc/modelbox/modelbox.conf";

class ToolCommandServer : public ToolCommand {
 public:
  ToolCommandServer();
  ~ToolCommandServer() override;

  int Run(int argc, char *argv[]) override;
  std::string GetHelp() override;

  std::string GetCommandName() override { return "server"; };
  std::string GetCommandDesc() override { return SERVER_DESC; };

 private:
  modelbox::Status InitClient(const std::string &connect_url);
  void CloseClient();
  modelbox::Status SendCommand(
      int argc, char *argv[],
      const std::string &connect_url = CONTROL_UNIX_PATH);
  modelbox::Status SendMsg(std::shared_ptr<ControlMsg> msg,
                         const std::string &connect_url = CONTROL_UNIX_PATH);
  int RecvCommand();
  modelbox::Status GetSockFile(const std::string &conf_file,
                             std::string &connect_url);
  int client_fd_{-1};
  std::string unix_path_;
  int temp_fd_{-1};
};

}  // namespace modelbox
#endif