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


#include "external_command.h"

#include <modelbox/base/crypto.h>
#include <modelbox/base/log.h>
#include <modelbox/base/utils.h>
#include <errno.h>
#include <getopt.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace modelbox {

ExternalCommandKey::ExternalCommandKey() {}
ExternalCommandKey::~ExternalCommandKey() {}

Status ExternalCommandKey::SetExecuteCmd(const std::string &cmd) {
  cmd_ = cmd;
  return STATUS_OK;
}

void ExternalCommandKey::SetCommandName(const std::string &name) {
  name_ = name;
}

void ExternalCommandKey::SetCommandDesc(const std::string &desc) {
  desc_ = desc;
}

void ExternalCommandKey::SetHelpCmd(const std::string &help_cmd) {
  help_cmd_ = help_cmd;
}

int ExternalCommandKey::Run(int argc, char *argv[]) {
  std::string cmd = cmd_;
  for (int i = 1; i < argc; i++) {
    cmd += " ";
    cmd += argv[i];
  }

  MBLOG_DEBUG << "run cmd: " << cmd;
  int ret = system(cmd.c_str());
  return WEXITSTATUS(ret);
}

std::string ExternalCommandKey::GetHelp() {
  auto ret = system(help_cmd_.c_str());
  UNUSED_VAR(ret);
  return "";
}

std::string ExternalCommandKey::GetCommandName() {
  return modelbox::GetBaseName(cmd_);
}

std::string ExternalCommandKey::GetCommandDesc() { return desc_; }

Status ExternalCommandLoader::LoadCmds(const std::string &cmd_json_file) {
  std::ifstream infile(cmd_json_file);
  if (infile.fail()) {
    std::cerr << "read file " << cmd_json_file << " failed, " << modelbox::StrError(errno)
              << std::endl;
    return {STATUS_BADCONF, modelbox::StrError(errno)};
  }

  Defer { infile.close(); };
  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());

  try {
    auto conf = nlohmann::json::parse(data);
    auto list = conf["cmd-list"];
    for (auto &cmd : list) {
      auto name = cmd["name"].get<std::string>();
      auto exec = cmd["exec"].get<std::string>();
      auto desc = cmd["desc"].get<std::string>();
      auto help_cmd = cmd["help-cmd"].get<std::string>();

      auto ext_cmd = std::make_shared<ExternalCommandKey>();
      ext_cmd->SetExecuteCmd(exec);
      ext_cmd->SetCommandName(name);
      ext_cmd->SetCommandDesc(desc);
      ext_cmd->SetHelpCmd(help_cmd);
      auto new_func = [ext_cmd]() -> std::shared_ptr<ExternalCommandKey> {
        auto new_cmd = std::make_shared<ExternalCommandKey>();
        *new_cmd = *ext_cmd;
        return new_cmd;
      };

      MODELBOX_TOOL_ADD_COMMAND(new_func);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Load external command failed, %s\n", e.what());
    return {STATUS_BADCONF, e.what()};
  }

  return STATUS_OK;
}

Status ExternalCommandLoader::Load(const std::string &path) {
  std::vector<std::string> files;
  auto ret = modelbox::ListFiles(path, "*.json", &files, LIST_FILES_FILE);
  if (!ret) {
    return ret;
  }

  for (auto &file : files) {
    ret = ExternalCommandLoader::LoadCmds(file);
    if (!ret) {
      continue;
    }
  }

  return STATUS_OK;
}

}  // namespace modelbox
