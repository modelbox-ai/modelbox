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

#ifndef MODELBOX_CONTROL_COMMAND_H_
#define MODELBOX_CONTROL_COMMAND_H_

#include "modelbox/base/memory_pool.h"
#include "modelbox/common/command.h"

namespace modelbox {

constexpr const char *LOG_CONTROL_DESC = "control server log";
constexpr const char *SLAB_CONTROL_DESC = "control server slab";
constexpr const char *STATISTICS_DESC = "control server statistics";

class ToolCommandLog : public modelbox::ToolCommand {
 public:
  ToolCommandLog();
  ~ToolCommandLog() override;

  int Run(int argc, char *argv[]) override;
  std::string GetHelp() override;

  std::string GetCommandName() override { return "log"; };
  std::string GetCommandDesc() override { return LOG_CONTROL_DESC; };
};

class ToolCommandSlab : public modelbox::ToolCommand {
 public:
  ToolCommandSlab();
  ~ToolCommandSlab() override;

  int Run(int argc, char *argv[]) override;
  std::string GetHelp() override;

  std::string GetCommandName() override { return "slab"; };
  std::string GetCommandDesc() override { return SLAB_CONTROL_DESC; };

 private:
  int RunDeviceOption(int argc, char *argv[]);
  int DisplaySlabsInfo(const std::string &type = "");
  void DisplaySlabInfo(std::shared_ptr<modelbox::MemoryPoolBase> &mem_pool,
                       const std::string &type, const std::string &id);
  bool GetMemPools(
      std::vector<std::shared_ptr<modelbox::MemoryPoolBase>> &mempools,
      const std::string &type, const std::string &id = "");
  int DeviceSlabInfo(const std::string &type, const std::string &id);
  bool DisplayMemPools(const std::string &type);
};

class ToolCommandStatistics : public modelbox::ToolCommand {
 public:
  ToolCommandStatistics();
  ~ToolCommandStatistics() override;

  int Run(int argc, char *argv[]) override;
  std::string GetHelp() override;
  std::string GetCommandName() override { return "stat"; };
  std::string GetCommandDesc() override { return STATISTICS_DESC; };
};

}  // namespace modelbox

#endif  // MODELBOX_CONTROL_COMMAND_H_
