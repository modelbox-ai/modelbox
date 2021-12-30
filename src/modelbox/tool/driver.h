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

#ifndef MODELBOX_TOOL_DRIVER_H
#define MODELBOX_TOOL_DRIVER_H

#include <modelbox/base/configuration.h>
#include <modelbox/base/driver.h>

#include "modelbox/common/command.h"
#include "modelbox/common/flowunit_info.h"

namespace modelbox {

enum DRIVER_OUTFORMAT {
  DRIVER_OUTFORMAT_LIST,
  DRIVER_OUTFORMAT_DETAILS,
  DRIVER_OUTFORMAT_JSON,
};

enum DRIVER_TYPE {
  DRIVER_TYPE_ALL,
  DRIVER_TYPE_FLOWUNIT,
};

constexpr const char *DRIVER_DESC = "List all driver information";

class ToolCommandDriver : public ToolCommand {
 public:
  ToolCommandDriver();
  virtual ~ToolCommandDriver();

  int Run(int argc, char *argv[]);

  std::string GetHelp();

  std::string GetCommandName() { return "driver"; };

  std::string GetCommandDesc() { return DRIVER_DESC; };

 protected:
  int RunInfoCommand(int argc, char *argv[]);
  Status OutputInfo(std::shared_ptr<Configuration> config,
                    enum DRIVER_TYPE type, enum DRIVER_OUTFORMAT format,
                    const std::string &filter_name);
  Status OutputDriverInfo(std::shared_ptr<Configuration> config,
                          enum DRIVER_OUTFORMAT format,
                          const std::string &filter_name);
  Status OutputFlowunitInfo(std::shared_ptr<Configuration> config,
                            enum DRIVER_OUTFORMAT format,
                            const std::string &filter_name);
  Status DisplayDriverInList(std::shared_ptr<Configuration> config);
  Status DisplayDriverInDetails(std::shared_ptr<Configuration> config,
                                const std::string &filter_name);
  Status DisplayDriverInJson(std::shared_ptr<Configuration> config);
  Status DisplayFlowunitInList(std::shared_ptr<Configuration> config);
  Status DisplayFlowunitInDetails(std::shared_ptr<Configuration> config,
                                  const std::string &filter_name);
  Status DisplayFlowunitInJson(std::shared_ptr<Configuration> config);
  void DisplayFlowunit(std::shared_ptr<FlowUnitDesc> flowunit);
  void DisplayFlowunitByFilter(std::shared_ptr<FlowUnitInfo> flowunit_info,
                               const std::string &filter_name);
};

}  // namespace modelbox

#endif