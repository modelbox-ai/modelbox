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

#include "driver.h"

#include <getopt.h>
#include <modelbox/base/config.h>
#include <modelbox/modelbox.h>
#include <stdio.h>

#include <nlohmann/json.hpp>

namespace modelbox {

constexpr const char *DRIVER_CONF = "driver";

enum MODELBOX_TOOL_DRIVER_COMMAND {
  MODELBOX_TOOL_DRIVER_INFO,
};

static struct option driver_options[] = {
    {"info", 0, 0, MODELBOX_TOOL_DRIVER_INFO},
    {0, 0, 0, 0},
};

enum MODELBOX_TOOL_DRIVER_INFO_COMMAND {
  MODELBOX_TOOL_DRIVER_INFO_PATH,
  MODELBOX_TOOL_DRIVER_INFO_FROM_CONF,
  MODELBOX_TOOL_DRIVER_INFO_DETAILS,
  MODELBOX_TOOL_DRIVER_INFO_DETAILS_FILTER_NAME,
  MODELBOX_TOOL_DRIVER_INFO_FORMAT_JSON,
  MODELBOX_TOOL_DRIVER_INFO_TYPE,
};

static struct option driver_info_options[] = {
    {"path", 1, 0, MODELBOX_TOOL_DRIVER_INFO_PATH},
    {"conf", 1, 0, MODELBOX_TOOL_DRIVER_INFO_FROM_CONF},
    {"details", 0, 0, MODELBOX_TOOL_DRIVER_INFO_DETAILS},
    {"name", 1, 0, MODELBOX_TOOL_DRIVER_INFO_DETAILS_FILTER_NAME},
    {"format-json", 0, 0, MODELBOX_TOOL_DRIVER_INFO_FORMAT_JSON},
    {"type", 1, 0, MODELBOX_TOOL_DRIVER_INFO_TYPE},
    {0, 0, 0, 0},
};

REG_MODELBOX_TOOL_COMMAND(ToolCommandDriver)

ToolCommandDriver::ToolCommandDriver() {}
ToolCommandDriver::~ToolCommandDriver() {}

std::string ToolCommandDriver::GetHelp() {
  char help[] =
      " driver options: \n"
      "   -info    List all driver information\n"
      "     -type               Filter driver type, support value: flowunit\n"
      "     -path               Scan additional path, format: dir1,dir2\n"
      "     -details            Show detail information\n"
      "        -name [name]     Filter name for details\n"
      "     -conf [toml file]   Read toml config, and list drivers\n"
      "     -format-json        export json format driver information\n"
      "\n";
  return help;
}

int ToolCommandDriver::Run(int argc, char *argv[]) {
  int cmdtype = 0;

  if (argc == 1) {
    std::cerr << GetHelp();
    return 1;
  }

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, driver_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_DRIVER_INFO:
      optind = 1;
      MODELBOX_COMMAND_SUB_UNLOCK();
      return RunInfoCommand(MODELBOX_COMMAND_SUB_ARGC,
                            MODELBOX_COMMAND_SUB_ARGV);
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return 0;
}

int ToolCommandDriver::RunInfoCommand(int argc, char *argv[]) {
  int cmdtype = 0;
  ConfigurationBuilder config_builder;
  std::shared_ptr<Configuration> config_merge;
  enum DRIVER_OUTFORMAT format = DRIVER_OUTFORMAT_LIST;
  enum DRIVER_TYPE type = DRIVER_TYPE_ALL;
  std::string filter_name;
  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, driver_info_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_DRIVER_INFO_PATH: {
      std::string path = optarg;
      std::vector<std::string> paths = modelbox::StringSplit(path, ',');
      config_builder.AddProperty(std::string(DRIVER_CONF) + "." + DRIVER_DIR,
                                 paths);
      config_builder.AddProperty(
          std::string(DRIVER_CONF) + "." + DRIVER_SKIP_DEFAULT, "0");
      break;
    }
    case MODELBOX_TOOL_DRIVER_INFO_DETAILS:
      format = DRIVER_OUTFORMAT_DETAILS;
      break;
    case MODELBOX_TOOL_DRIVER_INFO_DETAILS_FILTER_NAME:
      format = DRIVER_OUTFORMAT_DETAILS;
      filter_name = optarg;
      break;
    case MODELBOX_TOOL_DRIVER_INFO_FROM_CONF: {
      ConfigurationBuilder builder;
      std::string confile_file = optarg;
      config_merge = config_builder.Build(confile_file);
      if (config_merge == nullptr) {
        fprintf(stderr, "parser config '%s' failed, %s\n", confile_file.c_str(),
                StatusError.WrapErrormsgs().c_str());
        return 1;
      }
      break;
    }
    case MODELBOX_TOOL_DRIVER_INFO_FORMAT_JSON:
      format = DRIVER_OUTFORMAT_JSON;
    case MODELBOX_TOOL_DRIVER_INFO_TYPE: {
      std::string t = optarg;
      if (t == "flowunit") {
        type = DRIVER_TYPE_FLOWUNIT;
      }
      break;
    }
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  auto config = config_builder.Build();
  if (config_merge) {
    config->Add(*config_merge);
  }

  auto status = OutputInfo(config, type, format, filter_name);
  if (!status) {
    fprintf(stderr, "display driver info failed, %s\n",
            status.WrapErrormsgs().c_str());
    return 1;
  }

  return 0;
}

Status ToolCommandDriver::OutputDriverInfo(
    std::shared_ptr<Configuration> config, enum DRIVER_OUTFORMAT format,
    const std::string &filter_name) {
  if (format == DRIVER_OUTFORMAT_LIST) {
    return DisplayDriverInList(config);
  } else if (format == DRIVER_OUTFORMAT_JSON) {
    return DisplayDriverInJson(config);
  } else if (format == DRIVER_OUTFORMAT_DETAILS) {
    return DisplayDriverInDetails(config, filter_name);
  }

  return STATUS_NOTSUPPORT;
}

Status ToolCommandDriver::OutputFlowunitInfo(
    std::shared_ptr<Configuration> config, enum DRIVER_OUTFORMAT format,
    const std::string &filter_name) {
  if (format == DRIVER_OUTFORMAT_LIST) {
    return DisplayFlowunitInList(config);
  } else if (format == DRIVER_OUTFORMAT_JSON) {
    return DisplayFlowunitInJson(config);
  } else if (format == DRIVER_OUTFORMAT_DETAILS) {
    return DisplayFlowunitInDetails(config, filter_name);
  }

  return STATUS_NOTSUPPORT;
}

Status ToolCommandDriver::OutputInfo(std::shared_ptr<Configuration> config,
                                     enum DRIVER_TYPE type,
                                     enum DRIVER_OUTFORMAT format,
                                     const std::string &filter_name) {
  if (type == DRIVER_TYPE_ALL) {
    return OutputDriverInfo(config, format, filter_name);
  } else if (type == DRIVER_TYPE_FLOWUNIT) {
    return OutputFlowunitInfo(config, format, filter_name);
  }

  return STATUS_NOTSUPPORT;
}

Status ToolCommandDriver::DisplayDriverInList(
    std::shared_ptr<Configuration> config) {
  auto drivers = std::make_shared<Drivers>();
  auto status = drivers->Initialize(config->GetSubConfig(DRIVER_CONF));
  if (!status) {
    fprintf(stderr, "initialize drivers failed, %s\n",
            status.WrapErrormsgs().c_str());
    return {status, "initialize failed."};
  }

  status = drivers->Scan();
  if (!status) {
    fprintf(stderr, "scan failed, %s\n", status.WrapErrormsgs().c_str());
    return {status, "scan failed."};
  }
  int index = 0;
  auto drivers_list = drivers->GetAllDriverList();
  printf("Drivers Information:\n");
  printf("%-25s%-30s%-10s%-10s%s\n", "Class", "Name", "Type", "Version",
         "Path");
  for (const auto &driver : drivers_list) {
    auto desc = driver->GetDriverDesc();
    printf("%-25s%-30s%-10s%-10s%s\n", desc->GetClass().c_str(),
           desc->GetName().c_str(), desc->GetType().c_str(),
           desc->GetVersion().c_str(), desc->GetFilePath().c_str());
    index++;
  }

  return STATUS_OK;
}

void ToolCommandDriver::DisplayFlowunitByFilter(
    std::shared_ptr<FlowUnitInfo> flowunit_info,
    const std::string &filter_name) {
  if (flowunit_info == nullptr) {
    printf("DisplayFlowunitByFilter:  flowunit_info is nullptr.");
    return;
  }
  int index = 0;
  auto flow_list = flowunit_info->GetFlowUnitManager()->GetAllFlowUnitDesc();
  for (const auto &flow : flow_list) {
    auto driver_desc = flow->GetDriverDesc();
    if (filter_name.length() > 0 && (filter_name != driver_desc->GetName() &&
                                     filter_name != driver_desc->GetType() &&
                                     filter_name != flow->GetFlowUnitName())) {
      continue;
    }
    index++;
    if (index == 1) {
      printf("FlowUnit Information\t\t:\n");
    }
    DisplayFlowunit(flow);
  }
}

Status ToolCommandDriver::DisplayDriverInDetails(
    std::shared_ptr<Configuration> config, const std::string &filter_name) {
  auto flowunit_info = std::make_shared<FlowUnitInfo>();
  int index = 0;
  auto status = flowunit_info->Init(config);
  if (!status) {
    std::cerr << status << std::endl;
    return status;
  }

  auto device_desc_list = flowunit_info->GetDeviceManager()->GetDeviceDescList();
  for (const auto &itr_list : device_desc_list) {
    for (const auto &itr_device : itr_list.second) {
      auto desc = itr_device.second;
      if (filter_name.length() > 0 && filter_name != itr_device.first) {
        continue;
      }
      index++;
      if (index == 1) {
        printf("Device Information\t\t:\n");
        printf("--------------------------------\t\t:\n");
      }
      printf("name:\t\t%s\n", itr_device.first.c_str());
      printf("type:\t\t%s\n", desc->GetDeviceType().c_str());
      printf("version:\t\t%s\n", desc->GetDeviceVersion().c_str());
      printf("descryption: %s\n", desc->GetDeviceDesc().c_str());
      printf("\n");
    }
  }

  auto drivers_list = flowunit_info->GetDriverManager()->GetAllDriverList();
  index = 0;
  for (const auto &driver : drivers_list) {
    auto desc = driver->GetDriverDesc();
    if (filter_name.length() > 0 && filter_name != desc->GetName()) {
      continue;
    }
    index++;
    if (index == 1) {
      printf("Driver Information\t\t:\n");
      printf("--------------------------------\t\t:\n");
    }
    printf("driver name:\t\t%s\n", desc->GetName().c_str());
    printf("device type:\t\t%s\n", desc->GetType().c_str());
    printf("version:\t\t%s\n", desc->GetVersion().c_str());
    printf("class:\t\t%s\n", desc->GetClass().c_str());
    printf("descryption: %s\n", desc->GetDescription().c_str());
    printf("\n");
  }

  DisplayFlowunitByFilter(flowunit_info, filter_name);

  return STATUS_OK;
}

Status ToolCommandDriver::DisplayDriverInJson(
    std::shared_ptr<Configuration> config) {
  FlowUnitInfo flowunit_info;

  auto status = flowunit_info.Init(config);
  if (!status) {
    std::cerr << status << std::endl;
    return status;
  }

  std::string info;
  status = flowunit_info.GetInfoInJson(&info);
  if (!status) {
    std::cerr << status << std::endl;
    return status;
  }

  std::cout << info << std::endl;

  return STATUS_OK;
}

Status ToolCommandDriver::DisplayFlowunitInList(
    std::shared_ptr<Configuration> config) {
  FlowUnitInfo flowunit_info;

  auto status = flowunit_info.Init(config);
  if (!status) {
    std::cerr << status << std::endl;
    return status;
  }

  auto flowunit_list = flowunit_info.GetFlowUnitManager()->GetAllFlowUnitDesc();

  printf("Flowunit Information:\n");
  printf("%-30s%-15s%-15s%-30s%-15s%-30s%-30s\n", "FlowunitName", "DeviceType",
         "GroupType", "DriverName", "Version", "InputPort", "OutputPort");

  for (const auto &flowunit : flowunit_list) {
    std::string input_ports;
    std::string output_ports;
    for (const auto &input : flowunit->GetFlowUnitInput()) {
      auto s =
          (input_ports == "") ? input.GetPortName() : "," + input.GetPortName();
      input_ports += s;
    }
    for (const auto &output : flowunit->GetFlowUnitOutput()) {
      auto s = (output_ports == "") ? output.GetPortName()
                                    : "," + output.GetPortName();
      output_ports += s;
    }
    auto driverdesc = flowunit->GetDriverDesc();
    printf("%-30s%-15s%-15s%-30s%-15s%-30s%-30s\n",
           flowunit->GetFlowUnitName().c_str(), driverdesc->GetType().c_str(),
           flowunit->GetGroupType().c_str(), driverdesc->GetName().c_str(),
           driverdesc->GetVersion().c_str(), input_ports.c_str(),
           output_ports.c_str());
  }

  return STATUS_OK;
}

void ToolCommandDriver::DisplayFlowunit(std::shared_ptr<FlowUnitDesc> flow) {
  auto driverdesc = flow->GetDriverDesc();
  printf("--------------------------------------\n");
  printf("flowunit name\t: %s\n", flow->GetFlowUnitName().c_str());
  printf("type\t\t: %s\n", driverdesc->GetType().c_str());
  printf("driver name\t: %s\n", driverdesc->GetName().c_str());
  printf("version\t\t: %s\n", driverdesc->GetVersion().c_str());
  printf("descryption\t: %s\n", flow->GetDescription().c_str());
  printf("group\t\t: %s\n",
         [&]() -> std::string {
           auto type = flow->GetGroupType();
           if (type.empty()) {
             return "Generic";
           }

           return type;
         }()
                      .c_str());

  int index = 0;
  for (const auto &input : flow->GetFlowUnitInput()) {
    index++;
    if (index == 1) {
      printf("inputs\t\t:\n");
    }
    printf("  input index\t: %d\n", index);
    printf("    name\t: %s\n", input.GetPortName().c_str());
    printf("    type\t: %s\n", input.GetPortType().c_str());
    printf("    device\t: %s\n", input.GetDeviceType().c_str());
  }

  index = 0;
  for (const auto &output : flow->GetFlowUnitOutput()) {
    index++;
    if (index == 1) {
      printf("outputs\t\t:\n");
    }
    printf("  output index\t: %d\n", index);
    printf("    name\t: %s\n", output.GetPortName().c_str());
    printf("    device\t: %s\n", output.GetDeviceType().c_str());
  }

  index = 0;
  for (auto &option : flow->GetFlowUnitOption()) {
    index++;
    if (index == 1) {
      printf("options\t\t:\n");
    }
    printf("  option\t: %d\n", index);
    printf("    name\t: %s\n", option.GetOptionName().c_str());
    printf("    default\t: %s\n", option.GetOptionDefault().c_str());
    printf("    desc\t: %s\n", option.GetOptionDesc().c_str());
    printf("    required\t: %s\n", option.IsRequire() ? "true" : "false");
    printf("    type\t: %s\n", option.GetOptionType().c_str());
    auto values = option.GetOptionValues();
    if (values.size() > 0) {
      nlohmann::json json_values;
      for (const auto &value : values) {
        printf("        %s\t: %s\n", value.first.c_str(), value.second.c_str());
      }
    }
  }
  printf("\n");
}

Status ToolCommandDriver::DisplayFlowunitInDetails(
    std::shared_ptr<Configuration> config, const std::string &filter_name) {
  auto flowunit_info = std::make_shared<FlowUnitInfo>();
  auto status = flowunit_info->Init(config);
  if (!status) {
    std::cerr << status << std::endl;
    return status;
  }

  DisplayFlowunitByFilter(flowunit_info, filter_name);

  return STATUS_OK;
}

Status ToolCommandDriver::DisplayFlowunitInJson(
    std::shared_ptr<Configuration> config) {
  DisplayDriverInJson(config);
  return STATUS_OK;
}

}  // namespace modelbox