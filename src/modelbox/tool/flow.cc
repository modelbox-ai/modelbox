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


#include "flow.h"

#include <modelbox/modelbox.h>
#include <modelbox/flow.h>
#include <getopt.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace modelbox {

REG_MODELBOX_TOOL_COMMAND(ToolCommandFlow)

enum MODELBOX_TOOL_FLOW_COMMAND {
  MODELBOX_TOOL_FLOW_RUN,
  MODELBOX_TOOL_FLOW_CONF_CONVERT,
};

enum MODELBOX_TOOL_FLOW_CONVERT_COMMAND {
  MODELBOX_TOOL_FLOW_CONVERT_COMMAND_PATH,
  MODELBOX_TOOL_FLOW_CONVERT_COMMAND_OUTFORMAT,
};

static struct option flow_convert_options[] = {
    {"path", 1, nullptr, MODELBOX_TOOL_FLOW_CONVERT_COMMAND_PATH},
    {"out-format", 1, nullptr, MODELBOX_TOOL_FLOW_CONVERT_COMMAND_OUTFORMAT},
    {nullptr, 0, nullptr, 0},
};
static struct option flow_options[] = {
    {"run", 1, nullptr, MODELBOX_TOOL_FLOW_RUN},
    {"conf-convert", 0, nullptr, MODELBOX_TOOL_FLOW_CONF_CONVERT},
    {nullptr, 0, nullptr, 0},
};

ToolCommandFlow::ToolCommandFlow() = default;
ToolCommandFlow::~ToolCommandFlow() = default;

std::string ToolCommandFlow::GetHelp() {
  char help[] =
      " option:\n"
      "   -run [toml file]          run flow from file\n"
      "   -conf-convert             convert graph file format to json or "
      "toml\n"
      "     -path [conf file]       graph file path\n"
      "     -out-format [json|toml| output format, default is toml\n"
      "\n";
  return help;
}

int ToolCommandFlow::Run(int argc, char *argv[]) {
  int cmdtype = 0;
  int ret = -1;

  if (argc == 1) {
    std::cerr << GetHelp();
    return 1;
  }

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, flow_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_FLOW_RUN:
      return RunFlow(optarg);
    case MODELBOX_TOOL_FLOW_CONF_CONVERT:
      optind = 1;
      MODELBOX_COMMAND_SUB_UNLOCK();
      return RunConfConvertCommand(MODELBOX_COMMAND_SUB_ARGC,
                                   MODELBOX_COMMAND_SUB_ARGV);
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return ret;
}

int ToolCommandFlow::RunFlow(const std::string &file) {
  auto flow = std::make_shared<modelbox::Flow>();
  MBLOG_INFO << "run flow " << file;
  auto ret = flow->Init(file);
  if (!ret) {
    MBLOG_ERROR << "init flow failed, " << ret.WrapErrormsgs();
    return 1;
  }

  ret = flow->Build();
  if (!ret) {
    MBLOG_ERROR << "build flow failed, " << ret.WrapErrormsgs();
    return 1;
  }

  flow->RunAsync();

  ret = flow->Wait();
  if (!ret) {
    MBLOG_ERROR << "run flow failed, " << ret.WrapErrormsgs();
    return 1;
  }

  flow->Stop();
  MBLOG_INFO << "run flow " << file << " success";

  return 0;
}

int ToolCommandFlow::RunConfConvertCommand(int argc, char *argv[]) {
  int cmdtype = 0;
  ConfigurationBuilder config_builder;
  std::shared_ptr<Configuration> config_merge;
  std::string path;
  std::string format = "toml";
  std::string out_result;
  modelbox::Status ret;

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, flow_convert_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_FLOW_CONVERT_COMMAND_PATH:
      path = optarg;
      break;
    case MODELBOX_TOOL_FLOW_CONVERT_COMMAND_OUTFORMAT:
      format = optarg;
      break;
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()
  if (path.length() == 0 || format.length() == 0) {
    std::cerr << "please input conf file path and format." << std::endl;
    return 1;
  }

  std::ifstream infile(path);
  if (infile.fail()) {
    std::cerr << "read file " << path << " failed, " << modelbox::StrError(errno)
              << std::endl;
    return 1;
  }

  Defer { infile.close(); };
  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());

  if (format == "json") {
    ret = modelbox::TomlToJson(data, &out_result, true);
  } else if (format == "toml") {
    ret = modelbox::JsonToToml(data, &out_result);
  } else {
    std::cerr << "output format is not supported" << std::endl;
    return 1;
  }

  if (!ret) {
    std::cerr << "convert failed, " << ret.WrapErrormsgs() << std::endl;
    return 1;
  }

  std::cout << out_result << std::endl;

  return 0;
}  // namespace modelbox

}  // namespace modelbox