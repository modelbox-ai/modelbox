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

#include "control-command.h"

#include <modelbox/base/log.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/utils.h>
#include <modelbox/common/log.h>

#include "modelbox/statistics.h"

namespace modelbox {

REG_MODELBOX_TOOL_COMMAND(ToolCommandLog)

enum MODELBOX_SERVER_COMMAND_LOG {
  MODELBOX_SERVER_COMMAND_LOG_GET,
  MODELBOX_SERVER_COMMAND_LOG_SET,
};

static struct option server_log_options[] = {
    {"getlevel", no_argument, nullptr, MODELBOX_SERVER_COMMAND_LOG_GET},
    {"setlevel", required_argument, nullptr, MODELBOX_SERVER_COMMAND_LOG_SET},
    {nullptr, 0, nullptr, 0},
};

ToolCommandLog::ToolCommandLog() = default;
ToolCommandLog::~ToolCommandLog() = default;

std::string ToolCommandLog::GetHelp() {
  char help[] =
      "option:\n"
      "  --getlevel          get current log level\n"
      "  --setlevel [level]  set server log level\n"
      "\n";
  return help;
}

int ToolCommandLog::Run(int argc, char *argv[]) {
  int cmdtype = 0;

  if (argc <= 1) {
    TOOL_COUT << GetHelp();
    return 0;
  }

  auto logger = ModelBoxLogger.GetLogger();

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, server_log_options)
  switch (cmdtype) {
    case MODELBOX_SERVER_COMMAND_LOG_GET:
      TOOL_COUT << "Log Level : "
                << modelbox::LogLevelToString(logger->GetLogLevel())
                << std::endl;
      return 0;
    case MODELBOX_SERVER_COMMAND_LOG_SET: {
      auto level = modelbox::LogLevelStrToLevel(optarg);
      if (modelbox::StatusError != modelbox::STATUS_OK) {
        TOOL_CERR << "Log level is invalid.";
        return 1;
      }
      TOOL_COUT << "Set Log Level : " << modelbox::LogLevelToString(level)
                << std::endl;
      logger->SetLogLevel(level);
      return 0;
    } break;
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return 0;
}

REG_MODELBOX_TOOL_COMMAND(ToolCommandSlab)

enum MODELBOX_SERVER_COMMAND_SLAB {
  MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_GET,
};

static struct option server_slab_options[] = {
    {"device", no_argument, nullptr,
     MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_GET},
    {nullptr, 0, nullptr, 0},
};

enum AFILOG_SERVER_COMMAND_SLAB_DEVICE {
  MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_TYPE,
  MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_ID,
};

static struct option server_slab_device_options[] = {
    {"type", required_argument, nullptr,
     MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_TYPE},
    {"id", required_argument, nullptr,
     MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_ID},
    {nullptr, 0, nullptr, 0},
};

constexpr const char *CPU_MEMPOOL_TYPE = "cpu";
constexpr const char *CUDA_MEMPOOL_TYPE = "cuda";
constexpr const char *ASCEND_MEMPOOL_TYPE = "ascend";

ToolCommandSlab::ToolCommandSlab() = default;
ToolCommandSlab::~ToolCommandSlab() = default;

std::string ToolCommandSlab::GetHelp() {
  char help[] =
      "option:\n"
      "  --device                               get all device slab info\n"
      "       --type [cpu|cuda]                 specified type. e.g. --device "
      "--type cpu\n"
      "       --id [0|1|..]                     specified id. e.g. --device "
      "--type cuda --id 0\n"
      "\n";
  return help;
}

bool ToolCommandSlab::GetMemPools(
    std::vector<std::shared_ptr<modelbox::MemoryPoolBase>> &mempools,
    const std::string &type, const std::string &id) {
  std::shared_ptr<modelbox::MemoryPoolBase> mempool;

  std::string name = type;
  if (id.length() > 0) {
    name += "-" + id;
  }

  for (auto &p : modelbox::MemoryPoolBase::GetAllPools()) {
    if (p->GetName().find(name) == std::string::npos) {
      continue;
    }

    mempools.emplace_back(p);
  }

  if (mempools.size() == 0) {
    return false;
  }

  return true;
}

void ToolCommandSlab::DisplaySlabInfo(
    std::shared_ptr<modelbox::MemoryPoolBase> &mem_pool,
    const std::string &type, const std::string &id) {
  if (mem_pool == nullptr) {
    return;
  }

  auto slabcaches = mem_pool->GetSlabCaches();
  uint64_t total_memory = 0;
  for (size_t i = 0; i < slabcaches.size(); ++i) {
    if (i == 0) {
      TOOL_COUT << "object size\t\tactive_objs\t\tnum_objects\n";
    }
    TOOL_COUT << modelbox::GetBytesReadable(slabcaches[i]->ObjectSize())
              << "\t\t\t" << slabcaches[i]->GetActiveObjNumber() << "\t\t\t"
              << slabcaches[i]->GetObjNumber() << "\n";
    total_memory += slabcaches[i]->ObjectSize() * slabcaches[i]->GetObjNumber();
  }
  TOOL_COUT << "name: " << mem_pool->GetName()
            << "    total_active_objects: " << mem_pool->GetAllActiveObjectNum()
            << "    total_objects: " << mem_pool->GetAllObjectNum()
            << "    total_memory: " << modelbox::GetBytesReadable(total_memory)
            << "\n\n";
}

bool ToolCommandSlab::DisplayMemPools(const std::string &type) {
  std::vector<std::shared_ptr<modelbox::MemoryPoolBase>> mem_pools;
  auto mem_pool_flag = GetMemPools(mem_pools, type);
  if (mem_pool_flag) {
    for (size_t i = 0; i < mem_pools.size(); ++i) {
      DisplaySlabInfo(mem_pools[i], type, std::to_string(i));
    }
  }
  return mem_pool_flag;
}

int ToolCommandSlab::DisplaySlabsInfo(const std::string &type) {
  bool mem_pool_flag = false;

  std::vector<std::string> types;
  if (type.empty()) {
    types = {CPU_MEMPOOL_TYPE, CUDA_MEMPOOL_TYPE, ASCEND_MEMPOOL_TYPE};
  } else {
    types.emplace_back(type);
  }

  for (const auto &item : types) {
    mem_pool_flag |= DisplayMemPools(item);
  }

  if (!mem_pool_flag) {
    if (type.empty()) {
      TOOL_CERR << "There is no memory pools here.\n";
    } else {
      TOOL_CERR << "There is no " << type << " memory pools here.\n";
    }
    return 1;
  }

  return 0;
}

int ToolCommandSlab::DeviceSlabInfo(const std::string &type,
                                    const std::string &id) {
  if (id.empty()) {
    return DisplaySlabsInfo(type);
  }

  if (type.empty()) {
    TOOL_CERR << "Your format is wrong , not allow only --id, please try "
                 "modelbox-tool sever slab "
                 "--device --type [cpu|cuda] --id [0|1|...]\n";
    return 1;
  }

  std::vector<std::shared_ptr<modelbox::MemoryPoolBase>> mem_pools;
  auto res = GetMemPools(mem_pools, type, id);
  if (!res) {
    TOOL_CERR << type << " " << id << " memory pool does not exist.\n";
    return 1;
  }
  DisplaySlabInfo(mem_pools[0], type, id);

  return 0;
}

int ToolCommandSlab::RunDeviceOption(int argc, char *argv[]) {
  int cmdtype = 0;
  std::string type;
  std::string id;
  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, server_slab_device_options)
  switch (cmdtype) {
    case MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_TYPE: {
      type = optarg;
      break;
    }
    case MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_ID: {
      id = optarg;
      break;
    }
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return DeviceSlabInfo(type, id);
}

int ToolCommandSlab::Run(int argc, char *argv[]) {
  int cmdtype = 0;

  if (argc <= 1) {
    TOOL_COUT << GetHelp();
    return 1;
  }

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, server_slab_options)
  switch (cmdtype) {
    case MODELBOX_SERVER_COMMAND_SLAB_INFO_DEVICE_GET: {
      MODELBOX_COMMAND_SUB_UNLOCK();
      return RunDeviceOption(MODELBOX_COMMAND_SUB_ARGC,
                             MODELBOX_COMMAND_SUB_ARGV);
    } break;
    default:
      return 1;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return 0;
}

REG_MODELBOX_TOOL_COMMAND(ToolCommandStatistics)

enum MODELBOX_SERVER_COMMAND_STATISTICS {
  MODELBOX_SERVER_COMMAND_STAT_All,
  MODELBOX_SERVER_COMMAND_STAT_NODE,
};

static struct option server_statistics_options[] = {
    {"all", no_argument, nullptr, MODELBOX_SERVER_COMMAND_STAT_All},
    {"node", required_argument, nullptr, MODELBOX_SERVER_COMMAND_STAT_NODE},
    {nullptr, 0, nullptr, 0},
};

ToolCommandStatistics::ToolCommandStatistics() = default;
ToolCommandStatistics::~ToolCommandStatistics() = default;

std::string ToolCommandStatistics::GetHelp() {
  char help[] =
      "option:\n"
      "  --all               get all info\n"
      "  --node [name]       get specific node info\n"
      "\n";
  return help;
}

int ToolCommandStatistics::Run(int argc, char *argv[]) {
  int cmdtype = 0;
  auto root = modelbox::Statistics::GetGlobalItem();

  if (argc <= 1) {
    TOOL_COUT << GetHelp();
    return 0;
  }

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, server_statistics_options)

  if (root == nullptr) {
    TOOL_CERR << "Root of Statistics have not been created";
    return 1;
  }

  switch (cmdtype) {
    case MODELBOX_SERVER_COMMAND_STAT_All: {
      TOOL_COUT << "Display All Info: " << std::endl;
      root->ForEach(
          [&](const std::shared_ptr<modelbox::StatisticsItem> &item,
              const std::string &relative_path) {
            auto value = item->GetValue();
            TOOL_COUT << item->GetPath()
                      << (value ? " = " + value->ToString() : "") << std::endl;
            return modelbox::STATUS_OK;
          },
          true);
    } break;

    case MODELBOX_SERVER_COMMAND_STAT_NODE: {
      auto *name = optarg;
      bool if_found = false;
      root->ForEach(
          [&](const std::shared_ptr<modelbox::StatisticsItem> &item,
              const std::string &relative_path) {
            if (item->GetName() == name) {
              auto value = item->GetValue();
              TOOL_COUT << item->GetPath()
                        << (value ? " = " + value->ToString() : "")
                        << std::endl;
              if_found = true;
            }
            return modelbox::STATUS_OK;
          },
          true);
      if (!if_found) {
        TOOL_COUT << name << " is not found." << std::endl;
      }
    } break;

    default:
      TOOL_COUT << GetHelp();
      return 1;
  }

  MODELBOX_COMMAND_GETOPT_END()

  return 0;
}

}  // namespace modelbox