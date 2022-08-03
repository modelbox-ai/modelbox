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

#include "modelbox/flow.h"

#include <fstream>
#include <utility>

#include "modelbox/base/timer.h"

namespace modelbox {

Status FlowSetupLog(const std::shared_ptr<Configuration>& config) {
  if (!config) {
    return {STATUS_INVALID, "config is nullptr."};
  }

  auto ret = config->GetSubConfig("log");
  if (!StatusError) {
    return StatusError;
  }

  auto str_level = ret->GetString("level", "");
  if (str_level.length() == 0) {
    return {STATUS_NOTFOUND, "Config log level not found."};
  }

  auto level = LogLevelStrToLevel(str_level);
  if (level == LOG_OFF && !StatusError) {
    return {StatusError};
  }

  ModelBoxLogger.GetLogger()->SetLogLevel(level);

  return STATUS_OK;
}

Flow::Flow() = default;

Flow::~Flow() { Clear(); };

void Flow::Clear() {
  if (graph_) {
    graph_->Shutdown();
  }
  graph_ = nullptr;
  graphconfig_ = nullptr;
  flowunit_mgr_ = nullptr;
  device_mgr_ = nullptr;
  graphconf_mgr_ = nullptr;
  profiler_ = nullptr;
  args_ = nullptr;
  if (drivers_) {
    drivers_->Clear();
  }

  if (timer_run_) {
    TimerGlobal::Stop();
    timer_run_ = false;
  }
}

Status Flow::GetInputArgs(
    std::shared_ptr<Configuration>& config,
    const std::unordered_map<std::string, std::string>& input_args) {
  // get default args
  auto sub_keys = config->GetSubKeys("args");
  if (!sub_keys.empty()) {
    auto args_config = config->GetSubConfig("args");
    args_ = std::make_shared<std::unordered_map<std::string, std::string>>();
    for (const auto& key : sub_keys) {
      auto value = args_config->GetSubConfig(key)->GetString("default");
      args_->insert({key, value});
    }
  }

  // get final args
  for (const auto& input_arg : input_args) {
    if (args_ == nullptr || args_->find(input_arg.first) == args_->end()) {
      MBLOG_ERROR << "input args key:" << input_arg.first
                  << " is not in the flow.";
      return STATUS_FAULT;
    }

    MBLOG_INFO << "input args key:" << input_arg.first << " use custom value.";
    (*args_)[input_arg.first] = input_arg.second;
  }

  return STATUS_OK;
}

Status Flow::InitByName(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& args,
    const std::string& flow_dir) {
  Status ret;
  std::string flow_path;
  auto status = GetGraphFilePathByName(name, flow_dir, flow_path);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "failed find toml file, errmsg:" << status.Errormsg();
    return status;
  }

  std::shared_ptr<Configuration> config;
  ret = GetConfigByGraphFile(flow_path, config, FORMAT_AUTO);
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "read config from  toml:" << flow_path
                << "failed, err :" << ret.Errormsg();
    return ret;
  }

  ret = GetInputArgs(config, args);
  if (ret != STATUS_OK) {
    return ret;
  }

  return Init(config);
}

Status Flow::Init(const std::shared_ptr<FlowGraphDesc>& flow_graph_desc) {
  auto status = flow_graph_desc->GetStatus();
  if (status != STATUS_OK) {
    MBLOG_ERROR << "graph desc has error, ret " << status;
    return status;
  }

  graph_ = std::make_shared<Graph>();
  config_ = flow_graph_desc->GetConfig();
  profiler_ = std::make_shared<Profiler>(device_mgr_, config_);

  device_mgr_ = flow_graph_desc->GetDeviceManager();
  flowunit_mgr_ = flow_graph_desc->GetFlowUnitManager();
  gcgraph_ = flow_graph_desc->GetGCGraph();

  auto ret = profiler_->Init();
  if (!ret) {
    MBLOG_ERROR << "Initial profiler failed, " << ret.WrapErrormsgs();
    return {ret, "Initial profiler failed."};
  }

  ret = graph_->Initialize(flowunit_mgr_, device_mgr_, profiler_, config_);
  if (!ret) {
    MBLOG_ERROR << "Initial graph failed, " << ret.WrapErrormsgs();
    return {ret, "Initial graph failed."};
  }

  return STATUS_OK;
}

Status Flow::StartRun() {
  auto ret = Build();
  if (!ret) {
    return ret;
  }

  ret = RunAsync();
  if (!ret) {
    return ret;
  }

  return STATUS_OK;
}

Status Flow::Init(std::shared_ptr<Configuration> config) {
  config_ = std::move(config);
  drivers_ = std::make_shared<Drivers>();
  device_mgr_ = std::make_shared<DeviceManager>();
  flowunit_mgr_ = std::make_shared<FlowUnitManager>();
  graphconf_mgr_ = std::make_shared<GraphConfigManager>();
  profiler_ = std::make_shared<Profiler>(device_mgr_, config_);
  graph_ = std::make_shared<Graph>();
  graphconfig_ = nullptr;

  if (config_ == nullptr) {
    return {STATUS_INVALID, "Load config failed, config is invalid."};
  }

  if (config_->GetBool("flow.enable", true) == false) {
    return {STATUS_PERMIT, "flow is disabled"};
  }

  FlowSetupLog(config_);

  auto ret = drivers_->Initialize(config_->GetSubConfig("driver"));
  if (!ret) {
    MBLOG_ERROR << "driver init failed, " << ret.WrapErrormsgs();
    return {ret, "driver init failed."};
  }

  Defer {
    if (ret == STATUS_OK) {
      return;
    }

    Clear();
  };

  ret = drivers_->Scan();
  if (!ret) {
    MBLOG_ERROR << "Scan driver failed, " << ret.WrapErrormsgs();
    return {ret, "Scan driver failed."};
  }

  TimerGlobal::Start();
  timer_run_ = true;

  ret = graphconf_mgr_->Initialize(drivers_, config_);
  if (!ret) {
    MBLOG_ERROR << "Init graph config failed, " << ret.WrapErrormsgs();
    return {ret, "Init graph config failed."};
  }

  graphconfig_ = graphconf_mgr_->LoadGraphConfig(config_);
  if (graphconfig_ == nullptr) {
    MBLOG_ERROR << "Load graph config failed";
    return {StatusError, "load graph failed."};
  }

  ret = device_mgr_->Initialize(drivers_, config_);
  if (!ret) {
    MBLOG_ERROR << "Inital device failed, " << ret.WrapErrormsgs();
    return {ret, "Inital device failed."};
  }

  ret = flowunit_mgr_->Initialize(drivers_, device_mgr_, config_);
  if (!ret) {
    MBLOG_ERROR << "Initial flowunit manager failed, " << ret.WrapErrormsgs();
    return {ret, "Initial flowunit manager failed."};
  }

  ret = profiler_->Init();
  if (!ret) {
    MBLOG_ERROR << "Initial profiler failed, " << ret.WrapErrormsgs();
    return {ret, "Initial profiler failed."};
  }

  ret = graph_->Initialize(flowunit_mgr_, device_mgr_, profiler_, config_);
  if (!ret) {
    MBLOG_ERROR << "Initial graph failed, " << ret.WrapErrormsgs();
    return {ret, "Initial graph failed."};
  }

  return STATUS_OK;
}

Status Flow::GuessConfFormat(const std::string& configfile,
                             const std::string& data, enum Format* format) {
  *format = FORMAT_UNKNOWN;
  std::string extension = configfile.substr(configfile.find_last_of('.') + 1);
  if (extension == "json") {
    *format = FORMAT_JSON;
    return STATUS_OK;
  }

  if (extension == "toml") {
    *format = FORMAT_TOML;
    return STATUS_OK;
  }

  if (data.length() <= 0) {
    return {STATUS_NOTSUPPORT, "unknown file format"};
  }

  size_t i = 0;
  for (i = 0; i < data.length(); i++) {
    if (data[i] != ' ' && data[i] != '\t' && data[i] != '\n' &&
        data[i] != '\r') {
      break;
    }
  }

  if (i == data.length()) {
    i = data.length() - 1;
  }

  if (data[i] == '{') {
    *format = FORMAT_JSON;
  } else {
    *format = FORMAT_TOML;
  }

  if (*format == FORMAT_UNKNOWN) {
    return {STATUS_NOTSUPPORT, "unknown file format"};
  }

  return STATUS_OK;
}

Status Flow::ConfigFileRead(const std::string& configfile, Format format,
                            std::istringstream* ifs) {
  Status ret;
  std::string toml_data;
  std::string file_content;
  std::ifstream infile(configfile);
  Format config_format = format;

  if (infile.fail()) {
    std::string msg = "read file " + configfile + " failed, " + StrError(errno);
    return {STATUS_BADCONF, msg};
  }

  Defer { infile.close(); };
  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());

  if (config_format == FORMAT_AUTO) {
    ret = GuessConfFormat(configfile, data, &config_format);
    if (!ret) {
      return {ret, "unsupport file format"};
    }
  }

  if (config_format == FORMAT_JSON) {
    ret = JsonToToml(data, &toml_data);
    if (!ret) {
      return {ret, "json file is invalid."};
    }
  } else if (config_format == FORMAT_TOML) {
    toml_data = data;
  } else {
    return {STATUS_NOTSUPPORT, "unsupport file format"};
  }

  ifs->str(toml_data);
  return STATUS_OK;
}

Status Flow::Init(const std::string& name, const std::string& graph,
                  Format format) {
  Status ret;
  ConfigurationBuilder config_builder;
  std::shared_ptr<Configuration> config;
  std::istringstream ifs;

  if (graph.length() <= 0) {
    return {STATUS_NOTSUPPORT, "unknown file format"};
  }

  if (graph[0] == '{') {
    std::string toml_data;
    ret = JsonToToml(graph, &toml_data);
    if (!ret) {
      return {ret, "invalid graph"};
    }
    ifs.str(toml_data);
  } else {
    ifs.str(graph);
  }

  config = config_builder.Build(ifs, name);
  if (config == nullptr) {
    return {StatusError,
            "Load config file failed, detail: " + StatusError.Errormsg()};
  }

  ret = Init(config);
  if (!ret) {
    MBLOG_WARN << "Init failed, graph: " << name << " status: " << ret;
  }

  return ret;
}

Status Flow::GetGraphFilePathByName(const std::string& flow_name,
                                    const std::string& graph_dir,
                                    std::string& graph_path) {
  std::vector<std::string> toml_list;
  std::string filter = "*.toml";
  auto status = ListSubDirectoryFiles(graph_dir, filter, &toml_list);
  if (status != STATUS_OK) {
    MBLOG_WARN << "find " << flow_name << " toml file in directory "
               << graph_dir << " failed.";
  }

  filter = "*.json";
  std::vector<std::string> json_list;
  status = ListSubDirectoryFiles(graph_dir, filter, &json_list);
  if (status != STATUS_OK) {
    MBLOG_WARN << "find " << flow_name << " json file in directory "
               << graph_dir << " failed.";
  }

  toml_list.insert(toml_list.end(), json_list.begin(), json_list.end());
  if (toml_list.empty()) {
    std::string err_msg =
        "there is no graph file named " + flow_name + " in " + graph_dir;
    return {STATUS_NOTFOUND, err_msg};
  }

  for (const auto& iter : toml_list) {
    std::shared_ptr<Configuration> config;
    auto status = GetConfigByGraphFile(iter, config, Flow::FORMAT_AUTO);
    if (status != STATUS_OK) {
      continue;
    }

    if (config->GetString("flow.name") == flow_name) {
      graph_path = iter;
      MBLOG_DEBUG << "found flow name: " << config->GetString("flow.name")
                  << ", toml path = " << iter;
      return STATUS_OK;
    }
  }

  return {STATUS_NOTFOUND,
          "failed find flow name:" + flow_name + "'s toml file"};
}

Status Flow::GetConfigByGraphFile(const std::string& configfile,
                                  std::shared_ptr<Configuration>& config,
                                  Format format) {
  ConfigurationBuilder config_builder;
  std::istringstream ifs;

  auto ret = ConfigFileRead(configfile, format, &ifs);
  if (!ret) {
    return ret;
  }

  config = config_builder.Build(ifs, configfile);
  if (config == nullptr) {
    return {StatusError,
            "Load config file failed, detail: " + StatusError.Errormsg()};
  }
  return STATUS_OK;
}

Status Flow::Init(const std::string& configfile, Format format) {
  Status ret;
  std::shared_ptr<Configuration> config;

  ret = GetConfigByGraphFile(configfile, config, format);
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "read config from  toml:" << configfile
                << "failed, err :" << ret.Errormsg();
    return ret;
  }
  // TODO: Add args configuration
  ret = Init(config);
  if (!ret) {
    MBLOG_WARN << "Init failed, configfile: " << configfile
               << " status: " << ret;
  }

  return ret;
}

Status Flow::Init(std::istream& is, const std::string& fname) {
  ConfigurationBuilder config_builder;

  auto config = config_builder.Build(is, fname);
  if (config == nullptr) {
    return {StatusError,
            "Load config file failed, detail: " + StatusError.Errormsg()};
  }

  auto status = Init(config);
  if (!status) {
    MBLOG_WARN << "Init failed, configfile: " << fname << " status: " << status;
  }

  return status;
}

Status Flow::Build() {
  if (graph_ == nullptr || (graphconfig_ == nullptr && gcgraph_ == nullptr)) {
    return {STATUS_FAULT, "Flow not initialized."};
  }

  if (graphconfig_ != nullptr) {
    gcgraph_ = graphconfig_->Resolve();
    if (gcgraph_ == nullptr) {
      MBLOG_ERROR << "graph config resolve failed, "
                  << StatusError.WrapErrormsgs();
      return STATUS_FAULT;
    }
  }

  // update args
  if (args_ != nullptr && !args_->empty()) {
    auto nodes = gcgraph_->GetAllNodes();
    for (auto& node : nodes) {
      auto node_config = node.second->GetConfiguration();
      auto node_keys = node_config->GetKeys();
      for (const auto& key : node_keys) {
        auto value = node_config->GetString(key);
        auto value_name = value.substr(1, value.size());
        if (std::regex_match(value, std::regex("^\\$.*")) &&
            args_->find(value_name) != args_->end()) {
          node_config->SetProperty(key, (*args_)[value_name]);
        }
      }
    }
  }

  auto ret = graph_->Build(gcgraph_);
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "build graph failed, " << ret.WrapErrormsgs();
    return ret;
  }

  return STATUS_OK;
}

Status Flow::Run() {
  if (graph_ == nullptr) {
    return {STATUS_FAULT, "Flow not initialized."};
  }

  auto ret = graph_->Run();
  if (ret != STATUS_OK) {
    MBLOG_ERROR << "graph run failed, " << ret.WrapErrormsgs();
    return ret;
  }
  return STATUS_OK;
}

Status Flow::RunAsync() {
  if (graph_ == nullptr) {
    return {STATUS_FAULT, "Flow not initialized."};
  }

  graph_->RunAsync();
  return STATUS_OK;
}

Status Flow::Wait(int64_t millisecond, Status* ret_val) {
  if (graph_ == nullptr) {
    return {STATUS_INPROGRESS, "Flow not initialized."};
  }

  auto ret = graph_->Wait(millisecond, ret_val);
  if (ret != STATUS_OK) {
    if (ret == STATUS_BUSY || ret == STATUS_SHUTDOWN ||
        ret == STATUS_NORESPONSE) {
      return ret;
    }
    MBLOG_ERROR << "flow wait error, " << ret.WrapErrormsgs();
    return ret;
  }
  return STATUS_OK;
}

void Flow::Stop() {
  if (graph_ == nullptr) {
    MBLOG_ERROR << "Flow not initialized.";
    return;
  }
  graph_->Shutdown();

  graph_->Wait(1000);
}

std::shared_ptr<ExternalDataMap> Flow::CreateExternalDataMap() {
  if (graph_ == nullptr) {
    MBLOG_ERROR << "graph is nullptr";
    return nullptr;
  }
  return graph_->CreateExternalDataMap();
}

std::shared_ptr<FlowStreamIO> Flow::CreateStreamIO() {
  if (graph_ == nullptr) {
    MBLOG_ERROR << "graph is nullptr";
    return nullptr;
  }

  auto external_data_map = graph_->CreateExternalDataMap();
  if (external_data_map == nullptr) {
    MBLOG_ERROR << "create external data for graph failed";
    return nullptr;
  }

  return std::make_shared<FlowStreamIO>(external_data_map);
}

std::shared_ptr<Profiler> Flow::GetProfiler() { return profiler_; }

std::string Flow::GetGraphId() const {
  if (graph_ == nullptr) {
    MBLOG_ERROR << "graph is nullptr";
    return "";
  }

  return graph_->GetId();
}

std::string Flow::GetGraphName() const {
  if (graph_ == nullptr) {
    MBLOG_ERROR << "graph is nullptr";
    return "";
  }

  return graph_->GetName();
}
}  // namespace modelbox