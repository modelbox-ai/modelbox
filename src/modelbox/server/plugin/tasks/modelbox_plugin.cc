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

#include "modelbox_plugin.h"

#include <dirent.h>

#include <nlohmann/json.hpp>
#include <toml.hpp>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

using namespace modelbox;

const std::string SERVER_PATH = "/v1/modelbox/job";
constexpr const char* GRAPH_DISABLED_FLAG = "DISABLED_";
constexpr int MAX_FILES = 1 << 16;

std::map<std::string, std::string> ERROR_INFO = {
    {"MODELBOX_001", "server internal error"},
    {"MODELBOX_002", "request invalid, no such job"},
    {"MODELBOX_003", "request invalid, can not get jobId"},
    {"MODELBOX_004", "request invalid, can not get graph"},
    {"MODELBOX_005", "request invalid, job already exist"},
    {"MODELBOX_006", "request invalid, invalid command"},
    {"MODELBOX_007",
     "job id contain invalid characters or contains more than 64 "
     "characters, please rename job with valid "
     "charaters."}};

const std::string ERROR_CODE = "error_code";
const std::string ERROR_MSG = "error_msg";
const char* HTTP_GRAPH_FORMAT_JSON = "json";
const char* HTTP_GRAPH_FORMAT_TOML = "toml";

bool ModelboxPlugin::Init(std::shared_ptr<modelbox::Configuration> config) {
  MBLOG_INFO << "modelbox plugin init";

  bool ret = ParseConfig(config);
  if (!ret) {
    MBLOG_ERROR << "parse config file failed";
    return false;
  }

  auto endpoint = "http://" + ip_ + ":" + port_;
  listener_ = std::make_shared<modelbox::HttpListener>(endpoint);
  MBLOG_INFO << "run modelbox plugin on " << endpoint;
  RegistHandlers();

  return CreateLocalJobs();
}

std::shared_ptr<Plugin> CreatePlugin() {
  MBLOG_INFO << "create modelbox plugin";
  return std::make_shared<ModelboxPlugin>();
}

void ModelboxPlugin::RegistHandlers() {
  MBLOG_INFO << "modelbox plugin regist handlers";
  MBLOG_INFO << "regist url : " << SERVER_PATH;

  listener_->Register(SERVER_PATH, HttpMethods::PUT,
                      std::bind(&ModelboxPlugin::HandlerPut, this,
                                std::placeholders::_1, std::placeholders::_2));

  listener_->Register(SERVER_PATH, HttpMethods::DELETE,
                      std::bind(&ModelboxPlugin::HandlerDel, this,
                                std::placeholders::_1, std::placeholders::_2));

  listener_->Register(SERVER_PATH, HttpMethods::GET,
                      std::bind(&ModelboxPlugin::HandlerGet, this,
                                std::placeholders::_1, std::placeholders::_2));
}

bool ModelboxPlugin::Start() {
  listener_->SetAclWhiteList(acl_white_list_);
  listener_->Start();

  auto ret = listener_->GetStatus();
  if (!ret) {
    MBLOG_ERROR << "Start modelbox plugin failed, err " << ret;
    return false;
  }

  return true;
}

bool ModelboxPlugin::Stop() {
  listener_->Stop();

  return true;
}

bool ModelboxPlugin::CheckJobIdValid(std::string job_id) {
  constexpr int max_id_length = 64;
  const std::string valid_char =
      "^[0-9a-zA-Z\\-\\+\\~\\_][0-9a-zA-Z\\-\\+\\~\\_\\.]+";
  std::regex valid_str(valid_char);

  if (job_id.length() > max_id_length) {
    return false;
  }

  return std::regex_match(job_id, valid_str);
}

bool ModelboxPlugin::ParseConfig(
    std::shared_ptr<modelbox::Configuration> config) {
  ip_ = config->GetString("server.ip");
  if (ip_.length() <= 0) {
    MBLOG_ERROR << "can not find ip from config file";
    return false;
  }

  port_ = config->GetString("server.port");
  if (port_.length() <= 0) {
    MBLOG_ERROR << "can not find port from config file";
    return false;
  }

  default_flow_path_ = config->GetString("server.flow_path");
  if (default_flow_path_.length() <= 0) {
    MBLOG_ERROR << "can not find flow from config file";
    return false;
  }

  acl_white_list_ = config->GetStrings("acl.allow");

  oneshot_flow_path_ = default_flow_path_ + "/oneshot";

  return true;
}

modelbox::Status ModelboxPlugin::CreateLocalJobs() {
  MBLOG_INFO << "create local job";
  std::vector<std::string> files;
  auto ret = modelbox::ListFiles(default_flow_path_, "*", &files,
                                 modelbox::LIST_FILES_FILE);
  if (!ret) {
    return ret;
  }

  for (auto& file : files) {
    // do not check return
    auto jobname = modelbox::GetBaseName(file);
    if (jobname.length() == 0) {
      continue;
    }

    std::string job_id = jobname;
    if (job_id.find(GRAPH_DISABLED_FLAG) != std::string::npos) {
      MBLOG_INFO << "graph '" << file << "' is disabled.";
      continue;
    }

    MBLOG_INFO << "Create local job " << file;
    ret = CreateJobByFile(job_id, file);
    if (!ret) {
      MBLOG_WARN << "create job " << file << " failed, " << ret;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelboxPlugin::SaveGraphFile(const std::string& job_id,
                                               const std::string& toml_graph) {
  auto ret = modelbox::CreateDirectory(oneshot_flow_path_);
  if (!ret) {
    return {modelbox::STATUS_FAULT,
            std::string("create graph directory failed, ") +
                modelbox::StrError(errno) + ", path: " + oneshot_flow_path_};
  }

  std::vector<std::string> list_files;
  ret = modelbox::ListSubDirectoryFiles(oneshot_flow_path_, "*", &list_files);
  if (!ret) {
    return {modelbox::STATUS_FAULT,
            std::string("list subdirectoryfiles failed, ") +
                modelbox::StrError(errno) + ", path: " + oneshot_flow_path_};
  }

  while (list_files.size() > MAX_FILES) {
    size_t earliest_file_index = modelbox::FindTheEarliestFileIndex(list_files);
    auto& earliest_file_path = list_files[earliest_file_index];
    MBLOG_WARN << "the graph file nums is more than " << MAX_FILES
               << ", remove the earliest access one, path: "
               << earliest_file_path;
    auto ret = remove(earliest_file_path.c_str());
    if (ret) {
      return {modelbox::STATUS_FAULT,
              std::string("remove earlier access file failed, ") +
                  modelbox::StrError(errno)};
    }
    list_files.erase(list_files.begin() + earliest_file_index);
  }

  std::string path = oneshot_flow_path_ + "/" + job_id;
  std::ofstream out(path, std::ios::trunc);
  if (out.fail()) {
    return {modelbox::STATUS_FAULT, std::string("save graph file failed, ") +
                                        modelbox::StrError(errno) +
                                        ", path: " + path};
  }

  chmod(path.c_str(), 0600);
  Defer { out.close(); };

  out << toml_graph;
  if (out.fail()) {
    return {modelbox::STATUS_FAULT, std::string("save graph file failed, ") +
                                        modelbox::StrError(errno) +
                                        ", path: " + path};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelboxPlugin::StartJob(std::shared_ptr<modelbox::Job> job) {
  auto ret = job->Init();
  if (!ret) {
    MBLOG_ERROR << "start job init failed:" << ret;
    return ret;
  }

  ret = job->Build();
  if (!ret) {
    MBLOG_ERROR << "start job build failed:" << ret;
    return ret;
  }

  job->Run();

  return modelbox::STATUS_OK;
}

modelbox::Status ModelboxPlugin::CreateJobByFile(
    const std::string& job_id, const std::string& graph_file) {
  auto job = jobmanager_.CreateJob(job_id, graph_file);
  if (job == nullptr) {
    MBLOG_ERROR << "create job " << job_id << " from " << graph_file
                << "failed.";
    return modelbox::StatusError;
  }

  auto ret = StartJob(job);
  if (!ret) {
    MBLOG_ERROR << "create job " << job_id << " from file failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelboxPlugin::CreateJobByString(const std::string& job_id,
                                                   const std::string& graph,
                                                   const std::string& format) {
  std::string toml_data;
  modelbox::Status ret;
  if (format == HTTP_GRAPH_FORMAT_TOML || format.length() == 0) {
    toml_data = graph;
  } else if (format == HTTP_GRAPH_FORMAT_JSON) {
    if (modelbox::JsonToToml(graph, &toml_data) == false) {
      return {modelbox::STATUS_INVALID, "graph data is invalid."};
    }
  } else {
    return {modelbox::STATUS_INVALID, "graph type:" + format + " is invalid."};
  }

  auto job = jobmanager_.CreateJob(job_id, job_id, toml_data);
  if (job == nullptr) {
    return modelbox::StatusError;
  }
  Defer {
    if (!ret) {
      job->SetError(ret);
    }
  };

  ret = SaveGraphFile(job_id, toml_data);
  if (!ret) {
    return ret;
  }

  ret = StartJob(job);
  if (!ret) {
    MBLOG_ERROR << "create job " << job_id << " from string failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

std::string BuildErrorResponse(const std::string& error,
                               const std::string& msg = "") {
  nlohmann::json response;
  std::string errmsg = ERROR_INFO[error];
  if (msg.length() > 0) {
    errmsg += ", " + msg;
  }

  response[ERROR_CODE] = error;
  response[ERROR_MSG] = errmsg;

  return response.dump();
}

void ModelboxPlugin::HandlerPut(const httplib::Request& request,
                                httplib::Response& response) {
  std::string graph_format = HTTP_GRAPH_FORMAT_JSON;
  std::string error_code = "MODELBOX_001";
  std::string error_msg;
  AddSafeHeader(response);
  bool is_failed = true;
  Defer {
    if (is_failed == false) {
      return;
    }

    MBLOG_ERROR << "Create task failed, " << error_msg;
    const auto& response_content =
        BuildErrorResponse(error_code.c_str(), error_msg);
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(response_content, JSON);
  };

  try {
    nlohmann::json body;
    try {
      body = nlohmann::json::parse(request.body);
    } catch (const std::exception& e) {
      error_code = "MODELBOX_006";
      MBLOG_ERROR << "process request failed, " << e.what();
      error_msg = e.what();
      return;
    }

    if (body.find("job_id") == body.end()) {
      error_code = "MODELBOX_003";
      error_msg = ERROR_INFO[error_code];
      return;
    }

    auto jobid = body["job_id"].get<std::string>();
    if (body.find("job_graph") == body.end()) {
      error_code = "MODELBOX_004";
      error_msg = ERROR_INFO[error_code];
      return;
    }

    if (!CheckJobIdValid(jobid)) {
      error_code = "MODELBOX_007";
      error_msg = ERROR_INFO[error_code];
      return;
    }

    if (body.find("job_graph_format") != body.end()) {
      graph_format = body["job_graph_format"].get<std::string>();
    }

    std::string graph_data;
    if (graph_format == HTTP_GRAPH_FORMAT_JSON) {
      graph_data = body["job_graph"].dump();
    } else if (graph_format == HTTP_GRAPH_FORMAT_TOML) {
      graph_data = body["job_graph"].get<std::string>();
    } else {
      error_msg = "Unsupport graph format: " + graph_format;
      error_code = "MODELBOX_006";
      return;
    }

    if (modelbox::JobStatus::JOB_STATUS_NOTEXIST !=
        jobmanager_.QueryJobStatus(jobid)) {
      error_code = "MODELBOX_005";
      error_msg = ERROR_INFO[error_code];
      return;
    }

    auto status = CreateJobByString(jobid, graph_data, graph_format);
    if (!status) {
      error_code = "MODELBOX_001";
      error_msg = status.WrapErrormsgs();
      return;
    }

    is_failed = false;
  } catch (const std::exception& e) {
    MBLOG_ERROR << "process request failed, " << e.what();
    error_msg = e.what();
    return;
  }

  response.status = HttpStatusCodes::CREATED;
  return;
}

void ModelboxPlugin::HandlerGet(const httplib::Request& request,
                                httplib::Response& response) {
  AddSafeHeader(response);
  try {
    std::string relative_path = request.path.substr(SERVER_PATH.size());
    std::string pre_path;
    std::string job_id;
    SplitPath(relative_path, pre_path, job_id);
    if (pre_path.empty()) {
      if (modelbox::JobStatus::JOB_STATUS_NOTEXIST ==
          jobmanager_.QueryJobStatus(job_id)) {
        const auto& response_content = BuildErrorResponse("MODELBOX_002");
        response.status = HttpStatusCodes::NOT_FOUND;
        response.set_content(response_content, JSON);
        return;
      }

      auto job_status = jobmanager_.QueryJobStatusString(job_id);
      auto job_msg = jobmanager_.GetJobErrorMsg(job_id);
      nlohmann::json response_json;
      response_json["job_id"] = job_id;
      response_json["job_status"] = job_status;
      response_json["job_error_msg"] = job_msg;
      response.status = HttpStatusCodes::OK;
      response.set_content(response_json.dump(), JSON);
      return;
    }

    if (pre_path == "/list" && job_id == "all") {
      nlohmann::json response_json;
      response_json["job_list"] = nlohmann::json::array();
      auto jobs = jobmanager_.GetJobMap();
      for (const auto& job : jobs) {
        nlohmann::json job_state;
        auto job_id = job.first;
        auto job_status = jobmanager_.QueryJobStatusString(job_id);
        auto job_msg = jobmanager_.GetJobErrorMsg(job_id);
        job_state["job_id"] = job_id;
        job_state["job_status"] = job_status;
        job_state["job_error_msg"] = job_msg;
        response_json["job_list"].push_back(job_state);
      }

      response.status = HttpStatusCodes::OK;
      response.set_content(response_json.dump(), JSON);
      return;
    }

    const auto& response_content = BuildErrorResponse("MODELBOX_006");
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(response_content, JSON);
    return;
  } catch (const std::exception& e) {
    const auto& response_content = BuildErrorResponse("MODELBOX_001");
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(response_content, JSON);
    return;
  }

  return;
}

void ModelboxPlugin::HandlerDel(const httplib::Request& request,
                                httplib::Response& response) {
  AddSafeHeader(response);
  try {
    std::string relative_path = request.path.substr(SERVER_PATH.size());
    std::string pre_path;
    std::string job_id;
    SplitPath(relative_path, pre_path, job_id);
    if (modelbox::JobStatus::JOB_STATUS_NOTEXIST ==
        jobmanager_.QueryJobStatus(job_id)) {
      const auto& response_content = BuildErrorResponse("MODELBOX_002");
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(response_content, JSON);
      return;
    }

    auto job = jobmanager_.GetJob(job_id);
    job->Stop();
    bool ret = jobmanager_.DeleteJob(job_id);
    if (!ret) {
      const auto& response_content = BuildErrorResponse("MODELBOX_002");
      response.status = HttpStatusCodes::BAD_REQUEST;
      response.set_content(response_content, JSON);
    }
  } catch (const std::exception& e) {
    const auto& response_content = BuildErrorResponse("MODELBOX_001");
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(response_content, JSON);
    return;
  }

  response.status = HttpStatusCodes::NO_CONTENT;
  return;
}