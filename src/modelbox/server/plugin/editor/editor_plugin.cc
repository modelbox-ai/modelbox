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
#include "editor_plugin.h"

#include <modelbox/base/popen.h>
#include <stdlib.h>
#include <sys/wait.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <toml.hpp>
#include <typeinfo>
#include <vector>

#include "config.h"
#include "modelbox/base/log.h"
#include "modelbox/common/flowunit_info.h"
#include "modelbox/common/utils.h"
#include "modelbox/server/utils.h"

using namespace modelbox;

const std::string DEFAULT_WEB_ROOT = "/usr/local/share/modelbox/www";
const std::string DEFAULT_DEMO_GRAPHS_ROOT =
    std::string(MODELBOX_DEMO_PATH) + "/graphs";

const std::string UI_url = "/";
const std::string flowunit_info_url = "/editor/flow-info";
const std::string demo_url = "/editor/demo";
const std::string project_url = "/editor/project";
const std::string save_project_url = "/editor/graph";
const std::string flowunit_url = "/editor/flowunit";
const std::string search_url = "/editor/search";

const char* HTTP_GRAPH_FORMAT_JSON = "json";
const char* HTTP_GRAPH_FORMAT_TOML = "toml";

constexpr const char* HTTP_RESP_ERR_GETINFO_FAILED = "Get info failed";
constexpr const char* HTTP_RESP_ERR_PATH_NOT_FOUND = "Path not found";
constexpr const char* HTTP_RESP_ERR_PATH_NOT_FILE = "Path not a file";
constexpr const char* HTTP_RESP_ERR_CANNOT_READ = "Can not read file";
constexpr int MAX_FILES = 1 << 16;

const std::string ModelboxGetMimeType(const std::string& file) {
  std::string ext = file.substr(file.find_last_of(".") + 1);

  const static std::map<std::string, std::string> mime_map = {
      {"htm", "text/html"},         {"html", "text/html"},
      {"js", "text/javascript"},    {"css", "text/css"},
      {"json", "application/json"}, {"png", "image/png"},
      {"gif", "image/gif"},         {"jpeg", "image/jpeg"},
      {"svg", "image/svg+xml"},     {"tar", "application/x-tar"},
      {"txt", "text/plain"},        {"ico", "application/octet-stream"},
      {"xml", "text/xml"},          {"mpeg", "video/mpeg"},
      {"mp3", "audio/mpeg"},
  };

  auto itr = mime_map.find(ext);
  if (itr != mime_map.end()) {
    return itr->second;
  }

  return "application/octet-stream";
}

bool ModelboxEditorPlugin::CheckBlackDir(std::string dir) {
  if ((dir.find("\r") != dir.npos) || (dir.find("\n") != dir.npos)) {
    return true;
  }
  const std::string filter_dir =
      "(/bin)|(/boot)|(/sbin)|(/etc)|(/dev)|(/proc)|(/sys)|(/var)";
  const std::string black_dir_str =
      filter_dir + "|" + "((" + filter_dir + ")/.*)";
  std::regex invalid_str(black_dir_str);
  std::string path = modelbox::PathCanonicalize(dir);
  return std::regex_match(path, invalid_str);
}

bool ModelboxEditorPlugin::Init(
    std::shared_ptr<modelbox::Configuration> config) {
  MBLOG_INFO << "modelbox editor plugin init";

  bool ret = ParseConfig(config);
  if (!ret) {
    MBLOG_ERROR << "parse config file failed";
    return false;
  }

  if (enable_ == false) {
    MBLOG_INFO << "editor is disabled.";
    return true;
  }

  auto endpoint = "http://" + server_ip_ + ":" + server_port_;
  listener_ = std::make_shared<modelbox::HttpListener>(endpoint);
  MBLOG_INFO << "run editor on " << endpoint;
  RegistHandlers();

  return true;
}

std::shared_ptr<Plugin> CreatePlugin() {
  MBLOG_INFO << "create modelbox editor plugin";
  return std::make_shared<ModelboxEditorPlugin>();
}

void ModelboxEditorPlugin::RegistHandlers() {
  listener_->Register(UI_url, HttpMethods::GET,
                      std::bind(&ModelboxEditorPlugin::HandlerUIGet, this,
                                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(
      flowunit_info_url, HttpMethods::PUT,
      std::bind(&ModelboxEditorPlugin::HandlerFlowUnitInfoPut, this,
                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(
      flowunit_info_url, HttpMethods::GET,
      std::bind(&ModelboxEditorPlugin::HandlerFlowUnitInfoGet, this,
                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(demo_url, HttpMethods::GET,
                      std::bind(&ModelboxEditorPlugin::HandlerDemoGet, this,
                                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(project_url, HttpMethods::GET,
                      std::bind(&ModelboxEditorPlugin::HandlerProjectGet, this,
                                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(
      search_url, HttpMethods::GET,
      std::bind(&ModelboxEditorPlugin::HandlerDirectoryGet, this,
                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(project_url, HttpMethods::PUT,
                      std::bind(&ModelboxEditorPlugin::HandlerProjectPut, this,
                                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(flowunit_url, HttpMethods::PUT,
                      std::bind(&ModelboxEditorPlugin::HandlerFlowUnitPut, this,
                                std::placeholders::_1, std::placeholders::_2));
  listener_->Register(save_project_url, HttpMethods::PUT,
                      std::bind(&ModelboxEditorPlugin::SaveAllProject, this,
                                std::placeholders::_1, std::placeholders::_2));
}

bool ModelboxEditorPlugin::GetHtmlFile(const std::string& in_file,
                                       std::string* out_file,
                                       std::string* redirect_file) {
  auto in_file_canon = modelbox::PathCanonicalize(in_file, web_root_);
  std::string base_filename = in_file.substr(in_file.find_last_of("/\\") + 1);
  if (base_filename.length() == 0 || base_filename.c_str()[0] == '\0') {
    base_filename = "/";
  }

  auto file_name = in_file_canon;
  if (base_filename.find_first_of(".") == std::string::npos) {
    if (base_filename != "/") {
      // if not a specify file, then must be a directory
      *redirect_file = in_file + "/";
      return false;
    }
    auto default_file_name = file_name + "/index.htm";
    if (access(default_file_name.c_str(), R_OK) != 0) {
      default_file_name = file_name + "/index.html";
      if (access(default_file_name.c_str(), R_OK) == 0) {
        file_name = default_file_name;
      }
    } else {
      file_name = default_file_name;
    }
  }
  if (file_name.length() == 0) {
    return false;
  }

  *out_file = file_name;

  return true;
}

void ModelboxEditorPlugin::HandlerFlowUnitInfoGet(
    const httplib::Request& request, httplib::Response& response) {
  modelbox::ConfigurationBuilder config_builder;

  return HandlerFlowUnitInfo(request, response, config_builder.Build());
}

void ModelboxEditorPlugin::HandlerFlowUnitInfoPut(
    const httplib::Request& request, httplib::Response& response) {
  modelbox::ConfigurationBuilder config_builder;
  try {
    auto body = nlohmann::json::parse(request.body);
    if (body.find("skip-default") != body.end()) {
      config_builder.AddProperty(
          "driver." + std::string(DRIVER_SKIP_DEFAULT),
          std::to_string(body["skip-default"].get<bool>()));
    }

    if (body.find("dir") != body.end()) {
      std::vector<std::string> dirs;
      for (auto& it : body["dir"]) {
        auto dir = it.get<std::string>();
        if (!CheckBlackDir(dir)) {
          dirs.push_back(dir);
        }
      }
      config_builder.AddProperty("driver." + std::string(DRIVER_DIR), dirs);
    }
  } catch (const std::exception& e) {
    std::string errmsg = "Get info failed: ";
    errmsg += e.what();
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(errmsg, TEXT_PLAIN);
    AddSafeHeader(response);
    return;
  }

  return HandlerFlowUnitInfo(request, response, config_builder.Build());
}

modelbox::Status ModelboxEditorPlugin::CreateFlowunitByTool(
    const httplib::Request& request, httplib::Response& response) {
  auto body = nlohmann::json::parse(request.body);
  nlohmann::json error_json;
  std::string cmd;
  std::string out;
  std::string err;

  AddSafeHeader(response);

  GenerateCommand(cmd, body, response);

  ExecCommand(cmd, response);

  return STATUS_SUCCESS;
}

void ModelboxEditorPlugin::HandlerArgs(nlohmann::json& body, std::string& args,
                                       std::string key, std::string arg) {
  if (body.find(key) != body.end() &&
      body[key].get<std::string>().length() > 0) {
    std::string value = body[key].get<std::string>();
    if (key == "desc") {
      value = "\"" + value + "\"";
    }
    args += " " + arg;
    args += " " + value;
  }
}

void ModelboxEditorPlugin::HandlerFlowUnitPut(const httplib::Request& request,
                                              httplib::Response& response) {
  try {
    auto status = CreateFlowunitByTool(request, response);
    if (status != modelbox::STATUS_SUCCESS) {
      response.status = HttpStatusCodes::BAD_REQUEST;
      response.set_content("Failed to create Flowunit", TEXT_PLAIN);
      return;
    }
  } catch (const std::exception& e) {
    std::string errmsg = "Get info failed: ";
    errmsg += e.what();
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(errmsg, TEXT_PLAIN);
    AddSafeHeader(response);
    return;
  }

  response.status = HttpStatusCodes::CREATED;
  return;
}

void ModelboxEditorPlugin::SaveAllProject(const httplib::Request& request,
                                          httplib::Response& response) {
  try {
    auto body = nlohmann::json::parse(request.body);
    auto jobid = body["job_id"].get<std::string>();
    auto graph_data = body["job_graph"].dump();
    auto path = body["graph_path"].get<std::string>();
    std::string toml_data;

    MBLOG_INFO << "Save All Project Info: " << path;
    AddSafeHeader(response);

    if (modelbox::JsonToToml(graph_data, &toml_data) == false) {
      std::string errmsg = "Graph data is invalid.";
      response.status = HttpStatusCodes::INTERNAL_ERROR;
      response.set_content(errmsg, TEXT_PLAIN);
    }
    ConfigJobid(jobid);
    // save graph data
    auto ret = SaveGraphFile(jobid, toml_data, path);
    if (!ret) {
      std::string errmsg = "Failed to save file.";
      response.status = HttpStatusCodes::INTERNAL_ERROR;
      response.set_content(errmsg, TEXT_PLAIN);
      return;
    }

  } catch (const std::exception& e) {
    std::string errmsg = "Get info failed: ";
    errmsg += e.what();
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(errmsg, TEXT_PLAIN);
    return;
  }

  response.status = HttpStatusCodes::OK;
}

void ModelboxEditorPlugin::ConfigJobid(std::string& job_id) {
  auto type = ".toml";
  if (job_id.rfind(type) == std::string::npos) {
    job_id = job_id + type;
  }
}

modelbox::Status ModelboxEditorPlugin::SaveGraphFile(
    const std::string& job_id, const std::string& toml_graph,
    const std::string& path) {
  auto ret = modelbox::CreateDirectory(path);
  if (!ret) {
    return {modelbox::STATUS_FAULT,
            std::string("create graph directory failed, ") +
                modelbox::StrError(errno) + ", path: " + path};
  }

  std::vector<std::string> list_files;
  ret = modelbox::ListSubDirectoryFiles(path, "*", &list_files);
  if (!ret) {
    return {modelbox::STATUS_FAULT,
            std::string("list subdirectoryfiles failed, ") +
                modelbox::StrError(errno) + ", path: " + path};
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

  std::string path_graph = path + "/" + job_id;
  MBLOG_INFO << "path_graph: " << path_graph;
  std::ofstream out(path_graph, std::ios::trunc);
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

void ModelboxEditorPlugin::HandlerProjectGet(const httplib::Request& request,
                                             httplib::Response& response) {
  try {
    std::string relative_path = request.path;
    int pos_1 = relative_path.find("\"");
    int pos_2 = relative_path.find_last_of("\"");
    std::string project_path = "";
    project_path = relative_path.substr(pos_1 + 1, pos_2 - pos_1 - 1);
    MBLOG_INFO << "get project path: " << project_path;
    AddSafeHeader(response);

    auto project_name = modelbox::GetBaseName(project_path);
    MBLOG_INFO << "loading project: " << project_name;
    nlohmann::json json;
    std::string result;
    json["project_name"] = project_name;
    json["path"] = project_path.substr(0, project_path.find_last_of("/\\"));
    json["flowunits"] = nlohmann::json::array();
    json["graphs"] = nlohmann::json::array();

    MBLOG_INFO << "project path: " << json["path"];

    auto flowunit_path = project_path + "/src/flowunit";
    auto graph_path = project_path + "/src/graph";

    std::vector<std::string> graphs;
    auto ret = modelbox::ListSubDirectoryFiles(graph_path, "*.toml", &graphs);
    if (!ret) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
      return;
    }

    std::string json_data;
    nlohmann::json graph;
    for (auto g : graphs) {
      ret = GraphFileToJson(g, json_data);
      graph = nlohmann::json::parse(json_data);
      graph["name"] = modelbox::GetBaseName(g);
      json["graphs"].push_back(graph);
    }

    std::vector<std::string> flowunits;
    ret = modelbox::ListSubDirectoryFiles(flowunit_path, "*.toml", &flowunits);
    // nlohmann::json flowunit;
    for (auto f : flowunits) {
      ret = GraphFileToJson(f, json_data);
      json["flowunits"].push_back(json_data);
    }
    result = json.dump();
    MBLOG_INFO << "infos: " << result;
    response.set_content(result, JSON);
  } catch (const std::exception& e) {
    std::string errmsg = "Get info failed: ";
    errmsg += e.what();
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(errmsg, TEXT_PLAIN);
    return;
  }

  response.status = HttpStatusCodes::OK;
}

void ModelboxEditorPlugin::GenerateCommand(std::string& cmd,
                                           nlohmann::json& body,
                                           httplib::Response& response) {
  cmd = "modelbox-tool create";
  nlohmann::json error_json;
  for (auto& element : body.items()) {
    cmd += " -" + element.key();
    if (!element.value().is_null()) {
      int num = 0;
      if (nlohmann::json::value_t::array == element.value().type()) {
        for (auto port : element.value()) {
          if (num > 0) {
            cmd += " -" + element.key() + " ";
          } else {
            cmd += " ";
          }
          for (auto& i : port.items()) {
            cmd += i.key() + "=" + i.value().dump();
            if (i != port.items().end()) {
              cmd += ",";
            }
          }
          cmd = cmd.substr(0, cmd.length() - 1);
          num += 1;
        }
      } else {
        cmd += "=" + element.value().dump();
      }
    }
  }
  return;
}

void ModelboxEditorPlugin::ExecCommand(std::string& cmd,
                                       httplib::Response& response) {
  Popen p;
  nlohmann::json error_json;
  nlohmann::json json;
  p.Open(cmd, 2000, "re");
  MBLOG_INFO << "exec: " << cmd;
  std::string out;
  std::string err;
  auto ret = p.ReadAll(&out, &err);
  ret = p.Close();
  if (ret == 0){
    json["message"] = out;
    response.status = HttpStatusCodes::CREATED;
    response.set_content(json.dump(), JSON);
  }else{
    error_json["errmessage"] = err;
    error_json["errcode"] = ret;
    response.status = HttpStatusCodes::BAD_REQUEST;
    response.set_content(error_json.dump(), JSON);
  }
  
  return;
}

void ModelboxEditorPlugin::HandlerProjectPut(const httplib::Request& request,
                                             httplib::Response& response) {
  
  auto body = nlohmann::json::parse(request.body);
  nlohmann::json error_json;
  std::string cmd;
  std::string out;
  std::string err;

  AddSafeHeader(response);

  GenerateCommand(cmd, body, response);

  ExecCommand(cmd, response);

  return;
}

void ModelboxEditorPlugin::HandlerFlowUnitInfo(
    const httplib::Request& request, httplib::Response& response,
    std::shared_ptr<modelbox::Configuration> config) {
  modelbox::FlowUnitInfo flowunit_info;

  AddSafeHeader(response);
  auto status = flowunit_info.Init(config);
  if (!status) {
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(HTTP_RESP_ERR_GETINFO_FAILED, TEXT_PLAIN);
    MBLOG_ERROR << status;
    return;
  }

  std::string info;
  status = flowunit_info.GetInfoInJson(&info);
  if (!status) {
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(HTTP_RESP_ERR_GETINFO_FAILED, TEXT_PLAIN);
    MBLOG_ERROR << status;
    return;
  }

  response.status = HttpStatusCodes::OK;
  response.set_content(info, JSON);
}

void ModelboxEditorPlugin::HandlerDirectoryGet(const httplib::Request& request,
                                               httplib::Response& response) {
  AddSafeHeader(response);
  nlohmann::json response_json;
  nlohmann::json error_json;
  nlohmann::json project_json;
  response_json = nlohmann::json::array();
  try {
    std::string path;
    std::vector<std::string> listfiles;

    auto last_split_start_pos = request.path.find('"');
    auto last_split_end_pos = request.path.rfind('"');
    path = request.path.substr(last_split_start_pos + 1,
                               last_split_end_pos - last_split_start_pos - 1);
    ListFiles(path, "*", &listfiles, LIST_FILES_DIR);

    MBLOG_INFO << "Search path: " << path;
    for (auto f : listfiles) {
      project_json["dirname"]=modelbox::GetBaseName(f);
      project_json["isproject"]=IsProject(f);
      response_json.push_back(project_json);
    }
    

  } catch (const std::exception& e) {
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    error_json["errmessage"] = "internal error when searching path";
    response.set_content(error_json, JSON);
    return;
  }
  
  response.set_content(response_json.dump(), JSON);
  return;
}

bool ModelboxEditorPlugin::IsProject(std::string &path){
  std::vector<std::string> files;
  auto ret = modelbox::ListSubDirectoryFiles(path, "src", &files);
  for (const auto& file : files) {
    std::vector<std::string> flowunit_folder;
    std::vector<std::string> graph_folder;
    ret = modelbox::ListSubDirectoryFiles(file, "flowunit", &flowunit_folder);
    ret = modelbox::ListSubDirectoryFiles(file, "flowunit", &graph_folder);
    if (flowunit_folder.size() + graph_folder.size() == 2) {
      return true;
    }
  }
  return false;
}

void ModelboxEditorPlugin::HandlerUIGet(const httplib::Request& request,
                                        httplib::Response& response) {
  auto path = request.path;
  auto file_name = path;
  struct stat path_stat;
  std::string resolve_file;
  std::string redirect_file;

  AddSafeHeader(response);
  MBLOG_DEBUG << "request file:" << file_name;
  if (GetHtmlFile(file_name, &resolve_file, &redirect_file) == false) {
    if (!redirect_file.empty()) {
      response.status = HttpStatusCodes::FOUND;
      response.headers.insert(std::make_pair("location", redirect_file));
      return;
    }

    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
    return;
  }

  file_name = resolve_file;
  if (file_name.find(web_root_) != 0) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
    return;
  }

  if (stat(file_name.c_str(), &path_stat) != 0) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
    return;
  }

  if (!S_ISREG(path_stat.st_mode)) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_PATH_NOT_FILE, TEXT_PLAIN);
    return;
  }

  if (access(file_name.c_str(), R_OK | F_OK) != 0) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
    return;
  }

  SendFile(file_name, response);
}

void ModelboxEditorPlugin::SendFile(const std::string& file_name,
                                    httplib::Response& response) {
  auto content_type = ModelboxGetMimeType(file_name);
  auto file = std::shared_ptr<std::ifstream>(new std::ifstream(file_name),
                                             [](std::ifstream* ptr) {
                                               ptr->close();
                                               delete ptr;
                                             });
  if (!file->is_open()) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
    return;
  }

  size_t data_size = 4096;
  auto data = std::shared_ptr<char>(new (std::nothrow) char[data_size],
                                    [](char* ptr) { delete[] ptr; });
  if (data.get() == nullptr) {
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
    return;
  }

  response.status = HttpStatusCodes::OK;
  response.set_content_provider(
      content_type.c_str(),
      [file, data, data_size](size_t offset, httplib::DataSink& sink) {
        if (file->eof()) {
          sink.done();
          return true;
        }

        file->read(data.get(), data_size);
        if (!sink.is_writable()) {
          return false;
        }

        auto ret = sink.write(data.get(), file->gcount());
        if (!ret) {
          return false;
        }

        return true;
      });
}

modelbox::Status ModelboxEditorPlugin::GraphFileToJson(const std::string& file,
                                                       std::string& json_data) {
  std::ifstream infile(file);
  if (infile.fail()) {
    return {modelbox::STATUS_NOTFOUND, "Get demo failed"};
  }
  Defer { infile.close(); };

  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());
  if (data.length() <= 0) {
    return {modelbox::STATUS_BADCONF, "demo file is invalid."};
  }

  std::string extension = file.substr(file.find_last_of(".") + 1);
  if (extension == "json" || data[0] == '{') {
    json_data = data;
  } else {
    auto ret = modelbox::TomlToJson(data, &json_data);
    if (!ret) {
      return {modelbox::STATUS_FAULT, "demo format error"};
    }
  }

  return modelbox::STATUS_OK;
}

void ModelboxEditorPlugin::HandlerDemoGetList(const httplib::Request& request,
                                              httplib::Response& response) {
  AddSafeHeader(response);
  std::vector<std::string> dirs;
  std::vector<std::string> files;
  auto ret = modelbox::ListSubDirectoryFiles(demo_path_, "graph", &dirs);
  if (!ret) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
      return;
  }
  for (auto &f: dirs){
    ret = modelbox::ListSubDirectoryFiles(f, "*.toml", &files);
    ret = modelbox::ListSubDirectoryFiles(f, "*.json", &files);
    if (!ret) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
      return;
    }
  }

  nlohmann::json response_json;
  response_json["demo_list"] = nlohmann::json::array();
  for (const auto& file : files) {
    nlohmann::json demo;
    std::string filename = modelbox::GetBaseName(file);
    std::string name = filename;

    std::string json_data;
    std::string desc;
    auto ret = GraphFileToJson(file, json_data);
    if (ret) {
      try {
        auto response = nlohmann::json::parse(json_data);
        desc = response["flow"]["desc"].get<std::string>();
      } catch (const std::exception& e) {
        MBLOG_WARN << "parser json failed, " << e.what();
      }
    }

    demo["desc"] = desc;
    demo["name"] = name;
    demo["file"] = file;
    response_json["demo_list"].push_back(demo);
  }

  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);
  return;
}

void ModelboxEditorPlugin::HandlerDemoGet(const httplib::Request& request,
                                          httplib::Response& response) {
  try {
    std::string relative_path = request.path.substr(demo_url.size());
    std::string pre_path;
    std::string demo_file;
    std::string demo_name;
    SplitPath(relative_path, pre_path, demo_name);
    if (demo_name.length() == 0) {
      HandlerDemoGetList(request, response);
      return;
    }

    std::vector<std::string> files;
    std::vector<std::string> dirs;

    auto ret = modelbox::ListSubDirectoryFiles(demo_path_, "graph", &dirs);
    
    for (auto &f: dirs){
      modelbox::ListSubDirectoryFiles(f, "*.toml", &files);
      modelbox::ListSubDirectoryFiles(f, "*.json", &files);
    }
    for (const auto& file : files) {
      std::string filename = modelbox::GetBaseName(file);
      if (filename == demo_name) {
        demo_file = file;
        break;
      }
    }

    AddSafeHeader(response);
    if (demo_file.length() == 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    auto resolve_path = modelbox::PathCanonicalize(demo_file);
    if (resolve_path.length() == 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    demo_file = resolve_path;
    if (demo_file.find(demo_path_) != 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    std::string json_data;
    MBLOG_INFO << "load demo file " << demo_file;
    ret = GraphFileToJson(demo_file, json_data);
    if (!ret) {
      std::string msg = "demo file is invalid.";
      msg += ret.Errormsg();
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(msg, TEXT_PLAIN);
      MBLOG_WARN << "Get graph file failed, " << ret.WrapErrormsgs();
      return;
    }

    response.status = HttpStatusCodes::OK;
    response.set_content(json_data, JSON);
    return;
  } catch (const std::exception& e) {
    MBLOG_ERROR << "demo get failed, " << e.what();
    response.status = HttpStatusCodes::INTERNAL_ERROR;
    response.set_content(HTTP_RESP_ERR_GETINFO_FAILED, TEXT_PLAIN);
    return;
  }

  return;
}

bool ModelboxEditorPlugin::Start() {
  if (enable_ == false) {
    return true;
  }

  listener_->SetAclWhiteList(acl_white_list_);

  listener_->Start();

  auto ret = listener_->GetStatus();
  if (!ret) {
    MBLOG_ERROR << "Start editor failed, err " << ret;
    return false;
  }

  return true;
}

bool ModelboxEditorPlugin::Stop() {
  if (enable_ == false) {
    return true;
  }

  listener_->Stop();

  return true;
}

bool ModelboxEditorPlugin::ParseConfig(
    std::shared_ptr<modelbox::Configuration> config) {
  enable_ = config->GetBool("editor.enable", false);
  web_root_ = config->GetString("editor.root", DEFAULT_WEB_ROOT);
  server_ip_ = config->GetString("editor.ip",
                                 config->GetString("server.ip", "127.0.0.1"));
  server_port_ = config->GetString("editor.port",
                                   config->GetString("server.port", "1104"));
  demo_path_ =
      config->GetString("editor.demo_graphs", DEFAULT_DEMO_GRAPHS_ROOT);

  acl_white_list_ = config->GetStrings("acl.allow");
  return true;
}