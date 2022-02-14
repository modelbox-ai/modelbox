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

#include <dirent.h>

#include <nlohmann/json.hpp>
#include <toml.hpp>

#include "config.h"
#include "modelbox/base/log.h"
#include "modelbox/common/flowunit_info.h"
#include "modelbox/common/utils.h"
#include "modelbox/server/utils.h"

using namespace modelbox;

const std::string DEFAULT_WEB_ROOT = "/usr/local/share/modelbox/www";
const std::string DEFAULT_SOLUTION_GRAPHS_ROOT =
    std::string(MODELBOX_SOLUTION_PATH) + "/graphs";

const std::string UI_url = "/";
const std::string flowunit_info_url = "/editor/flow-info";
const std::string solution_url = "/editor/solution";

constexpr const char* HTTP_RESP_ERR_GETINFO_FAILED = "Get info failed";
constexpr const char* HTTP_RESP_ERR_PATH_NOT_FOUND = "Path not found";
constexpr const char* HTTP_RESP_ERR_PATH_NOT_FILE = "Path not a file";
constexpr const char* HTTP_RESP_ERR_CANNOT_READ = "Can not read file";

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
  listener_->Register(solution_url, HttpMethods::GET,
                      std::bind(&ModelboxEditorPlugin::HandlerSolutionGet, this,
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
    return {modelbox::STATUS_NOTFOUND, "Get solution failed"};
  }
  Defer { infile.close(); };

  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());
  if (data.length() <= 0) {
    return {modelbox::STATUS_BADCONF, "solution file is invalid."};
  }

  std::string extension = file.substr(file.find_last_of(".") + 1);
  if (extension == "json" || data[0] == '{') {
    json_data = data;
  } else {
    auto ret = modelbox::TomlToJson(data, &json_data);
    if (!ret) {
      return {modelbox::STATUS_FAULT, "solution format error"};
    }
  }

  return modelbox::STATUS_OK;
}

void ModelboxEditorPlugin::HandlerSolutionGetList(
    const httplib::Request& request, httplib::Response& response) {
  AddSafeHeader(response);
  std::vector<std::string> files;
  auto ret = modelbox::ListSubDirectoryFiles(solution_path_, "*.toml", &files);
  if (!ret) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
    return;
  }

  ret = modelbox::ListSubDirectoryFiles(solution_path_, "*.json", &files);
  if (!ret) {
    response.status = HttpStatusCodes::NOT_FOUND;
    response.set_content(HTTP_RESP_ERR_CANNOT_READ, TEXT_PLAIN);
    return;
  }

  nlohmann::json response_json;
  response_json["solution_list"] = nlohmann::json::array();
  for (const auto& file : files) {
    nlohmann::json solution;
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

    solution["desc"] = desc;
    solution["name"] = name;
    solution["file"] = file;
    response_json["solution_list"].push_back(solution);
  }

  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);
  return;
}

void ModelboxEditorPlugin::HandlerSolutionGet(const httplib::Request& request,
                                              httplib::Response& response) {
  try {
    std::string relative_path = request.path.substr(solution_url.size());
    std::string pre_path;
    std::string solution_file;
    std::string solution_name;
    SplitPath(relative_path, pre_path, solution_name);
    if (solution_name.length() == 0) {
      HandlerSolutionGetList(request, response);
      return;
    }

    std::vector<std::string> files;
    modelbox::ListSubDirectoryFiles(solution_path_, "*.toml", &files);
    modelbox::ListSubDirectoryFiles(solution_path_, "*.json", &files);
    for (const auto& file : files) {
      std::string filename = modelbox::GetBaseName(file);
      if (filename == solution_name) {
        solution_file = file;
        break;
      }
    }

    AddSafeHeader(response);
    if (solution_file.length() == 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    auto resolve_path = modelbox::PathCanonicalize(solution_file);
    if (resolve_path.length() == 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    solution_file = resolve_path;
    if (solution_file.find(solution_path_) != 0) {
      response.status = HttpStatusCodes::NOT_FOUND;
      response.set_content(HTTP_RESP_ERR_PATH_NOT_FOUND, TEXT_PLAIN);
      return;
    }

    std::string json_data;
    MBLOG_INFO << "load solution file " << solution_file;
    auto ret = GraphFileToJson(solution_file, json_data);
    if (!ret) {
      std::string msg = "solution file is invalid.";
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
    MBLOG_ERROR << "solution get failed, " << e.what();
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
  solution_path_ =
      config->GetString("editor.solution_graphs", DEFAULT_SOLUTION_GRAPHS_ROOT);

  acl_white_list_ = config->GetStrings("acl.allow");
  return true;
}