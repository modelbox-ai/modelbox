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
#include <pwd.h>
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

const std::string DEFAULT_WEB_ROOT =
    "${MODELBOX_ROOT}/usr/local/share/modelbox/www";
const std::string DEFAULT_PROJECT_TEMPLATE_DIR =
    "${MODELBOX_ROOT}/usr/local/share/modelbox/project-template";
const std::string DEFAULT_DEMO_ROOT_DIR = MODELBOX_DEMO_PATH;
constexpr const char* DEFAULT_MODELBOX_TEMPLATE_CMD =
    "${MODELBOX_ROOT}/usr/local/bin/modelbox-tool template";

const std::string UI_url = "/";
const std::string flowunit_info_url = "/editor/flow-info";
const std::string basic_info = "/editor/basic-info";
const std::string demo_url = "/editor/demo";
const std::string save_graph_url = "/editor/graph";
const std::string flowunit_create_url = "/editor/flowunit/create";
const std::string project_url = "/editor/project";
const std::string project_template_url = "/editor/project/template";
const std::string project_list_url = "/editor/project/list";
const std::string project_create_url = "/editor/project/create";
const std::string pass_encode_url = "/editor/password/encode";
const std::string postman_url = "/editor/postman";

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
  struct Handler_Map {
    const std::string path;
    const HttpMethod method;
    void (ModelboxEditorPlugin::*func)(const httplib::Request& request,
                                       httplib::Response& response);
  };
  Handler_Map handler_list[] = {
      {UI_url, HttpMethods::GET, &ModelboxEditorPlugin::HandlerUIGet},
      {basic_info, HttpMethods::GET, &ModelboxEditorPlugin::HanderBasicInfoGet},
      {flowunit_info_url, HttpMethods::PUT,
       &ModelboxEditorPlugin::HandlerFlowUnitInfoPut},
      {flowunit_info_url, HttpMethods::GET,
       &ModelboxEditorPlugin::HandlerFlowUnitInfoGet},
      {demo_url, HttpMethods::GET, &ModelboxEditorPlugin::HandlerDemoGet},
      {project_url, HttpMethods::GET, &ModelboxEditorPlugin::HandlerProjectGet},
      {project_template_url, HttpMethods::GET,
       &ModelboxEditorPlugin::HandlerProjectTemplateListGet},
      {project_list_url, HttpMethods::GET,
       &ModelboxEditorPlugin::HandlerProjectListGet},
      {project_create_url, HttpMethods::PUT,
       &ModelboxEditorPlugin::HandlerProjectCreate},
      {flowunit_create_url, HttpMethods::PUT,
       &ModelboxEditorPlugin::HandlerFlowUnitCreate},
      {save_graph_url, HttpMethods::PUT,
       &ModelboxEditorPlugin::HandlerSaveGraph},
      {pass_encode_url, HttpMethods::PUT,
       &ModelboxEditorPlugin::HandlerPassEncode},
      {postman_url, HttpMethods::POST, &ModelboxEditorPlugin::HandlerPostman},
  };

  for (const auto& hander : handler_list) {
    listener_->Register(hander.path, hander.method,
                        std::bind(hander.func, this, std::placeholders::_1,
                                  std::placeholders::_2));
  }
}

std::string ModelboxEditorPlugin::ResultMsg(const std::string& code,
                                            const std::string& msg) {
  nlohmann::json result_json;
  result_json["code"] = code;
  result_json["msg"] = msg;

  return result_json.dump();
}

std::string ModelboxEditorPlugin::ResultMsg(modelbox::Status& status) {
  return ResultMsg(status.StrCode(), status.WrapErrormsgs());
}

void ModelboxEditorPlugin::SetUpResponse(httplib::Response& response,
                                         modelbox::Status& status) {
  switch (status.Code()) {
    case modelbox::STATUS_SUCCESS:
      response.status = HttpStatusCodes::OK;
      break;
    case modelbox::STATUS_NOTFOUND:
      response.status = HttpStatusCodes::NOT_FOUND;
      break;
    case modelbox::STATUS_INVALID:
    case modelbox::STATUS_BADCONF:
      response.status = HttpStatusCodes::BAD_REQUEST;
      break;
    default:
      response.status = HttpStatusCodes::INTERNAL_ERROR;
      break;
  }

  AddSafeHeader(response);
  response.set_content(ResultMsg(status), JSON);

  return;
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
    modelbox::Status ret = {modelbox::STATUS_INVALID, errmsg};
    SetUpResponse(response, ret);
    return;
  }

  return HandlerFlowUnitInfo(request, response, config_builder.Build());
}

modelbox::Status ModelboxEditorPlugin::RunTemplateCommand(
    const httplib::Request& request, httplib::Response& response,
    const std::string& cmd) {
  modelbox::Status ret = modelbox::STATUS_FAULT;

  try {
    std::string runcmd;
    runcmd = template_cmd_ + " " + cmd;
    auto body = nlohmann::json::parse(request.body);
    ret = GenerateCommandFromJson(body, runcmd);
    if (ret == modelbox::STATUS_OK) {
      ret = RunCommand(runcmd);
    }
  } catch (const std::exception& e) {
    modelbox::Status errret(
        ret, std::string("run modelbox-tool failed, ") + e.what());
    ret = errret;
  }

  SetUpResponse(response, ret);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_WARN << ret.WrapErrormsgs();
    return ret;
  }

  return ret;
}

void ModelboxEditorPlugin::HandlerFlowUnitCreate(
    const httplib::Request& request, httplib::Response& response) {
  auto ret = RunTemplateCommand(request, response, "--flowunit");
  if (ret == modelbox::STATUS_OK) {
    response.status = HttpStatusCodes::CREATED;
  }

  return;
}

void ModelboxEditorPlugin::HandlerSaveGraph(const httplib::Request& request,
                                            httplib::Response& response) {
  auto ret = SaveGraph(request);
  SetUpResponse(response, ret);
  if (ret == modelbox::STATUS_OK) {
    response.status = HttpStatusCodes::CREATED;
  }

  return;
}

modelbox::Status ModelboxEditorPlugin::SaveGraph(
    const httplib::Request& request) {
  try {
    auto body = nlohmann::json::parse(request.body);
    std::string toml_data;

    auto graph_data = body["graph"].dump();
    auto graph_name = body["graph_name"].get<std::string>();
    auto path = body["graph_path"].get<std::string>();
    MBLOG_INFO << "Save graph to : " << path;

    if (IsModelboxProjectDir(path) == false) {
      return {modelbox::STATUS_INVALID, "path is not a modelbox project"};
    }

    auto ret = modelbox::JsonToToml(graph_data, &toml_data);
    if (!ret) {
      return {ret, "convert json failed"};
    }

    std::string graphfile = path + "/src/graph/" + graph_name + ".toml";
    std::ofstream out(graphfile, std::ios::trunc);
    if (out.fail()) {
      return {modelbox::STATUS_FAULT, std::string("save graph file failed, ") +
                                          modelbox::StrError(errno) +
                                          ", path: " + graphfile};
    }

    chmod(graphfile.c_str(), 0600);
    Defer { out.close(); };

    out << toml_data;
    if (out.fail()) {
      return {modelbox::STATUS_FAULT, std::string("save graph file failed, ") +
                                          modelbox::StrError(errno) +
                                          ", path: " + graphfile};
    }

  } catch (const std::exception& e) {
    std::string errmsg = "save graph info failed: ";
    errmsg += e.what();
    return {STATUS_INVALID, errmsg};
  }

  return STATUS_OK;
}

modelbox::Status ModelboxEditorPlugin::ReadProjectName(const std::string& path,
                                                       std::string& name) {
  auto ret = RunCommand(template_cmd_ + " -project -getname \"" + path + "\"",
                        nullptr, &name);
  if (!ret) {
    return ret;
  }

  name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());
  name.erase(std::remove(name.begin(), name.end(), '\r'), name.end());

  return STATUS_OK;
}

void ModelboxEditorPlugin::HandlerProjectGet(const httplib::Request& request,
                                             httplib::Response& response) {
  std::string project_name;

  try {
    if (request.has_param("path") == false) {
      modelbox::Status ret = {modelbox::STATUS_INVALID,
                              "argument path is not set."};
      SetUpResponse(response, ret);
      return;
    }

    std::string project_path = "";
    project_path = request.params.find("path")->second;
    MBLOG_INFO << "loading project: " << project_path;

    auto ret = ReadProjectName(project_path, project_name);
    if (!ret) {
      modelbox::Status rspret = {ret, "Get project name failed."};
      SetUpResponse(response, rspret);
      return;
    }

    nlohmann::json json;
    std::string result;
    std::vector<std::string> graphs;
    auto flowunit_path = project_path + "/src/flowunit";
    auto graph_path = project_path + "/src/graph";

    json["project_name"] = project_name;
    json["project_path"] = project_path;
    json["flowunits"] = nlohmann::json::array();
    json["graphs"] = nlohmann::json::array();

    ret = modelbox::ListSubDirectoryFiles(graph_path, "*.toml", &graphs);
    if (!ret) {
      modelbox::Status ret = {modelbox::STATUS_NOTFOUND,
                              HTTP_RESP_ERR_CANNOT_READ};
      SetUpResponse(response, ret);
      return;
    }

    std::string json_data;
    nlohmann::json graph;
    for (auto g : graphs) {
      ret = GraphFileToJson(g, json_data);
      graph["name"] = modelbox::GetBaseName(g);
      graph = nlohmann::json::parse(json_data);
      json["graphs"].push_back(graph);
    }

    std::vector<std::string> flowunits;
    ret = modelbox::ListSubDirectoryFiles(flowunit_path, "*.toml", &flowunits);
    for (auto f : flowunits) {
      ret = GraphFileToJson(f, json_data);
      if (!ret) {
        modelbox::Status rspret = {ret, "toml"};
        SetUpResponse(response, rspret);
        return;
      }
      json["flowunits"].push_back(nlohmann::json::parse(json_data));
    }
    result = json.dump();
    MBLOG_DEBUG << "infos: " << result;
    response.set_content(result, JSON);
  } catch (const std::exception& e) {
    std::string errmsg = "Get info failed: ";
    errmsg += e.what();
    MBLOG_ERROR << errmsg;
    modelbox::Status ret = {STATUS_INVALID, errmsg};
    SetUpResponse(response, ret);
    return;
  }

  response.status = HttpStatusCodes::OK;
}

modelbox::Status ModelboxEditorPlugin::GenerateCommandFromJson(
    const nlohmann::json& body, std::string& cmd) {
  nlohmann::json error_json;
  for (auto& element : body.items()) {
    cmd += " -" + element.key();
    if (element.value().is_null()) {
      continue;
    }

    if (element.value().type() != nlohmann::json::value_t::array) {
      cmd += "=" + element.value().dump();
      continue;
    }

    int num = 0;
    for (auto port : element.value()) {
      if (num > 0) {
        cmd += " -" + element.key() + "=";
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
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelboxEditorPlugin::RunCommand(const std::string& cmd,
                                                  const std::string* in,
                                                  std::string* out) {
  Popen p;
  std::string outmsg;
  std::string err;
  std::string mode;

  MBLOG_INFO << "exec: " << cmd;

  if (in) {
    mode = "wre";
  } else {
    mode = "re";
  }

  auto retstatus = p.Open(cmd, 3000, mode.c_str(), template_cmd_env_);
  if (!retstatus) {
    return retstatus;
  }

  if (in) {
    p.WriteString(*in);
  }

  auto ret = p.ReadAll(&outmsg, &err);
  ret = p.Close();
  if (ret == 0) {
    retstatus = modelbox::STATUS_OK;
  } else {
    std::string errmsg = "Execute command failed, ret: ";
    errmsg += std::to_string(ret);
    errmsg += " error: " + err;
    retstatus = {modelbox::STATUS_FAULT, errmsg};
  }

  if (out) {
    *out = outmsg;
  }

  return retstatus;
}

void ModelboxEditorPlugin::HandlerProjectCreate(const httplib::Request& request,
                                                httplib::Response& response) {
  auto ret = RunTemplateCommand(request, response, "--project");
  if (ret == modelbox::STATUS_OK) {
    response.status = HttpStatusCodes::CREATED;
  }
}

void ModelboxEditorPlugin::HandlerFlowUnitInfo(
    const httplib::Request& request, httplib::Response& response,
    std::shared_ptr<modelbox::Configuration> config) {
  modelbox::FlowUnitInfo flowunit_info;

  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  auto status = flowunit_info.Init(config);
  if (!status) {
    rspret = {status, HTTP_RESP_ERR_GETINFO_FAILED};
    return;
  }

  std::string info;
  status = flowunit_info.GetInfoInJson(&info);
  if (!status) {
    rspret = {status, HTTP_RESP_ERR_GETINFO_FAILED};
    return;
  }

  AddSafeHeader(response);
  response.status = HttpStatusCodes::OK;
  response.set_content(info, JSON);
}

void ModelboxEditorPlugin::HandlerProjectTemplateListGet(
    const httplib::Request& request, httplib::Response& response) {
  std::vector<std::string> dirs;
  std::map<std::string, std::vector<std::string>> graphs;
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  std::vector<std::string> files;
  std::string template_dir = template_dir_ + "/project";
  MBLOG_INFO << "template_dir:" << template_dir;
  auto ret = modelbox::ListSubDirectoryFiles(template_dir, "desc.toml", &files);
  if (!ret) {
    rspret = {ret, HTTP_RESP_ERR_CANNOT_READ};
    MBLOG_INFO << "read template dir " << template_dir << " failed, " << ret;
    return;
  }

  nlohmann::json response_json;
  response_json["project_template_list"] = nlohmann::json::array();

  for (const auto& file : files) {
    std::string dirname = modelbox::GetBaseName(modelbox::GetDirName(file));
    std::string name = dirname;

    std::string json_data;
    std::string desc;
    auto ret = GraphFileToJson(file, json_data);
    if (ret) {
      try {
        auto desc_json = nlohmann::json::parse(json_data);
        desc_json["dirname"] = dirname;
        response_json["project_template_list"].push_back(desc_json);
      } catch (const std::exception& e) {
        MBLOG_WARN << "parser json " << file << " failed, " << e.what();
      }
    }
  }

  AddSafeHeader(response);
  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);
  return;
}

void ModelboxEditorPlugin::HandlerProjectListGet(
    const httplib::Request& request, httplib::Response& response) {
  nlohmann::json response_json;
  nlohmann::json subdir_json;
  std::vector<std::string> listfiles;
  subdir_json = nlohmann::json::array();
  modelbox::Status ret;
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  try {
    if (request.has_param("path") == false) {
      rspret = {modelbox::STATUS_INVALID, "argument path is not set."};
      return;
    }

    std::string list_path = "";
    list_path = request.params.find("path")->second;
    MBLOG_DEBUG << "list path: " << list_path;
    ret = ListFiles(list_path, "*", &listfiles, LIST_FILES_DIR);
    if (!ret) {
      rspret = {ret, "List path failed."};
      return;
    }

    response_json["dirname"] = list_path;
    response_json["isproject"] = IsModelboxProjectDir(list_path);

    for (auto f : listfiles) {
      nlohmann::json dirname;
      dirname["dirname"] = modelbox::GetBaseName(f);
      dirname["isproject"] = IsModelboxProjectDir(f);
      subdir_json.push_back(dirname);
    }

    response_json["subdir"] = subdir_json;
  } catch (const std::exception& e) {
    std::string errmsg = "internal error when searching path, ";
    errmsg += e.what();
    MBLOG_ERROR << errmsg;
    rspret = {STATUS_FAULT, errmsg};
    return;
  }

  AddSafeHeader(response);
  response.set_content(response_json.dump(), JSON);
  return;
}

bool ModelboxEditorPlugin::IsModelboxProjectDir(std::string& path) {
  struct stat statbuf;
  std::string checkfile;
  checkfile = path + "/CMakeLists.txt";
  if (stat(checkfile.c_str(), &statbuf) || !S_ISREG(statbuf.st_mode)) {
    return false;
  }

  checkfile = path + "/src";
  if (stat(checkfile.c_str(), &statbuf) || !S_ISDIR(statbuf.st_mode)) {
    return false;
  }

  checkfile = path + "/src/flowunit";
  if (stat(checkfile.c_str(), &statbuf) || !S_ISDIR(statbuf.st_mode)) {
    return false;
  }

  checkfile = path + "/src/graph";
  if (stat(checkfile.c_str(), &statbuf) || !S_ISDIR(statbuf.st_mode)) {
    return false;
  }

  return true;
}

void ModelboxEditorPlugin::HandlerUIGet(const httplib::Request& request,
                                        httplib::Response& response) {
  auto path = request.path;
  auto file_name = path;
  struct stat path_stat;
  std::string resolve_file;
  std::string redirect_file;
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  MBLOG_DEBUG << "request file:" << file_name;
  if (GetHtmlFile(file_name, &resolve_file, &redirect_file) == false) {
    if (!redirect_file.empty()) {
      response.status = HttpStatusCodes::FOUND;
      response.headers.insert(std::make_pair("location", redirect_file));
      return;
    }

    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FOUND};
    return;
  }

  file_name = resolve_file;
  if (file_name.find(web_root_) != 0) {
    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FOUND};
    return;
  }

  if (stat(file_name.c_str(), &path_stat) != 0) {
    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FOUND};
    return;
  }

  if (!S_ISREG(path_stat.st_mode)) {
    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FILE};
    return;
  }

  if (access(file_name.c_str(), R_OK | F_OK) != 0) {
    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_CANNOT_READ};
    return;
  }

  SendFile(file_name, response);
}

void ModelboxEditorPlugin::SendFile(const std::string& file_name,
                                    httplib::Response& response) {
  auto content_type = ModelboxGetMimeType(file_name);

  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  auto file = std::shared_ptr<std::ifstream>(new std::ifstream(file_name),
                                             [](std::ifstream* ptr) {
                                               ptr->close();
                                               delete ptr;
                                             });
  if (!file->is_open()) {
    rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_CANNOT_READ};
    return;
  }

  size_t data_size = 4096;
  auto data = std::shared_ptr<char>(new (std::nothrow) char[data_size],
                                    [](char* ptr) { delete[] ptr; });
  if (data.get() == nullptr) {
    rspret = {STATUS_FAULT, HTTP_RESP_ERR_CANNOT_READ};
    return;
  }

  AddSafeHeader(response);
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
    return {modelbox::STATUS_NOTFOUND,
            "Get graph failed" + modelbox::StrError(errno)};
  }
  Defer { infile.close(); };

  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());
  if (data.length() <= 0) {
    return {modelbox::STATUS_BADCONF, "graph file is invalid."};
  }

  std::string extension = file.substr(file.find_last_of(".") + 1);
  if (extension == "json" || data[0] == '{') {
    json_data = data;
  } else {
    auto ret = modelbox::TomlToJson(data, &json_data);
    if (!ret) {
      return {ret, "graph format error"};
    }
  }

  return modelbox::STATUS_OK;
}

void ModelboxEditorPlugin::HanderBasicInfoGet(const httplib::Request& request,
                                              httplib::Response& response) {
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  nlohmann::json response_json;
  struct passwd pwd;
  struct passwd* result;
  std::vector<char> buff;
  buff.resize(sysconf(_SC_GETPW_R_SIZE_MAX));
  getpwuid_r(getuid(), &pwd, buff.data(), buff.size(), &result);
  if (result == nullptr) {
    rspret = {modelbox::STATUS_FAULT,
              "Get pw info failed, " + modelbox::StrError(errno)};
    return;
  }

  response_json["user"] = result->pw_name;
  response_json["home-dir"] = result->pw_dir;

  AddSafeHeader(response);
  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);
  return;
}

void ModelboxEditorPlugin::HandlerDemoGetList(const httplib::Request& request,
                                              httplib::Response& response) {
  std::vector<std::string> dirs;
  std::map<std::string, std::vector<std::string>> graphs;
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  auto ret =
      modelbox::ListFiles(demo_path_, "*", &dirs, modelbox::LIST_FILES_DIR);
  if (!ret) {
    rspret = {ret, HTTP_RESP_ERR_CANNOT_READ};
    return;
  }

  for (auto const& dir : dirs) {
    std::vector<std::string> files;
    std::string graphdir = dir + "/graph";

    auto ret = modelbox::ListSubDirectoryFiles(graphdir, "*.toml", &files);
    if (!ret) {
      MBLOG_INFO << "list directory " << demo_path_ << "failed.";
    }

    ret = modelbox::ListSubDirectoryFiles(graphdir, "*.json", &files);
    if (!ret) {
      MBLOG_INFO << "list directory " << demo_path_ << "failed.";
    }

    if (files.size() == 0) {
      continue;
    }

    graphs[dir] = files;
  }

  nlohmann::json response_json;
  response_json["demo_list"] = nlohmann::json::array();
  for (auto it : graphs) {
    std::string demoname = modelbox::GetBaseName(it.first);
    for (const auto& file : it.second) {
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
          name = response["flow"]["name"].get<std::string>();
        } catch (const std::exception& e) {
          MBLOG_WARN << "parser json " << file << " failed, " << e.what();
        }
      }

      demo["demo"] = demoname;
      demo["name"] = name;
      demo["desc"] = desc;
      demo["graphfile"] = filename;
      response_json["demo_list"].push_back(demo);
    }
  }

  AddSafeHeader(response);
  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);
  return;
}

void ModelboxEditorPlugin::HandlerDemoGet(const httplib::Request& request,
                                          httplib::Response& response) {
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  try {
    std::string relative_path = request.path.substr(demo_url.size());
    std::string graph_file;
    std::string demo_file;
    std::string demo_name;
    SplitPath(relative_path, demo_name, graph_file);
    if (demo_name.length() == 0 && graph_file.length() == 0) {
      HandlerDemoGetList(request, response);
      return;
    }

    if (graph_file.length() == 0 || demo_name.length() == 0) {
      rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FOUND};
      return;
    }

    demo_file =
        PathCanonicalize(demo_name + "/graph/" + graph_file, demo_path_);
    if (demo_file.length() == 0) {
      rspret = {modelbox::STATUS_NOTFOUND, HTTP_RESP_ERR_PATH_NOT_FOUND};
      return;
    }

    std::string json_data;
    MBLOG_INFO << "load demo file " << demo_file;
    rspret = GraphFileToJson(demo_file, json_data);
    if (!rspret) {
      MBLOG_WARN << "Get graph file failed, " << rspret.WrapErrormsgs();
      std::string msg = "demo file is invalid.";
      rspret = {STATUS_BADCONF, msg};
      return;
    }

    AddSafeHeader(response);
    response.status = HttpStatusCodes::OK;
    response.set_content(json_data, JSON);
    return;
  } catch (const std::exception& e) {
    MBLOG_ERROR << "demo get failed, " << e.what();
    rspret = {STATUS_FAULT,
              std::string(HTTP_RESP_ERR_GETINFO_FAILED) + e.what()};
    return;
  }

  return;
}

void ModelboxEditorPlugin::HandlerPassEncode(const httplib::Request& request,
                                             httplib::Response& response) {
  std::string out;
  nlohmann::json response_json;
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  try {
    std::string keypass;
    std::string plainpass;
    std::string sysrelated = " -n";
    keypass = "modelbox-tool key -pass";

    auto body = nlohmann::json::parse(request.body);

    if (body.find("password") == body.end()) {
      std::string errmsg = "password key not found";
      rspret = {modelbox::STATUS_NOTFOUND, errmsg};
      return;
    }

    plainpass = body["password"].get<std::string>();

    if (body.find("sysrelated") != body.end()) {
      auto issys = body["sysrelated"].get<bool>();
      if (issys == false) {
        sysrelated = "";
      }
    }

    keypass += sysrelated;
    plainpass += "\n";
    rspret = RunCommand(keypass, &plainpass, &out);
  } catch (const std::exception& e) {
    std::string errmsg = "encrypt password failed, " + std::string(e.what());
    rspret = {STATUS_FAULT, errmsg};
    return;
  }

  if (rspret != modelbox::STATUS_SUCCESS) {
    MBLOG_WARN << rspret.WrapErrormsgs();
    return;
  }

  auto lines = modelbox::StringSplit(out, '\n');
  if (lines.size() != 2) {
    rspret = {STATUS_FAULT, "Run key command failed."};
    return;
  }

  for (auto const& line : lines) {
    auto values = modelbox::StringSplit(line, ':');
    if (values.size() != 2) {
      rspret = {STATUS_FAULT, "Get values failed."};
      return;
    }

    if (values[0].find("Key") != std::string::npos) {
      response_json["key"] =
          std::string(values[1].begin() + 1, values[1].end());
    } else if (values[0].find("Encrypted password") >= 0) {
      response_json["enpass"] =
          std::string(values[1].begin() + 1, values[1].end());
    }
  }

  AddSafeHeader(response);
  response.status = HttpStatusCodes::OK;
  response.set_content(response_json.dump(), JSON);

  return;
}

void ModelboxEditorPlugin::HandlerPostman(const httplib::Request& request,
                                          httplib::Response& response) {
  modelbox::Status rspret;

  Defer {
    if (!rspret) {
      SetUpResponse(response, rspret);
    }
  };

  try {
    std::string method;
    std::string url;
    bool hasbody = false;
    bool hasheader = false;
    nlohmann::json rheader;
    nlohmann::json rbody;
    modelbox::HttpMethod hmethod;

    auto body = nlohmann::json::parse(request.body);
    if (body.find("method") != body.end()) {
      method = body["method"].get<std::string>();
    } else {
      rspret = {STATUS_FAULT, "Get method failed."};
      return;
    }

    if (body.find("url") != body.end()) {
      url = body["url"].get<std::string>();
    } else {
      rspret = {STATUS_FAULT, "Get url failed."};
      return;
    }

    if (body.find("header") != body.end()) {
      rheader = nlohmann::json::parse(body["header"].get<std::string>());
      hasheader = true;
    }

    if (body.find("body") != body.end()) {
      rbody = nlohmann::json::parse(body["body"].get<std::string>());
      hasbody = true;
    }

    if (method == "POST") {
      hmethod = HttpMethods::POST;
    } else if (method == "GET") {
      hmethod = HttpMethods::GET;
    } else if (method == "DELETE") {
      hmethod = HttpMethods::DELETE;
    } else if (method == "PUT") {
      hmethod = HttpMethods::PUT;
    }

    HttpRequest hrequest(hmethod, url);

    if (hasbody) {
      hrequest.SetBody(rbody);
    }

    if (hasheader) {
      hrequest.SetHeaders(rheader);
    }

    rspret = SendHttpRequest(hrequest);
    if (rspret != modelbox::STATUS_SUCCESS) {
      return;
    }
    auto test_response = hrequest.GetResponse();
    AddSafeHeader(response);

    nlohmann::json response_json;
    nlohmann::json test_response_json;

    test_response_json["status"] = test_response.status;
    test_response_json["body"] = test_response.body;
    test_response_json["headers"] = test_response.headers;

    response_json["body"] = test_response_json;

    response.status = HttpStatusCodes::OK;
    response.set_content(response_json.dump(), JSON);

  } catch (const std::exception& e) {
    std::string errmsg = "internal error when debugging";
    errmsg += e.what();
    MBLOG_ERROR << errmsg;
    rspret = {STATUS_FAULT, errmsg};
    return;
  }
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
  demo_path_ = config->GetString("editor.demo_root", DEFAULT_DEMO_ROOT_DIR);

  template_cmd_ = config->GetString("editor.test.template_cmd",
                                    DEFAULT_MODELBOX_TEMPLATE_CMD);
  template_cmd_env_ = config->GetString("editor.test.template_cmd_env", "");
  template_dir_ =
      config->GetString("editor.template_dir", DEFAULT_PROJECT_TEMPLATE_DIR);
  acl_white_list_ = config->GetStrings("acl.allow");

  web_root_ = modelbox_full_path(web_root_);
  demo_path_ = modelbox_full_path(demo_path_);
  template_dir_ = modelbox_full_path(template_dir_);
  template_cmd_ = modelbox_full_path(template_cmd_);

  return true;
}