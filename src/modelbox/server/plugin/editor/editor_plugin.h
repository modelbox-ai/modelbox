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

#ifndef MODELBOX_MODELBOX_EDITOR_PLUGIN_H_
#define MODELBOX_MODELBOX_EDITOR_PLUGIN_H_

#include <nlohmann/json.hpp>

#include "modelbox/server/http_helper.h"
#include "modelbox/server/plugin.h"

class ModelboxEditorPlugin : public modelbox::Plugin {
 public:
  ModelboxEditorPlugin(){};
  virtual ~ModelboxEditorPlugin(){};

  bool Init(std::shared_ptr<modelbox::Configuration> config) override;
  bool Start() override;
  bool Stop() override;

  void RegistHandlers();
  bool ParseConfig(std::shared_ptr<modelbox::Configuration> config);

 private:
  void HandlerPassEncode(const httplib::Request &request,
                         httplib::Response &response);
  void HandlerDemoGet(const httplib::Request &request,
                      httplib::Response &response);
  void HandlerDemoGetList(const httplib::Request &request,
                          httplib::Response &response);
  void HandlerUIGet(const httplib::Request &request,
                    httplib::Response &response);
  void HanderBasicInfoGet(const httplib::Request &request,
                    httplib::Response &response);
  void SendFile(const std::string &file_name, httplib::Response &response);
  void HandlerFlowUnitInfoPut(const httplib::Request &request,
                              httplib::Response &response);
  void HandlerFlowUnitInfoGet(const httplib::Request &request,
                              httplib::Response &response);
  void HandlerFlowUnitInfo(const httplib::Request &request,
                           httplib::Response &response,
                           std::shared_ptr<modelbox::Configuration> config);
  void HandlerFlowUnitCreate(const httplib::Request &request,
                             httplib::Response &response);
  void HandlerFlowUnitGet(const httplib::Request &request,
                          httplib::Response &response);
  void HandlerProjectCreate(const httplib::Request &request,
                            httplib::Response &response);
  void HandlerProjectTemplateListGet(const httplib::Request &request,
                            httplib::Response &response);
  void HandlerProjectGet(const httplib::Request &request,
                         httplib::Response &response);
  void HandlerProjectListGet(const httplib::Request &request,
                             httplib::Response &response);
  void HandlerSaveGraph(const httplib::Request &request,
                        httplib::Response &response);
  bool GetHtmlFile(const std::string &in_file, std::string *out_file,
                   std::string *redirect_file);
  modelbox::Status GraphFileToJson(const std::string &file,
                                   std::string &json_data);
  bool CheckBlackDir(std::string dir);

 private:
  std::string ResultMsg(const std::string &code, const std::string &msg);
  std::string ResultMsg(modelbox::Status &status);
  void SetUpResponse(httplib::Response &response, modelbox::Status &status);
  modelbox::Status GenerateCommandFromJson(const nlohmann::json &body,
                                           std::string &cmd);
  modelbox::Status RunCommand(const std::string &cmd,
                              const std::string *in = nullptr,
                              std::string *out = nullptr);
  modelbox::Status RunTemplateCommand(const httplib::Request &request,
                                      httplib::Response &response,
                                      const std::string &cmd);
  modelbox::Status SaveGraph(const httplib::Request &request);
  modelbox::Status ReadProjectName(const std::string &path, std::string &name);
  bool IsModelboxProjectDir(std::string &path);
  std::shared_ptr<modelbox::HttpListener> listener_;
  std::string web_root_;
  std::string demo_path_;
  std::string server_ip_;
  std::string server_port_;
  // for test setting
  std::string template_cmd_;
  std::string template_cmd_env_;
  std::string template_dir_;
  std::vector<std::string> acl_white_list_;
  bool enable_{false};
  std::string url_;
};

#endif  // MODELBOX_MODELBOX_EDITOR_PLUGIN_H_
