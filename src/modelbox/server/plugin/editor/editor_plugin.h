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
  void HandlerSolutionGet(const httplib::Request &request, httplib::Response &response);
  void HandlerSolutionGetList(const httplib::Request &request, httplib::Response &response);
  void HandlerUIGet(const httplib::Request &request, httplib::Response &response);
  void SendFile(const std::string &file_name, httplib::Response &response);
  void HandlerFlowUnitInfoPut(const httplib::Request &request, httplib::Response &response);
  void HandlerFlowUnitInfoGet(const httplib::Request &request, httplib::Response &response);
  void HandlerFlowUnitInfo(const httplib::Request &request, httplib::Response &response,
                           std::shared_ptr<modelbox::Configuration> config);
  void HandlerFlowUnitPut(const httplib::Request &request, httplib::Response &response);
  void HandlerFlowUnitGet(const httplib::Request &request, httplib::Response &response);
  void HandlerProjectPut(const httplib::Request &request, httplib::Response &response);
  void HandlerProjectGet(const httplib::Request &request, httplib::Response &response);
  void HandlerDirectoryGet(const httplib::Request &request, httplib::Response &response);
  void HandlerProject(const httplib::Request &request, httplib::Response &response,
                           std::shared_ptr<modelbox::Configuration> config);
  void SaveAllProject(const httplib::Request &request, httplib::Response &response);
  modelbox::Status SaveGraphFile(const std::string& job_id,
                                 const std::string& toml_graph,
                                 const std::string& path);
  void ConfigJobid(std::string& job_id);
  bool GetHtmlFile(const std::string &in_file, std::string *out_file, std::string *redirect_file);
  modelbox::Status GraphFileToJson(const std::string &file,
                                 std::string &json_data);
  bool CheckBlackDir(std::string dir);

 private:
  std::shared_ptr<modelbox::HttpListener> listener_;
  std::string web_root_;
  std::string solution_path_;
  std::string server_ip_;
  std::string server_port_;
  std::vector<std::string> acl_white_list_;
  bool enable_{false};
  std::string url_;
};

#endif  // MODELBOX_MODELBOX_EDITOR_PLUGIN_H_
