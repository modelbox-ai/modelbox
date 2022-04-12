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

#ifndef MODELBOX_MODELBOX_PLUGIN_H_
#define MODELBOX_MODELBOX_PLUGIN_H_

#include "memory"
#include "modelbox/server/http_helper.h"
#include "modelbox/server/job_manager.h"
#include "modelbox/server/plugin.h"

class ModelboxPlugin : public modelbox::Plugin {
 public:
  ModelboxPlugin(){};
  virtual ~ModelboxPlugin(){};

  bool Init(std::shared_ptr<modelbox::Configuration> config) override;
  bool Start() override;
  bool Stop() override;

  void RegistHandlers();
  bool ParseConfig(std::shared_ptr<modelbox::Configuration> config);
  void RegistCallbacks();
  bool CheckMethodVaild(std::string method);
  bool CheckUrlVaild(std::string url);

 private:
  void HandlerPut(const httplib::Request& request, httplib::Response& response);
  void HandlerGet(const httplib::Request& request, httplib::Response& response);
  void HandlerDel(const httplib::Request& request, httplib::Response& response);

  modelbox::Status CreateLocalJobs();
  modelbox::Status CreateJobByFile(const std::string& job_id,
                                   const std::string& graph_file);
  modelbox::Status CreateJobByString(const std::string& job_id,
                                     const std::string& graph,
                                     const std::string& format);

  modelbox::Status StartJob(std::shared_ptr<modelbox::Job> job);

  modelbox::Status SaveGraphFile(const std::string& job_id,
                                 const std::string& toml_graph);
  bool CheckJobIdValid(std::string job_id);

 private:
  std::string ip_;
  std::string port_;
  std::string default_flow_path_;
  std::string default_project_path_;
  std::string oneshot_flow_path_;

  std::vector<std::string> acl_white_list_;
  std::shared_ptr<modelbox::HttpListener> listener_;
  modelbox::JobManager jobmanager_;
};

#endif  // MODELBOX_MODELBOX_PLUGIN_H_
