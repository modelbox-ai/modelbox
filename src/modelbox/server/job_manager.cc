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

#include <modelbox/base/log.h>
#include <modelbox/server/job_manager.h>

namespace modelbox {

std::shared_ptr<modelbox::Job> JobManager::CreateJob(
    const std::string& job_name, const std::string& graph_file_path) {
  std::lock_guard<std::mutex> lock(job_lock_);
  if (jobs_.find(job_name) != jobs_.end()) {
    MBLOG_WARN << "job " << job_name << " is running";
    modelbox::StatusError = {modelbox::STATUS_ALREADY, "job already running"};
    return nullptr;
  }
  auto job = std::make_shared<modelbox::Job>(job_name, graph_file_path);
  jobs_.emplace(std::make_pair(job_name, job));
  return job;
}

std::shared_ptr<modelbox::Job> JobManager::CreateJob(
    const std::string& job_name, const std::string& graph_name,
    const std::string& graph) {
  std::lock_guard<std::mutex> lock(job_lock_);
  if (jobs_.find(job_name) != jobs_.end()) {
    std::string msg = "job " + job_name + " is running";
    MBLOG_WARN << msg;
    modelbox::StatusError = {modelbox::STATUS_INPROGRESS, msg};
    return nullptr;
  }
  auto job = std::make_shared<modelbox::Job>(job_name, graph_name, graph);
  jobs_.emplace(std::make_pair(job_name, job));
  return job;
}

modelbox::JobStatus JobManager::QueryJobStatus(const std::string& job_name) {
  std::lock_guard<std::mutex> lock(job_lock_);
  auto iter = jobs_.find(job_name);
  if (iter != jobs_.end()) {
    return iter->second->GetJobStatus();
  }

  return modelbox::JobStatus::JOB_STATUS_NOTEXIST;
}

std::string JobManager::QueryJobStatusString(const std::string& job_name) {
  std::lock_guard<std::mutex> lock(job_lock_);
  auto iter = jobs_.find(job_name);
  if (iter != jobs_.end()) {
    return iter->second->JobStatusString();
  }

  return modelbox::Job::JobStatusToString(
      modelbox::JobStatus::JOB_STATUS_NOTEXIST);
}

std::string JobManager::GetJobErrorMsg(const std::string& job_name) {
  std::lock_guard<std::mutex> lock(job_lock_);
  auto iter = jobs_.find(job_name);
  if (iter != jobs_.end()) {
    return iter->second->GetErrorMsg();
  }

  return "";
}

bool JobManager::DeleteJob(std::string job_name) {
  MBLOG_INFO << "delete job : " << job_name;
  std::lock_guard<std::mutex> lock(job_lock_);
  if (jobs_.find(job_name) != jobs_.end()) {
    jobs_.erase(job_name);
  } else {
    MBLOG_WARN << "can not delete job : " << job_name << ", no this job";
    return false;
  }

  return true;
}

std::shared_ptr<modelbox::Job> JobManager::GetJob(std::string job_name) {
  std::lock_guard<std::mutex> lock(job_lock_);
  if (jobs_.find(job_name) == jobs_.end()) {
    return nullptr;
  }

  return jobs_[job_name];
}

std::vector<std::shared_ptr<modelbox::Job>> JobManager::GetJobList() {
  std::lock_guard<std::mutex> lock(job_lock_);
  std::vector<std::shared_ptr<modelbox::Job>> jobs;
  for (auto job : jobs_) {
    jobs.push_back(job.second);
  }
  return jobs;
}

std::unordered_map<std::string, std::shared_ptr<modelbox::Job>>
JobManager::GetJobMap() {
  return jobs_;
}

}  // namespace modelbox