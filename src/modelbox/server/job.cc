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
#include <modelbox/flow.h>
#include <modelbox/server/job.h>
#include <modelbox/server/timer.h>

constexpr uint64_t HEART_BEAT_PERIOD_MS = 60 * 1000;

namespace modelbox {

Job::Job(const std::string& job_name, const std::string& graph_path)
    : job_name_(job_name), graph_path_(graph_path) {}

Job::Job(const std::string& job_name, const std::string& graph_name,
         const std::string& graph)
    : job_name_(job_name), graph_name_(graph_name), graph_(graph) {}

Job::~Job() {
  if (flow_ != nullptr) {
    flow_->Stop();
    flow_ = nullptr;
  }
}

std::string Job::JobStatusToString(JobStatus status) {
  if ((int)status >= (int)(JOB_STATUS_NOTEXIST + 1)) {
    return "";
  }

  return job_str_status[status];
}

std::string Job::JobStatusString() { return JobStatusToString(GetJobStatus()); }

Status Job::Init() {
  flow_ = std::make_shared<modelbox::Flow>();
  Status status;
  status_ = JOB_STATUS_CREATING;
  if (graph_path_.length() > 0) {
    status = flow_->Init(graph_path_);
  }
  if (graph_.length() > 0) {
    status = flow_->Init(graph_name_, graph_);
  }

  if (!status) {
    MBLOG_ERROR << "flow init failed: " << status;
    SetError(status);
  }

  return status;
}

Status Job::Build() {
  if (flow_ == nullptr) {
    Status status = {STATUS_SHUTDOWN, "Job is shutdown"};
    SetError(status);
    return status;
  }

  auto retval = flow_->Build();
  if (!retval) {
    SetError(retval);
  }

  return retval;
}

void Job::Run() {
  if (flow_ == nullptr) {
    return;
  }

  flow_->RunAsync();
  status_ = JOB_STATUS_RUNNING;

  heart_beat_task_ = std::make_shared<modelbox::TimerTask>([this]() {
    auto job_status = this->GetJobStatus();

    if (job_status != JOB_STATUS_RUNNING) {
      MBLOG_ERROR << "get job[" << this->GetJobName()
                  << "] status:" << this->JobStatusToString(job_status);
    }
  });

  kServerTimer->Schedule(heart_beat_task_, 0, HEART_BEAT_PERIOD_MS, false);
}

void Job::Stop() {
  if (flow_ == nullptr) {
    return;
  }

  flow_->Stop();
}

void Job::Join() {
  if (flow_ == nullptr) {
    return;
  }

  flow_->Wait();
}

JobStatus Job::GetJobStatus() {
  modelbox::Status retval;
  if (status_ != JOB_STATUS_RUNNING) {
    return status_;
  }

  if (flow_ == nullptr) {
    return JOB_STATUS_PENDING;
  }

  auto status = flow_->Wait(-1, &retval);
  switch (status.Code()) {
    case modelbox::STATUS_SUCCESS:
      if (retval == modelbox::STATUS_OK || retval == modelbox::STATUS_STOP ||
          retval == STATUS_SHUTDOWN) {
        status_ = JOB_STATUS_SUCCEEDED;
        return status_;
      }
      break;
    case modelbox::STATUS_NORESPONSE:
    case modelbox::STATUS_BUSY:
      return JOB_STATUS_RUNNING;
      break;
    default:
      SetError(status);
      flow_->Stop();
      return JOB_STATUS_FAILED;
      break;
  }

  return JOB_STATUS_UNKNOWN;
}

ErrorInfo Job::GetErrorInfo() { return error_info_; }

std::string Job::GetErrorMsg() {
  std::string msg;

  if (error_info_.error_code_.length() > 0) {
    msg = error_info_.error_code_;
  }

  if (error_info_.error_msg_.length() > 0) {
    if (msg.length() > 0) {
      msg += ", ";
    }

    msg += error_info_.error_msg_;
  }

  return msg;
}

void Job::SetErrorInfo(ErrorInfo& errorInfo) { error_info_ = errorInfo; }

void Job::ClearErrorInfo() {
  error_info_.error_code_ = "";
  error_info_.error_msg_ = "";
}

void Job::SetErrorInfo(const std::string& code, const std::string& msg) {
  error_info_.error_code_ = code;
  error_info_.error_msg_ = msg;
}

void Job::SetError(const modelbox::Status& status) {
  error_info_.error_msg_ = status.WrapErrormsgs();
  status_ = JOB_STATUS_FAILED;
}

void Job::SetJobName(const std::string& job_name) { job_name_ = job_name; }

std::string Job::GetJobName() { return job_name_; }

std::shared_ptr<modelbox::Flow> Job::GetFlow() { return flow_; }

std::shared_ptr<TaskManager> Job::CreateTaskManger(int limit_task_count) {
  auto task_manager = std::make_shared<TaskManager>(flow_, limit_task_count);
  return task_manager;
}

}  // namespace modelbox
