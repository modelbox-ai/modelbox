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

#ifndef MODELBOX_JOB_H_
#define MODELBOX_JOB_H_

#include <modelbox/flow.h>
#include <modelbox/server/task_manager.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace modelbox {

/**
 * @brief Job status
 */
enum JobStatus {
  /// @brief Job is creating
  JOB_STATUS_CREATING,
  /// @brief Job is running
  JOB_STATUS_RUNNING,
  /// @brief Job ran success.
  JOB_STATUS_SUCCEEDED,
  /// @brief Job ran failed.
  JOB_STATUS_FAILED,
  /// @brief Job is pending.
  JOB_STATUS_PENDING,
  /// @brief Job is deleting.
  JOB_STATUS_DELETEING,
  /// @brief Job status is unknown.
  JOB_STATUS_UNKNOWN,
  /// @brief Job is not exist.
  JOB_STATUS_NOTEXIST
};

constexpr const char* job_str_status[] = {"CREATEING", "RUNNING", "SUCCEEDED",
                                          "FAILED",    "PENDING", "DELETEING",
                                          "UNKNOWN",   "NOTEXIST"};

/**
 * @brief Job error info
 */
struct ErrorInfo {
  /// @brief Job error code
  std::string error_code_;
  /// @brief Job error message
  std::string error_msg_;
};

class Job {
 public:
  /**
   * @brief Create job
   * @param job_name job name
   * @param graph_path graph file
   */
  Job(std::string job_name, std::string graph_path);

  /**
   * @brief Create job
   * @param job_name job name
   * @param graph_name graph name
   * @param graph graph in string
   */
  Job(std::string job_name, std::string graph_name, std::string graph);

  virtual ~Job();

  /**
   * @brief Init job
   * @return init result
   */
  Status Init();

  /**
   * @brief Build graph
   * @return build result
   */
  Status Build();

  /**
   * @brief Run graph
   */
  void Run();

  /**
   * @brief Stop graph
   */
  void Stop();

  /**
   * @brief Wait graph finish
   */
  void Join();

  /**
   * @brief Get job status
   * @return job status
   */
  JobStatus GetJobStatus();

  /**
   * @brief Get job status in string
   * @return job status in string
   */
  std::string JobStatusString();

  /**
   * @brief Convert job status to string
   * @param status job status
   * @return job status in string
   */
  static std::string JobStatusToString(JobStatus status);

  /**
   * @brief Get job error info
   * @return job error info
   */
  ErrorInfo GetErrorInfo();

  /**
   * @brief Get job error info in string
   * @return job error info in string
   */
  std::string GetErrorMsg();

  /**
   * @brief Clear job error info
   */
  void ClearErrorInfo();

  /**
   * @brief Set error info to job
   * @param errorInfo error info
   */
  void SetErrorInfo(ErrorInfo& errorInfo);

  /**
   * @brief Set error info to job
   * @param code error code
   * @param msg error msg
   */
  void SetErrorInfo(const std::string& code, const std::string& msg);

  /**
   * @brief Set error info to job
   * @param status status
   */
  void SetError(const modelbox::Status& status);

  /**
   * @brief Set job name
   * @param job_name job name
   */
  void SetJobName(const std::string& job_name);

  /**
   * @brief Get job name
   * @return job name
   */
  std::string GetJobName();

  /**
   * @brief Get job flow
   * @return job nflow
   */
  std::shared_ptr<modelbox::Flow> GetFlow();

  /**
   * @brief Create task manager
   * @param limit_task_count task threshold
   * @return task manager
   */
  std::shared_ptr<TaskManager> CreateTaskManger(int limit_task_count);

 private:
  std::string job_name_;
  std::string graph_path_;
  std::string graph_name_;
  std::string graph_;
  JobStatus status_{JOB_STATUS_UNKNOWN};
  ErrorInfo error_info_;
  std::shared_ptr<modelbox::Flow> flow_;
  std::shared_ptr<modelbox::TimerTask> heart_beat_task_;
};

}  // namespace modelbox
#endif  // MODELBOX_JOB_H_
