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

#ifndef MODELBOX_JOB_MANAGER_H_
#define MODELBOX_JOB_MANAGER_H_

#include <modelbox/server/job.h>

#include <utility>

namespace modelbox {
/**
 * @brief Job management, used to manage multiple flows, provide management api
 * and query api of flowcharts.
 */
class JobManager {
 public:
  /**
   * @brief Create a new job with flow graph
   * @param job_name name of the job
   * @param graph_file_path file path of flow graph in modelbox graph format
   * @return a new job
   */
  std::shared_ptr<modelbox::Job> CreateJob(const std::string& job_name,
                                           const std::string& graph_file_path);

  /**
   * @brief Create a new job with inline flow graph
   * @param job_name name of the job
   * @param graph_name name of flow graph
   * @param graph inline string of graph, in modelbox graph format.
   * @return a new job
   */
  std::shared_ptr<modelbox::Job> CreateJob(const std::string& job_name,
                                           const std::string& graph_name,
                                           const std::string& graph);

  /**
   * @brief Delete job
   * @param job_name name of the job
   * @return delete result
   */
  bool DeleteJob(const std::string& job_name);

  /**
   * @brief Get job by job name
   * @param job_name name of the job
   * @return the job object
   */
  std::shared_ptr<modelbox::Job> GetJob(const std::string& job_name);

  /**
   * @brief Get all jobs
   * @return all jobs
   */
  std::vector<std::shared_ptr<modelbox::Job>> GetJobList();

  /**
   * @brief Get all jobs in map container, key is job name, value is job object
   * @return all jobs in map container.
   */
  std::unordered_map<std::string, std::shared_ptr<modelbox::Job>> GetJobMap();

  /**
   * @brief Get job status
   * @param job_name name of the job
   * @return job status
   */
  modelbox::JobStatus QueryJobStatus(const std::string& job_name);

  /**
   * @brief Get job status in string
   * @param job_name name of the job
   * @return job status in string
   */
  std::string QueryJobStatusString(const std::string& job_name);

  /**
   * @brief Get job error message
   * @param job_name name of the job
   * @return job error message.
   */
  std::string GetJobErrorMsg(const std::string& job_name);

 private:
  std::unordered_map<std::string, std::shared_ptr<modelbox::Job>> jobs_;
  std::mutex job_lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_JOB_MANAGER_H_