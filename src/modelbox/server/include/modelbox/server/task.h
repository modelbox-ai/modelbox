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

#ifndef MODELBOX_SERVER_TASK_H_
#define MODELBOX_SERVER_TASK_H_
#include <modelbox/base/status.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "modelbox/base/uuid.h"
#include "modelbox/flow.h"
#include "modelbox/session_context.h"
#include "modelbox/virtual_node.h"
namespace modelbox {

enum TaskStatus { UNKNOWN, WAITING, WORKING, STOPPED, ABNORMAL, FINISHED };
class TaskManager;
class OneShotTask;

using TaskDataCallback = std::function<void(OneShotTask*, const OutputBufferList &)>;
using TaskStatusCallback = std::function<void(OneShotTask*, TaskStatus)>;

class Task : public std::enable_shared_from_this<Task> {
 public:
  Task();
  virtual ~Task();

  /**
   * @brief Create buffer list
   * @return buffer list pointer
   */
  std::shared_ptr<BufferList> CreateBufferList();

  /**
   * @brief Set content to session
   * @param key content key
   * @param content pointer to content
   * @return set result
   */
  Status SetSessionContent(const std::string& key,
                           std::shared_ptr<void> content);

  /**
   * @brief Get session config to set before task run
   * @return ref to session config
   */
  std::shared_ptr<modelbox::Configuration> GetSessionConfig();

  /**
   * @brief Set data meta to port
   * @param port_name port name
   * @param meta pointer to meta
   * @return set result
   */
  Status SetDataMeta(const std::string& port_name,
                     std::shared_ptr<DataMeta> meta);

  /**
   * @brief Get task status
   * @return task status
   */
  TaskStatus GetTaskStatus();

  /**
   * @brief Get task last error
   * @return task error
   */
  std::shared_ptr<FlowUnitError> GetLastError();

  /**
   * @brief Get task id
   * @return task id
   */
  std::string GetTaskId();

  /**
   * @brief Start task
   * @return task start result
   */
  Status Start();

  /**
   * @brief Stop task
   * @return task stop result
   */
  Status Stop();

  /**
   * @brief Get task uuid
   * @return task uuid
   */
  std::string GetUUID();

 protected:
  /**
   * @brief Feed data to task
   * @return feed result
   */
  virtual Status FeedData() = 0;

  /**
   * @brief Fetch data from task
   * @param fetch_status task status
   * @param output_buf task output data
   */
  virtual void FetchData(Status fetch_status, OutputBufferList& output_buf) = 0;

  /**
   * @brief Pointer to external data
   */
  std::weak_ptr<ExternalDataMap> external_data_;

 private:
  friend class TaskManager;
  void SetTaskManager(const std::shared_ptr<TaskManager>& task_manager);
  void UpdateTaskStatus(TaskStatus task_status);
  Status SendData();
  std::shared_ptr<ExternalDataMap> GetExternalData();
  bool IsRready();

  std::atomic<TaskStatus> status_{UNKNOWN};
  std::shared_ptr<Flow> flow_;
  std::weak_ptr<TaskManager> task_manager_;
  std::atomic<bool> already_submit_;
  std::string task_uuid_;
  std::mutex lock_;
  std::condition_variable cv_;
};

class OneShotTask : public Task {
 public:
  friend class TaskManager;

  /**
   * @brief One off data task
   */
  OneShotTask();
  ~OneShotTask() override;

  /**
   * @brief Fill data to task
   * @param data data list
   * @return feed result
   */
  Status FillData(
      std::unordered_map<std::string, std::shared_ptr<BufferList>>& data);

  /**
   * @brief Register data callback function
   * @param callback data callback function
   */
  void RegisterDataCallback(const TaskDataCallback& callback);

  /**
   * @brief Register task status function
   * @param callback status callback function
   */
  void RegisterStatusCallback(const TaskStatusCallback& callback);

 protected:
  /**
   * @brief Feed data to task
   * @return feed result
   */
  Status FeedData() override;

  /**
   * @brief Fetch data from task
   * @param fetch_status task status
   * @param output_buf task output data
   */
  void FetchData(Status fetch_status, OutputBufferList& output_buf) override;

 private:
  TaskDataCallback GetDataCallback();
  TaskStatusCallback GetStatusCallback();
  std::unordered_map<std::string, std::shared_ptr<BufferList>> data_;

  TaskDataCallback data_callback_;
  TaskStatusCallback status_callback_;
};

}  // namespace modelbox
#endif
