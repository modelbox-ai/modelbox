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

#ifndef MODELBOX_SERVER_TASK_MANAGER_H_
#define MODELBOX_SERVER_TASK_MANAGER_H_

#include <modelbox/server/task.h>
#include <modelbox/virtual_node.h>

namespace modelbox {
enum TaskType { TASK_ONESHOT };
class TaskManager : public std::enable_shared_from_this<TaskManager> {
 public:
  /**
   * @brief Task manager
   * @param flow pointer to flow
   * @param task_num_limits task number threshold
   */
  TaskManager(std::shared_ptr<Flow> flow, uint32_t task_num_limits);
  virtual ~TaskManager();

  /**
   * @brief Start task manager
   * @return start result
   */
  Status Start();

  /**
   * @brief Stop task manager
   */
  void Stop();

  /**
   * @brief Create task
   * @return pointer to task
   */
  std::shared_ptr<Task> CreateTask(TaskType task_type);

  /**
   * @brief Delete task by id
   * @param taskid task id
   * @return delete result
   */
  Status DeleteTaskById(const std::string& taskid);

  /**
   * @brief Get task by id
   * @param taskid task id
   * @return task pointer
   */
  std::shared_ptr<Task> GetTaskById(const std::string& taskid);

  /**
   * @brief Get task number
   * @return task number
   */
  uint32_t GetTaskCount();

  /**
   * @brief Get all tasks
   * @return task list
   */
  std::vector<std::shared_ptr<Task>> GetAllTasks();

  /**
   * @brief Set task threshold
   * @param task_limits task threshold
   */
  void SetTaskNumLimit(int task_limits);

  /**
   * @brief Register new task
   * @param task task pointer
   */
  void RegisterTask(const std::shared_ptr<Task>& task);

 private:
  friend class Task;
  void ReceiveWork();
  Status Submit(const std::shared_ptr<Task>& task);
  void StartWaittingTask();
  std::shared_ptr<Flow> GetFlow();
  std::shared_ptr<ExternalDataSelect> GetSelector();

  std::shared_ptr<ThreadPool> thread_pool_;
  std::mutex new_del_lock_;
  std::mutex map_lock_;
  std::shared_ptr<Flow> flow_;
  std::atomic<uint32_t> task_num_limits_;
  std::atomic<uint32_t> avaiable_task_counts_;
  std::shared_ptr<ExternalDataSelect> selector_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_maps_;
  std::map<std::shared_ptr<ExternalDataMap>, std::shared_ptr<Task>>
      external_task_maps_;
  std::shared_ptr<std::thread> receive_thread_;
  std::atomic<bool> thread_run_;
};
}  // namespace modelbox
#endif