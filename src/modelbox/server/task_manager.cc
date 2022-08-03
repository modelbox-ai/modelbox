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

#include <modelbox/server/task.h>
#include <modelbox/server/task_manager.h>

#include <utility>
namespace modelbox {

TaskManager::TaskManager(std::shared_ptr<Flow> flow, uint32_t task_num_limits) {
  flow_ = std::move(flow);
  thread_pool_ = std::make_shared<ThreadPool>(0, task_num_limits);
  task_num_limits_ = task_num_limits;
  avaiable_task_counts_ = 0;
  thread_run_ = true;
  selector_ = std::make_shared<ExternalDataSelect>();
}
TaskManager::~TaskManager() { Stop(); }

Status TaskManager::Submit(const std::shared_ptr<Task>& task) {
  std::unique_lock<std::mutex> guard(new_del_lock_);
  if (avaiable_task_counts_ >= task_num_limits_) {
    MBLOG_INFO << "Running Task exceed task_num_limits "
               << avaiable_task_counts_;
    return STATUS_SUCCESS;
  }
  if (thread_pool_ == nullptr) {
    return {STATUS_NOTFOUND, "thread_pool not exist"};
  }
  avaiable_task_counts_++;
  guard.unlock();

  thread_pool_->Submit(task->GetTaskId(), &Task::SendData, task.get());

  return STATUS_SUCCESS;
}

void TaskManager::StartWaittingTask() {
  std::unique_lock<std::mutex> guard(map_lock_);
  auto task_iter = task_maps_.begin();
  if (avaiable_task_counts_ >= task_num_limits_) {
    return;
  }

  while (task_iter != task_maps_.end()) {
    auto task = task_iter->second;
    if (task->IsRready()) {
      Submit(task);
    }
    task_iter++;
  }
}

void TaskManager::ReceiveWork() {
  while (thread_run_) {
    std::list<std::shared_ptr<ExternalDataMap>> external_list;
    auto select_status = selector_->SelectExternalData(
        external_list, std::chrono::milliseconds(200));

    for (const auto& external : external_list) {
      std::unique_lock<std::mutex> map_guard(map_lock_);
      auto task_iter = external_task_maps_.find(external);
      if (task_iter == external_task_maps_.end()) {
        MBLOG_DEBUG << "task already deleted";
        continue;
      }
      auto task = task_iter->second;
      map_guard.unlock();

      modelbox::OutputBufferList map_buffer_list;
      auto status = external->Recv(map_buffer_list);

      std::unique_lock<std::mutex> guard(new_del_lock_);
      if (status == STATUS_INVALID) {
        MBLOG_WARN << "recv external failed";
        auto error = external->GetLastError();
        if (error->GetDesc() == "EOF") {
          task->UpdateTaskStatus(STOPPED);
        } else {
          task->UpdateTaskStatus(ABNORMAL);
        }
        avaiable_task_counts_--;
      } else if (status == STATUS_EOF) {
        MBLOG_DEBUG << "recv external finished";
        task->UpdateTaskStatus(FINISHED);
        avaiable_task_counts_--;
      }
      guard.unlock();
      task->FetchData(status, map_buffer_list);
    }
    StartWaittingTask();
  }
}

Status TaskManager::Start() {
  receive_thread_ =
      std::make_shared<std::thread>(&TaskManager::ReceiveWork, this);
  return STATUS_SUCCESS;
}

void TaskManager::Stop() {
  if (receive_thread_) {
    thread_run_ = false;
    receive_thread_->join();
    receive_thread_ = nullptr;
  }
  if (thread_pool_) {
    thread_pool_->Shutdown();
  }
  selector_ = nullptr;
}

std::shared_ptr<Task> TaskManager::CreateTask(TaskType task_type) {
  std::shared_ptr<Task> task = nullptr;
  switch (task_type) {
    case TASK_ONESHOT:
    default:
      task = std::make_shared<OneShotTask>();
      break;
  }
  RegisterTask(task);
  return task;
}

void TaskManager::SetTaskNumLimit(int task_limits) {
  std::lock_guard<std::mutex> guard(new_del_lock_);
  task_num_limits_ = task_limits;
}

std::shared_ptr<Flow> TaskManager::GetFlow() { return flow_; }
std::shared_ptr<ExternalDataSelect> TaskManager::GetSelector() {
  return selector_;
}

Status TaskManager::DeleteTaskById(const std::string& taskid) {
  std::unique_lock<std::mutex> guard(map_lock_);
  if (task_maps_.find(taskid) == task_maps_.end()) {
    return {STATUS_NOTFOUND, "task can not be found"};
  }
  auto task = task_maps_[taskid];
  auto external_data = task->GetExternalData();
  task_maps_.erase(taskid);
  guard.unlock();

  task->Stop();

  selector_->RemoveExternalData(external_data);
  external_task_maps_.erase(external_data);
  return STATUS_SUCCESS;
}

std::shared_ptr<Task> TaskManager::GetTaskById(const std::string& taskid) {
  std::unique_lock<std::mutex> guard(map_lock_);
  if (task_maps_.find(taskid) == task_maps_.end()) {
    return nullptr;
  }
  return task_maps_[taskid];
}

uint32_t TaskManager::GetTaskCount() {
  std::unique_lock<std::mutex> guard(map_lock_);
  return task_maps_.size();
}

std::vector<std::shared_ptr<Task>> TaskManager::GetAllTasks() {
  std::unique_lock<std::mutex> guard(map_lock_);
  std::vector<std::shared_ptr<Task>> task_list;
  auto task_iter = task_maps_.begin();
  while (task_iter != task_maps_.end()) {
    task_list.push_back(task_iter->second);
    task_iter++;
  }
  return task_list;
}

void TaskManager::RegisterTask(const std::shared_ptr<Task>& task) {
  std::unique_lock<std::mutex> guard(map_lock_);
  task->SetTaskManager(shared_from_this());
  auto external_data = task->GetExternalData();
  task_maps_.emplace(task->GetTaskId(), task);
  external_task_maps_.emplace(external_data, task);
}

}  // namespace modelbox