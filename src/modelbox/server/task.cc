
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

#include <modelbox/base/utils.h>
#include <modelbox/server/task.h>
#include <modelbox/server/task_manager.h>

namespace modelbox {

Task::Task() {
  flow_ = nullptr;
  status_ = WAITING;
  already_submit_ = false;
}

Task::~Task() {}

std::string Task::GetUUID() { return task_uuid_; }

std::shared_ptr<BufferList> Task::CreateBufferList() {
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return nullptr;
  }
  return external_data->CreateBufferList();
}

Status Task::SetDataMeta(const std::string& port_name,
                         std::shared_ptr<DataMeta> meta) {
  std::unique_lock<std::mutex> guard(lock_);
  auto external_data = external_data_.lock();
  if (external_data != nullptr) {
    return {STATUS_NOTFOUND, "external_data not found"};
  }

  if (status_ != WAITING) {
    return {STATUS_INVALID, "the task already start"};
  }

  return external_data->SetOutputMeta(port_name, meta);
}

Status Task::SetSessionContent(const std::string& key,
                               std::shared_ptr<void> content) {
  std::unique_lock<std::mutex> guard(lock_);
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return {STATUS_NOTFOUND, "external_data not found"};
  }

  if (status_ != WAITING) {
    return {STATUS_INVALID, "the task already start"};
  }

  auto session_ctx = external_data->GetSessionContext();
  if (session_ctx == nullptr) {
    return {STATUS_NOTFOUND, "session_ctx not found"};
  }

  session_ctx->SetPrivate(key, content);
  return STATUS_SUCCESS;
}

std::shared_ptr<modelbox::Configuration> Task::GetSessionConfig() {
  std::unique_lock<std::mutex> guard(lock_);
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return nullptr;
  }

  return external_data->GetSessionConfig();
}

std::string Task::GetTaskId() { return task_uuid_; }

TaskStatus Task::GetTaskStatus() {
  std::unique_lock<std::mutex> guard(lock_);
  return status_;
}

std::shared_ptr<FlowUnitError> Task::GetLastError() {
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return nullptr;
  }
  return external_data->GetLastError();
}

Status Task::Stop() {
  std::unique_lock<std::mutex> guard(lock_);
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return {STATUS_NOTFOUND, "external_data not found"};
  }

  if (status_ == STOPPED || status_ == ABNORMAL || status_ == FINISHED) {
    return {STATUS_INVALID, "task is already finished"};
  }

  auto status = external_data->Close();
  if (status != STATUS_SUCCESS) {
    return status;
  }
  cv_.wait(guard);
  return STATUS_SUCCESS;
}

bool Task::IsRready() {
  std::unique_lock<std::mutex> guard(lock_);
  if (already_submit_ && status_ == WAITING) {
    return true;
  }
  return false;
}

Status Task::Start() {
  std::lock_guard<std::mutex> guard(lock_);
  auto task_manager = task_manager_.lock();
  if (task_manager == nullptr) {
    return {STATUS_NOTFOUND, "task manger is empty"};
  }

  if (status_ != WAITING) {
    return {STATUS_PERMIT, "task is already started"};
  }

  if (already_submit_) {
    return {STATUS_PERMIT, "task is in the waitting queue"};
  }
  already_submit_ = true;

  auto status = task_manager->Submit(shared_from_this());
  if (status != STATUS_SUCCESS) {
    return status;
  }

  return STATUS_SUCCESS;
}

void Task::SetTaskManager(std::shared_ptr<TaskManager> task_manager) {
  task_manager_ = task_manager;
  flow_ = task_manager->GetFlow();
  std::shared_ptr<ExternalDataMap> external_data;
  if (flow_ != nullptr) {
    external_data = flow_->CreateExternalDataMap();
  }

  std::shared_ptr<SessionContext> session_ctx;
  if (external_data != nullptr) {
    session_ctx = external_data->GetSessionContext();
  }

  if (session_ctx != nullptr) {
    // Set task_uuid equal to session_id to establish connection between task
    // and graph session
    task_uuid_ = session_ctx->GetSessionId();
  }

  auto selector = task_manager->GetSelector();
  selector->RegisterExternalData(external_data);
  external_data_ = external_data;
}

std::shared_ptr<ExternalDataMap> Task::GetExternalData() {
  auto external_data = external_data_.lock();
  return external_data;
}

Status Task::SendData() {
  std::unique_lock<std::mutex> guard(lock_);
  if (status_ != WAITING) {
    MBLOG_ERROR << "Task " << task_uuid_ << " already started";
    return {STATUS_INVALID, "the task already start"};
  }

  status_ = WORKING;
  guard.unlock();
  while (true) {
    auto status = FeedData();
    if (status == STATUS_EOF) {
      break;
    }
    if (status != STATUS_SUCCESS) {
      MBLOG_ERROR << "Feed data to task " << task_uuid_
                  << " failed, ret: " << status;
      Stop();
      return status;
    }
  }
  return STATUS_SUCCESS;
}

void Task::UpdateTaskStatus(TaskStatus task_status) {
  status_ = task_status;
  cv_.notify_one();
}

OneShotTask::OneShotTask() : Task() {}
OneShotTask::~OneShotTask() {}

Status PreCheckData(
    std::unordered_map<std::string, std::shared_ptr<BufferList>> datas) {
  if (datas.empty()) {
    return {STATUS_NODATA, "no data available to start task"};
  }
  int buffer_number = -1;
  auto data_iter = datas.begin();
  while (data_iter != datas.end()) {
    auto port_name = data_iter->first;
    auto buffer_list = data_iter->second;
    if (buffer_number == -1) {
      buffer_number = buffer_list->Size();
    }

    if (size_t(buffer_number) != buffer_list->Size()) {
      return {STATUS_FAULT, port_name + " size not equal to the first"};
    }
    data_iter++;
  }
  return STATUS_OK;
}

Status OneShotTask::FillData(
    std::unordered_map<std::string, std::shared_ptr<BufferList>>& data) {
  auto status = PreCheckData(data);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  auto data_iter = data.begin();
  while (data_iter != data.end()) {
    data_.emplace(data_iter->first, data_iter->second);
    data_iter++;
  }
  return STATUS_SUCCESS;
}

Status OneShotTask::FeedData() {
  auto data_iter = data_.begin();
  auto external_data = external_data_.lock();
  if (external_data == nullptr) {
    return {STATUS_NOTFOUND, "external_data not found"};
  }
  if (data_.size() == 0) {
    return {STATUS_NODATA, "No data avalaible"};
  }
  while (data_iter != data_.end()) {
    auto port_name = data_iter->first;
    auto buffer_list = data_iter->second;
    if (buffer_list->Size() == 0) {
      return {STATUS_NODATA, port_name + " buffer_list size is 0"};
    }
    auto status = external_data->Send(port_name, buffer_list);
    if (status != STATUS_SUCCESS) {
      return status;
    }
    data_iter++;
  }
  external_data->Shutdown();
  return STATUS_EOF;
}

void OneShotTask::FetchData(Status fetch_status, OutputBufferList& output_buf) {
  if (fetch_status == STATUS_SUCCESS) {
    auto data_callback = GetDataCallback();
    if (data_callback) {
      data_callback(this, output_buf);
    }
    MBLOG_DEBUG << "recv external";
  } else {
    auto status_callback = GetStatusCallback();
    if (status_callback) {
      status_callback(this, GetTaskStatus());
    }
  }
}

void OneShotTask::RegisterDataCallback(TaskDataCallback callback) {
  data_callback_ = callback;
}

TaskDataCallback OneShotTask::GetDataCallback() { return data_callback_; }

void OneShotTask::RegisterStatusCallback(TaskStatusCallback callback) {
  status_callback_ = callback;
}

TaskStatusCallback OneShotTask::GetStatusCallback() { return status_callback_; }

}  // namespace modelbox