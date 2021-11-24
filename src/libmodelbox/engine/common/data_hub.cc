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


#include "engine/common/data_hub.h"

#include <algorithm>

#include "modelbox/base/log.h"

namespace modelbox {

PriorityPort::PriorityPort(const std::shared_ptr<IPort>& port)
    : active_time_(GetCurrentTime()), is_running_(false), port_(port) {
  if (port) {
    priority_ = port->GetPriority();
  } else {
    priority_ = std::numeric_limits<int>::min();
  }
}

const std::shared_ptr<IPort>& PriorityPort::GetPort() const { return port_; }
std::shared_ptr<IPort> PriorityPort::GetPort() { return port_; }

void PriorityPort::UpdateActiveTime() { active_time_ = GetCurrentTime(); }

int32_t PriorityPort::GetPriority() const { return priority_; }
void PriorityPort::SetPriority(int32_t priority) { priority_ = priority; }
// TODO port update dynamic priority
void PriorityPort::UpdatePriority() { priority_ = port_->GetPriority(); }

void PriorityPort::SetPushEventCallBack(const PushCallBack& func) {
  port_->SetPushEventCallBack(func);
};

void PriorityPort::SetPopEventCallBack(const PopCallBack& func) {
  port_->SetPopEventCallBack(func);
};

bool PriorityPort::HasData() { return !port_->Empty(); }

void PriorityPort::SetRuning(bool flag) { is_running_ = flag; }
bool PriorityPort::IsRunning() { return is_running_; }

bool PriorityPort::IsActivated() { return port_->IsActivated(); }

std::shared_ptr<NodeBase> PriorityPort::GetNode() const {
  auto port = GetPort();
  if (!port) {
    return nullptr;
  }

  return port->GetNode();
}

bool PortCompare::operator()(const std::shared_ptr<PriorityPort>& left,
                             const std::shared_ptr<PriorityPort>& right) {
  if (left->port_ == right->port_) {
    return false;
  }

  if (left->priority_ != right->priority_) {
    return left->priority_ > right->priority_;
  }

  if (left->active_time_ != right->active_time_) {
    return left->active_time_ < right->active_time_;
  }

  return left->port_ < right->port_;
}

DefaultDataHub::DefaultDataHub()
    : priority_ports_(), active_ports_(), active_mutex_(), cv_() {}

DefaultDataHub::~DefaultDataHub() {
  std::lock_guard<std::mutex> guard(active_mutex_);
  active_ports_.clear();

  for (auto priority_port : priority_ports_) {
    priority_port->SetPushEventCallBack(nullptr);
    priority_port->SetPopEventCallBack(nullptr);
  }

  priority_ports_.clear();
}

Status DefaultDataHub::AddPort(const std::shared_ptr<PriorityPort>& port) {
  std::lock_guard<std::mutex> lock(active_mutex_);
  if (!port) {
    MBLOG_WARN << "port is nullptr";
    return STATUS_INVALID;
  }

  auto iter = std::find(priority_ports_.begin(), priority_ports_.end(), port);
  if (priority_ports_.end() != iter) {
    MBLOG_WARN << port << " port has been added to the data hub.";
    return STATUS_OK;
  }

  priority_ports_.push_back(port);

  auto push_call_back = std::bind(&DefaultDataHub::PortEventCallback, this,
                                  port, std::placeholders::_1);
  port->SetPushEventCallBack(push_call_back);

  auto pop_call_back =
      std::bind(&DefaultDataHub::PortEventCallback, this, port, false);
  port->SetPopEventCallBack(pop_call_back);

  // If there is data in the port before the port is added, it needs to be added
  // to the active port to solve the problem that the flow unit starts to run
  // before the scheduler is initialized
  if (port->HasData() && !port->IsRunning() && port->IsActivated()) {
    UpdateActivePort(port, true);
  }

  return STATUS_OK;
}

Status DefaultDataHub::AddToActivePort(
    const std::shared_ptr<PriorityPort>& port) {
  std::lock_guard<std::mutex> lock(active_mutex_);
  if (!port) {
    return {STATUS_INVALID, "active_port is nullptr"};
  }

  port->SetRuning(false);
  if (port->HasData() && port->IsActivated()) {
    active_ports_.insert(port);
    cv_.notify_one();
  }

  return STATUS_OK;
}

Status DefaultDataHub::AddToActivePort(
    std::vector<std::shared_ptr<PriorityPort>>& ports) {
  std::lock_guard<std::mutex> lock(active_mutex_);
  for (auto& port : ports) {
    if (!port) {
      MBLOG_WARN << "active_port is nullptr";
      continue;
    }

    port->SetRuning(false);
    if (port->HasData() && port->IsActivated()) {
      active_ports_.insert(port);
    }
  }

  cv_.notify_all();
  return STATUS_OK;
}

void DefaultDataHub::UpdateActivePort(std::shared_ptr<PriorityPort> port,
                                      bool update_active_time) {
  auto it = active_ports_.find(port);
  if (active_ports_.end() != it) {
    active_ports_.erase(it);
  }

  if (update_active_time) {
    port->UpdateActiveTime();
  }

  port->UpdatePriority();
  active_ports_.insert(port);
};

void DefaultDataHub::PortEventCallback(std::shared_ptr<PriorityPort> port,
                                       bool update_active_time) {
  std::lock_guard<std::mutex> lock(active_mutex_);
  if (!port) {
    MBLOG_WARN << "port is nullptr";
    return;
  }

  if (port->HasData() && !port->IsRunning() && port->IsActivated()) {
    UpdateActivePort(port, update_active_time);
    cv_.notify_one();
  }
}

/**
 * @brief Get the highest priority port that may contain data
 *
 * @param active_port Return the port containing the data
 * @param timeout Specify the timeout period, 0 means blocking
 * @return Status
 *  @retval STATUS_TIMEDOUT means timeout
 *  @retval STATUS_NODATA means no active data
 *  @retval STATUS_OK means success
 */
Status DefaultDataHub::SelectActivePort(
    std::shared_ptr<PriorityPort>* active_port, int64_t timeout) {
  auto pred = [this] { return !active_ports_.empty(); };
  std::unique_lock<std::mutex> lock(active_mutex_);

  if (timeout > 0) {
    if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout), pred)) {
      return STATUS_TIMEDOUT;
    }
  } else if (0 == timeout) {
    cv_.wait(lock, pred);
  } else {
    if (active_ports_.empty()) {
      return STATUS_NODATA;
    }
  }

  auto it = active_ports_.begin();
  *active_port = *it;
  (*active_port)->SetRuning(true);
  active_ports_.erase(it);

  return STATUS_OK;
}

void DefaultDataHub::RemoveFromActivePort(
    std::vector<std::shared_ptr<PriorityPort>>& ports) {
  std::unique_lock<std::mutex> lock(active_mutex_);
  auto it = active_ports_.begin();
  for (auto& port : ports) {
    port->SetRuning(true);
    it = active_ports_.find(port);
    if (it != active_ports_.end()) {
      active_ports_.erase(it);
    }
  }
}

size_t DefaultDataHub::GetPortNum() const { return priority_ports_.size(); }

size_t DefaultDataHub::GetActivePortNum() const { return active_ports_.size(); }

}  // namespace modelbox
