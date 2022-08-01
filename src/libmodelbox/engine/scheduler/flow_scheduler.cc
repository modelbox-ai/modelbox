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

#include "flow_scheduler.h"

#include <modelbox/base/os.h>

namespace modelbox {

constexpr const char* TASK_FLOW_SCHEDUER_NAME = "Flow-Scheduler";
constexpr const char* TASK_FLOW_POOL_NAME = "Flow-Workers";

SchedulerPort::SchedulerPort(const std::string& name)
    : SchedulerPort(name, SIZE_MAX) {}

SchedulerPort::SchedulerPort(const std::string& name, size_t event_capacity)
    : NotifyPort(name, nullptr, 0, event_capacity) {
  queue_ = std::make_shared<SchedulerQueue>(event_capacity);
};

FlowScheduler::FlowScheduler() = default;

FlowScheduler::~FlowScheduler() {
  if (tp_) {
    tp_ = nullptr;
  }
}

Status FlowScheduler::Init(std::shared_ptr<Configuration> config,
                           std::shared_ptr<StatisticsItem> stats,
                           std::shared_ptr<ThreadPool> thread_pool) {
  stats_ = stats;
  if (stats) {
    stats_status_ = stats_->AddItem("status", std::string("initial", true));
  } else {
    stats_status_ = std::make_shared<StatisticsItem>();
  }

  if (thread_pool == nullptr) {
    auto threads = config->GetUint32("graph.thread-num",
                                     std::thread::hardware_concurrency() * 2);
    auto max_threads = config->GetUint32("graph.max-thread-num", threads * 32);
    auto thread_pool = std::make_shared<ThreadPool>(threads, max_threads);
    thread_pool->SetName(TASK_FLOW_POOL_NAME);
    tp_ = thread_pool;
    thread_create_ = true;

    if (data_hub_ == nullptr) {
      data_hub_ = std::make_shared<DefaultDataHub>();
    }

    if (scheduler_event_port_ == nullptr) {
      scheduler_event_port_ =
          std::make_shared<SchedulerPort>("_Scheduler_Event");
      scheduler_event_port_->Init();
      auto priority_port =
          std::make_shared<PriorityPort>(scheduler_event_port_);
      if (!(data_hub_->AddPort(priority_port))) {
        MBLOG_ERROR << "failed to add port to data hub";
        return STATUS_FAULT;
      }
    }

    MBLOG_INFO << "init scheduler with " << threads << " threads, max "
               << max_threads;
  }

  return STATUS_OK;
}

Status FlowScheduler::Build(const Graph& graph) {
  if (data_hub_ == nullptr || tp_ == nullptr) {
    return {STATUS_SHUTDOWN, "Scheduler not init."};
  }

  auto node_port_list = graph.GetNotifyPort();
  if (node_port_list.empty()) {
    MBLOG_ERROR << "graph has no flow unit group";
    return STATUS_FAULT;
  }

  for (const auto& iter_pair : node_port_list) {
    std::vector<std::shared_ptr<PriorityPort>> priority_ports;
    priority_ports.reserve(iter_pair.second.size());
    for (const auto& port : iter_pair.second) {
      if (!port) {
        MBLOG_ERROR << "port must not be nullptr.";
        return STATUS_FAULT;
      }

      auto priority_port = std::make_shared<PriorityPort>(port);
      if (!(data_hub_->AddPort(priority_port))) {
        MBLOG_ERROR << "failed to add port to data hub";
        return STATUS_FAULT;
      }

      priority_ports.emplace_back(priority_port);
    }

    node_port_map_.emplace(iter_pair.first, std::move(priority_ports));
  }

  MBLOG_DEBUG << "flow scheduler build.";
  return STATUS_OK;
}

void FlowScheduler::RunAsync() {
  if (tp_ == nullptr) {
    return;
  }

  mode_ = ASYNC;
  is_stop_ = false;
  run_fut_ =
      tp_->Submit(TASK_FLOW_SCHEDUER_NAME, &FlowScheduler::RunImpl, this);
  MBLOG_DEBUG << "flow scheduler is running.";
}

Status FlowScheduler::Run() {
  if (tp_ == nullptr) {
    return {STATUS_SHUTDOWN, "Scheduler not init."};
  }

  mode_ = SYNC;
  is_stop_ = false;
  MBLOG_DEBUG << "flow scheduler is running.";
  return RunImpl();
}

void FlowScheduler::EnableActivePort(const std::shared_ptr<NodeBase>& node) {
  auto iter = node_port_map_.find(node);
  if (iter != node_port_map_.end()) {
    data_hub_->AddToActivePort(iter->second);
  }

  std::unique_lock<std::mutex> lock(status_mutex_);
  nodes_runing_status_[node] = false;
}
void FlowScheduler::DisableActivePort(const std::shared_ptr<NodeBase>& node) {
  auto iter = node_port_map_.find(node);
  if (iter != node_port_map_.end()) {
    data_hub_->RemoveFromActivePort(iter->second);
  }

  std::unique_lock<std::mutex> lock(status_mutex_);
  nodes_runing_status_[node] = true;
}

void FlowScheduler::SendSchedulerCommand(
    SchedulerCommandType type, const std::shared_ptr<PriorityPort>& port) {
  auto cmd = std::make_shared<SchedulerCommand>(type, port);
  scheduler_event_port_->Send(cmd);
  scheduler_event_port_->NotifyPushEvent();
}

void FlowScheduler::RunWapper(std::shared_ptr<NodeBase> node, RunType type,
                              std::shared_ptr<PriorityPort> active_port) {
  Status status = STATUS_FAULT;
  try {
    MBLOG_DEBUG << "run " << node->GetName() << " begin";
    status = node->Run(type);
    MBLOG_DEBUG << "run " << node->GetName() << " end";
  } catch (std::exception& e) {
    MBLOG_WARN << "node " << node->GetName()
               << " run exception caught: " << e.what();
    status = {STATUS_FAULT, e.what()};
  }

  if (!status) {
    MBLOG_ERROR << "node (" << node->GetName()
                << ") run return: " << status.WrapErrormsgs();
    auto cmd_type = (status == STATUS_STOP)
                        ? SchedulerCommandType::COMMAND_STOP
                        : SchedulerCommandType::COMMAND_ERROR;
    SendSchedulerCommand(cmd_type, active_port);
  }

  EnableActivePort(node);
  std::unique_lock<std::mutex> lock(notify_mutex_);
  running_node_count_--;
  if (is_wait_stop_) {
    cv_.notify_one();
  }
}

Status FlowScheduler::RunNode(std::shared_ptr<PriorityPort> active_port) {
  if (tp_ == nullptr) {
    return {STATUS_FAULT, "scheduler not init."};
  }

  if (!active_port) {
    return {STATUS_INVALID, "active port must not be nullptr."};
  }

  auto port = active_port->GetPort();
  if (!port) {
    return {STATUS_INVALID,
            "unexcept! port must not be nullptr, flow scheduler will be "
            "shutdown."};
  }

  auto node = active_port->GetNode();
  if (!node) {
    return {STATUS_INVALID,
            "unexcept! can not find node in graph, flow scheduler will be "
            "shutdown."};
  }

  DisableActivePort(node);

  auto type = (typeid(*port) == typeid(EventPort) ? EVENT : DATA);
  MBLOG_DEBUG << "begin run node " << node->GetName() << " for type: " << type;
  running_node_count_++;
  auto fut = tp_->Submit(node->GetName(), &FlowScheduler::RunWapper, this, node,
                         type, active_port);
  if (!fut.valid()) {
    MBLOG_ERROR << "Submit task " << node->GetName() << "failed.";
    EnableActivePort(node);
    running_node_count_--;
  }

  return STATUS_OK;
}

Status FlowScheduler::RunCommand(std::shared_ptr<PriorityPort> active_port) {
  auto port = active_port->GetPort();
  if (!port) {
    return {STATUS_INVALID,
            "unexcept! port must not be nullptr, flow scheduler will be "
            "shutdown."};
  }

  auto scheduler_port = std::dynamic_pointer_cast<SchedulerPort>(port);
  auto command = scheduler_port->Recv();
  scheduler_event_port_->NotifyPopEvent();
  data_hub_->AddToActivePort(active_port);
  if (!command) {
    return STATUS_OK;
  }

  switch (command->GetType()) {
    case SchedulerCommandType::COMMAND_STOP:
      return STATUS_STOP;
    case SchedulerCommandType::COMMAND_ERROR:
      return STATUS_FAULT;
    default:
      return {STATUS_INVALID, "invalid scheduler command type."};
  }

  return STATUS_OK;
}

bool FlowScheduler::IsSchedulerCommand(
    const std::shared_ptr<PriorityPort>& active_port) {
  return scheduler_event_port_ == active_port->GetPort();
}

void FlowScheduler::WaitNodeFinish() {
  auto pred = [this] { return running_node_count_ == 0; };
  std::unique_lock<std::mutex> lock(notify_mutex_);
  is_wait_stop_ = true;

  cv_.wait(lock, pred);
}

void FlowScheduler::CheckScheduleStatus(const bool& printlog) {
  // If the current status is normal, no information is printed. The logic for
  // determining abnormalities is as follows:
  // 1. If the node is running, it indicates that the current node has been
  // running for 60 seconds and no response is returned. In this case, an
  // exception may occur. It is probably blocked in the Send function.
  // 2. The node is not in the running state, but the port contains data. The
  // possible cause is that the threads in the thread pool are exhausted.
  bool is_print_threadpool = false;
  bool is_no_response = false;
  for (auto iter : node_port_map_) {
    auto node = iter.first;
    for (auto port_iter : iter.second) {
      if (!port_iter->GetPort()) {
        continue;
      }

      std::unique_lock<std::mutex> lock(status_mutex_);
      auto node_status = nodes_runing_status_.find(node);
      // node is not running
      if (node_status == nodes_runing_status_.end() ||
          node_status->second == false) {
        if (port_iter->GetPort()->Empty()) {
          continue;
        }

        is_no_response = true;

        if (printlog) {
          MBLOG_WARN << "node:" << node->GetName()
                     << " is not running, but port:"
                     << port_iter->GetPort()->GetName()
                     << " has data:" << port_iter->GetPort()->GetDataCount()
                     << " priority:" << port_iter->GetPriority()
                     << ", scheduler may be blocking.";
          is_print_threadpool = true;
        }
      } else {
        is_no_response = true;

        // node is running
        if (printlog == false) {
          continue;
        }
        MBLOG_WARN << "node:" << node->GetName()
                   << " running long time, and port:"
                   << port_iter->GetPort()->GetName()
                   << " has data:" << port_iter->GetPort()->GetDataCount()
                   << " priority:" << port_iter->GetPriority()
                   << ", node may be blocking "
                   << ((port_iter->GetPort()->GetDataCount() > 0)
                           ? ""
                           : "in Send function ")
                   << "or thread pool is busy.";
        is_print_threadpool = true;
      }
    }
  }

  // If no exception occurs, do not print any information to prevent excessive
  // information from being printed when the HTTP server is used.
  if (is_print_threadpool) {
    MBLOG_INFO << "Thread Pool Status:";
    MBLOG_INFO << "                    max thread size: "
               << tp_->GetMaxThreadsNum();
    MBLOG_INFO << "                    worker thread size: "
               << tp_->GetThreadsNum();
    MBLOG_INFO << "                    wating work count: "
               << tp_->GetWaitingWorkCount();

    MBLOG_INFO << "running_node_count: " << running_node_count_;
  }

  is_no_response_ = is_no_response;
  check_count_++;
}

Status FlowScheduler::RunImpl() {
  MBLOG_DEBUG << "flow schedule is begin run.";
  os->Thread->SetName("Flow-Scheduler");
  std::shared_ptr<PriorityPort> active_port = nullptr;
  Status status = STATUS_OK;
  is_stop_ = false;
  is_wait_stop_ = false;
  bool has_print = false;
  is_no_response_ = false;
  int timeout_count = 0;
  time_t last_check_time = 0;
  stats_status_->SetValue(std::string("running"));

  while (!is_stop_) {
    if (is_no_response_) {
      time_t currtime;
      time(&currtime);
      if (last_check_time != currtime) {
        CheckScheduleStatus(false);
        last_check_time = currtime;
      }
    }

    status = data_hub_->SelectActivePort(&active_port, check_timeout_);
    if (status == STATUS_TIMEDOUT) {
      // The system displays the current status information every 60 seconds if
      // the system is idle.
      if (!has_print && timeout_count >= max_check_timeout_count_) {
        CheckScheduleStatus(!has_print);
        has_print = true;
        timeout_count = 0;
        if (is_no_response_) {
          stats_status_->SetValue(std::string("blocking"));
        }
      } else if (is_no_response_ == false) {
        stats_status_->SetValue(std::string("idle"));
      }

      timeout_count++;
      continue;
    }

    if (status == STATUS_NODATA) {
      timeout_count = 0;
      continue;
    }

    if (timeout_count > 0 && is_no_response_ == false) {
      stats_status_->SetValue(std::string("running"));
    }

    has_print = false;
    timeout_count = 0;
    status = IsSchedulerCommand(active_port) ? RunCommand(active_port)
                                             : RunNode(active_port);
    if (!status) {
      break;
    }
  }

  is_stop_ = true;

  if (!status) {
    MBLOG_ERROR << "the scheduler caught an error : " << status;
  }

  ShutdownNodes();
  WaitNodeFinish();
  stats_status_->SetValue(std::string("shutdown"));
  return status;
}

void FlowScheduler::ShutdownNodes() {
  for (auto& iter : node_port_map_) {
    iter.first->Shutdown();
  }
}

void FlowScheduler::Shutdown() {
  is_stop_ = true;
  if (tp_ == nullptr) {
    return;
  }

  MBLOG_INFO << "shutdown flow scheduler.";
  SendSchedulerCommand(SchedulerCommandType::COMMAND_STOP, nullptr);
  if (run_fut_.valid()) {
    run_fut_.wait();
  }
  if (thread_create_) {
    tp_->Shutdown();
    thread_create_ = false;
  }

  tp_ = nullptr;
}

Status FlowScheduler::Wait(int64_t milliseconds, Status* ret_val) {
  if (is_stop_ == true || tp_ == nullptr) {
    return STATUS_SHUTDOWN;
  }

  if (!run_fut_.valid()) {
    MBLOG_WARN << "async run future is invalid.";
    return STATUS_FAULT;
  }

  if (0 == milliseconds) {
    run_fut_.wait();
  } else if (milliseconds > 0) {
    auto status = run_fut_.wait_for(std::chrono::milliseconds(milliseconds));
    if (status != std::future_status::ready) {
      return STATUS_TIMEDOUT;
    }
  }

  if (is_no_response_) {
    return STATUS_NORESPONSE;
  }

  if (is_stop_ == false && milliseconds < 0) {
    return STATUS_BUSY;
  }

  auto ret = run_fut_.get();
  if (ret_val != nullptr) {
    *ret_val = ret;
  }

  // TODO thread pool should provide force stop api
  return STATUS_OK;
}

}  // namespace modelbox
