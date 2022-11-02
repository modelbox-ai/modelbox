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


#ifndef MODELBOX_PIPELINE_SCHEDULER_H_
#define MODELBOX_PIPELINE_SCHEDULER_H_

#include <modelbox/base/thread_pool.h>
#include <modelbox/graph.h>

#include <atomic>
#include <thread>
#include <utility>

#include "../common/data_hub.h"
#include "modelbox/scheduler.h"

namespace modelbox {

enum class SchedulerCommandType {
  COMMAND_STOP = 1,
  COMMAND_ERROR = 2,
};

constexpr const int SCHED_CHECK_TIMEOUT_MS = 1000;
constexpr const int SCHED_MAX_CHECK_TIMEOUT_COUNT = 60;

class SchedulerCommand {
 public:
  SchedulerCommand(SchedulerCommandType type,
                   std::shared_ptr<PriorityPort> port);
  virtual ~SchedulerCommand();

  SchedulerCommandType GetType();

  std::shared_ptr<PriorityPort> GetPort();

  int GetPriority();

 private:
  SchedulerCommandType type_;
  std::shared_ptr<PriorityPort> port_;
};

struct SchedulerCommandCompare {
  auto operator()(std::shared_ptr<SchedulerCommand> const& a,
                  std::shared_ptr<SchedulerCommand> const& b) const -> bool {
    if (nullptr == a->GetPort() && b->GetPort()) {
      return false;
    }

    if (a->GetPort() && nullptr == b->GetPort()) {
      return true;
    }

    if (nullptr == a->GetPort() && nullptr == b->GetPort()) {
      return true;
    }

    return a->GetPort()->GetPriority() < b->GetPort()->GetPriority();
  }
};

using SchedulerQueue = PriorityBlockingQueue<std::shared_ptr<SchedulerCommand>,
                                             SchedulerCommandCompare>;

class SchedulerPort
    : public NotifyPort<SchedulerCommand, SchedulerCommandCompare> {
 public:
  SchedulerPort(const std::string& name);
  SchedulerPort(const std::string& name, size_t event_capacity);

  ~SchedulerPort() override;

  Status Init() override;
};

class FlowScheduler : public Scheduler {
 public:
  FlowScheduler();
  ~FlowScheduler() override;
  Status Init(std::shared_ptr<Configuration> config,
              std::shared_ptr<StatisticsItem> stats = nullptr,
              std::shared_ptr<ThreadPool> thread_pool = nullptr) override;
  Status Build(const Graph& graph) override;
  Status Run() override;
  void RunAsync() override;
  void Shutdown() override;

  Status Wait(int64_t milliseconds, Status* ret_val = nullptr) override;

  void SetCheckTimeout(int timeout) { check_timeout_ = timeout; }
  void SetMaxCheckTimeoutCount(int max_timeout_count) {
    max_check_timeout_count_ = max_timeout_count;
  }
  int64_t GetCheckCount() const { return check_count_; }

 private:
  std::shared_ptr<DataHub> data_hub_;
  std::shared_ptr<ThreadPool> tp_;
  bool thread_create_ = false;

  std::atomic<bool> is_stop_{false};
  std::atomic<bool> is_built_{false};

  std::future<Status> run_fut_;
  std::atomic<int> mode_{SYNC};

  std::shared_ptr<SchedulerPort> scheduler_event_port_;
  std::unordered_map<std::shared_ptr<NodeBase>,
                     std::vector<std::shared_ptr<PriorityPort>>>
      node_port_map_;

  std::atomic<uint32_t> running_node_count_{0};
  bool is_wait_stop_{false};
  bool is_no_response_{false};
  std::mutex notify_mutex_;
  std::condition_variable cv_;

  std::shared_ptr<StatisticsItem> stats_;
  std::shared_ptr<StatisticsItem> stats_status_;

  std::mutex status_mutex_;
  std::unordered_map<std::shared_ptr<NodeBase>, bool> nodes_runing_status_;

  int check_timeout_{SCHED_CHECK_TIMEOUT_MS};
  int max_check_timeout_count_{SCHED_MAX_CHECK_TIMEOUT_COUNT};
  std::atomic<int64_t> check_count_{0};

  Status RunImpl();
  void RunWapper(const std::shared_ptr<NodeBase>& node, RunType type,
                 const std::shared_ptr<PriorityPort>& active_port);

  Status RunNode(const std::shared_ptr<PriorityPort>& active_port);
  Status RunCommand(const std::shared_ptr<PriorityPort>& active_port);
  void SendSchedulerCommand(SchedulerCommandType type,
                            const std::shared_ptr<PriorityPort>& port);
  bool IsSchedulerCommand(const std::shared_ptr<PriorityPort>& active_port);

  void EnableActivePort(const std::shared_ptr<NodeBase>& node);
  void DisableActivePort(const std::shared_ptr<NodeBase>& node);
  void WaitNodeFinish();
  void ShutdownNodes();
  void CheckScheduleStatus(const bool &printlog);
};

}  // namespace modelbox

#endif
