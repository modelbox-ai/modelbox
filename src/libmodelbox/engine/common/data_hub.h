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


#ifndef MODELBOX_MULTI_QUEUE_H_
#define MODELBOX_MULTI_QUEUE_H_

#include <modelbox/base/blocking_queue.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer.h>
#include <modelbox/port.h>

#include <memory>
#include <set>
#include <unordered_map>

namespace modelbox {

class PortCompare;

class PriorityPort {
 public:
  PriorityPort(const std::shared_ptr<IPort>& port);

  virtual ~PriorityPort() = default;

  const std::shared_ptr<IPort>& GetPort() const;
  std::shared_ptr<IPort> GetPort();

  std::shared_ptr<NodeBase> GetNode() const;

  void UpdateActiveTime();

  int32_t GetPriority() const;
  void SetPriority(int32_t priority);
  // TODO port优先级更新
  void UpdatePriority();

  void SetPushEventCallBack(const PushCallBack& func);
  void SetPopEventCallBack(const PopCallBack& func);

  bool HasData();

  void SetRuning(bool flag);

  bool IsRunning();

  bool IsActivated();

  friend PortCompare;

 private:
  int32_t priority_{0};
  int64_t active_time_{0};
  bool is_running_{false};
  std::shared_ptr<IPort> port_;
};

/**
 * @brief DataHub Class interface
 * Pure virtual class, can not instantiable
 *
 */
class DataHub {
 public:
  virtual ~DataHub() = default;

  virtual Status AddPort(const std::shared_ptr<PriorityPort>& port) = 0;
  virtual Status SelectActivePort(std::shared_ptr<PriorityPort>* active_port,
                                  int64_t timeout = 0) = 0;

  virtual size_t GetPortNum() const = 0;
  virtual size_t GetActivePortNum() const = 0;

  virtual void RemoveFromActivePort(
      std::vector<std::shared_ptr<PriorityPort>>& ports) = 0;
  virtual Status AddToActivePort(
      std::vector<std::shared_ptr<PriorityPort>>& ports) = 0;
  virtual Status AddToActivePort(const std::shared_ptr<PriorityPort>& port) = 0;
};

class PortCompare {
 public:
  bool operator()(const std::shared_ptr<PriorityPort>& left,
                  const std::shared_ptr<PriorityPort>& right);
};

/**
 * @brief DataHub default implementation class
 *
 */
class DefaultDataHub : public DataHub {
 public:
  DefaultDataHub();
  ~DefaultDataHub() override;

  Status AddPort(const std::shared_ptr<PriorityPort>& port) override;
  Status SelectActivePort(std::shared_ptr<PriorityPort>* active_port,
                          int64_t timeout = 0) override;

  size_t GetPortNum() const override;

  size_t GetActivePortNum() const override;

  void RemoveFromActivePort(
      std::vector<std::shared_ptr<PriorityPort>>& ports) override;
  Status AddToActivePort(
      std::vector<std::shared_ptr<PriorityPort>>& ports) override;
  Status AddToActivePort(const std::shared_ptr<PriorityPort>& port) override;

 private:
  void PortEventCallback(std::shared_ptr<PriorityPort> port,
                         bool update_active_time);
  void UpdateActivePort(std::shared_ptr<PriorityPort> port,
                        bool update_active_time = true);

  std::vector<std::shared_ptr<PriorityPort>> priority_ports_;
  std::set<std::shared_ptr<PriorityPort>, PortCompare> active_ports_;

  std::mutex active_mutex_;
  std::condition_variable cv_;
};

}  // namespace modelbox
#endif