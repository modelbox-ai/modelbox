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

#ifndef MODELBOX_PORT_H_
#define MODELBOX_PORT_H_

#include "modelbox/base/blocking_queue.h"
#include "modelbox/base/status.h"
#include "modelbox/inner_event.h"
#include "modelbox/match_buffer.h"
#include "modelbox/node.h"

namespace modelbox {

class NodeBase;
class SingleMatchCache;

struct CustomCompare {
  auto operator()(std::shared_ptr<IndexBuffer> const& a,
                  std::shared_ptr<IndexBuffer> const& b) const -> bool {
    return a->GetPriority() < b->GetPriority();
  }
};

struct EventCompare {
  auto operator()(std::shared_ptr<FlowUnitInnerEvent> const& a,
                  std::shared_ptr<FlowUnitInnerEvent> const& b) const -> bool {
    return a->GetPriority() < b->GetPriority();
  }
};

typedef PriorityBlockingQueue<std::shared_ptr<IndexBuffer>, CustomCompare>
    BufferQueue;

using EventQueue =
    PriorityBlockingQueue<std::shared_ptr<FlowUnitInnerEvent>, EventCompare>;

template <typename QueueType, typename Compare>
using PortQueue = PriorityBlockingQueue<std::shared_ptr<QueueType>, Compare>;

class Port {
 public:
  /**
   * @brief Construct a new Port object
   *
   * @param name the port name
   * @param node the parent node contains the port
   */
  Port(const std::string& name, std::shared_ptr<NodeBase> node);

  /**
   * @brief destructor
   */
  virtual ~Port();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  const std::string& GetName();

  /**
   * @brief Get the parent Node object
   *
   * @return std::shared_ptr<NodeBase>
   */
  std::shared_ptr<NodeBase> GetNode();

  virtual void Shutdown();

 protected:
  std::string name_;
  std::weak_ptr<NodeBase> node_;
};

using PushCallBack = std::function<void(bool)>;
using PopCallBack = std::function<void(void)>;

class IPort : public Port {
 public:
  IPort(const std::string& name, std::shared_ptr<NodeBase> node)
      : Port(name, node){};
  virtual ~IPort() = default;
  virtual int32_t GetPriority() const = 0;
  virtual int32_t GetDataCount() const = 0;
  virtual void SetPriority(int32_t priority) = 0;
  virtual void SetPushEventCallBack(const PushCallBack& func) = 0;
  virtual void SetPopEventCallBack(const PopCallBack& func) = 0;
  virtual void NotifyPushEvent(bool update_active_time) = 0;
  virtual void NotifyPushEvent() = 0;
  virtual void NotifyPopEvent() = 0;
  virtual bool Empty() const = 0;

  virtual bool IsActivated() = 0;
  virtual void SetActiveState(bool flag) = 0;

  virtual Status Init() = 0;
};

template <typename QueueType, typename Compare>
class NotifyPort : public IPort {
 public:
  NotifyPort(const std::string& name, std::shared_ptr<NodeBase> node,
             uint32_t priority = 0, size_t event_capacity = SIZE_MAX)
      : IPort(name, node),
        priority_(priority),
        mutex_(),
        push_call_back_(nullptr),
        pop_call_back_(nullptr),
        is_activated_(true),
        queue_(
            std::make_shared<PortQueue<QueueType, Compare>>(event_capacity)) {}

  virtual ~NotifyPort() { queue_->Clear(); }

  /**
   * @brief Get the Priority
   *
   * @return int
   */
  virtual int32_t GetPriority() const override {
    std::shared_ptr<QueueType> data = nullptr;
    if (queue_->Front(&data)) {
      return data->GetPriority();
    }

    return priority_;
  }

  /**
   * @brief Get the data count in port
   *
   * @return int
   */
  virtual int32_t GetDataCount() const override {
    return queue_ ? queue_->Size() : 0;
  }

  /**
   * @brief Set the Priority
   *
   * @param priority
   */
  virtual void SetPriority(int32_t priority) override { priority_ = priority; }

  /**
   * @brief Set the Push Event Call Back Function
   *
   * @param func PushEvent Callback Function
   */
  virtual void SetPushEventCallBack(const PushCallBack& func) override {
    push_call_back_ = func;
  }

  /**
   * @brief Set the Pop Event Call Back Function
   *
   * @param func PopEvent Callback Function
   */
  virtual void SetPopEventCallBack(const PopCallBack& func) override {
    pop_call_back_ = func;
  }

  /**
   * @brief Notify PushEvent
   *
   */
  virtual void NotifyPushEvent(bool update_active_time) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (push_call_back_) {
      push_call_back_(update_active_time);
    }
  }

  virtual void NotifyPushEvent() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (push_call_back_) {
      push_call_back_(true);
    }
  }

  /**
   * @brief Notify PopEvent
   *
   */
  virtual void NotifyPopEvent() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pop_call_back_) {
      pop_call_back_();
    }
  }

  /**
   * @brief
   *
   * @return true
   * @return false
   */
  virtual bool Empty() const override { return queue_->Empty(); };

  virtual bool IsActivated() override { return is_activated_; }

  virtual void SetActiveState(bool flag) override { is_activated_ = flag; }

  virtual void Shutdown() override {
    std::lock_guard<std::mutex> lock(mutex_);
    push_call_back_ = nullptr;
    pop_call_back_ = nullptr;
    queue_->Shutdown();
  }

  Status Send(const std::shared_ptr<QueueType>& data) {
    if (!data) {
      MBLOG_WARN << "data must not be nullptr.";
      return STATUS_INVALID;
    }

    return queue_->Push(data, 0);
  }

  Status Recv(std::shared_ptr<std::vector<std::shared_ptr<QueueType>>>& datas) {
    if (!datas) {
      datas = std::make_shared<std::vector<std::shared_ptr<QueueType>>>();
    }

    queue_->PopBatch(datas.get(), -1);
    return STATUS_OK;
  }

  std::shared_ptr<QueueType> Recv() {
    std::shared_ptr<QueueType> data = nullptr;
    queue_->Pop(&data, -1);
    return data;
  }

  std::shared_ptr<PortQueue<QueueType, Compare>> GetQueue() { return queue_; }

 protected:
  int32_t priority_{0};
  std::mutex mutex_;

  PushCallBack push_call_back_;
  PopCallBack pop_call_back_;
  std::atomic<bool> is_activated_{false};

  std::shared_ptr<PortQueue<QueueType, Compare>> queue_;
};

class EventPort : public NotifyPort<FlowUnitInnerEvent, EventCompare> {
 public:
  EventPort(const std::string& name, std::shared_ptr<NodeBase> node,
            uint32_t priority = 0, size_t event_capacity = SIZE_MAX)
      : NotifyPort(name, node, priority, event_capacity){};
  virtual ~EventPort() override = default;

  Status Init() override { return STATUS_OK; };

  Status SendBatch(
      std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list);

  Status Send(std::shared_ptr<FlowUnitInnerEvent>& event);
};

class OutPort;
class InPort : public NotifyPort<IndexBuffer, CustomCompare> {
  friend class OutPort;

 public:
  /**
   * @brief Construct a new InPort object
   *
   * @param name the port name
   * @param node the parent node contains the port
   * @param priority priority
   * @param event_capacity event queue size
   */
  InPort(const std::string& name, std::shared_ptr<NodeBase> node,
         uint32_t priority = 0, size_t event_capacity = SIZE_MAX)
      : NotifyPort(name, node, priority, event_capacity) {}

  virtual ~InPort() override = default;

  /**
   * @brief Init the InPort generate the queue
   *
   * @return Status {status} if success return STATUS_SUCCESS else return
   * STATUS_INVALID
   */
  Status Init() override;

  void Recv(std::vector<std::shared_ptr<IndexBuffer>>& buffer_vector,
            uint32_t left_buffer_num);

  /**
   * @brief Get the All Out Port object
   *
   * @return std::vector<std::weak_ptr<OutPort>>
   */
  std::vector<std::weak_ptr<OutPort>> GetAllOutPort();

 private:
  bool SetOutputPort(std::shared_ptr<OutPort> output_port);

  std::vector<std::weak_ptr<OutPort>> output_ports;
};

class OutPort : public Port, public std::enable_shared_from_this<OutPort> {
 public:
  OutPort(const std::string& name, std::shared_ptr<NodeBase> node);
  virtual ~OutPort();
  Status Init();
  Status Send(std::vector<std::shared_ptr<IndexBuffer>>& buffer);
  std::set<std::shared_ptr<InPort>> GetConnectInPort();
  bool AddPort(std::shared_ptr<InPort>);
  void Shutdown();

 private:
  std::set<std::shared_ptr<InPort>> input_ports_;
};

}  // namespace modelbox
#endif
