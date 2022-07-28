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

#include "modelbox/port.h"

namespace modelbox {

Port::Port(const std::string& name, std::shared_ptr<NodeBase> node)
    : name_(name), node_(node) {}

Port::~Port() {}

const std::string& Port::GetName() { return name_; }
std::shared_ptr<NodeBase> Port::GetNode() {
  auto node = node_.lock();
  return node;
}

void Port::Shutdown() {}

Status InPort::Init() {
  auto node = node_.lock();
  if (node == nullptr) {
    return STATUS_INVALID;
  }

  return STATUS_SUCCESS;
}

void InPort::Recv(std::vector<std::shared_ptr<Buffer>>& buffer_vector,
                  uint32_t left_buffer_num) {
  if (left_buffer_num == 0) {
    if (queue_->RemainCapacity() == 0) {
      SetActiveState(false);
    }
    return;
  }
  queue_->PopBatch(&buffer_vector, -1, left_buffer_num);

  if (!buffer_vector.empty()) {
    NotifyPopEvent();
  }
}

bool InPort::SetOutputPort(std::shared_ptr<OutPort> output_port) {
  for (auto output_exist_port : output_ports) {
    if (output_port == output_exist_port.lock()) {
      return false;
    }
  }

  output_ports.push_back(output_port);
  return true;
}

size_t InPort::GetConnectedPortNumber() { return output_ports.size(); }

std::vector<std::weak_ptr<OutPort>> InPort::GetAllOutPort() {
  return output_ports;
}

OutPort::OutPort(const std::string& name, std::shared_ptr<NodeBase> node)
    : Port(name, node) {}

OutPort::~OutPort() {}

Status OutPort::Init() {
  auto node = node_.lock();
  if (node == nullptr) {
    return STATUS_INVALID;
  }
  return STATUS_SUCCESS;
}

Status OutPort::Send(std::vector<std::shared_ptr<Buffer>>& buffers) {
  bool loop;
  auto real_node = std::dynamic_pointer_cast<Node>(GetNode());
  LoopType loop_type{NOT_LOOP};
  if (real_node != nullptr) {
    loop_type = real_node->GetLoopType();
  }

  for (auto input_port : connected_input_ports_) {
    loop = false;
    auto queue = input_port->GetQueue();
    auto priority = input_port->GetPriority();
    std::vector<std::shared_ptr<Buffer>> port_buffers;
    port_buffers.reserve(buffers.size());
    for (auto& origin_buffer : buffers) {
      auto buffer = origin_buffer->Copy();
      BufferManageView::SetIndexInfo(
          buffer, BufferManageView::GetIndexInfo(origin_buffer));
      BufferManageView::SetPriority(buffer, real_node->GetPriority());
      port_buffers.push_back(buffer);
      // only loop flowunit itself in the loop structure
      auto buffer_priority = BufferManageView::GetPriority(buffer);
      if (loop_type == LOOP) {
        BufferManageView::SetPriority(buffer, buffer_priority + 1);
        continue;
      }

      if (buffer_priority < priority) {
        BufferManageView::SetPriority(buffer, priority);
        continue;
      }

      // during loop
      loop = true;
    }

    while (port_buffers.size() > 0) {
      if (loop_type == LOOP || loop) {
        if (queue->PushBatchForce(&port_buffers, false, 0) == 0) {
          break;
        }
      } else {
        if (0 == queue->PushBatchForce(&port_buffers, true, 0)) {
          break;
        }
      }

      if (port_buffers.size() > 0) {
        input_port->NotifyPushEvent();
      }
    }
  }

  for (const auto& input_port : connected_input_ports_) {
    input_port->NotifyPushEvent();
  }

  return STATUS_SUCCESS;
}

bool OutPort::ConnectPort(std::shared_ptr<InPort> inport) {
  if (inport == nullptr) {
    return false;
  }
  if (!inport->SetOutputPort(shared_from_this())) {
    return false;
  }
  auto pair = connected_input_ports_.emplace(inport);
  return pair.second;
}

void OutPort::Shutdown() {
  for (auto inport : connected_input_ports_) {
    inport->Shutdown();
  }
}

std::set<std::shared_ptr<InPort>> OutPort::GetConnectInPort() {
  return connected_input_ports_;
}

Status EventPort::SendBatch(
    std::vector<std::shared_ptr<FlowUnitInnerEvent>>& event_list) {
  queue_->PushBatchForce(&event_list, true, 0);
  return STATUS_SUCCESS;
}

Status EventPort::Send(std::shared_ptr<FlowUnitInnerEvent>& event) {
  queue_->PushForce(event);
  return STATUS_SUCCESS;
}

}  // namespace modelbox
