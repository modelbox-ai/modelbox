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

#include "modelbox/stream_matcher.h"

#include "modelbox/virtual_node.h"

namespace modelbox {

StreamMatcher::~StreamMatcher() {
  root_stream_ = nullptr;
  port_stream_map_.clear();
  port_connect_map_.clear();
  checking_nodes_.clear();
  checked_nodes_.clear();
  all_nodes_.clear();
}

StreamMatcher::StreamMatcher(std::set<std::shared_ptr<NodeBase>> start_nodes,
                             std::set<std::shared_ptr<NodeBase>> all_nodes) {
  for (auto start_node : start_nodes) {
    checking_nodes_.insert(start_node);
  }

  for (auto node : all_nodes) {
    all_nodes_.insert(node);
  }

  root_stream_ = nullptr;
}

Status StreamMatcher::StartCheck() {
  while (true) {
    for (auto checking_node : checking_nodes_) {
      auto input_stream = GetInputStream(checking_node);
      if (input_stream == nullptr) {
        auto msg = checking_node->GetName() + " input stream is not match";
        return {STATUS_INVALID, msg};
      }
      auto status = GenerateOutputStream(checking_node, input_stream);
      if (status != STATUS_OK) {
        return status;
      }
    }
    auto update_status = UpdateCheckingNode();
    if (update_status != STATUS_CONTINUE) {
      return update_status;
    }
  }
  return STATUS_OK;
}

std::shared_ptr<GraphStream> StreamMatcher::GetInputStream(
    std::shared_ptr<NodeBase> node) {
  auto real_node = std::dynamic_pointer_cast<Node>(node);
  std::shared_ptr<GraphStream> input_stream;
  if (node->GetInputNum() == 0) {
    auto root_stream = GenerateRootStream();
    input_stream = root_stream->GenerateChildStream();
  } else {
    auto input_ports = node->GetInputPorts();
    std::shared_ptr<GraphStream> first_stream;
    for (auto input_port : input_ports) {
      auto output_port_stream = port_stream_map_[input_port];
      if (output_port_stream == nullptr) {
        return nullptr;
      }

      if (input_port->GetAllOutPort().size() > 1 &&
          real_node->GetLoopType() != LOOP) {
        output_port_stream = output_port_stream->GetFullStream();
      }

      if (first_stream == nullptr) {
        first_stream = output_port_stream;
      }

      if (first_stream != output_port_stream) {
        return nullptr;
      }
    }
    input_stream = first_stream;
  }

  return input_stream;
}

std::shared_ptr<GraphStream> GenOutStreamFromInput(
    std::shared_ptr<NodeBase> node, std::shared_ptr<GraphStream> input_stream,
    std::shared_ptr<GraphStream> last_output_stream,
    std::shared_ptr<OutPort> output_port) {
  std::shared_ptr<GraphStream> output_stream;
  if (typeid(*node) != typeid(Node)) {
    return input_stream;
  }

  auto actual_node = std::dynamic_pointer_cast<Node>(node);
  if (actual_node->GetConditionType() == IF_ELSE) {
    output_stream = input_stream->GenerateConditionStream(last_output_stream);
  } else {
    if (last_output_stream == nullptr) {
      if (actual_node->GetOutputType() == COLLAPSE) {
        output_stream = input_stream->GetParentStream();
      } else if (actual_node->GetOutputType() == EXPAND) {
        output_stream = input_stream->GenerateChildStream();
      } else {
        if (!actual_node->IsStreamSameCount()) {
          output_stream = input_stream->GenerateSiblingStream();
        } else {
          output_stream = input_stream;
        }
      }
    } else {
      output_stream = last_output_stream;
    }
  }

  return output_stream;
}

Status StreamMatcher::BindOutputStream(
    std::shared_ptr<NodeBase> node, std::shared_ptr<OutPort> output_port,
    std::shared_ptr<GraphStream> output_stream) {
  auto connect_ports = output_port->GetConnectInPort();
  for (auto connect_port : connect_ports) {
    auto connect_ports_size = connect_port->GetAllOutPort().size();

    auto real_next_node =
        std::dynamic_pointer_cast<Node>(connect_port->GetNode());

    if (connect_ports_size != 1 && real_next_node->GetLoopType() != LOOP) {
      if (output_stream->GetFullStream() == nullptr) {
        auto msg = node->GetName() + " : " + output_port->GetName() +
                   " stream is not a condition stream,but connect to a "
                   "condition flow";
        return {STATUS_INVALID, msg};
      }

      auto find_it = port_stream_map_.find(connect_port);
      if (find_it != port_stream_map_.end()) {
        if (!output_stream->IsSameConditonGroupStream(
                port_stream_map_[connect_port])) {
          auto msg = node->GetName() + " : " + output_port->GetName() +
                     " stream is not connect to a right condition flow";
          return {STATUS_INVALID, msg};
        }

        for (auto connect_output_port : port_connect_map_[connect_port]) {
          auto output_ports = node->GetOutputPorts();
          for (auto outside_output_port : output_ports) {
            if (connect_output_port == outside_output_port) {
              auto msg = node->GetName() + " : " +
                         outside_output_port->GetName() +
                         " more condition stream is connect to one input port";
              return {STATUS_INVALID, msg};
            }
          }
        }
        port_connect_map_[connect_port].push_back(output_port);
        continue;
      }
    }

    port_stream_map_[connect_port] = output_stream;
    if (port_connect_map_[connect_port].size() != 0) {
      port_connect_map_[connect_port].push_back(output_port);
      continue;
    }

    std::vector<std::shared_ptr<OutPort>> outport_vector;
    outport_vector.push_back(output_port);
    port_connect_map_[connect_port] = outport_vector;
  }
  return STATUS_OK;
}

Status StreamMatcher::GenerateOutputStream(
    std::shared_ptr<NodeBase> node, std::shared_ptr<GraphStream> input_stream) {
  if (node->GetOutputNum() == 0) {
    return STATUS_OK;
  }

  auto output_ports = node->GetOutputPorts();
  std::shared_ptr<GraphStream> output_stream = nullptr;
  for (auto output_port : output_ports) {
    output_stream =
        GenOutStreamFromInput(node, input_stream, output_stream, output_port);

    if (output_stream == nullptr) {
      auto msg = node->GetName() + " : " + output_port->GetName() +
                 " generate stream failed.please check in conditon branch,it's "
                 "only allowed to use the normal flow";
      return {STATUS_INVALID, msg};
    }

    if (output_stream == root_stream_) {
      auto msg =
          node->GetName() + " : " + output_port->GetName() +
          " generate stream failed,we can't collapse the Outermost stream";
      return {STATUS_INVALID, msg};
    }

    auto status = BindOutputStream(node, output_port, output_stream);
    if (status != STATUS_OK) {
      return status;
    }
  }

  return STATUS_OK;
}

Status StreamMatcher::UpdateCheckingNode() {
  std::vector<std::shared_ptr<NodeBase>> added_nodes;
  std::vector<std::shared_ptr<NodeBase>> untouched_nodes;

  for (auto checking_node : checking_nodes_) {
    checked_nodes_.insert(checking_node);
  }
  checking_nodes_.clear();

  for (auto node : all_nodes_) {
    if (checked_nodes_.find(node) != checked_nodes_.end()) {
      continue;
    }

    auto input_ports = node->GetInputPorts();
    bool ready_flag = false;
    for (auto input_port : input_ports) {
      if (port_connect_map_.find(input_port) == port_connect_map_.end()) {
        ready_flag = false;
        break;
      }

      auto real_node = std::dynamic_pointer_cast<Node>(node);
      if (real_node == nullptr) {
        auto msg = "node: " + node->GetName() + " convert failed.";
        return {modelbox::STATUS_FAULT, msg};
      }

      if (port_connect_map_[input_port].size() !=
              input_port->GetAllOutPort().size() &&
          real_node->GetLoopType() != LOOP) {
        ready_flag = false;
        break;
      } else {
        ready_flag = true;
      }
    }

    if (ready_flag) {
      added_nodes.push_back(node);
    } else {
      untouched_nodes.push_back(node);
    }
  }

  if (added_nodes.size() == 0) {
    if (checked_nodes_.size() != all_nodes_.size()) {
      std::string msg = " in the graph is untouchable";
      for (auto untouched_node : untouched_nodes) {
        msg = untouched_node->GetName() + "," + msg;
      }
      return {STATUS_INVALID, msg};
    } else {
      return STATUS_OK;
    }
  }

  for (auto added_node : added_nodes) {
    checking_nodes_.insert(added_node);
  }

  return STATUS_CONTINUE;
}

std::shared_ptr<GraphStream> StreamMatcher::GenerateRootStream() {
  root_stream_ = std::make_shared<GraphStream>();
  return root_stream_;
}

GraphStream::GraphStream() {
  parent_stream_ = nullptr;
  full_stream_ = nullptr;
}

bool GraphStream::IsSameConditonGroupStream(
    std::shared_ptr<GraphStream> other_stream) {
  if (full_stream_ != other_stream->full_stream_) {
    return false;
  }

  std::shared_ptr<GraphStream> next_stream = next_stream_.lock();
  while (true) {
    if (next_stream == other_stream) {
      return true;
    }

    if (next_stream == shared_from_this()) {
      break;
    }
    next_stream = next_stream->next_stream_.lock();
  }

  return false;
}

std::shared_ptr<GraphStream> GraphStream::GenerateConditionStream(
    std::shared_ptr<GraphStream> other_stream) {
  auto condition_stream = std::make_shared<GraphStream>();
  condition_stream->full_stream_ = shared_from_this();
  if (other_stream != nullptr) {
    condition_stream->next_stream_ = other_stream;
    auto next_stream = other_stream;
    while (true) {
      if (other_stream == next_stream->next_stream_.lock()) {
        break;
      }
      next_stream = next_stream->next_stream_.lock();
    }
    next_stream->next_stream_ = condition_stream;
  } else {
    condition_stream->next_stream_ = condition_stream;
  }

  return condition_stream;
}

std::shared_ptr<GraphStream> GraphStream::GenerateChildStream() {
  auto child_stream = std::make_shared<GraphStream>();
  child_stream->parent_stream_ = shared_from_this();
  return child_stream;
}

std::shared_ptr<GraphStream> GraphStream::GenerateSiblingStream() {
  if (parent_stream_ == nullptr) {
    return nullptr;
  }
  return parent_stream_->GenerateChildStream();
}

std::shared_ptr<GraphStream> GraphStream::GetParentStream() {
  return parent_stream_;
}

std::shared_ptr<GraphStream> GraphStream::GetFullStream() {
  return full_stream_;
}
}  // namespace modelbox
