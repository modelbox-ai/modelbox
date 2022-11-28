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

#include <utility>

#include "modelbox/single_node.h"

namespace modelbox {
#define DEFAULT_SINGLE_NODE_QUEUE_SIZE 8192
SingleNode::SingleNode(const std::string& unit_name,
                       const std::string& unit_type,
                       const std::string& unit_device_id,
                       std::shared_ptr<FlowUnitManager> flowunit_mgr,
                       std::shared_ptr<Configuration> config,
                       std::shared_ptr<Profiler> profiler,
                       std::shared_ptr<StatisticsItem> graph_stats)
    : config_(std::move(config)) {
  SetFlowUnitInfo(unit_name, unit_type, unit_device_id,
                  std::move(flowunit_mgr));
  SetProfiler(std::move(profiler));
  SetStats(std::move(graph_stats));
}

Status SingleNode::Init() {
  flowunit_group_ = std::make_shared<FlowUnitGroup>(
      flowunit_name_, flowunit_type_, flowunit_device_id_, config_, profiler_);
  std::set<std::string> input_port_names;
  auto input_ports =
      flowunit_manager_->GetFlowUnitDesc(flowunit_type_, flowunit_name_)
          ->GetFlowUnitInput();

  for (auto& input_port : input_ports) {
    auto input_port_name = input_port.GetPortName();
    queue_size_ = config_->GetUint64("queue_size", DEFAULT_SINGLE_NODE_QUEUE_SIZE);
    if (0 == queue_size_) {
      return {STATUS_INVALID, "invalid queue_size config: 0"};
    }
    auto in_queue_size =
        config_->GetUint64("queue_size_" + input_port_name, queue_size_);
    if (0 == in_queue_size) {
      return {STATUS_INVALID,
              "invalid queue_size_" + input_port_name + " config: 0"};
    }

    input_ports_.emplace_back(std::make_shared<InPort>(
        input_port_name,
        std::dynamic_pointer_cast<NodeBase>(shared_from_this()), GetPriority(),
        in_queue_size));
  }

  auto out_ports =
      flowunit_manager_->GetFlowUnitDesc(flowunit_type_, flowunit_name_)
          ->GetFlowUnitOutput();

  for (auto& output_port : out_ports) {
    auto output_port_name = output_port.GetPortName();
    output_ports_.emplace_back(
        std::make_shared<OutPort>(output_port_name, shared_from_this()));
  }
  std::set<std::string> input_ports_name;
  std::set<std::string> output_ports_name;
  auto status = flowunit_group_->Init(input_ports_name, output_ports_name,
                                      flowunit_manager_, false);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "failed init flowunit group";
    return status;
  }

  flowunit_group_->SetNode(std::dynamic_pointer_cast<Node>(shared_from_this()));
  return STATUS_OK;
}

std::shared_ptr<FlowUnitDataContext> SingleNode::CreateDataContext() {
  auto flowunit_desc =
      flowunit_manager_->GetFlowUnitDesc(flowunit_type_, flowunit_name_);
  if (flowunit_desc->GetFlowType() == NORMAL) {
    data_context_ =
        std::make_shared<NormalFlowUnitDataContext>(this, nullptr, nullptr);
  } else {
    MBLOG_ERROR << "flowunit type is stream, return null";
  }
  return data_context_;
}

Status SingleNode::RecvData(const std::shared_ptr<DataHandler>& data) {
  auto input_ports = GetInputPorts();
  auto data_map = std::make_shared<PortDataMap>();

  for (auto& iter : input_ports) {
    auto name = iter->GetName();
    if (input_ports.size() == 1) {
      name = DEFAULT_PORT_NAME;
    }

    auto bufferlist = data->GetBufferList(name);
    if (!bufferlist) {
      MBLOG_ERROR << "bufferlist is nullptr, RecvData error ";
      return STATUS_INVALID;
    }

    bufferlist->Swap(data_map->at(name));
  }

  if (data_context_ == nullptr) {
    data_context_ =
        std::make_shared<NormalFlowUnitDataContext>(this, nullptr, nullptr);
  }

  auto data_ctx =
      std::static_pointer_cast<NormalFlowUnitDataContext>(data_context_);
  data_ctx->WriteInputData(data_map);
  return STATUS_OK;
}

Status SingleNode::Process() {
  if (!flowunit_group_) {
    MBLOG_ERROR << "flowunit_group not created . ";
    return STATUS_INVALID;
  }
  std::list<std::shared_ptr<FlowUnitDataContext>> data_ctx_list;
  data_ctx_list.push_back(data_context_);
  auto status = flowunit_group_->Run(data_ctx_list);
  if (status != STATUS_OK) {
    return STATUS_FAULT;
  }
  return STATUS_OK;
}

Status SingleNode::PushDataToDataHandler(
    std::shared_ptr<DataHandler>& data_handler) {
  if (data_context_ == nullptr || data_handler == nullptr) {
    return STATUS_INVALID;
  }
  PortDataMap port_data_map;
  data_context_->PopOutputData(port_data_map);
  if (port_data_map.size() == 0) {
    return STATUS_NODATA;
  }
  for (auto& iter : port_data_map) {
    std::string port_name = iter.first;

    for (auto buffer : iter.second) {
      data_handler->PushData(buffer, port_name);
    }
  }

  data_context_->ClearData();
  return STATUS_OK;
}

void SingleNode::Run(const std::shared_ptr<DataHandler>& data) {
  auto status = RecvData(data);
  if (status != STATUS_OK) {
    MBLOG_ERROR << "failed recv data ...";
    return;
  }

  status = Process();
  if (status != STATUS_OK) {
    MBLOG_ERROR << "process data failed ...";
    return;
  }
}

}  // namespace modelbox