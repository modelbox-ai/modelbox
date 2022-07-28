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

#include "modelbox/flowunit_balancer.h"

namespace modelbox {

static std::unordered_map<FlowUnitBalanceStrategy, std::string,
                          FUBalanceStrategyHash>
    g_strategy_name_map = {
        {FlowUnitBalanceStrategy::FU_ROUND_ROBIN, "RoundRobin"},
        {FlowUnitBalanceStrategy::FU_CAPABILITY, "Capability"},
        {FlowUnitBalanceStrategy::FU_NULL, "Null"}};

std::ostream& operator<<(std::ostream& os, const FlowUnitBalanceStrategy& s) {
  os << g_strategy_name_map[s];
  return os;
}

Status FlowUnitBalancer::Init(
    const std::vector<std::shared_ptr<FlowUnit>>& flowunits) {
  if (flowunits.empty()) {
    return {STATUS_FAULT, "no flowunit available"};
  }

  flowunits_ = flowunits;
  return OnInit();
}

std::shared_ptr<FlowUnit> FlowUnitBalancer::GetFlowUnit(
    const std::shared_ptr<FlowUnitDataContext>& data_ctx) {
  {
    std::lock_guard<std::mutex> lock(ctx_to_flowunit_map_lock_);
    auto item = ctx_to_flowunit_map_.find(data_ctx.get());
    if (item != ctx_to_flowunit_map_.end()) {
      return item->second;
    }
  }

  return FirstBind(data_ctx);
}

std::shared_ptr<FlowUnit> FlowUnitBalancer::FirstBind(
    const std::shared_ptr<FlowUnitDataContext>& data_ctx) {
  auto fu = BindFlowUnit(data_ctx);
  if (fu == nullptr) {
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(ctx_to_flowunit_map_lock_);
    ctx_to_flowunit_map_[data_ctx.get()] = fu;
  }
  std::weak_ptr<FlowUnitBalancer> balancer_ref = shared_from_this();
  auto* data_ctx_ptr = data_ctx.get();
  data_ctx->AddDestroyCallback([data_ctx_ptr, balancer_ref]() {
    auto balancer = balancer_ref.lock();
    if (balancer == nullptr) {
      return;
    }

    balancer->UnbindFlowUnit(data_ctx_ptr);
  });
  return fu;
}

void FlowUnitBalancer::UnbindFlowUnit(const FlowUnitDataContext* data_ctx_ptr) {
  std::lock_guard<std::mutex> lock(ctx_to_flowunit_map_lock_);
  ctx_to_flowunit_map_.erase(data_ctx_ptr);
}

FlowUnitBalancerFactory& FlowUnitBalancerFactory::GetInstance() {
  static FlowUnitBalancerFactory factory;
  return factory;
}

std::shared_ptr<FlowUnitBalancer> FlowUnitBalancerFactory::CreateBalancer(
    FlowUnitBalanceStrategy strategy) {
  auto item = balancer_creator_map_.find(strategy);
  if (item == balancer_creator_map_.end()) {
    MBLOG_ERROR << "Flowunit balance strategy " << strategy
                << " is not supported";
    return nullptr;
  }

  return item->second();
}

void FlowUnitBalancerFactory::RegistBalancer(
    const FUBalancerCreateFunc& create_func) {
  auto balancer = create_func();
  auto strategy = balancer->GetType();
  balancer_creator_map_[strategy] = create_func;
}

FlowUnitBalancerRegister::FlowUnitBalancerRegister(
    const FUBalancerCreateFunc& create_func) {
  auto& factory = FlowUnitBalancerFactory::GetInstance();
  factory.RegistBalancer(create_func);
}

void FlowUnitBalancerUtil::Init(
    const std::vector<std::shared_ptr<FlowUnit>>& flowunits) {
  for (const auto& fu : flowunits) {
    device_to_fu_map_[fu->GetBindDevice().get()] = fu;
  }
}

std::shared_ptr<FlowUnit> FlowUnitBalancerUtil::GetFlowUnitByDevice(
    const std::shared_ptr<Device>& device) {
  auto item = device_to_fu_map_.find(device.get());
  if (item != device_to_fu_map_.end()) {
    return item->second;
  }

  return nullptr;
}

std::set<std::shared_ptr<Device>> FlowUnitBalancerUtil::GetInputDevices(
    const std::shared_ptr<FlowUnitDataContext>& data_ctx) {
  const auto& inputs = data_ctx->GetInputs();
  std::set<std::shared_ptr<Device>> devices;
  for (const auto& port_item : inputs) {
    const auto& port_buffer_list = port_item.second;
    if (port_buffer_list.empty()) {
      continue;
    }

    auto first_buffer = port_buffer_list.front();
    if (first_buffer == nullptr) {
      continue;
    }

    auto dev = first_buffer->GetDevice();
    if (dev == nullptr) {
      continue;
    }

    devices.insert(dev);
  }

  return devices;
}

REGIST_FLOWUNIT_BALANCER(FURoundRobinBalancer);

FlowUnitBalanceStrategy FURoundRobinBalancer::GetType() {
  return FlowUnitBalanceStrategy::FU_ROUND_ROBIN;
}

Status FURoundRobinBalancer::OnInit() {
  util.Init(flowunits_);
  return STATUS_OK;
}

std::shared_ptr<FlowUnit> FURoundRobinBalancer::BindFlowUnit(
    const std::shared_ptr<FlowUnitDataContext>& data_ctx) {
  std::list<std::shared_ptr<FlowUnit>> candidate_fu_list;
  auto devices = util.GetInputDevices(data_ctx);
  for (const auto& device : devices) {
    auto fu = util.GetFlowUnitByDevice(device);
    if (fu == nullptr) {
      continue;
    }

    candidate_fu_list.push_back(fu);
  }

  std::shared_ptr<FlowUnit> fu;
  if (candidate_fu_list.empty()) {
    fu = GetNextFU();
  } else {
    // Use first directly
    fu = candidate_fu_list.front();
  }

  return fu;
}

std::shared_ptr<FlowUnit> FURoundRobinBalancer::GetNextFU() {
  auto fu = flowunits_[fu_index_];
  fu_index_ = (fu_index_ + 1) % flowunits_.size();
  return fu;
}

}  // namespace modelbox