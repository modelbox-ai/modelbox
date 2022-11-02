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

#ifndef MODELBOX_FLOW_UNIT_BALANCER_H_
#define MODELBOX_FLOW_UNIT_BALANCER_H_

#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "flowunit.h"
#include "flowunit_data_executor.h"
#include "modelbox/base/status.h"

namespace modelbox {

enum class FlowUnitBalanceStrategy : int32_t {
  FU_ROUND_ROBIN,
  FU_CAPABILITY,
  FU_NULL
};

class FUBalanceStrategyHash {
 public:
  std::size_t operator()(const FlowUnitBalanceStrategy& value) const {
    return (size_t)value;
  }
};

std::ostream& operator<<(std::ostream& os, const FlowUnitBalanceStrategy& s);

class FlowUnitBalancer : public std::enable_shared_from_this<FlowUnitBalancer> {
 public:
  FlowUnitBalancer();

  virtual ~FlowUnitBalancer();

  Status Init(const std::vector<std::shared_ptr<FlowUnit>>& flowunits);

  std::shared_ptr<FlowUnit> GetFlowUnit(
      const std::shared_ptr<FlowUnitDataContext>& data_ctx);

  void UnbindFlowUnit(const FlowUnitDataContext* data_ctx_ptr);

  virtual FlowUnitBalanceStrategy GetType() = 0;

 protected:
  virtual Status OnInit();

  virtual std::shared_ptr<FlowUnit> BindFlowUnit(
      const std::shared_ptr<FlowUnitDataContext>& data_ctx) = 0;

  std::vector<std::shared_ptr<FlowUnit>> flowunits_;
  std::mutex ctx_to_flowunit_map_lock_;
  std::unordered_map<const DataContext*, std::shared_ptr<FlowUnit>>
      ctx_to_flowunit_map_;

 private:
  std::shared_ptr<FlowUnit> FirstBind(
      const std::shared_ptr<FlowUnitDataContext>& data_ctx);
};

using FUBalancerCreateFunc = std::function<std::shared_ptr<FlowUnitBalancer>()>;

class FlowUnitBalancerFactory {
 public:
  virtual ~FlowUnitBalancerFactory();

  static FlowUnitBalancerFactory& GetInstance();

  std::shared_ptr<FlowUnitBalancer> CreateBalancer(
      FlowUnitBalanceStrategy strategy =
          FlowUnitBalanceStrategy::FU_ROUND_ROBIN);

  void RegistBalancer(const FUBalancerCreateFunc& create_func);

 private:
  FlowUnitBalancerFactory();

  std::unordered_map<FlowUnitBalanceStrategy, FUBalancerCreateFunc,
                     FUBalanceStrategyHash>
      balancer_creator_map_;
};

class FlowUnitBalancerRegister {
 public:
  FlowUnitBalancerRegister(const FUBalancerCreateFunc& create_func);
};

#define REGIST_FLOWUNIT_BALANCER(balancer_class)                         \
  FlowUnitBalancerRegister g_flowunit_balancer_regiter_##balancer_class( \
      []() { return std::make_shared<balancer_class>(); });

class FlowUnitBalancerUtil {
 public:
  void Init(const std::vector<std::shared_ptr<FlowUnit>>& flowunits);

  std::shared_ptr<FlowUnit> GetFlowUnitByDevice(
      const std::shared_ptr<Device>& device);

  std::set<std::shared_ptr<Device>> GetInputDevices(
      const std::shared_ptr<FlowUnitDataContext>& data_ctx);

 private:
  std::unordered_map<const Device*, std::shared_ptr<FlowUnit>>
      device_to_fu_map_;
};

class FURoundRobinBalancer : public FlowUnitBalancer {
 public:
  FlowUnitBalanceStrategy GetType() override;

 protected:
  Status OnInit() override;

  std::shared_ptr<FlowUnit> BindFlowUnit(
      const std::shared_ptr<FlowUnitDataContext>& data_ctx) override;

 private:
  std::shared_ptr<FlowUnit> GetNextFU();

  FlowUnitBalancerUtil util;
  size_t fu_index_{0};
};

};  // namespace modelbox

#endif  // MODELBOX_FLOW_UNIT_BALANCER_H_