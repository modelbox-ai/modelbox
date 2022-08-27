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

#ifndef MODELBOX_FLOWUNIT_GROUP_H_
#define MODELBOX_FLOWUNIT_GROUP_H_

#include <algorithm>
#include <list>
#include <set>
#include <utility>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_data_executor.h"
#include "modelbox/profiler.h"

namespace modelbox {

using OutputBuffer =
    std::unordered_map<std::string, std::vector<std::shared_ptr<BufferList>>>;

using SchedulerEventList =
    std::shared_ptr<std::vector<std::shared_ptr<SchedulerEvent>>>;
class FlowUnitBalancer;
class Node;
class FlowUnitGroup {
 public:
  FlowUnitGroup(std::string unit_name, std::string unit_type,
                std::string unit_device_id,
                std::shared_ptr<Configuration> config,
                std::shared_ptr<Profiler> profiler);

  virtual ~FlowUnitGroup();

  Status Init(const std::set<std::string> &input_ports_name,
              const std::set<std::string> &output_ports_name,
              const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
              bool checkport = true);

  Status CheckInputAndOutput(const std::set<std::string> &input_ports_name,
                             const std::set<std::string> &output_ports_name);

  Status Run(std::list<std::shared_ptr<FlowUnitDataContext>> &data_ctx_list);

  Status Destory();

  std::shared_ptr<FlowUnit> GetExecutorUnit();

  void SetNode(const std::shared_ptr<Node> &node);

  Status Open(const CreateExternalDataFunc &create_func);

  Status Close();

 private:
  std::weak_ptr<Node> node_;
  uint32_t batch_size_;

  std::vector<std::shared_ptr<FlowUnit>> flowunit_group_;
  std::string unit_name_;
  std::string unit_type_;
  std::string unit_device_id_;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<Profiler> profiler_;
  std::shared_ptr<FlowUnitTrace> flowunit_trace_;
  std::once_flag trace_init_flag_;

  std::shared_ptr<FlowUnitBalancer> balancer_;
  std::shared_ptr<FlowUnitDataExecutor> executor_;

  void InitTrace();

  std::shared_ptr<TraceSlice> StartTrace(FUExecContextList &exec_ctx_list);

  void StopTrace(std::shared_ptr<TraceSlice> &slice);

  void PreProcess(FUExecContextList &exec_ctx_list);

  Status Process(FUExecContextList &exec_ctx_list);

  Status PostProcess(FUExecContextList &exec_ctx_list);

  void PostProcessEvent(FUExecContextList &exec_ctx_list);

  FUExecContextList CreateExecCtx(
      std::list<std::shared_ptr<FlowUnitDataContext>> &data_ctx_list);
};
}  // namespace modelbox
#endif  // MODELBOX_FLOWUNIT_GROUP_H_
