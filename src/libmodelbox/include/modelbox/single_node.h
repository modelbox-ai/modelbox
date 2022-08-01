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

#ifndef MODELBOX_UNMATCH_NODE_H_
#define MODELBOX_UNMATCH_NODE_H_
#include "data_handler.h"
#include "modelbox/node.h"

namespace modelbox {

class SingleNode : public Node {
 public:
  using Node::Init;
  using Node::Run;
  SingleNode(const std::string& unit_name, const std::string& unit_type,
             const std::string& unit_device_id,
             std::shared_ptr<FlowUnitManager> flowunit_mgr,
             std::shared_ptr<Configuration> config,
             std::shared_ptr<Profiler> profiler = nullptr,
             std::shared_ptr<StatisticsItem> graph_stats = nullptr);

  /**
   * @brief init node amnd port
   * @return init result.
   */
  Status Init();

  /**
   * @brief run node process
   * @param data  run node with input data
   * @return process result.
   */
  void Run(const std::shared_ptr<DataHandler>& data);

  /**
   * @brief  push data to output datahandler
   * @param data_handler output datahandler
   * @return process result.
   */
  Status PushDataToDataHandler(std::shared_ptr<DataHandler>& data_handler);

 private:
  std::shared_ptr<FlowUnitDataContext> CreateDataContext();
  Status RecvData(const std::shared_ptr<DataHandler>& data);
  Status Process();

  std::shared_ptr<FlowUnitDataContext> data_context_;
  std::shared_ptr<Configuration> config_;
};

}  // namespace modelbox
#endif
