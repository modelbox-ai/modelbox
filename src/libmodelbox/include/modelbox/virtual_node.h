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

#ifndef MODELBOX_VIRTUAL_NODE_H_
#define MODELBOX_VIRTUAL_NODE_H_

#include <chrono>

#include "external_data_map.h"
#include "modelbox/base/device.h"
#include "modelbox/error.h"
#include "modelbox/external_data_map.h"
#include "modelbox/node.h"
#include "modelbox/statistics.h"

namespace modelbox {

class InputVirtualNode : public Node {
 public:
  InputVirtualNode(std::string device_name, std::string device_id,
                   std::shared_ptr<DeviceManager> device_manager);

  ~InputVirtualNode() override;

  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              const std::shared_ptr<Configuration>& config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  std::shared_ptr<Device> GetDevice() override;

 private:
  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;
};

class OutputVirtualNode : public Node {
 public:
  OutputVirtualNode(const std::string& device_name,
                    const std::string& device_id,
                    std::shared_ptr<DeviceManager> device_manager);

  ~OutputVirtualNode() override;

  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              const std::shared_ptr<Configuration>& config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  std::shared_ptr<Device> GetDevice() override;

 private:
  void EraseInvalidData();

  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;

  std::shared_ptr<Device> target_device_;
  bool need_move_to_device_{false};
};

class SessionUnmatchCache {
 public:
  SessionUnmatchCache(const std::set<std::string>& port_names);

  void SetTargetDevice(std::shared_ptr<Device> target_device);

  Status CacheBuffer(const std::string& port_name,
                     const std::shared_ptr<Buffer>& buffer);

  std::shared_ptr<FlowUnitError> GetLastError();

  Status PopCache(OutputBufferList& output_buffer_list);

 private:
  std::shared_ptr<Device> target_device_;

  std::unordered_map<std::string, std::map<std::shared_ptr<Stream>,
                                           std::vector<std::shared_ptr<Buffer>>,
                                           StreamPtrOrderCmp>>
      port_streams_map_;

  std::shared_ptr<FlowUnitError> last_error_;
};

class OutputUnmatchVirtualNode : public Node {
 public:
  OutputUnmatchVirtualNode(const std::string& device_name,
                           const std::string& device_id,
                           std::shared_ptr<DeviceManager> device_manager);

  ~OutputUnmatchVirtualNode() override;

  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              const std::shared_ptr<Configuration>& config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  std::shared_ptr<Device> GetDevice() override;

 private:
  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;

  std::shared_ptr<Device> target_device_;
  bool need_move_to_device_{false};

  std::unordered_map<std::shared_ptr<Session>,
                     std::shared_ptr<SessionUnmatchCache>>
      session_cache_map_;
};

}  // namespace modelbox
#endif  // MODELBOX_VIRTUAL_NODE_H_
