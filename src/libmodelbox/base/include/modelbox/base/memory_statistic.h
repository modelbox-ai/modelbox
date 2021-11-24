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


#ifndef MODELBOX_MEMORY_H
#define MODELBOX_MEMORY_H

#include <modelbox/base/device.h>

namespace modelbox {
/**
 * @brief Device memory statistics
 */
class MemoryStatistics {
 public:
  MemoryStatistics(DeviceManager &deviceManager);
  virtual ~MemoryStatistics();

  /**
   * @brief Get total memory of one device
   * @param device_type
   * @param device_id
   * @return int
   */
  int GetTotalMemory(std::string &device_type, std::string &device_id);

  /**
   * @brief Get already used memory of one device
   * @param device_type
   * @param device_id
   * @return int
   */
  int GetUsedMemory(std::string &device_type, std::string &device_id);

  /**
   * @brief Get total memory of all devices
   * @return std::map<std::string, int>
   */
  std::map<std::string, int> &GetTotalMemory();

  /**
   * @brief Get already used memory of all devices
   * @return std::map<std::string, int>
   */
  std::map<std::string, int> &GetUsedMemory();

  /**
   * @brief Get memory trace log of one device
   * @param device_type
   * @param device_id
   * @return std::vector<CircleQueue>
   */
  std::vector<CircleQueue> &GetMemoryTrace(std::string &device_type,
                                           std::string &device_id);

  /**
   * @brief Get memory trace log of all device
   * @return std::map<std::string, std::vector<CircleQueue>>
   */
  std::map<std::string, std::vector<CircleQueue>> &GetMemoryTrace();

  /**
   * @brief Get allocated device memory of one device
   * @param device_type
   * @param device_id
   */
  std::vector<std::shared_ptr<DeviceMemory>> GetAllocatedMemory(
      std::string &device_type, std::string &device_id);

 private:
  DeviceManager device_mgr_;
};
}  // namespace modelbox

#endif  // MODELBOX_MEMORY_H
