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

#ifndef MODELBOX_MODELBOX_EDITOR_FLOWUNIT_INFO_H_
#define MODELBOX_MODELBOX_EDITOR_FLOWUNIT_INFO_H_

#include <modelbox/base/config.h>
#include <modelbox/modelbox.h>

namespace modelbox {

class FlowUnitInfo {
 public:
  FlowUnitInfo();
  virtual ~FlowUnitInfo();

  Status Init(const std::shared_ptr<Configuration>& config);

  Status GetInfoInJson(std::string *result);

  std::shared_ptr<DeviceManager> GetDeviceManager();

  std::shared_ptr<FlowUnitManager> GetFlowUnitManager();

  std::shared_ptr<Drivers> GetDriverManager();

 private:
  std::shared_ptr<Drivers> drivers_;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<DeviceManager> device_;
  std::shared_ptr<FlowUnitManager> flowunit_;
  std::vector<std::string> flowunits_from_files_;
};
}  // namespace modelbox
#endif  // MODELBOX_MODELBOX_EDITOR_FLOWUNIT_INFO_H_
