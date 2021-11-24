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

#ifndef MODELBOX_DEVICE_ASCEND_H_
#define MODELBOX_DEVICE_ASCEND_H_

#include <modelbox/base/device.h>
#include <modelbox/data_context.h>
#include <modelbox/device/ascend/ascend_memory.h>
#include <modelbox/flow.h>

namespace modelbox {

constexpr const char *DEVICE_TYPE = "ascend";
constexpr const char *DEVICE_DRIVER_NAME = "device-ascend";
constexpr const char *DEVICE_DRIVER_DESCRIPTION = "A ascend device driver";

class Ascend : public Device {
 public:
  Ascend(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  virtual ~Ascend();
  const std::string GetType() const override;

  Status DeviceExecute(DevExecuteCallBack fun, int32_t priority,
                       size_t count) override;

  bool NeedResourceNice() override;
};

class AscendFactory : public DeviceFactory {
 public:
  AscendFactory();
  virtual ~AscendFactory();

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe();
  const std::string GetDeviceFactoryType();
  std::shared_ptr<Device> CreateDevice(const std::string &device_id);
};

class AscendDesc : public DeviceDesc {
 public:
  AscendDesc() = default;
  virtual ~AscendDesc() = default;
};

class AscendFlowUnit : public FlowUnit {
 public:
  AscendFlowUnit() = default;
  virtual ~AscendFlowUnit() = default;

  virtual Status AscendProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               aclrtStream stream) = 0;

  virtual Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_ASCEND_H_