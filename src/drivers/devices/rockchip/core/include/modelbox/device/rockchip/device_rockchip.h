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

#ifndef MODELBOX_DEVICE_ROCKCHIP_H_
#define MODELBOX_DEVICE_ROCKCHIP_H_

#include <modelbox/base/device.h>
#include <modelbox/data_context.h>
#include <modelbox/device/rockchip/rockchip_memory.h>
#include <modelbox/flow.h>

namespace modelbox {

typedef void MppBufHdl;

constexpr const char *DEVICE_TYPE = "rockchip";
constexpr const char *DEVICE_DRIVER_NAME = "device-rockchip";
constexpr const char *DEVICE_DRIVER_DESCRIPTION = "A rockchip device driver";

class RockChip : public Device {
 public:
  RockChip(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  ~RockChip() override = default;
  std::string GetType() const override;

  Status DeviceExecute(const DevExecuteCallBack &rkfun, int32_t priority,
                       size_t rkcount) override;
  bool NeedResourceNice() override;
};

class RockChipFactory : public DeviceFactory {
 public:
  RockChipFactory() = default;
  ~RockChipFactory() override = default;

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe() override;
  std::string GetDeviceFactoryType() override;
  std::shared_ptr<Device> CreateDevice(const std::string &device_id) override;

 private:
  std::map<std::string, std::shared_ptr<DeviceDesc>> ProbeRKNNDevice();
};

class RockChipDesc : public DeviceDesc {
 public:
  RockChipDesc() = default;
  ~RockChipDesc() override = default;
};

// use it to store the rknn device names
class RKNNDevs {
 public:
  typedef enum {
    RKNN_DEVICE_TYPE_OTHERS = 0,
    RKNN_DEVICE_TYPE_RK3399PRO,
    RKNN_DEVICE_TYPE_RK356X,
    RKNN_DEVICE_TYPE_RK358X,
    RKNN_DEVICE_TYPE_RV110X
  } RKNN_DEVICE_TYPE;

  static RKNNDevs &Instance() {
    static RKNNDevs rk_nndevs;
    return rk_nndevs;
  }

  void SetNames(std::vector<std::string> &dev_names);
  const std::vector<std::string> &GetNames();
  void UpdateDeviceType();
  RKNN_DEVICE_TYPE GetDeviceType();

 private:
  RKNNDevs() = default;
  virtual ~RKNNDevs() = default;
  RKNNDevs(const RKNNDevs &) = delete;
  RKNNDevs &operator=(const RKNNDevs &) = delete;

  std::vector<std::string> dev_names_;
  RKNN_DEVICE_TYPE type_{RKNN_DEVICE_TYPE_OTHERS};
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_ROCKCHIP_H_
