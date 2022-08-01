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

#ifndef MODELBOX_DEVICE_CUDA_H_
#define MODELBOX_DEVICE_CUDA_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>
#include <modelbox/device/cuda/cuda_memory.h>
#include <modelbox/flow.h>

#include <list>

#define GET_CUDA_API_ERROR(api, err_code, err_str)    \
  const char *err_name = NULL;                        \
  cuGetErrorName(err_code, &err_name);                \
  std::ostringstream error_log;                       \
  error_log << #api << ", return code : " << err_code \
            << ", error : " << err_name;              \
  auto err_str = error_log.str();

#define CUDA_API_CALL(api)                                      \
  do {                                                          \
    CUresult err_code = api;                                    \
    if (err_code != CUDA_SUCCESS) {                             \
      GET_CUDA_API_ERROR(api, err_code, err_str);               \
      throw NVDECException::MakeNVDECException(                 \
          err_str, err_code, __FUNCTION__, __FILE__, __LINE__); \
    }                                                           \
  } while (0)

// This is a no-exception version of the above MACRO CUDA_API_CALL(api)
#define CHECK_CUDA_API_RETURN(api)                           \
  do {                                                       \
    CUresult err_code = api;                                 \
    if (err_code != CUDA_SUCCESS) {                          \
      GET_CUDA_API_ERROR(api, err_code, err_str);            \
      MBLOG_ERROR << "Failed to call CUDA API: " << err_str; \
      return {modelbox::STATUS_FAULT, err_str};              \
    }                                                        \
  } while (0)

namespace modelbox {

constexpr const char *DEVICE_TYPE = "cuda";
constexpr const char *DEVICE_DRIVER_NAME = "device-cuda";
constexpr const char *DEVICE_DRIVER_DESCRIPTION = "A gpu device driver";

class Cuda : public Device {
 public:
  Cuda(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  ~Cuda() override;
  std::string GetType() const override;

  Status DeviceExecute(DevExecuteCallBack fun, int32_t priority,
                       size_t count) override;

  bool NeedResourceNice() override;
};

class CudaFactory : public DeviceFactory {
 public:
  CudaFactory();
  ~CudaFactory() override;

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe() override;
  std::string GetDeviceFactoryType() override;
  std::vector<std::string> GetDeviceList() override;
  std::shared_ptr<Device> CreateDevice(const std::string &device_id) override;
};

class CudaDesc : public DeviceDesc {
 public:
  CudaDesc() = default;
  ~CudaDesc() override = default;
};

class CudaFlowUnit : public FlowUnit {
 public:
  CudaFlowUnit() = default;
  ~CudaFlowUnit() override = default;

  virtual Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                             cudaStream_t stream) = 0;

  Status Process(std::shared_ptr<modelbox::DataContext> data_ctx) override;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_CUDA_H_
