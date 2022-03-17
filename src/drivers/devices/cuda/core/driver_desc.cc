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

#include <cuda_runtime.h>
#include <modelbox/base/timer.h>
#include <stdio.h>

#include <memory>

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/device/cuda/device_cuda.h"

#define CUDA_DEVICE_SCHEDULE_FLAG "CUDA_DEVICE_SCHEDULE_FLAG"

static std::map<std::string, unsigned int> device_flags_map = {
  {"cudaDeviceScheduleAuto", cudaDeviceScheduleAuto},
  {"cudaDeviceScheduleSpin", cudaDeviceScheduleSpin},
  {"cudaDeviceScheduleYield", cudaDeviceScheduleYield},
  {"cudaDeviceScheduleBlockingSync", cudaDeviceScheduleBlockingSync},
  {"cudaDeviceMapHost", cudaDeviceMapHost}
};

std::shared_ptr<modelbox::Timer> kDeviceTimer;

modelbox::Timer *GetTimer() { return kDeviceTimer.get(); }

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<modelbox::CudaFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetClass(modelbox::DRIVER_CLASS_DEVICE);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetName(modelbox::DEVICE_DRIVER_NAME);
  desc->SetDescription(modelbox::DEVICE_DRIVER_DESCRIPTION);

  return;
}

modelbox::Status DriverInit() {
  if (kDeviceTimer != nullptr) {
    return modelbox::STATUS_OK;
  }

  kDeviceTimer = std::make_shared<modelbox::Timer>();
  kDeviceTimer->SetName("Cuda-Timer");
  kDeviceTimer->Start();

  auto env_flag = getenv(CUDA_DEVICE_SCHEDULE_FLAG);
  std::string cudaDeviceScheduleFlag;
  if (env_flag != nullptr) {
    cudaDeviceScheduleFlag = std::string(env_flag);
  }

  unsigned int flag = cudaDeviceScheduleAuto;
  if (device_flags_map.find(cudaDeviceScheduleFlag) != device_flags_map.end()) {
    flag = device_flags_map[cudaDeviceScheduleFlag];
  }

  auto cuda_ret = cudaSetDeviceFlags(flag);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "set cuda device flags " << flag << " failed, cuda ret " << cuda_ret;
    return modelbox::STATUS_OK;
  }

  return modelbox::STATUS_OK;
}

void DriverFini() {
  if (kDeviceTimer == nullptr) {
    return;
  }

  // Driver Fini.
  kDeviceTimer->Stop();
  kDeviceTimer = nullptr;
}

