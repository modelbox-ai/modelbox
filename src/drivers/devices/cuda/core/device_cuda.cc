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

#include "modelbox/device/cuda/device_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <thread>

#include "device_stream.h"
#include "modelbox/base/log.h"

namespace modelbox {

Cuda::Cuda(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

Cuda::~Cuda() = default;

std::string Cuda::GetType() const { return DEVICE_TYPE; }

Status Cuda::DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                           size_t count) {
  if (0 == count) {
    return STATUS_OK;
  }

  for (size_t i = 0; i < count; ++i) {
    auto status = fun(i);
    if ((status != STATUS_OK) && (status != STATUS_CONTINUE)) {
      MBLOG_WARN << "executor func failed: " << status;
      return status;
    }
  }

  return STATUS_OK;
};

bool Cuda::NeedResourceNice() { return true; }

CudaFactory::CudaFactory() = default;
CudaFactory::~CudaFactory() = default;

std::map<std::string, std::shared_ptr<DeviceDesc>> CudaFactory::DeviceProbe() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> return_map;
  std::vector<std::string> device_list = GetDeviceList();
  cudaDeviceProp prop;
  for (auto &device : device_list) {
    auto cuda_ret = cudaGetDeviceProperties(&prop, std::stoi(device));
    if (cudaSuccess != cuda_ret) {
      MBLOG_WARN << "Get device " << device << " properties failed, cuda_ret "
                 << cuda_ret;
      continue;
    }

    auto device_desc = std::make_shared<CudaDesc>();
    device_desc->SetDeviceDesc("This is a cuda device description.");
    device_desc->SetDeviceId(device);
    device_desc->SetDeviceMemory(GetBytesReadable(prop.totalGlobalMem));
    device_desc->SetDeviceType("cuda");
    return_map.insert(std::make_pair(device, device_desc));
  }
  return return_map;
}

std::string CudaFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::vector<std::string> CudaFactory::GetDeviceList() {
  std::vector<std::string> device_list;
  int count;
  auto cuda_ret = cudaGetDeviceCount(&count);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "count device failed, cuda ret " << cuda_ret;
    return device_list;
  }

  for (int i = 0; i < count; i++) {
    device_list.push_back(std::to_string(i));
  }

  return device_list;
}

std::shared_ptr<Device> CudaFactory::CreateDevice(
    const std::string &device_id) {
  auto mem_mgr = std::make_shared<CudaMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }
  return std::make_shared<Cuda>(mem_mgr);
}

CudaFlowUnit::CudaFlowUnit() = default;

CudaFlowUnit::~CudaFlowUnit() = default;

Status CudaFlowUnit::Process(std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto cuda_ret = cudaSetDevice(dev_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Set cuda device " << dev_id_ << " failed, cuda ret "
                << cuda_ret;
    return STATUS_FAULT;
  }

  auto stream = GetDevSyncStream<CudaStream, CudaMemory>(data_ctx);
  modelbox::Status status;
  if (stream == nullptr) {
    return {modelbox::STATUS_NOTFOUND, "get sync stream failed."};
  }

  auto process_status = CudaProcess(data_ctx, stream->Get());
  if (process_status != modelbox::STATUS_OK &&
      process_status != modelbox::STATUS_CONTINUE) {
    return process_status;
  }

  status = SetDevStream<CudaStream, CudaMemory>(data_ctx, stream);
  if (!status) {
    return status;
  }

  status = HoldMemory<CudaStream>(data_ctx, stream);
  if (!status) {
    return status;
  }

  return process_status;
}

CudaDesc::CudaDesc() = default;

CudaDesc::~CudaDesc() = default;

}  // namespace modelbox
