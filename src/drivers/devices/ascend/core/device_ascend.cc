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

#include "modelbox/device/ascend/device_ascend.h"

#include <acl/acl.h>
#include <dsmi_common_interface.h>
#include <stdio.h>

#include "device_stream.h"
#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/device/ascend/ascend_memory.h"

const size_t SIZE_MB = 1024 * 1024;

namespace modelbox {
Ascend::Ascend(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

Ascend::~Ascend() {}

const std::string Ascend::GetType() const { return DEVICE_TYPE; }

Status Ascend::DeviceExecute(DevExecuteCallBack fun, int32_t priority,
                             size_t count) {
  if (0 == count) {
    return STATUS_OK;
  }

  for (size_t i = 0; i < count; ++i) {
    auto status = fun(i);
    if (!status) {
      MBLOG_WARN << "executor func failed: " << status
                 << " stack trace:" << GetStackTrace();
      return status;
    }
  }

  return STATUS_OK;
};

bool Ascend::NeedResourceNice() { return true; }

AscendFactory::AscendFactory() {}
AscendFactory::~AscendFactory() {}

std::map<std::string, std::shared_ptr<DeviceDesc>>
AscendFactory::DeviceProbe() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> device_desc_map;
  int32_t count = 0;
  auto ret = dsmi_get_device_count(&count);
  if (ret != 0) {
    MBLOG_ERROR << "dsmi_get_device_count failed, ret " << ret;
    return device_desc_map;
  }

  for (int32_t id = 0; id < count; ++id) {
    dsmi_memory_info_stru mem_info;
    ret = dsmi_get_memory_info(id, &mem_info);
    if (ret != 0) {
      MBLOG_ERROR << "dsmi_get_memory_info id:" << id << "failed, ret " << ret;
      continue;
    }

    auto device_desc = std::make_shared<AscendDesc>();
    device_desc->SetDeviceDesc("This is a ascend device description.");
    auto id_str = std::to_string(id);
    device_desc->SetDeviceId(id_str);
    device_desc->SetDeviceMemory(
        GetBytesReadable(mem_info.memory_size * SIZE_MB));
    device_desc->SetDeviceType("ascend");
    device_desc_map.insert(std::make_pair(id_str, device_desc));
  }

  return device_desc_map;
}

const std::string AscendFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::shared_ptr<Device> AscendFactory::CreateDevice(
    const std::string &device_id) {
  auto mem_mgr = std::make_shared<AscendMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }

  return std::make_shared<Ascend>(mem_mgr);
}

Status AscendFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto ret = aclrtSetDevice(dev_id_);
  if (ret != ACL_SUCCESS) {
    MBLOG_ERROR << "Set ascend device " << dev_id_ << " failed, acl ret "
                << ret;
    return STATUS_FAULT;
  }

  auto stream = GetDevSyncStream<AscendStream, AscendMemory>(data_ctx);
  modelbox::Status status;
  if (stream == nullptr) {
    return {modelbox::STATUS_NOTFOUND, "get sync stream failed."};
  }

  auto process_status = AscendProcess(data_ctx, stream->Get());
  if (process_status != modelbox::STATUS_OK &&
      process_status != modelbox::STATUS_CONTINUE) {
    return process_status;
  }

  status = SetDevStream<AscendStream, AscendMemory>(data_ctx, stream);
  if (!status) {
    return status;
  }

  status = HoldMemory<AscendStream>(data_ctx, stream);
  if (!status) {
    return status;
  }

  return process_status;
}

}  // namespace modelbox
