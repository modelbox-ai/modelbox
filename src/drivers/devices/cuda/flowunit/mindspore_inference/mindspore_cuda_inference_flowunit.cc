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

#include "mindspore_cuda_inference_flowunit.h"

MindSporeInferenceCudaFlowUnit::MindSporeInferenceCudaFlowUnit(){};

MindSporeInferenceCudaFlowUnit::~MindSporeInferenceCudaFlowUnit(){};

std::shared_ptr<mindspore::DeviceInfoContext>
MindSporeInferenceCudaFlowUnit::GetDeviceInfoContext(
    std::shared_ptr<modelbox::Configuration> &config) {
  auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  auto device_id = config->GetInt32("deviceid", 0);
  device_info->SetDeviceID(device_id);

  return device_info;
}

std::shared_ptr<modelbox::FlowUnit>
MindSporeInferenceCudaFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  auto inference_flowunit = std::make_shared<MindSporeInferenceCudaFlowUnit>();
  return inference_flowunit;
};
