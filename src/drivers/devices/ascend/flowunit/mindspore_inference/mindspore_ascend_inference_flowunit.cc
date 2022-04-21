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

#include "mindspore_ascend_inference_flowunit.h"

constexpr const char *ASCEND_D310_TYPE = "ascend310";
constexpr const char *ASCEND_D910_TYPE = "ascend910";

MindSporeInferenceAsendFlowUnit::MindSporeInferenceAsendFlowUnit(){};

MindSporeInferenceAsendFlowUnit::~MindSporeInferenceAsendFlowUnit(){};

std::shared_ptr<mindspore::DeviceInfoContext>
MindSporeInferenceAsendFlowUnit::GetDeviceInfoContext(
    std::shared_ptr<modelbox::Configuration> &config) {
  std::shared_ptr<mindspore::DeviceInfoContext> device_info;

  auto device_type = config->GetString("device_type", ASCEND_D310_TYPE);
  auto device_id = config->GetInt32("deviceid", 0);

  if (device_type == ASCEND_D310_TYPE) {
    auto d310_device_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    // NCHW or NHWC
    auto input_format = config->GetString("input_format", "NCHW");
    d310_device_info->SetDeviceID(device_id);
    d310_device_info->SetInputFormat(input_format);
    device_info = d310_device_info;
  } else if (device_type == ASCEND_D910_TYPE) {
    device_info = std::make_shared<mindspore::Ascend910DeviceInfo>();
  } else {
    modelbox::StatusError = {modelbox::STATUS_NOTSUPPORT,
                             "Not support card type: " + device_type};
    return nullptr;
  }

  return device_info;
}

std::shared_ptr<modelbox::FlowUnit>
MindSporeInferenceAsendFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  auto inference_flowunit = std::make_shared<MindSporeInferenceAsendFlowUnit>();
  return inference_flowunit;
};
