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


#include "skip_frame.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

SkipFrameFlowUnit::SkipFrameFlowUnit(){};
SkipFrameFlowUnit::~SkipFrameFlowUnit(){};

modelbox::Status SkipFrameFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  skip_rate_ = opts->GetUint32("process_frame_per_second", 5);
  if (skip_rate_ <= 0) {
    auto errMsg = "skip_rate is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status SkipFrameFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_buffer_list = data_ctx->Input("input_frame");
  auto output_buffer_list = data_ctx->Output("output_frame");
  int64_t real_index;
  int32_t rate_num;

  for (auto &buffer : *input_buffer_list) {
    if (!buffer->Get("index", real_index)) {
      MBLOG_ERROR << "skip_frame flowunit can not get input 'index' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key index"};
    }
    if (!buffer->Get("rate_num", rate_num)) {
      MBLOG_ERROR
          << "skip_frame flowunit can not get input 'rate_num' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key rate_num"};
    }
    int32_t interval = rate_num / skip_rate_;

    if (0 != (real_index % interval)) {
      continue;
    }

    output_buffer_list->PushBack(buffer);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_FLOWUNIT(SkipFrameFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput({"input_frame", modelbox::DEVICE_TYPE});
  desc.AddFlowUnitOutput({"output_frame", modelbox::DEVICE_TYPE});
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("process_frame_per_second", "int", true, "5",
                             "skipping frames for acceleration"));
  desc.SetFlowType(modelbox::STREAM);
  desc.SetStreamSameCount(false);
}