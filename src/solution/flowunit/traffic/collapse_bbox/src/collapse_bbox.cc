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


#include "collapse_bbox.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

CollapseBBoxFlowUnit::CollapseBBoxFlowUnit(){};
CollapseBBoxFlowUnit::~CollapseBBoxFlowUnit(){};

modelbox::Status CollapseBBoxFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status CollapseBBoxFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status CollapseBBoxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  // input: bbox  {[bboxes], ... , [bboxes]}
  auto input_bufs = ctx->Input("input");
  // output : bbox  {[bbox,bbox,...,bbox]}
  auto output_bufs = ctx->Output("output");

  MBLOG_DEBUG << "collapse_bbox input size: " << input_bufs->Size();

  size_t data_size = sizeof(BBox);
  std::vector<std::shared_ptr<BBox>> bboxes;
  for (size_t i = 0; i < input_bufs->Size(); i++) {
    size_t num_bboxes = input_bufs->At(i)->GetBytes() / data_size;
    for (size_t j = 0; j < num_bboxes; j++) {
      std::shared_ptr<BBox> b = std::make_shared<BBox>();
      auto ret = memcpy_s(
          b.get(), data_size,
          (const char *)(input_bufs->ConstBufferData(i)) + (data_size * j),
          data_size);
      if (EOK != ret) {
        MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size "
                   << data_size << ", dest size " << data_size
                   << ", skip an bbox!";
        continue;
      }
      bboxes.push_back(b);
    }
  }
  MBLOG_DEBUG << "collapse bboxes size: " << bboxes.size();

  std::vector<size_t> shape(1, bboxes.size() * data_size);
  output_bufs->Build(shape);

  auto output_data = (BBox *)(output_bufs->MutableBufferData(0));
  if (output_data == nullptr) {
    MBLOG_ERROR << "get output buffer failed.";
    return modelbox::STATUS_NOMEM;
  }

  for (auto &b : bboxes) {
    auto ret = memcpy_s(output_data, data_size, b.get(), data_size);
    if (EOK != ret) {
      MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size "
                  << data_size << ", dest size " << data_size
                  << ", skip an bbox!";
      continue;
    }
    output_data++;

    MBLOG_DEBUG << "collapse box: w: " << b->w << " h: " << b->h;
  }
  MBLOG_DEBUG << "collapse output size: " << output_bufs->Size();
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(CollapseBBoxFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("input", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("output", modelbox::DEVICE_TYPE));
  desc.SetOutputType(modelbox::COLLAPSE);
  desc.SetCollapseAll(true);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}