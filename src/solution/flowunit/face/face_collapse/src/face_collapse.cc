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


#include "face_collapse.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

FaceCollapseFlowUnit::FaceCollapseFlowUnit(){};
FaceCollapseFlowUnit::~FaceCollapseFlowUnit(){};

modelbox::Status FaceCollapseFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status FaceCollapseFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceCollapseFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "process collapse faces";
  // input: faces  {[faces], ... , [faces]}
  auto input_bufs = ctx->Input(COLLAPSE_UNIT_IN_NAME[0]);
  if (input_bufs->Size() <= 0) {
    auto errMsg = "In_img batch is " + std::to_string(input_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // output : faces  {[faces,faces,...,faces]}
  auto output_bufs = ctx->Output(COLLAPSE_UNIT_OUT_NAME[0]);

  size_t data_size = sizeof(FaceExpressionScore);
  std::vector<std::shared_ptr<FaceExpressionScore>> faces;
  for (size_t i = 0; i < input_bufs->Size(); i++) {
    size_t num_faces = input_bufs->At(i)->GetBytes() / data_size;
    for (size_t j = 0; j < num_faces; j++) {
      std::shared_ptr<FaceExpressionScore> b =
          std::make_shared<FaceExpressionScore>();
      (void)memcpy_s(
          b.get(), data_size,
          (const char *)(input_bufs->ConstBufferData(i)) + (data_size * j),
          data_size);
      faces.push_back(b);
    }
  }

  std::vector<size_t> shape(1, faces.size() * data_size);
  auto ret = output_bufs->Build(shape);
  if (!ret) {
    auto errMsg = "build output failed in face collapse unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  auto output_data = (FaceExpressionScore *)(output_bufs->MutableBufferData(0));
  for (auto &face : faces) {
    (void)memcpy_s(output_data, data_size, face.get(), data_size);
    output_data++;
  }
  MBLOG_DEBUG << "collapse faces output size: " << output_bufs->Size();

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceCollapseFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : COLLAPSE_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }
  for (auto port : COLLAPSE_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
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