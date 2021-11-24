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

modelbox::Status CollapseBBoxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto input_bufs = ctx->Input("input");
  auto output_bufs = ctx->Output("output");

  size_t data_size = sizeof(Person);
  std::vector<Person> Persons;
  for (size_t i = 0; i < input_bufs->Size(); i++) {
    Person person_data;
    auto input_ret =
        memcpy_s(&person_data, data_size,
                 (const char *)(input_bufs->ConstBufferData(i)), data_size);
    if (EOK != input_ret) {
      MBLOG_ERROR << "Cpu collapse_bbox failed, input_ret " << input_ret;
      return modelbox::STATUS_FAULT;
    }
    Persons.emplace_back(person_data);
  }

  for (auto &person : Persons) {
    float mold;
    float sum = 0.0;
    for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
      sum += person.emb[i] * person.emb[i];
    }

    mold = sqrt(sum);
    if (mold != 0) {
      for (int j = 0; j < EMBEDDING_LENGTH; ++j) {
        person.emb[j] /= mold;
      }
    }
  }

  std::vector<size_t> shape(1, Persons.size() * data_size);

  auto shape_ret = output_bufs->Build(shape);
  if (!shape_ret) {
    MBLOG_ERROR << "collapse_bbox: get output memory failed.";
    return modelbox::STATUS_NOMEM;
  }

  auto output_data = (Person *)(output_bufs->MutableBufferData(0));

  for (auto &person : Persons) {
    auto output_ret = memcpy_s(output_data, data_size, &person, data_size);
    if (EOK != output_ret) {
      MBLOG_ERROR << "Cpu collapse_bbox failed, output_ret " << output_ret;
      return modelbox::STATUS_FAULT;
    }
    output_data++;
  }

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