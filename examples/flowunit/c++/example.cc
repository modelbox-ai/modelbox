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

#include "example.h"
#include "modelbox/flowunit_api_helper.h"

ExampleFlowUnit::ExampleFlowUnit(){};
ExampleFlowUnit::~ExampleFlowUnit(){};

modelbox::Status ExampleFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status ExampleFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_bufs = data_ctx->Input("in");
  auto output_bufs = data_ctx->Output("out");

  // Your code goes here
  //

  return modelbox::STATUS_OK;
}

modelbox::Status ExampleFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleFlowUnit::DataGroupPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleFlowUnit::DataGroupPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ExampleFlowUnit, desc) {
  /*set flowunit attributes*/
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Undefined");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  /*set flowunit parameter
  example code:
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "parameter0", "int", true, "640", "parameter0 describe detail"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "parameter1", "int", true, "480", "parameter1 describe detail"));
  */
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion(FLOWUNIT_VERSION);
}
