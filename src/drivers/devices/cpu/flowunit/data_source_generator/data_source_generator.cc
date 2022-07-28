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

#include "data_source_generator.h"

#include <securec.h>

#include <sstream>
#include <unordered_set>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

DataSourceGeneratorFlowUnit::DataSourceGeneratorFlowUnit(){};
DataSourceGeneratorFlowUnit::~DataSourceGeneratorFlowUnit(){};

static std::unordered_set<std::string> g_predefined_keys{"type",
                                                         "flowunit",
                                                         "device",
                                                         "deviceid",
                                                         "label",
                                                         "batch_size",
                                                         "queue_size",
                                                         "queue_size_event",
                                                         "queue_size_external",
                                                         "source_type"};

modelbox::Status DataSourceGeneratorFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto source_type = opts->GetString("source_type");
  auto all_config_keys = opts->GetKeys();
  std::stringstream ss;
  ss << "{";
  std::unordered_set<std::string> output_keys;
  for (const auto &key : all_config_keys) {
    if (g_predefined_keys.find(key) != g_predefined_keys.end()) {
      continue;
    }

    output_keys.insert(key);
    ss << "\"" << key << "\":\"" << opts->GetString(key) << "\",";
  }
  ss.seekp(-1, std::stringstream::end);
  ss << "}";

  if (source_type.empty() || output_keys.empty()) {
    MBLOG_ERROR << "source_type and source config must be set in config";
    return modelbox::STATUS_BADCONF;
  }

  auto source_config = ss.str();
  MBLOG_INFO << "source type is : " << source_type;
  MBLOG_INFO << "source config is : " << source_config;

  auto ext_data = CreateExternalData();
  if (!ext_data) {
    MBLOG_ERROR << "can not get external data.";
  }

  auto output_buffers = ext_data->CreateBufferList();
  output_buffers->BuildFromHost({source_config.size()},
                                (void *)source_config.data(),
                                source_config.size());
  auto buffer = output_buffers->At(0);
  buffer->Set("source_type", source_type);

  auto status = ext_data->Send(output_buffers);
  if (!status) {
    MBLOG_ERROR << "external data send buffer list failed:" << status;
  }

  status = ext_data->Close();
  if (!status) {
    MBLOG_ERROR << "external data close failed:" << status;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DataSourceGeneratorFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status DataSourceGeneratorFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto output_buffers = data_ctx->Output("out_data");
  auto input_buffers = data_ctx->External();
  for (auto &buffer : *input_buffers) {
    output_buffers->PushBack(buffer);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(DataSourceGeneratorFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Input");
  desc.AddFlowUnitOutput({"out_data"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
