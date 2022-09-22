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

#include "data_source_parser_flowunit.h"

#include <securec.h>

#include "driver_util.h"
#include "modelbox/base/config.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status DataSourceParserFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto dev_mgr = GetBindDevice()->GetDeviceManager();
  if (dev_mgr == nullptr) {
    MBLOG_ERROR << "Can not get device manger";
    return modelbox::STATUS_FAULT;
  }

  auto drivers = dev_mgr->GetDrivers();
  if (drivers == nullptr) {
    MBLOG_ERROR << "Can not get drivers";
    return modelbox::STATUS_FAULT;
  }

  auto ret = driverutil::GetPlugin<modelbox::DataSourceParserPlugin>(
      DRIVER_CLASS_DATA_SOURCE_PARSER_PLUGIN, drivers, factories_, plugins_);
  if (!ret) {
    return ret;
  }

  for (auto &item : plugins_) {
    auto ret = item.second->Init(opts);
    if (!ret) {
      MBLOG_ERROR << "Init plugin " << item.first
                  << " failed, detail : " << ret.Errormsg();
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DataSourceParserFlowUnit::Close() {
  for (auto &item : plugins_) {
    item.second->Deinit();
  }

  plugins_.clear();

  return modelbox::STATUS_OK;
}

modelbox::Status DataSourceParserFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto session_ctx = data_ctx->GetSessionContext();
  if (!session_ctx) {
    MBLOG_ERROR << "Session data_ctx is null";
    return modelbox::STATUS_FAULT;
  }

  auto input_buffer_list = data_ctx->Input(INPUT_DATA_SOURCE_CFG);
  std::string source_type;
  std::vector<std::string> uri_list;
  if (input_buffer_list->Size() != 1) {
    MBLOG_ERROR << "Only support one data source config";
    return modelbox::STATUS_FAULT;
  }

  auto buffer = input_buffer_list->At(0);
  const auto *inbuff_data = (const char *)buffer->ConstData();
  if (inbuff_data == nullptr) {
    return {modelbox::STATUS_INVALID, "input buffer is invalid."};
  }

  buffer->Get(INPUT_META_SOURCE_TYPE, source_type);
  MBLOG_INFO << "Try parse input config " << source_type << " for "
             << session_ctx->GetSessionId();
  std::string data_source_cfg(inbuff_data, buffer->GetBytes());
  std::shared_ptr<std::string> uri;
  auto session_config = data_ctx->GetSessionConfig();
  std::shared_ptr<modelbox::SourceContext> source_context =
      Parse(session_ctx, session_config, source_type, data_source_cfg, uri);
  if (source_context) {
    source_context->SetDataSourceCfg(data_source_cfg);
  } else {
    MBLOG_ERROR << "Parse data source " << source_type << " failed";
  }

  auto ret =
      WriteData(data_ctx, uri, source_type, data_source_cfg, source_context);
  if (!ret) {
    return ret;
  }

  MBLOG_INFO << "parse input config ok for " << session_ctx->GetSessionId();
  return modelbox::STATUS_OK;
}

std::shared_ptr<modelbox::SourceContext> DataSourceParserFlowUnit::Parse(
    const std::shared_ptr<modelbox::SessionContext> &session_context,
    const std::shared_ptr<modelbox::Configuration> &session_config,
    const std::string &source_type, const std::string &data_source_cfg,
    std::shared_ptr<std::string> &uri) {
  auto plugin = GetPlugin(source_type);
  if (plugin == nullptr) {
    MBLOG_ERROR << "Can not find data source parse plugin for : " << source_type
                << ", please check whether plugin loaded";
    return nullptr;
  }

  std::string uri_str;
  modelbox::DestroyUriFunc destroy_uri_func;
  std::string stream_type;
  auto ret = plugin->Parse(session_context, session_config, data_source_cfg,
                           uri_str, destroy_uri_func);
  if (!ret) {
    MBLOG_ERROR << "Parse config failed, source uri is empty";
  }

  std::shared_ptr<modelbox::SourceContext> source_context =
      std::make_shared<modelbox::SourceContext>(plugin, source_type);
  source_context->SetRetryParam(plugin->GetRetryEnabled(),
                                plugin->GetRetryInterval(),
                                plugin->GetRetryTimes());
  plugin->GetStreamType(data_source_cfg, stream_type);
  source_context->SetStreamType(stream_type);
  source_context->SetSessionContext(session_context);
  source_context->SetSessionConfig(session_config);

  uri = std::shared_ptr<std::string>(new std::string(uri_str),
                                     [destroy_uri_func](std::string *ptr) {
                                       if (destroy_uri_func) {
                                         destroy_uri_func(*ptr);
                                       }
                                       delete ptr;
                                     });
  return source_context;
}

std::shared_ptr<modelbox::DataSourceParserPlugin>
DataSourceParserFlowUnit::GetPlugin(const std::string &source_type) {
  auto item = plugins_.find(source_type);
  if (item == plugins_.end()) {
    return nullptr;
  }

  return item->second;
}

modelbox::Status DataSourceParserFlowUnit::WriteData(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::shared_ptr<std::string> &uri, const std::string &source_type,
    const std::string &data_source_cfg,
    std::shared_ptr<modelbox::SourceContext> &source_context) {
  auto input_buffer_list = data_ctx->Input(INPUT_DATA_SOURCE_CFG);
  auto buffer = input_buffer_list->At(0);
  auto data_meta = std::make_shared<modelbox::DataMeta>();
  data_meta->SetMeta(STREAM_META_SOURCE_URL, uri);
  data_meta->SetMeta(PARSER_RETRY_CONTEXT, source_context);
  data_ctx->SetOutputMeta(OUTPUT_STREAM_META, data_meta);

  auto buffer_list = data_ctx->Output(OUTPUT_STREAM_META);
  buffer_list->Build({1});
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(DataSourceParserFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Input");
  desc.AddFlowUnitInput({INPUT_DATA_SOURCE_CFG});
  desc.AddFlowUnitOutput({OUTPUT_STREAM_META});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetStreamSameCount(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "retry_enable", "bool", false, "false", "enable source parser retry"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("retry_interval_ms", "int", false, "1000",
                               "the source parser retry interval in ms"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("retry_count_limit", "int", false, "-1",
                               "the source parser retry count limit"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
