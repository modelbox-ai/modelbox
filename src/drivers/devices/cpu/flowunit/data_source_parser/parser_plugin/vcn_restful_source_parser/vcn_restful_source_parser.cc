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

#include "vcn_restful_source_parser.h"

modelbox::Status VcnRestfulSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  ReadConf(opts);

  ReadConfVcnCommon(opts, retry_enabled_, retry_interval_, retry_max_times_);

  keep_alive_interval_ =
      opts->GetInt32("vcn_keep_alive_interval_sec", keep_alive_interval_);
  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulSourceParser::Deinit() {
  return modelbox::STATUS_OK;
}

modelbox::Status VcnRestfulSourceParser::Parse(
    std::shared_ptr<modelbox::SessionContext> session_context,
    const std::string &config, std::string &uri,
    DestroyUriFunc &destroy_uri_func) {
  modelbox::VcnRestfulInfo vcn_info;
  uri = "";

  // read info from cfg
  auto ret = GetVcnInfo(vcn_info, config);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << "failed to get vcn info.";
    return ret;
  }

  auto vcn_client =
      modelbox::VcnRestfulClient::GetInstance(keep_alive_interval_);
  if (nullptr == vcn_client) {
    MBLOG_ERROR << "failed to get vcn restful client instance.";
    return modelbox::STATUS_FAULT;
  }

  std::shared_ptr<modelbox::VcnStreamRestful> stream;
  ret = vcn_client->AddVcnStream(vcn_info, stream);
  if (modelbox::STATUS_OK != ret) {
    MBLOG_ERROR << ret.Errormsg();
    return ret;
  }

  uri = stream->GetUrl();
  destroy_uri_func = [stream](const std::string &uri) {
    MBLOG_DEBUG << "destory " << uri;
  };

  return modelbox::STATUS_OK;
}