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


#include "httpserver_sync_reply.h"

#include <cpprest/containerstream.h>
#include <cpprest/rawptrstream.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit_api_helper.h"
#include "receive/httpserver_sync_receive.h"

HTTPServerReplySync::HTTPServerReplySync(){};
HTTPServerReplySync::~HTTPServerReplySync(){};

const static std::map<std::string, std::string> content_type_map_ = {
    {"htm", U("text/html")},
    {"html", U("text/html")},
    {"js", U("text/javascript")},
    {"css", U("text/css")},
    {"json", U("application/json")},
    {"png", U("image/png")},
    {"gif", U("image/gif")},
    {"jpeg", U("image/jpeg")},
    {"svg", U("image/svg+xml")},
    {"tar", U("application/x-tar")},
    {"txt", U("text/plain;charset=utf-8")},
    {"ico", U("application/octet-stream")},
    {"xml", U("text/xml")},
    {"mpeg", U("video/mpeg")},
    {"mp3", U("audio/mpeg")},
};

modelbox::Status HTTPServerReplySync::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  auto content_type = opts->GetString("content_type", "txt");
  auto iter = content_type_map_.find(content_type);
  if (iter == content_type_map_.end()) {
    auto err_msg = "unsupport content type " + content_type;
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_BADCONF, err_msg};
  }

  content_type_ = iter->second;
  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerReplySync::Close() { return modelbox::STATUS_OK; }

modelbox::Status HTTPServerReplySync::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto session_ctx = ctx->GetSessionContext();
  auto reply =
      std::static_pointer_cast<ReplyHandle>(session_ctx->GetPrivate("reply"));
  if (reply == nullptr) {
    auto err_msg = "http reply handler is nullptr.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  auto input_data = ctx->Input("in_reply_info")->At(0);
  if (input_data == nullptr) {
    auto err_msg = "http reply flowunit get input data failed.";
    MBLOG_ERROR << err_msg;
    reply->Reply(
        web::http::status_codes::InternalError,
        concurrency::streams::bytestream::open_istream<std::string>(err_msg),
        "text/plain;charset=utf-8");
    return {modelbox::STATUS_NOMEM, err_msg};
  }

  auto bytes = input_data->GetBytes();
  auto data = input_data->ConstData();
  std::string ss((char *)data, bytes);
  auto resp_body =  concurrency::streams::bytestream::open_istream<std::string>(ss);
  reply->Reply(web::http::status_codes::OK, resp_body, content_type_);

  return modelbox::STATUS_OK;
}


MODELBOX_FLOWUNIT(HTTPServerReplySync, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME_REPLY);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_reply_info", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::STREAM);
  desc.SetFlowUnitGroupType("Output");
  desc.SetDescription(FLOWUNIT_DESC_REPLY);
}
