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

#include "httpserver_sync_receive.h"

#include <securec.h>

#include "modelbox/base/crypto.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit_api_helper.h"

HTTPServerReceiveSync::HTTPServerReceiveSync(){};
HTTPServerReceiveSync::~HTTPServerReceiveSync(){};

modelbox::Status HTTPServerReceiveSync::HandleFunc(
    web::http::http_request request) {
  if (request.request_uri().to_string() == "/health") {
    HandleHealthCheck(request);
    return modelbox::STATUS_OK;
  }

  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (*sum_cnt_ > max_requests_) {
      SafeReply(request, web::http::status_codes::TooManyRequests);
      return modelbox::STATUS_BUSY;
    }

    ++*sum_cnt_;
  }

  RequestInfo request_info;
  request_info.method = request.method();
  request_info.uri = request.request_uri().to_string();
  for (auto &head : request.headers()) {
    request_info.headers_map[head.first] = head.second;
  }
  request.extract_string().then(
      [this, request_info, request](pplx::task<utility::string_t> t) mutable {
        try {
          request_info.request_body = t.get();
          HandleTask(request, request_info);
        } catch (const std::exception &e) {
          MBLOG_ERROR << "get request body error" << e.what();
          SafeReply(request, web::http::status_codes::BadRequest);
          --*sum_cnt_;
        }
      });

  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerReceiveSync::HandleTask(
    web::http::http_request request, const RequestInfo &request_info) {
  auto return_ret = modelbox::STATUS_OK;
  Defer {
    if (return_ret != modelbox::STATUS_OK) {
      SafeReply(request, web::http::status_codes::InternalError);
      --*sum_cnt_;
    }
  };
  auto ext_data = this->CreateExternalData();
  if (!ext_data) {
    MBLOG_ERROR << "can not get external data.";
    return_ret = modelbox::STATUS_FAULT;
    return return_ret;
  }

  auto output_buf = ext_data->CreateBufferList();
  if (output_buf == nullptr) {
    MBLOG_ERROR << "Create buffer list failed.";
    return_ret = modelbox::STATUS_NOMEM;
    return return_ret;
  }

  auto size = request_info.request_body.size();
  std::vector<std::size_t> shape = {size};
  output_buf->Build(shape);
  if (size > 0) {
    auto outmem = output_buf->MutableBufferData(0);
    if (outmem == nullptr) {
      MBLOG_ERROR << "outmem buffer is nullptr.";
      return_ret = modelbox::STATUS_NOMEM;
      return return_ret;
    }

    auto ret = memcpy_s(outmem, size, request_info.request_body.data(), size);
    if (EOK != ret) {
      MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size " << size
                  << ", dest size " << size;
      return_ret = modelbox::STATUS_FAULT;
      return return_ret;
    }
  }

  output_buf->At(0)->Set("size", size);
  output_buf->At(0)->Set("method", (std::string)request_info.method);
  output_buf->At(0)->Set("uri", (std::string)request_info.uri);
  output_buf->At(0)->Set("headers", request_info.headers_map);
  output_buf->At(0)->Set("endpoint", request_url_);
  output_buf->At(0)->SetGetBufferType(modelbox::BufferEnumType::STR);

  auto replied = std::make_shared<std::atomic_bool>(false);
  auto timeout_task = std::make_shared<modelbox::TimerTask>(
      [](web::http::http_request request,
         std::shared_ptr<std::atomic_bool> replied,
         std::shared_ptr<std::atomic<uint64_t>> sum_cnt_) {
        auto replied_before = replied->exchange(true);
        if (!replied_before) {
          SafeReply(request, web::http::status_codes::RequestTimeout);
          --*sum_cnt_;
        }
      },
      request, replied, this->sum_cnt_);

  auto reply = std::make_shared<ReplyHandle>(
      [request, replied, timeout_task, this](
          uint16_t status, const concurrency::streams::istream &body_data,
          const utility::string_t &content_type) mutable {
        auto replied_before = replied->exchange(true);
        if (replied_before) {
          return;
        }

        SafeReply(request, status, body_data, content_type);
        timeout_task->Stop();
        --*(this->sum_cnt_);
      });
  auto session_ctx = ext_data->GetSessionContext();
  session_ctx->SetPrivate("reply", reply);

  auto status = ext_data->Send(output_buf);
  if (!status) {
    MBLOG_ERROR << "external data send buffer list failed:" << status;
    return_ret = modelbox::STATUS_FAULT;
    return return_ret;
  }

  timer_.Schedule(timeout_task, time_out_ms_, 0, false);
  status = ext_data->Close();
  if (!status) {
    MBLOG_ERROR << "external data close failed:" << status;
  }

  return return_ret;
}

modelbox::Status HTTPServerReceiveSync::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  timer_.Start();
  request_url_ = opts->GetString("endpoint", "");
  if (request_url_.empty()) {
    request_url_ = "http://127.0.0.1:8080";
    MBLOG_WARN << "endpoint not set, use default endpoint: " << request_url_;
  }
  max_requests_ = opts->GetUint64("max_requests", 1000);
  time_out_ms_ = opts->GetUint64("time_out_ms", 5000);
  keep_alive_time_out_sec_ = opts->GetUint64("keepalive_timeout_sec", 200);
  std::string key;
  std::string enpass;
  std::string keypass;
  const std::string cert = opts->GetString("cert", "");
  if (cert.length() > 0) {
    if (access(cert.c_str(), R_OK) != 0) {
      return {modelbox::STATUS_BADCONF, "certificate file is invalid."};
    }
    key = opts->GetString("key", "");
    if (access(key.c_str(), R_OK) != 0) {
      return {modelbox::STATUS_BADCONF, "key file is invalid."};
    }
    enpass = opts->GetString("passwd", "");
    if (enpass.empty()) {
      MBLOG_ERROR << "password not set";
      return {modelbox::STATUS_BADCONF, "password not set"};
    }
    keypass = opts->GetString("key_pass", "");
    if (keypass.empty()) {
      MBLOG_ERROR << "password key not set";
      return {modelbox::STATUS_BADCONF, "password key not set"};
    }
  }

  web::http::experimental::listener::http_listener_config server_config;
  server_config.set_timeout(std::chrono::seconds(keep_alive_time_out_sec_));
  if (cert.length() > 0 && key.length() > 0) {
    server_config.set_ssl_context_callback(
        [cert, key, enpass, keypass](boost::asio::ssl::context &ctx) {
          ctx.set_options(boost::asio::ssl::context::default_workarounds);
          modelbox::HardeningSSL(ctx.native_handle());
          ctx.native_handle();
          if (enpass.length() > 0) {
            ctx.set_password_callback(
                [enpass, keypass](
                    std::size_t max_length,
                    boost::asio::ssl::context::password_purpose purpose)
                    -> std::string {
                  std::vector<char> pass;
                  auto ret = modelbox::PassDecrypt(enpass, keypass, &pass);
                  if (!ret) {
                    MBLOG_ERROR << "key password is invalid";
                    return "";
                  }
                  std::string res;
                  res.insert(res.begin(), pass.begin(), pass.end());
                  return res;
                });
          }
          ctx.use_certificate_file(
              cert, boost::asio::ssl::context_base::file_format::pem);
          ctx.use_private_key_file(key, boost::asio::ssl::context::pem);
        });
  }
  listener_ =
      std::make_shared<web::http::experimental::listener::http_listener>(
          request_url_, server_config);
  listener_->support(
      web::http::methods::POST,
      [this](web::http::http_request request) { this->HandleFunc(request); });
  listener_->support(
      web::http::methods::PUT,
      [this](web::http::http_request request) { this->HandleFunc(request); });
  listener_->support(
      web::http::methods::GET,
      [this](web::http::http_request request) { this->HandleFunc(request); });
  listener_->support(
      web::http::methods::DEL,
      [this](web::http::http_request request) { this->HandleFunc(request); });

  listener_->support(web::http::methods::TRCE, HandleUnSupportMethod);
  listener_->support(web::http::methods::OPTIONS, HandleUnSupportMethod);
  try {
    listener_->open().wait();
    MBLOG_INFO << "start to listen : " << request_url_;
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return {modelbox::STATUS_FAULT, e.what()};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerReceiveSync::Close() {
  timer_.Stop();
  listener_->close().wait();
  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerReceiveSync::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto output_buf = ctx->Output("out_request_info");
  auto input_buf = ctx->External();

  for (auto &buf : *input_buf) {
    output_buf->PushBack(buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(HTTPServerReceiveSync, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME_RECEIVE);
  desc.AddFlowUnitOutput({"out_request_info"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetFlowUnitGroupType("Input");
  desc.SetDescription(FLOWUNIT_DESC_RECEIVE);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("endpoint", "string", true,
                                                  "https://127.0.0.1:8080",
                                                  "http server listen URL."));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_requests", "integer", false, "1000", "max http request."));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("keepalive_timeout_sec", "integer", false, "200",
                               "keep-alive timeout time(sec)"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("time_out", "integer", false, "100",
                               "max http request timeout. measured in 100ms"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("cert", "string", false, "", "cert file path"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("key", "string", false, "", "key file path"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "passwd", "string", false, "", "encrypted key file password."));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "key_pass", "string", false, "", "key for encrypted password."));
}
