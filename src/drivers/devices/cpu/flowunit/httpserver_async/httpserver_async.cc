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

#include "httpserver_async.h"

#include <securec.h>

#include "modelbox/base/crypto.h"
#include "modelbox/flowunit_api_helper.h"

HTTPServerAsync::HTTPServerAsync() = default;
HTTPServerAsync::~HTTPServerAsync() = default;

modelbox::Status HTTPServerAsync::HandleFunc(web::http::http_request request) {
  if (request.request_uri().to_string() == "/health") {
    HandleHealthCheck(request);
    return modelbox::STATUS_OK;
  }

  RequestInfo request_info;
  request_info.method = request.method();
  request_info.uri = request.request_uri().to_string();
  for (auto &head : request.headers()) {
    request_info.headers_map[head.first] = head.second;
  }

  request.extract_string().then(
      [request, request_info,
       this](const pplx::task<utility::string_t> &t) mutable {
        try {
          request_info.request_body = t.get();
          auto handle_status = HandleTask(request, request_info);
          if (handle_status == modelbox::STATUS_BUSY) {
            SafeReply(request, web::http::status_codes::TooManyRequests);
          } else if (handle_status == modelbox::STATUS_FAULT ||
                     handle_status == modelbox::STATUS_NOMEM) {
            SafeReply(request, web::http::status_codes::InternalError);
          }
        } catch (const std::exception &e) {
          MBLOG_ERROR << "get request body error" << e.what();
          SafeReply(request, web::http::status_codes::BadRequest);
        }
      });

  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerAsync::HandleTask(
    const web::http::http_request &request, const RequestInfo &request_info) {
  auto http_limiter = HttpRequestLimiter::GetInstance();
  if (http_limiter == nullptr) {
    return modelbox::STATUS_BUSY;
  }

  auto size = request_info.request_body.size();
  std::vector<std::size_t> shape = {size};
  auto ext_data = this->CreateExternalData();
  if (!ext_data) {
    MBLOG_ERROR << "can not get external data.";
    return modelbox::STATUS_FAULT;
  }
  auto session_cxt = ext_data->GetSessionContext();
  session_cxt->SetPrivate("http_limiter_" + session_cxt->GetSessionId(),
                          http_limiter);
  auto output_buf = ext_data->CreateBufferList();
  output_buf->Build(shape);
  if (size > 0) {
    auto *outmem = output_buf->MutableBufferData(0);
    if (outmem == nullptr) {
      MBLOG_ERROR << "outmem buffer is nullptr.";
      return modelbox::STATUS_NOMEM;
    }

    auto ret = memcpy_s(outmem, size, request_info.request_body.data(), size);
    if (EOK != ret) {
      MBLOG_ERROR << "Cpu memcpy failed, ret " << ret << ", src size " << size
                  << ", dest size " << size;
      return modelbox::STATUS_FAULT;
    }
  }

  output_buf->At(0)->Set("size", size);
  output_buf->At(0)->Set("method", (std::string)request_info.method);
  output_buf->At(0)->Set("uri", (std::string)request_info.uri);
  output_buf->At(0)->Set("headers", request_info.headers_map);
  output_buf->At(0)->Set("endpoint", request_url_);
  output_buf->At(0)->SetGetBufferType(modelbox::BufferEnumType::STR);
  auto status = ext_data->Send(output_buf);
  if (!status) {
    MBLOG_ERROR << "external data send buffer list failed:" << status;
    return modelbox::STATUS_FAULT;
  }

  SafeReply(request, web::http::status_codes::Accepted);

  status = ext_data->Close();
  if (!status) {
    MBLOG_ERROR << "external data close failed:" << status;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerAsync::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  request_url_ = opts->GetString("endpoint", "");
  if (request_url_.empty()) {
    request_url_ = "http://127.0.0.1:8080";
    MBLOG_WARN << "endpoint not set, use default endpoint: " << request_url_;
  }

  HttpRequestLimiter::max_request_ = opts->GetUint64("max_requests", 1000);
  std::atomic_init(&HttpRequestLimiter::request_count_, (size_t)0);
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
  listener_->support(web::http::methods::POST,
                     [this](const web::http::http_request &request) {
                       this->HandleFunc(request);
                     });
  listener_->support(web::http::methods::PUT,
                     [this](const web::http::http_request &request) {
                       this->HandleFunc(request);
                     });
  listener_->support(web::http::methods::GET,
                     [this](const web::http::http_request &request) {
                       this->HandleFunc(request);
                     });
  listener_->support(web::http::methods::DEL,
                     [this](const web::http::http_request &request) {
                       this->HandleFunc(request);
                     });

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

modelbox::Status HTTPServerAsync::Close() {
  listener_->close().wait();
  return modelbox::STATUS_OK;
}

modelbox::Status HTTPServerAsync::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto output_buf = data_ctx->Output("out_request_info");
  auto input_buf = data_ctx->External();

  for (auto &buf : *input_buf) {
    output_buf->PushBack(buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(HTTPServerAsync, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitOutput({"out_request_info"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetFlowUnitGroupType("Input");
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("endpoint", "string", true,
                                                  "https://127.0.0.1:8080",
                                                  "http server listen URL."));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_requests", "integer", true, "1000", "max http request."));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("keepalive_timeout_sec", "integer", false, "200",
                               "keep-alive timeout time(sec)"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("cert", "string", false, "", "cert file path"));
  desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("key", "string", false, "", "key file path"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "passwd", "string", false, "", "encrypted key file password."));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "key_pass", "string", false, "", "key for encrypted password."));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
