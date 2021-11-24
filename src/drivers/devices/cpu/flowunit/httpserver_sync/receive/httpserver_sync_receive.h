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

#ifndef MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_RECEIVE_CPU_H_
#define MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_RECEIVE_CPU_H_

#include "cpprest/http_listener.h"
#include "http_util.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME_RECEIVE = "httpserver_sync_receive";
constexpr const char *FLOWUNIT_DESC_RECEIVE =
    "\n\t@Brief: Start a http/https server, output request info to next "
    "flowunit. \n"
    "\t@Port parameter: The output port buffer contain the following meta "
    "fields:\n"
    "\t\tField Name: size,        Type: size_t\n"
    "\t\tField Name: method,      Type: string\n"
    "\t\tField Name: uri,         Type: string\n"
    "\t\tField Name: headers,     Type: map<string,string>\n"
    "\t\tField Name: endpoint,    Type: string\n"
    "\t  The the output port buffer data type is char * .\n"
    "\t@Constraint: The flowuint 'httpserver_sync_receive' must be used pair with 'httpserver_sync_reply'.";
;

struct RequestInfo {
  web::http::method method;
  utility::string_t uri;
  std::map<std::string, std::string> headers_map;
  utility::string_t request_body;
};

class ReplyHandle {
 public:
  ReplyHandle(
      const std::function<
          void(uint16_t status, const concurrency::streams::istream &body_data,
               const utility::string_t &content_type)> &reply_func) {
    reply_func_ = reply_func;
  }

  virtual ~ReplyHandle() {}
  void Reply(uint16_t status, const concurrency::streams::istream &body_data,
             const utility::string_t &content_type) {
    reply_func_(status, body_data, content_type);
  }

 private:
  std::function<void(uint16_t status,
                     const concurrency::streams::istream &body_data,
                     const utility::string_t &content_type)>
      reply_func_;
};

class HTTPServerReceiveSync : public modelbox::FlowUnit {
 public:
  HTTPServerReceiveSync();
  virtual ~HTTPServerReceiveSync();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status HandleFunc(web::http::http_request request);

  modelbox::Status HandleTask(web::http::http_request request,
                              const RequestInfo &request_info);

 private:
  std::shared_ptr<std::atomic<uint64_t>> sum_cnt_ =
      std::make_shared<std::atomic<uint64_t>>(0);
  std::shared_ptr<web::http::experimental::listener::http_listener> listener_;
  std::string request_url_;
  uint64_t max_requests_{1000};
  uint64_t time_out_ms_{5000};
  std::mutex request_mutex_;
  modelbox::Timer timer_;
};

#endif  // MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_RECEIVE_CPU_H_