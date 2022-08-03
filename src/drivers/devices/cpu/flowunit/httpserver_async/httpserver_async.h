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

#ifndef MODELBOX_FLOWUNIT_HTTPSERVER_ASYNC_CPU_H_
#define MODELBOX_FLOWUNIT_HTTPSERVER_ASYNC_CPU_H_

#include "cpprest/http_listener.h"
#include "http_util.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "httpserver_async";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Start a http/https server, reply to the response immediately "
    "when a request is received, and output request info to next flowunit. \n"
    "\t@Port parameter: The output port buffer contain the following meta "
    "fields:\n"
    "\t\tField Name: size,        Type: size_t\n"
    "\t\tField Name: method,      Type: string\n"
    "\t\tField Name: uri,         Type: string\n"
    "\t\tField Name: headers,     Type: map<string,string>\n"
    "\t\tField Name: endpoint,    Type: string\n"
    "\t  The the output port buffer data type is char * .\n"
    "\t@Constraint: ";

struct RequestInfo {
  web::http::method method;
  utility::string_t uri;
  std::map<std::string, std::string> headers_map;
  utility::string_t request_body;
};

class HTTPServerAsync : public modelbox::FlowUnit {
 public:
  HTTPServerAsync();
  ~HTTPServerAsync() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  modelbox::Status HandleFunc(web::http::http_request request);

  modelbox::Status HandleTask(const web::http::http_request &request,
                              const RequestInfo &request_info);

  std::shared_ptr<web::http::experimental::listener::http_listener> listener_;
  uint64_t keep_alive_time_out_sec_{200};
  std::string request_url_;
};

#endif  // MODELBOX_FLOWUNIT_HTTPSERVER_ASYNC_CPU_H_