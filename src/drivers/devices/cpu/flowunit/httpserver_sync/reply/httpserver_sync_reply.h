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

#ifndef MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_REPLY_CPU_H_
#define MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_REPLY_CPU_H_

#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME_REPLY = "httpserver_sync_reply";
constexpr const char *FLOWUNIT_DESC_REPLY =
    "\n\t@Brief: Send reply when receive a response info."
    "flowunit.\n"
    "\t@Port parameter: The input port buffer contain the following meta "
    "fields:\n"
    "\t\tField Name: status,        Type: int32_t\n"
    "\t\tField Name: headers,       Type: map<string,string>\n"
    "\t  The the input port buffer data type is char * .\n"
    "\t@Constraint: The flowuint 'httpserver_sync_reply' must be used pair "
    "with 'httpserver_sync_receive'.";
;

class HTTPServerReplySync : public modelbox::FlowUnit {
 public:
  HTTPServerReplySync();
  virtual ~HTTPServerReplySync();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  std::string content_type_;
};

#endif  // MODELBOX_FLOWUNIT_HTTPSERVER_SYNC_REPLY_CPU_H_