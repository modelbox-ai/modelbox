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

#include <string>

modelbox::Status ExampleSourceParser::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleSourceParser::Deinit() { return modelbox::STATUS_OK; }

modelbox::Status ExampleSourceParser::Parse(
    const std::shared_ptr<modelbox::SessionContext> &session_context,
    const std::string &config, std::string &uri,
    modelbox::DestroyUriFunc &destroy_uri_func) {
  // Your code goes here

  return modelbox::STATUS_OK;
}

modelbox::Status ExampleSourceParser::GetStreamType(const std::string &config,
                                                    std::string &stream_type) {
  stream_type = "file";  // "file" or  "stream"

  return modelbox::STATUS_OK;
}