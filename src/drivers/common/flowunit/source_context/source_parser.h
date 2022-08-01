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

#ifndef MODELBOX_SOURCE_PARSER_H_
#define MODELBOX_SOURCE_PARSER_H_

#include <modelbox/base/config.h>
#include <modelbox/base/device.h>
#include <modelbox/base/log.h>

#include <iostream>

using DestroyUriFunc = std::function<void(const std::string &uri)>;

namespace modelbox {

enum RetryStatus { RETRY_NONEED = 0, RETRY_NEED = 1, RETRY_STOP = 2 };

class SourceParser {
 public:
  virtual modelbox::Status Parse(
      std::shared_ptr<modelbox::SessionContext> session_context,
      const std::string &config, std::string &uri,
      DestroyUriFunc &destroy_uri_func) = 0;
  virtual RetryStatus NeedRetry(std::string &stream_type,
                                modelbox::Status &last_status,
                                int32_t retry_times) = 0;
  virtual ~SourceParser() = default;
};
}  // namespace modelbox

#endif