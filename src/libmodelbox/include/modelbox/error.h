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

#ifndef MODELBOX_ERROR_H_
#define MODELBOX_ERROR_H_

#include <string>

#include "modelbox/base/status.h"

namespace modelbox {

class FlowUnitError {
 public:
  FlowUnitError(std::string desc);
  FlowUnitError(const std::string& node, const std::string& error_pos,
                const Status& error_status);
  virtual ~FlowUnitError();
  std::string GetDesc();
  Status GetStatus();

 private:
  std::string desc_;
  Status error_status_;
};

class DataError {
 public:
  DataError(const std::string &error_code, const std::string &error_msg);

  virtual ~DataError();

  std::string GetErrorCode();

  std::string GetErrorMsg();

  size_t GetErrorDeepth();

  void SetErrorDeepth(size_t error_deepth);

 private:
  bool new_error_{false};

  std::string error_msg_;

  std::string error_code_;

  size_t error_deepth_{0};
};

}  // namespace modelbox

#endif  // MODELBOX_ERROR_H_