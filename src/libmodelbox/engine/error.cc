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

#include "modelbox/error.h"

#include <utility>

namespace modelbox {

FlowUnitError::FlowUnitError(std::string desc) { desc_ = std::move(desc); }

FlowUnitError::FlowUnitError(const std::string& node,
                             const std::string& error_pos,
                             const Status& error_status) {
  desc_ = "node:" + node + " error pos:" + error_pos +
          " status:" + error_status.StrCode() +
          " error:" + error_status.Errormsg();
  error_status_ = error_status;
}

FlowUnitError::~FlowUnitError() = default;

std::string FlowUnitError::GetDesc() { return desc_; };
Status FlowUnitError::GetStatus() { return error_status_; };

DataError::DataError(const std::string& error_code,
                     const std::string& error_msg) {
  error_code_ = error_code;
  error_msg_ = error_msg;
  new_error_ = true;
}

std::string DataError::GetErrorCode() { return error_code_; }

std::string DataError::GetErrorMsg() { return error_msg_; }

void DataError::SetErrorDeepth(size_t error_deepth) {
  // only new error need set deepth
  if (new_error_) {
    error_deepth_ = error_deepth;
    new_error_ = false;
  }
}

size_t DataError::GetErrorDeepth() { return error_deepth_; }

}  // namespace modelbox
