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

namespace modelbox {

FlowUnitError::FlowUnitError(std::string desc) { desc_ = desc; }

FlowUnitError::FlowUnitError(std::string node, std::string error_pos,
                             Status error_status) {
  desc_ = "node:" + node + " error pos:" + error_pos +
          " status:" + error_status.StrCode() +
          " error:" + error_status.Errormsg();
  error_status_ = error_status;
};

FlowUnitError::~FlowUnitError(){};

std::string FlowUnitError::GetDesc() { return desc_; };

}  // namespace modelbox
