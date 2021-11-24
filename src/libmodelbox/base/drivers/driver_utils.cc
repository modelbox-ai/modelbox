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

#include "modelbox/base/driver_utils.h"

#include "modelbox/base/crypto.h"

namespace modelbox {

std::string GenerateKey(int64_t check_sum) {
  std::vector<unsigned char> output;
  auto status = HmacEncode("sha256", &check_sum, sizeof(uint64_t), &output);
  if (!status) {
    StatusError = status;
    return "";
  }

  return HmacToString(output.data(), output.size());
}

}  // namespace modelbox