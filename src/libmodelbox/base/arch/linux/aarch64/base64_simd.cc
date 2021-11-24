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


#include <modelbox/base/base64_simd.h>

#include <arm_neon.h>

namespace modelbox {

Status Base64EncodeSIMD(const uint8_t *input, size_t input_len,
                        std::string *output) {
  return {STATUS_NOTFOUND, "To be implemented linux arrch64 simd."};
}

}  // namespace modelbox