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


#ifndef MODELBOX_BASE64_SIMD_H
#define MODELBOX_BASE64_SIMD_H

#include <modelbox/base/status.h>

namespace modelbox {

/**
 * @brief base64 endoce by SIMD
 * @param input input data
 * @param input_len input data len
 * @param output encode base64 string
 * @return wheter success
 */
Status Base64EncodeSIMD(const uint8_t *input, size_t input_len,
                        std::string *output);
}  // namespace modelbox

#endif  // MODELBOX_BASE64_SIMD_H