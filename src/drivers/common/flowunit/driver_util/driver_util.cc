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

#include "driver_util.h"

namespace driverutil {

std::string string_masking(const std::string &input) {
  std::regex url_auth_pattern("://[^ /]*?:[^ /]*?@");
  auto output = std::regex_replace(input, url_auth_pattern, "://*:*@");

  std::regex pattern_ak(R"("ak"[ ]*?:[ ]*?".*?")");
  output = std::regex_replace(output, pattern_ak, R"("ak":"*")");

  std::regex pattern_sk(R"("sk"[ ]*?:[ ]*?".*?")");
  output = std::regex_replace(output, pattern_sk, R"("sk":"*")");

  std::regex pattern_token(R"("securityToken"[ ]*?:[ ]*?".*?")");
  output = std::regex_replace(output, pattern_token, R"("securityToken":"*")");

  std::regex pattern_vcn_pwd(R"("vcn_stream_pwd"[ ]*?:[ ]*?".*?")");
  output =
      std::regex_replace(output, pattern_vcn_pwd, R"("vcn_stream_pwd":"*")");

  return std::move(output);
}

}  // namespace driverutil