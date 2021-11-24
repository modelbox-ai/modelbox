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


#include "mock_cert.h"

#include "modelbox/base/crypto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {

Status GenerateCert(std::string* enPass, std::string* ekRootKey,
                    const std::string& private_key,
                    const std::string& public_key) {
  std::string pass_str = "password";
  std::string openssl_cmd =
      "openssl req -x509 -newkey rsa:4096 -passout pass:" + pass_str +
      " -keyout ";
  openssl_cmd += modelbox::PathCanonicalize(private_key) + " -out " +
                 modelbox::PathCanonicalize(public_key);
  openssl_cmd +=
      " -days 36500 -subj '/C=CN/ST=SZ/L=SZ/O=HW/OU=OU/CN=localhost'";

  std::vector<char> pass(pass_str.begin(), pass_str.end());
  auto ret = PassEncrypt(pass, true, ekRootKey, enPass);
  if (!ret) {
    return ret;
  }

  if (system(openssl_cmd.c_str()) != 0) {
    std::string errmsg = "run command failed, ";
    errmsg += strerror(errno);
    return {STATUS_FAULT, errmsg};
  }

  return STATUS_OK;
}

Status GenerateCert(const std::string& private_key,
                    const std::string& public_key) {
  std::string openssl_cmd = "openssl req -nodes -x509 -newkey rsa:4096 ";
  openssl_cmd += "-keyout " + modelbox::PathCanonicalize(private_key) + " -out " +
                 modelbox::PathCanonicalize(public_key);
  openssl_cmd +=
      " -days 36500 -subj '/C=CN/ST=SZ/L=SZ/O=HW/OU=OU/CN=localhost'";

  if (system(openssl_cmd.c_str()) != 0) {
    std::string errmsg = "run command failed, ";
    errmsg += strerror(errno);
    return {STATUS_FAULT, errmsg};
  }

  return STATUS_OK;
}

}  // namespace modelbox
