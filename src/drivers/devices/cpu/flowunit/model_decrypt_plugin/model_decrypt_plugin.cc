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
#include "model_decrypt_plugin.h"

#include <sys/mman.h>

#include "modelbox/base/crypto.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"

using namespace modelbox;

Status ModelDecryptPlugin::Init(
    const std::string &fname,
    const std::shared_ptr<modelbox::Configuration> config) {
  rootkey_ = config->GetString("encryption.rootkey");
  en_pass_ = config->GetString("encryption.passwd");
  if (rootkey_.empty() || en_pass_.empty()) {
    MBLOG_ERROR << "passwd is empty";
    return STATUS_FAULT;
  }
  return STATUS_SUCCESS;
}

Status ModelDecryptPlugin::ModelDecrypt(uint8_t *raw_buf, int64_t raw_len,
                                        uint8_t *plain_buf,
                                        int64_t &plain_len) {
  std::vector<char> pass;
  auto ret = PassDecrypt(en_pass_, rootkey_, &pass, DEFAULT_CIPHER_AES256_CBC);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "decrypt passwd err:" << ret;
    return ret;
  }

  std::vector<unsigned char> iv;
  iv.resize(IV_LEN + MAX_PASSWORD_LEN);
  Base64Decode(en_pass_, &iv);

  int out_len;
  ret = Decrypt(DEFAULT_CIPHER_AES256_CBC, raw_buf, raw_len, plain_buf,
                &out_len, plain_len, (unsigned char *)pass.data(), iv.data());
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "decrypt model err:" << ret;
    return ret;
  }

  plain_len = out_len;
  return STATUS_SUCCESS;
}
