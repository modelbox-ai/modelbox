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

#ifndef MODELBOX_CRYPTO_H_
#define MODELBOX_CRYPTO_H_

#include <modelbox/base/status.h>
#include <time.h>

#include <string>
#include <vector>

namespace modelbox {

/// default ciphter
constexpr const char *DEFAULT_CIPHER_AES256_CBC = "aes-256-cbc";
constexpr int MAX_PASSWORD_LEN = 1024;
constexpr int  IV_LEN = 16;

/**
 * @brief hmac encode
 * @param algorithm algorithm, support sha512, sha256, sha1, md5, sha224, sha384
 * @param input input data
 * @param output output data
 * @return whether success
 */
Status HmacEncode(const std::string &algorithm,
                  const std::vector<unsigned char> &input,
                  std::vector<unsigned char> *output);

/**
 * @brief hmac encode
 * @param algorithm algorithm, support sha512, sha256, sha1, md5, sha224, sha384
 * @param input input data pointer
 * @param input_len input data len
 * @param output output data
 * @return whether success
 */
Status HmacEncode(const std::string &algorithm, const void *input,
                  size_t input_len, std::vector<unsigned char> *output);

/**
 * @brief Conver Hmac to string
 * @param input input data pointer
 * @param input_len input data len
 * @return Hmac in string
 */
std::string HmacToString(const void *input, size_t input_len);

/**
 * @brief Encrypt password
 * @param pass password in plain text
 * @param sysrelated Whether encryption system related
 * @param rootkey output rootkey
 * @param en_pass encrypted password
 * @param ciphername ciphter name, like aes-256-cbc
 * @return whether success
 */
Status PassEncrypt(const std::vector<char> &pass, bool sysrelated,
                   std::string *rootkey, std::string *en_pass,
                   const std::string &ciphername = DEFAULT_CIPHER_AES256_CBC);

/**
 * @brief Decrypt password
 * @param en_pass encrypted password
 * @param rootkey rootkey
 * @param pass output password in plain text
 * @param ciphername ciphter name, like aes-256-cbc
 * @return whether success
 */
Status PassDecrypt(const std::string &en_pass, const std::string &rootkey,
                   std::vector<char> *pass,
                   const std::string &ciphername = DEFAULT_CIPHER_AES256_CBC);

/**
 * @brief Generic encrypt function
 * @param ciphername ciphter name, like aes-256-cbc
 * @param input input data
 * @param input_len input data len
 * @param output output data
 * @param output_len output len
 * @param max_output max output len
 * @param key encrypt key
 * @param iv encrypt iv
 * @return whether success
 */
Status Encrypt(const std::string &ciphername, unsigned char *input,
               int input_len, unsigned char *output, int *output_len,
               int max_output, unsigned char *key, unsigned char *iv);

/**
 * @brief Generic decrypt function
 * @param ciphername ciphter name, like aes-256-cbc
 * @param input input data
 * @param input_len input data len
 * @param output output data
 * @param output_len output len
 * @param max_output max output len
 * @param key encrypt key
 * @param iv encrypt iv
 * @return whether success
 */
Status Decrypt(const std::string &ciphername, unsigned char *input,
               int input_len, unsigned char *output, int *output_len,
               int max_output, unsigned char *key, unsigned char *iv);

/**
 * @brief Base64 encode
 * @param input input data
 * @param output encoded base64 string
 * @return whether success
 */
Status Base64Encode(const std::vector<unsigned char> &input,
                    std::string *output);

/**
 * @brief Base64 encode
 * @param input input data
 * @param input_len input data len
 * @param output encoded base64 string
 * @return whether success
 */
Status Base64Encode(const unsigned char *input, size_t input_len,
                    std::string *output);

/**
 * @brief Base64 decode
 * @param input encoded base64 string
 * @param output decoded data
 * @return whether success
 */
Status Base64Decode(const std::string &input,
                    std::vector<unsigned char> *output);

}  // namespace modelbox

#endif
