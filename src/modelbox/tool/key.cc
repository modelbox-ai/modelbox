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

#include "key.h"

#include <errno.h>
#include <getopt.h>
#include <modelbox/base/crypto.h>
#include <modelbox/base/utils.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>

namespace modelbox {

REG_MODELBOX_TOOL_COMMAND(ToolCommandKey)

enum MODELBOX_TOOL_KEY_COMMAND {
  MODELBOX_TOOL_KEY_PASS,
  MODELBOX_TOOL_KEY_MODEL,
};

static struct option key_options[] = {
    {"pass", 0, 0, MODELBOX_TOOL_KEY_PASS},
    {"model", 1, 0, MODELBOX_TOOL_KEY_MODEL},
    {0, 0, 0, 0},
};

enum MODELBOX_TOOL_KEY_PASS_COMMAND {
  MODELBOX_TOOL_KEY_PASS_NON_SYSRELATED,
};

static struct option key_pass_option[] = {
    {"n", 0, 0, MODELBOX_TOOL_KEY_PASS_NON_SYSRELATED},
    {0, 0, 0, 0},
};

constexpr int ASCII_ETX = 0x3;
constexpr int ASCII_BACKSPACE = 127;
constexpr int ASCII_DEL = 126;
constexpr int AES256_KEY_LEN = 32;
constexpr int ENCRYPT_BLOCK_SIZE = (AES256_KEY_LEN * 256);

ToolCommandKey::ToolCommandKey() {}
ToolCommandKey::~ToolCommandKey() {}

std::string ToolCommandKey::GetHelp() {
  char help[] =
      " action:\n"
      "   -pass     Encrypt password, the password can be environment\n"
      "variables 'MODELBOX_PASSWORD' or read from stdin\n"
      "   -model [model file]          Encrypt model or file, the password can "
      "be environment\n"
      "variables 'MODELBOX_PASSWORD' or read from stdin\n"
      "Important! Model Encryption may be unsafe if rootkey and en_pass are "
      "exposured\n"
      "   -n       None system related password\n"
      "\n";
  return help;
}

Status OpenFile(const std::string &plain_path, const std::string &encypt_path,
                std::ifstream &fplain, std::ofstream &fencypt) {
  fplain.open(plain_path, std::ios::binary);
  if (fplain.fail() || !fplain.is_open()) {
    std::cout << "open model '" << plain_path << "' failed, "
              << modelbox::StrError(errno);
    return STATUS_FAULT;
  }

  fencypt.open(encypt_path, std::ios::binary);
  if (fencypt.fail() || !fencypt.is_open()) {
    std::cout << "write en_model '" << encypt_path << " failed, "
              << modelbox::StrError(errno);
    return STATUS_FAULT;
  }
  return STATUS_SUCCESS;
}

Status EncryptWithFile(const std::string &plain_path,
                       const std::string &encypt_path, unsigned char *key,
                       unsigned char *iv) {
  std::ifstream fplain;
  std::ofstream fencypt;
  auto ret = OpenFile(plain_path, encypt_path, fplain, fencypt);
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  Defer {
    fplain.close();
    fencypt.close();
  };

  std::shared_ptr<uint8_t> read_buf(new (std::nothrow)
                                        uint8_t[ENCRYPT_BLOCK_SIZE],
                                    [](uint8_t *p) { delete[] p; });
  std::shared_ptr<uint8_t> en_buf(
      new (std::nothrow) uint8_t[ENCRYPT_BLOCK_SIZE + EVP_MAX_BLOCK_LENGTH + 1],
      [](uint8_t *p) { delete[] p; });

  if (en_buf.get() == nullptr || read_buf.get() == nullptr) {
    return {STATUS_NOMEM, "no memory to encode"};
  }

  std::shared_ptr<EVP_CIPHER_CTX> ctx;
  const EVP_CIPHER *cipher = nullptr;
  EVP_CIPHER_CTX *ctx_new = nullptr;
  int len;

  if (read_buf == nullptr) {
    return {STATUS_FAULT, "read_buf new err"};
  }

  cipher = EVP_get_cipherbyname(DEFAULT_CIPHER_AES256_CBC);
  if (cipher == nullptr) {
    return {STATUS_NOTSUPPORT, "cipher not support aes256_cbc"};
  }

  /* Create and initialise the context */
  ctx_new = EVP_CIPHER_CTX_new();
  if (ctx_new == nullptr) {
    return {STATUS_NOMEM, "create cipher failed."};
  }

  ctx.reset(ctx_new, [](EVP_CIPHER_CTX *ctx) { EVP_CIPHER_CTX_free(ctx); });

  /* Initialise the encryption operation. IMPORTANT - ensure you use a key
   * and IV size appropriate for your cipher
   * In this example we are using 256 bit AES (i.e. a 256 bit key). The
   * IV size for *most* modes is the same as the block size. For AES this
   * is 128 bits */
  if (1 != EVP_EncryptInit_ex(ctx.get(), cipher, NULL, key, iv)) {
    return {STATUS_FAULT, "encrypt init failed."};
  }

  /* Provide the message to be encrypted, and obtain the encrypted output.
   * EVP_EncryptUpdate can be called multiple times if necessary
   */
  while (!fplain.eof()) {
    fplain.read((char *)read_buf.get(), ENCRYPT_BLOCK_SIZE);
    int read_len = fplain.gcount();
    if (read_len != ENCRYPT_BLOCK_SIZE && !fplain.eof()) {
      return {STATUS_FAULT, "Read file fail."};
    }
    if (1 != EVP_EncryptUpdate(ctx.get(), en_buf.get(), &len, read_buf.get(),
                               read_len)) {
      return {STATUS_FAULT, "encrypt update failed."};
    }
    fencypt.write((char *)en_buf.get(), len);
  }

  /* Finalise the encryption. Further ciphertext bytes may be written at
   * this stage.
   */
  uint8_t *pend = en_buf.get() + len;
  if (1 != EVP_EncryptFinal_ex(ctx.get(), pend, &len)) {
    return {STATUS_FAULT, "encrypt final failed."};
  }
  fencypt.write((char *)pend, len);

  return STATUS_OK;
}

/**
 * @brief Encrypt model
 * @param model_path model path
 * @param pass password in plain text
 * @param sysrelated Whether encryption system related
 * @param rootkey output rootkey
 * @param en_pass encrypted password
 * @param ciphername ciphter name, like aes-256-cbc
 * @return whether success
 */
Status ModelEncrypt(const std::string &model_path,
                    const std::vector<char> &pass, bool sysrelated,
                    std::string *rootkey, std::string *en_pass) {
  std::vector<char> aes256_pass(pass);
  // fill key with "0"
  aes256_pass.resize(AES256_KEY_LEN, '\0');
  auto ret = PassEncrypt(aes256_pass, sysrelated, rootkey, en_pass,
                         DEFAULT_CIPHER_AES256_CBC);
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  std::vector<unsigned char> iv;
  iv.resize(IV_LEN + MAX_PASSWORD_LEN);
  Base64Decode(*en_pass, &iv);

  ret = EncryptWithFile(model_path, model_path + ".en",
                        (unsigned char *)aes256_pass.data(), iv.data());
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  return STATUS_SUCCESS;
}

int ToolCommandKey::Run(int argc, char *argv[]) {
  int cmdtype = 0;
  std::string fname("");
#if OPENSSL_API_COMPAT < 0x10100000L
  OpenSSL_add_all_algorithms();
  Defer { EVP_cleanup(); };
#endif

  if (argc == 1) {
    std::cerr << GetHelp();
    return 1;
  }

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, key_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_KEY_MODEL:
      fname = optarg;
    case MODELBOX_TOOL_KEY_PASS:
      optind = 1;
      MODELBOX_COMMAND_SUB_UNLOCK();
      return RunPassCommand(MODELBOX_COMMAND_SUB_ARGC,
                            MODELBOX_COMMAND_SUB_ARGV, fname);
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  return 0;
}

Status ToolCommandKey::ReadPassword(std::string *pass) {
  struct termios oldt, newt;
  char ch;
  int num = 0;
  char c_pass[MAX_PASSWORD_LEN];

  if (isatty(STDIN_FILENO) == 0) {
    std::cin >> *pass;
    return STATUS_OK;
  }

  std::cout << "Please input password: ";
  if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
    return {STATUS_FAULT, modelbox::StrError(errno)};
  }
  Defer { tcsetattr(STDIN_FILENO, TCSANOW, &oldt); };

  newt = oldt;
  newt.c_lflag &= ~(ECHO | ICANON | ISIG);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  while (((ch = getchar()) != '\n') && (num < MAX_PASSWORD_LEN - 1)) {
    if (ch == ASCII_ETX) {
      std::cout << std::endl;
      return {STATUS_STOP};
    } else if (ch == ASCII_BACKSPACE || ch == ASCII_DEL) {
      if (num > 0) {
        num--;
      }
      continue;
    } else if (ch == EOF) {
      if (num == 0) {
        return {STATUS_EOF, "Get input failed"};
      }
      break;
    }

    c_pass[num] = ch;
    num++;
  }

  c_pass[num] = '\0';
  *pass = c_pass;
  std::cout << std::endl;

  return STATUS_OK;
}

int ToolCommandKey::RunPassCommand(int argc, char *argv[], std::string &fname) {
  int cmdtype = 0;
  std::string rootkey;
  std::string enpass;
  bool sysrelated = true;

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, key_pass_option)
  switch (cmdtype) {
    case MODELBOX_TOOL_KEY_PASS_NON_SYSRELATED:
      sysrelated = false;
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()

  auto ret = EnKey(sysrelated, &rootkey, &enpass, fname);
  if (ret == STATUS_STOP) {
    return -1;
  } else if (!ret) {
    std::cerr << std::endl << "encrypt password failed, " << ret << std::endl;
    return -1;
  }

  std::cout << "Key: " << rootkey << std::endl;
  std::cout << "Encrypted password: " << enpass << std::endl;
  if (!fname.empty()) {
    std::cout << "Encrypted Model Path: " << fname + ".en" << std::endl;
  }
  return 0;
}

Status ToolCommandKey::EnKey(bool sysrelated, std::string *rootkey,
                             std::string *enpass, std::string &fname) {
  std::string pass;
  auto ret = STATUS_OK;

  const char *env_pass = getenv("MODELBOX_PASSWORD");
  if (env_pass) {
    pass = env_pass;
  } else {
    ret = ReadPassword(&pass);
    if (ret == STATUS_STOP) {
      return ret;
    } else if (ret != STATUS_OK) {
      std::cerr << "Read password failed, " << ret << std::endl;
      return STATUS_INVALID;
    }

    if (pass.length() == 0) {
      return STATUS_NODATA;
    }
  }
  std::vector<char> pass_vec(pass.begin(), pass.end());
  if (!fname.empty()) {
    return ModelEncrypt(fname, pass_vec, sysrelated, rootkey, enpass);
  }
  return PassEncrypt(pass_vec, sysrelated, rootkey, enpass);
}

}  // namespace modelbox