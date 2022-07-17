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

#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <modelbox/base/base64_simd.h>
#include <modelbox/base/crypto.h>
#include <modelbox/base/os.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <stdint.h>
#include <stdlib.h>  // for endian type
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iomanip>
#include <sstream>
#include <vector>

#include "modelbox/base/log.h"
#include "securec.h"

namespace modelbox {

#define MODELBOX_SIGN_LEN 4096
#define KEY_LEN 48
#define SALT_LEN 32
#define ITERATION_NUM 10000
#define KEY_PATH_MAX (1024)
#define KEY_BUFF_LEN (4096)
#define SEED_LEN 55
#define RANDOM_SOURCE "/dev/random"

struct key_gen_info {
  unsigned char sysrelated;
  unsigned char rootKey[KEY_LEN];
  unsigned char salt[SALT_LEN];
} __attribute__((packed, aligned(1)));

struct cipher_context {
  unsigned char iv[IV_LEN];
  unsigned char ciph[MAX_PASSWORD_LEN];
} __attribute__((packed, aligned(1)));

Status HmacEncode(const std::string &algorithm, const void *input,
                  size_t input_len, std::vector<unsigned char> *output) {
  EVP_MD_CTX *mdctx = nullptr;
  const EVP_MD *md = nullptr;
  unsigned char md_value[EVP_MAX_MD_SIZE];
  unsigned int md_len;

  md = EVP_get_digestbyname(algorithm.c_str());

  if (!md) {
    return {STATUS_NOTSUPPORT, "unknown digest " + algorithm};
  }

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  mdctx = EVP_MD_CTX_new();
#else
  mdctx = EVP_MD_CTX_create();
#endif
  if (mdctx == nullptr) {
    return {STATUS_NOMEM, "create md ctx failed."};
  }

  EVP_DigestInit_ex(mdctx, md, nullptr);
  EVP_DigestUpdate(mdctx, input, input_len);
  EVP_DigestFinal_ex(mdctx, md_value, &md_len);
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  EVP_MD_CTX_free(mdctx);
#else
  EVP_MD_CTX_destroy(mdctx);
#endif

  output->insert(output->end(), &md_value[0], &md_value[md_len]);

  return STATUS_OK;
}

Status HmacEncode(const std::string &algorithm,
                  const std::vector<unsigned char> &input,
                  std::vector<unsigned char> *output) {
  return HmacEncode(algorithm, input.data(), input.size(), output);
}

std::string HmacToString(const void *input, size_t input_len) {
  std::stringstream ss;
  const unsigned char *data = (unsigned char *)input;
  for (size_t i = 0; i < input_len; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
  }
  return ss.str();
}

Status Base64Encode(const unsigned char *input, size_t input_len,
                    std::string *output) {
  auto ret = Base64EncodeSIMD(input, input_len, output);
  if (ret == STATUS_NOTFOUND) {
    std::vector<unsigned char> out;
    int base64_len = 0;
    int output_len = (((input_len + 2) / 3) * 4) + 1;
    out.resize(output_len);

    base64_len = EVP_EncodeBlock(out.data(), input, input_len);
    if (base64_len <= 0) {
      return {STATUS_FAULT, "base64 encode failed."};
    }

    output->assign(out.begin(), out.begin() + base64_len);
  } else {
    return ret;
  }

  return STATUS_OK;
}

Status Base64Encode(const std::vector<unsigned char> &input,
                    std::string *output) {
  return Base64Encode(input.data(), input.size(), output);
}

Status Base64Decode(const std::string &input,
                    std::vector<unsigned char> *output) {
  int base64_len = 0;
  int out_max_len = (input.length() * 6 + 7) / 8;

  output->resize(out_max_len);
  base64_len = EVP_DecodeBlock(output->data(), (unsigned char *)input.c_str(),
                               input.length());
  if (base64_len < 0) {
    return {STATUS_FAULT, "Decode base64 failed: " + input};
  }

  for (int i = input.length() - 1; i >= 0; i--) {
    if (input.c_str()[i] != '=') {
      break;
    }

    base64_len--;
  }

  output->resize(base64_len);

  return STATUS_OK;
}

static const signed char ROOT_MATERIAL_INIT[] = {
    -58,  -85,  80,   55,   -26,  -5,   110,  -63,  71,   37,   104,  -9,
    -45,  58,   32,   33,   6,    -22,  23,   121,  79,   -62,  -96,  0,
    125,  -45,  -68,  53,   116,  95,   108,  12,   20,   -105, -102, -11,
    -62,  -8,   121,  -90,  80,   55,   35,   -27,  -91,  -51,  29,   74,
    -88,  -51,  -115, 94,   3,    -85,  -72,  -99,  -65,  93,   44,   86,
    58,   -55,  48,   -52,  -33,  -60,  -77,  -74,  107,  -8,   -2,   -6,
    115,  -27,  -84,  -11,  -39,  43,   -34,  11,   3,    3,    -95,  28,
    98,   -59,  -96,  -88,  -89,  14,   104,  -104, 99,   63,   12,   61,
    68,   -121, 122,  27,   -68,  -71,  113,  -112, -34,  63,   -51,  119,
    109,  -81,  1,    20,   -103, -6,   -28,  12,   25,   13,   25,   -97,
    -51,  -72,  -71,  -112, -29,  41,   -12,  -52,  95,   -96,  73,   41,
    42,   115,  98,   -82,  -112, -92,  19,   -16,  -72,  -15,  -69,  62,
    -60,  21,   116,  -54,  -11,  -110, -2,   73,   -20,  70,   56,   94,
    35,   -49,  50,   -88,  -76,  70,   -121, 52,   -58,  -43,  98,   -45,
    113,  -94,  -97,  -95,  -96,  9,    -88,  -25,  -26,  97,   123,  -83,
    -48,  -5,   -22,  -79,  87,   40,   18,   -57,  -86,  12,   -107, 101,
    118,  53,   -97,  64,   -13,  125,  -27,  58,   -85,  -49,  -23,  77,
    -100, -88,  -28,  65,   -92,  100,  -9,   -49,  -128, -28,  -64,  43,
    -35,  33,   -103, -62,  31,   59,   115,  63,   2,    21,   102,  117,
    -66,  -71,  -115, -90,  37,   -53,  -125, -48,  -89,  -45,  1,    36,
    102,  91,   -125, -123, 114,  63,   -92,  81,   115,  66,   -42,  -78,
    81,   -94,  -91,  -51,  54,   -40,  62,   19,   -31,  107,  34,   45,
    110,  -8,   -75,  104,  58,   97,   65,   83,   -33,  117,  -80,  125,
    -103, -87,  -37,  50,   31,   -32,  -9,   -54,  76,   -108, 38,   116,
    41,   18,   115,  -15,  -110, 54,   90,   87,   28,   118,  -90,  -127,
    -59,  4,    -33,  31,   68,   11,   -116, -48,  64,   -25,  -25,  -31,
    -32,  17,   -92,  103,  17,   -5,   61,   -125, -105, 36,   15,   0,
    65,   -3,   97,   -71,  114,  -103, -81,  -28,  39,   55,   119,  69,
    88,   -59,  -96,  -102, -61,  123,  -105, 20,   -40,  -45,  114,  33,
    -57,  3,    -57,  115,  -80,  -39,  108,  -79,  114,  45,   -5,   114,
    -50,  81,   105,  15,   51,   99,   -37,  -105, 27,   124,  -20,  -68,
    7,    -20,  110,  -119, -63,  51,   -67,  -85,  109,  24,   79,   -123,
    -121, -6,   -35,  -69,  62,   76,   21,   48,   -109, -128, -9,   127,
    -106, 9,    -42,  -85,  110,  113,  50,   -46,  29,   -3,   -17,  -84,
    82,   -122, -27,  3,    67,   -83,  -30,  50,   100,  -99,  -92,  -68,
    -59,  -72,  39,   0,    -54,  -107, -83,  31,   86,   -123, 9,    -69,
    -23,  121,  -65,  70,   -64,  -16,  31,   20,   123,  -88,  0,    -125,
    18,   -87,  64,   96,   -67,  17,   -119, 34,   -19,  -36,  37,   -25,
    105,  -69,  -30,  -12,  -72,  -104, -52,  63,   -69,  29,   -117, -17,
    122,  -124, -52,  23,   -72,  -106, 119,  -82,  -102, 115,  -71,  -71,
    -105, -111, -42,  -71,  -8,   81,   4,    -64,  -90,  37,   66,   10,
    76,   -14,  -8,   -63,  72,   74,   -14,  -3,   -114, -63,  12,   -106,
    -18,  5,    -19,  44,   -93,  -66,  -33,  -94};

static bool seed_set = false;

static Status ReadBytes(int fd, void *buffer, int len) {
  int bytesRead = 0;
  int result;
  while (bytesRead < len) {
    result = read(fd, (char *)(buffer) + bytesRead, len - bytesRead);
    if (result == -1) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      }
      MBLOG_ERROR << "errno is " << StrError(errno);
      return {STATUS_FAULT, "Generate Seed Failed."};
    }
    bytesRead += result;
  }
  return STATUS_OK;
}

Status GetTrueRandom(void *random, int len) {
  int fd;
  fd = open(RANDOM_SOURCE, O_RDONLY);
  if (fd <= 0) {
    return {STATUS_FAULT, "Open /dev/random failed"};
  }
  Defer { close(fd); };
  auto status = ReadBytes(fd, random, len);
  return status;
}

Status HmacGenRootKey(int sysrelated, std::string *en_key) {
  struct key_gen_info keyGenInfo;
  unsigned char sysrelate_num = 0;

  memset_s(&keyGenInfo, sizeof(keyGenInfo), 0, sizeof(keyGenInfo));

  if (!seed_set) {
    unsigned char seed[SEED_LEN];
    Status status = GetTrueRandom(seed, SEED_LEN);
    if (status != STATUS_SUCCESS) {
      return status;
    }
    RAND_seed(&seed, SEED_LEN);
    seed_set = true;
  }

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  RAND_priv_bytes((unsigned char *)keyGenInfo.rootKey, KEY_LEN);
  RAND_priv_bytes(keyGenInfo.salt, SALT_LEN);
  RAND_priv_bytes(&sysrelate_num, sizeof(sysrelate_num));
#else
  RAND_bytes((unsigned char *)keyGenInfo.rootKey, KEY_LEN);
  RAND_bytes(keyGenInfo.salt, SALT_LEN);
  RAND_bytes(&sysrelate_num, sizeof(sysrelate_num));
#endif

  if (sysrelated) {
    if ((sysrelate_num % 2) != 0) {
      sysrelate_num++;
    }
  } else {
    if ((sysrelate_num % 2) == 0) {
      sysrelate_num++;
    }
  }

  keyGenInfo.sysrelated = sysrelate_num;
  std::vector<unsigned char> keyinfo(
      (unsigned char *)&keyGenInfo,
      (unsigned char *)&keyGenInfo + sizeof(keyGenInfo));

  return Base64Encode(keyinfo, en_key);
}

Status HmacGetRootKey(const std::string &en_key,
                      std::vector<unsigned char> *outkey) {
  unsigned int i;
  int iRet;

  int MATERIAL_LEN = sizeof(ROOT_MATERIAL_INIT);
  struct key_gen_info *keyGenInfo;
  std::vector<unsigned char> raw_key;

  auto ret = Base64Decode(en_key, &raw_key);
  if (raw_key.size() < sizeof(struct key_gen_info)) {
    return {STATUS_INVALID, "enkey is invalid."};
  }

  keyGenInfo = (struct key_gen_info *)raw_key.data();

  for (i = 0; i < sizeof(keyGenInfo->rootKey); ++i) {
    keyGenInfo->rootKey[i] =
        keyGenInfo->rootKey[i] ^ ROOT_MATERIAL_INIT[i % MATERIAL_LEN];
  }

  if (keyGenInfo->sysrelated % 2 == 0) {
    std::string sysID = os->GetSystemID();
    std::string mac_addr = os->GetMacAddress();
    std::vector<unsigned char> syskey(KEY_LEN);
#if OPENSSL_VERSION_NUMBER >= 0x1000100fL
    iRet = PKCS5_PBKDF2_HMAC(sysID.c_str(), sysID.length(), keyGenInfo->salt,
                             SALT_LEN, ITERATION_NUM, EVP_sha256(),
                             syskey.size(), syskey.data());
    if (mac_addr.length() > 0) {
      iRet = PKCS5_PBKDF2_HMAC(mac_addr.c_str(), mac_addr.length(),
                               keyGenInfo->salt, SALT_LEN, ITERATION_NUM,
                               EVP_sha256(), syskey.size(), syskey.data());
    }
#else
    iRet = PKCS5_PBKDF2_HMAC_SHA1(sysID.c_str(), sysID.length(),
                                  keyGenInfo->salt, SALT_LEN, ITERATION_NUM,
                                  syskey.size(), syskey.data());
    if (mac_addr.length() > 0) {
      iRet = PKCS5_PBKDF2_HMAC_SHA1(mac_addr.c_str(), mac_addr.length(),
                                    keyGenInfo->salt, SALT_LEN, ITERATION_NUM,
                                    syskey.size(), syskey.data());
    }
#endif
    for (i = 0; i < sizeof(keyGenInfo->rootKey); ++i) {
      keyGenInfo->rootKey[i] = keyGenInfo->rootKey[i] ^ syskey[i % KEY_LEN];
    }
  }

  outkey->resize(KEY_LEN);
#if OPENSSL_VERSION_NUMBER >= 0x1000100fL
  iRet = PKCS5_PBKDF2_HMAC((const char *)keyGenInfo->rootKey, KEY_LEN,
                           keyGenInfo->salt, SALT_LEN, ITERATION_NUM,
                           EVP_sha256(), outkey->size(), outkey->data());
#else
  iRet = PKCS5_PBKDF2_HMAC_SHA1(keyGenInfo->rootKey, KEY_LEN, keyGenInfo->salt,
                                SALT_LEN, ITERATION_NUM, outkey->size(),
                                outkey->data());
#endif

  if (iRet == 0) {
    return {STATUS_FAULT, "Create HMAC failed."};
  }

  return STATUS_OK;
}

Status Encrypt(const std::string &ciphername, unsigned char *input,
               int input_len, unsigned char *output, int *output_len,
               int max_output, unsigned char *key, unsigned char *iv) {
  std::shared_ptr<EVP_CIPHER_CTX> ctx;
  const EVP_CIPHER *cipher = nullptr;
  EVP_CIPHER_CTX *ctx_new = nullptr;
  int len;
  *output_len = 0;

  if (input_len + EVP_MAX_BLOCK_LENGTH >= max_output) {
    return {STATUS_NOSPACE, "output buffer is not enough."};
  }

  cipher = EVP_get_cipherbyname(ciphername.c_str());
  if (cipher == nullptr) {
    return {STATUS_NOTSUPPORT, "cipher not support, " + ciphername};
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
  if (1 != EVP_EncryptInit_ex(ctx.get(), cipher, nullptr, key, iv)) {
    return {STATUS_FAULT, "encrypt init failed."};
  }

  /* Provide the message to be encrypted, and obtain the encrypted output.
   * EVP_EncryptUpdate can be called multiple times if necessary
   */
  if (1 != EVP_EncryptUpdate(ctx.get(), output, &len, input, input_len)) {
    return {STATUS_FAULT, "encrypt update failed."};
  }
  *output_len += len;

  /* Finalise the encryption. Further ciphertext bytes may be written at
   * this stage.
   */
  if (1 != EVP_EncryptFinal_ex(ctx.get(), output + *output_len, &len)) {
    return {STATUS_FAULT, "encrypt final failed."};
  }
  *output_len += len;

  return STATUS_OK;
}

std::string EvpGetErrorMsg() {
  const auto *errmsg = ERR_reason_error_string(ERR_get_error());
  if (errmsg == nullptr) {
    return "";
  }

  return errmsg;
}

Status Decrypt(const std::string &ciphername, unsigned char *input,
               int input_len, unsigned char *output, int *output_len,
               int max_output, unsigned char *key, unsigned char *iv) {
  std::shared_ptr<EVP_CIPHER_CTX> ctx;
  const EVP_CIPHER *cipher = nullptr;
  EVP_CIPHER_CTX *ctx_new = nullptr;
  int len = 0;

  if (input_len + EVP_MAX_BLOCK_LENGTH >= max_output) {
    return {STATUS_NOSPACE, "output buffer is not enough."};
  }

  *output_len = 0;

  cipher = EVP_get_cipherbyname(ciphername.c_str());
  if (cipher == nullptr) {
    return {STATUS_NOTSUPPORT, "cipher not support, " + ciphername};
  }

  /* Create and initialise the context */
  ctx_new = EVP_CIPHER_CTX_new();
  if (ctx_new == nullptr) {
    return {STATUS_NOMEM, "create cipher failed."};
  }

  ctx.reset(ctx_new, [](EVP_CIPHER_CTX *ctx) { EVP_CIPHER_CTX_free(ctx); });

  /* Initialise the decryption operation. IMPORTANT - ensure you use a key
   * and IV size appropriate for your cipher
   * In this example we are using 256 bit AES (i.e. a 256 bit key). The
   * IV size for *most* modes is the same as the block size. For AES this
   * is 128 bits */
  if (1 != EVP_DecryptInit_ex(ctx.get(), cipher, nullptr, key, iv)) {
    std::string msg = "decrypt failed, " + EvpGetErrorMsg();
    return {STATUS_FAULT, msg};
  }

  /* Provide the message to be decrypted, and obtain the plaintext output.
   * EVP_DecryptUpdate can be called multiple times if necessary
   */
  if (1 != EVP_DecryptUpdate(ctx.get(), output, &len, input, input_len)) {
    std::string msg = "decrypt update failed, " + EvpGetErrorMsg();
    return {STATUS_FAULT, msg};
  }

  *output_len += len;

  /* Finalise the decryption. Further plaintext bytes may be written at
   * this stage.
   */
  if (1 != EVP_DecryptFinal_ex(ctx.get(), output + *output_len, &len)) {
    std::string msg = "decrypt final failed, " + EvpGetErrorMsg();
    return {STATUS_FAULT, msg};
  }

  *output_len += len;

  return STATUS_OK;
}

Status PassEncrypt(const std::vector<char> &pass, bool sysrelated,
                   std::string *rootkey, std::string *en_pass,
                   const std::string &ciphername) {
  Status ret;
  struct cipher_context *contex = nullptr;
  int cipher_len = 0;
  std::vector<unsigned char> key;

  if (en_pass == nullptr || rootkey == nullptr) {
    return STATUS_INVALID;
  }

  std::vector<unsigned char> encrypt_raw_pass(sizeof(struct cipher_context));
  contex = (struct cipher_context *)encrypt_raw_pass.data();
  memset_s(contex, sizeof(*contex), 0, sizeof(*contex));

  if (rootkey->length() == 0) {
    /* Generate root key */
    ret = HmacGenRootKey(sysrelated, rootkey);
    if (ret != STATUS_OK) {
      return ret;
    }
  }

  /* Get root key */
  ret = HmacGetRootKey(*rootkey, &key);
  if (ret != STATUS_OK) {
    return ret;
  }

  RAND_bytes(contex->iv, IV_LEN);
  ret = Encrypt(ciphername, (unsigned char *)pass.data(), pass.size(),
                contex->ciph, &cipher_len, MAX_PASSWORD_LEN, key.data(),
                contex->iv);
  if (ret != STATUS_OK) {
    return ret;
  }

  encrypt_raw_pass.resize(IV_LEN + cipher_len);
  memset_s(key.data(), key.size(), 0, key.size());
  ret = Base64Encode(encrypt_raw_pass, en_pass);
  return ret;
}

Status PassDecrypt(const std::string &en_pass, const std::string &rootkey,
                   std::vector<char> *pass, const std::string &ciphername) {
  Status ret;
  struct cipher_context *contex;
  std::vector<unsigned char> key;
  std::vector<unsigned char> raw_pass(sizeof(struct cipher_context));

  ret = HmacGetRootKey(rootkey, &key);
  if (ret != STATUS_OK) {
    return ret;
  }

  ret = Base64Decode(en_pass, &raw_pass);
  if (ret != STATUS_OK) {
    return ret;
  }

  int en_pass_len = raw_pass.size();
  contex = (struct cipher_context *)raw_pass.data();
  std::vector<unsigned char> encrypt_raw_pass(contex->ciph,
                                              contex->ciph + en_pass_len);
  // fill key with "0"
  pass->resize(en_pass.length() + EVP_MAX_BLOCK_LENGTH, '\0');
  int passwordlen = 0;
  ret = Decrypt(ciphername, contex->ciph, en_pass_len - IV_LEN,
                (unsigned char *)pass->data(), &passwordlen, MAX_PASSWORD_LEN,
                key.data(), contex->iv);
  if (!ret) {
    return ret;
  }

  pass->resize(passwordlen);
  return ret;
}

}  // namespace modelbox