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

#ifndef MODELBOX_FLOWUNIT_MODEL_DECRYPT_INTERFACE_H_
#define MODELBOX_FLOWUNIT_MODEL_DECRYPT_INTERFACE_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/status.h>

#include <cstdint>

constexpr const char* MAGIC_FLAG = "MB_EnModel";

#define MAGIC_SIZE 10  // len of MB_EnModel
#define PLUGIN_SIZE 50

struct PrefixInfo {
  char magic[MAGIC_SIZE];        /* magicnumber */
  uint8_t ver_major;             /* version number X*/
  uint8_t ver_minor;             /* version number Y*/
  uint8_t ver_patch;             /* version number Z*/
  uint8_t reserve;               /* reserve flag*/
  char plugin_name[PLUGIN_SIZE]; /* plugin name*/
} __attribute__((packed, aligned(1)));

class IModelDecryptPlugin : public modelbox::Driver {
 public:
  /**
   * @brief model decrypt Init
   * @param fname model file path name
   * @param config encryption.rootkey and encryption.passwd will pass here if
   * passwd is deliverd by config toml file
   * @return Success or not
   */
  virtual modelbox::Status Init(
      const std::string& fname,
      std::shared_ptr<modelbox::Configuration> config) = 0;

  /**
   * @brief model decrypt implement
   * @param raw_buf model encrypted buffer
   * @param raw_len model encrypted buffer len
   * @param plain_buf model plain buffer, plain_len will pass the max plain_buf len
   * @param plain_len set the real len for plain buffer 
   * @return Success or not
   */
  virtual modelbox::Status ModelDecrypt(uint8_t* raw_buf, int64_t raw_len,
                                        uint8_t* plain_buf,
                                        int64_t& plain_len) = 0;
};

#endif  // MODELBOX_FLOWUNIT_MODEL_DECRYPT_INTERFACE_H_