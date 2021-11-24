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

#ifndef MODELBOX_FLOWUNIT_MODEL_DECRYPT_H_
#define MODELBOX_FLOWUNIT_MODEL_DECRYPT_H_

#include <fstream>
#include <memory>
#include <string>

#include "model_decrypt_interface.h"

class ModelDecryption {
 public:
  typedef enum {
    MODEL_STATE_ENCRYPT,
    MODEL_STATE_PLAIN,
    MODEL_STATE_ERROR
  } MODEL_STATE;
  ModelDecryption() = default;
  virtual ~ModelDecryption();

  /**
   * @brief init funciton
   * @param model_path a model filename path
   * @param drivers_ptr drivers point to get plugin
   * @param config a toml config
   * @return Success if pass
   */
  modelbox::Status Init(const std::string& model_path,
                        const std::shared_ptr<modelbox::Drivers>& drivers_ptr,
                        const std::shared_ptr<modelbox::Configuration>& config);
                        
  /**
   * @brief model decrypt implement
   * @param model_len a return value: the plain model buffer length
   * @return plain buffer，note ,call free for this buffer by yourself!
   */
  uint8_t* GetModelBuffer(int64_t& model_len);

  /**
   * @brief model decrypt implement
   * @param model_path model file path name
   * @param model_len a return value: the plain model buffer length
   * @return plain buffer smart point, recommand to call this function
   */
  std::shared_ptr<uint8_t> GetModelSharedBuffer(int64_t& model_len);

  /**
   * @brief call it to know whether it's a encrypt model
   * @return MODEL_STATE enum
   */
  inline MODEL_STATE GetModelState() { return model_state_; }

 private:
  ModelDecryption(const ModelDecryption&) = delete;
  ModelDecryption& operator=(const ModelDecryption&) = delete;
  void GetInfoFromHeader(std::string& plugin_name, std::string& plugin_version,
                         const std::shared_ptr<modelbox::Configuration>& config);
                         
  int64_t fsize_ = 0;
  int32_t header_offset_ = 0;
  std::ifstream fmodel_;
  std::shared_ptr<modelbox::DriverFactory> cur_factory_ = nullptr;
  std::shared_ptr<IModelDecryptPlugin> cur_plugin_ = nullptr;
  MODEL_STATE model_state_ = MODEL_STATE_ERROR;
};

#endif  // MODELBOX_FLOWUNIT_MODEL_DECRYPT_H_