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

#ifndef MODELBOX_FLOWUNIT_MODEL_DECRYPT_PLUGIN_H_
#define MODELBOX_FLOWUNIT_MODEL_DECRYPT_PLUGIN_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "model_decrypt_interface.h"

class ModelDecryptPlugin : public IModelDecryptPlugin {
 public:
  ModelDecryptPlugin() = default;
  virtual ~ModelDecryptPlugin() = default;
  virtual modelbox::Status Init(
      const std::string &fname,
      const std::shared_ptr<modelbox::Configuration> config);
  virtual modelbox::Status ModelDecrypt(uint8_t *raw_buf, int64_t raw_len,
                                        uint8_t *plain_buf, int64_t &plain_len);

 private:
  std::string rootkey_;
  std::string en_pass_;
};

class ModelDecryptFactory : public modelbox::DriverFactory {
 public:
  ModelDecryptFactory() = default;
  virtual ~ModelDecryptFactory() = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> model_plugin =
        std::make_shared<ModelDecryptPlugin>();
    return model_plugin;
  }
};

#endif  // MODELBOX_FLOWUNIT_MODEL_DECRYPT_PLUGIN_H_
