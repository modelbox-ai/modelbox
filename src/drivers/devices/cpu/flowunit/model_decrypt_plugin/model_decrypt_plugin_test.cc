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

#include <dlfcn.h>
#include <gmock/gmock-actions.h>
#include <securec.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "model_decrypt_interface.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"
#include "modelbox/base/config.h"
#include "modelbox/buffer.h"

#define DLL_NAME_SUB "libmodelbox-unit-cpu-model-decrypt-plugin.so."
#define MODELBOX_VERSION_MAJORSTR(R) #R
#define MODELBOX_VERSION_MAJORSTRING(R) MODELBOX_VERSION_MAJORSTR(R)
#define DLL_NAME DLL_NAME_SUB MODELBOX_VERSION_MAJORSTRING(MODELBOX_VERSION_MAJOR)

using ::testing::_;

namespace modelbox {

typedef std::shared_ptr<DriverFactory> (*CreateDriverFactory)();

class ModelDecryptPluginTest : public testing::Test {
 public:
  ModelDecryptPluginTest() = default;
  CreateDriverFactory driver_func_ = nullptr;

 protected:
  void SetUp() override {
    auto ret = OpenDriver();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override {
    if (driver_handler_) {
      dlclose(driver_handler_);
    }
  };

 private:
  modelbox::Status OpenDriver();
  void *driver_handler_ = nullptr;
};

modelbox::Status ModelDecryptPluginTest::OpenDriver() {
  std::string so_path = TEST_DRIVER_DIR;
  so_path.append("/").append(DLL_NAME);
  driver_handler_ = dlopen(so_path.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (driver_handler_ == nullptr) {
    MBLOG_ERROR << "dll open fail :" << dlerror();
    return STATUS_FAULT;
  }
  driver_func_ =
      (CreateDriverFactory)dlsym(driver_handler_, "CreateDriverFactory");
  if (driver_func_ == nullptr) {
    MBLOG_ERROR << "dll func fail :" << dlerror();
    return STATUS_FAULT;
  }

  return STATUS_OK;
}

TEST_F(ModelDecryptPluginTest, ModelDecryptTest) {
  if (driver_func_ == nullptr) {
    MBLOG_ERROR << "driver_func is null";
    return;
  }
  // This test would be skipped, if no auth info is provided.
  auto model_decrypt_func = std::dynamic_pointer_cast<IModelDecryptPlugin>(
      driver_func_()->GetDriver());
  std::shared_ptr<Configuration> config = std::make_shared<Configuration>();
  config->SetProperty("encryption.rootkey",
                      "JRNd6slbpA08mRxnMwZZZJYBR5gHhtJASjgSiRNTiLgTNrC8DGEfKuYF"
                      "SDashsuU/eHB1ybr+Fm7kgjDcoCYk71nv4LIHrHZL6QZiVqL9CfT");
  config->SetProperty("encryption.passwd",
                      "IudbJKZB+7lenEjHkPO+AaMmoloOv5MMDbbZwqPSTpsANBWF/C/"
                      "eDJGnDvARVpUV3EIgXm4oS28RBtNT27c+5Q==");

  auto ret = model_decrypt_func->Init("", config);
  EXPECT_EQ(ret, STATUS_OK);

  std::string test_str("this is a test");
  uint8_t enbuf[] = {0x87, 0xAC, 0xDD, 0x3D, 0x3F, 0x47, 0xCC, 0x87,
                     0x3C, 0x1A, 0x1B, 0x31, 0x3B, 0xB5, 0x34, 0x70};
  char plainbuf[sizeof(enbuf) + EVP_MAX_BLOCK_LENGTH + 1];
  int64_t plain_len = sizeof(plainbuf);
  ret = model_decrypt_func->ModelDecrypt((uint8_t *)enbuf, sizeof(enbuf),
                                         (uint8_t *)plainbuf, plain_len);
  EXPECT_EQ(ret, STATUS_OK);
  std::string result_str(plainbuf, plain_len);
  EXPECT_EQ(result_str.compare(test_str), 0);
}

}  // namespace modelbox