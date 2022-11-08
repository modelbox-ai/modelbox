/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_DLENGINE_INFERENCE_FLOWUNIT_TEST_H_
#define MODELBOX_DLENGINE_INFERENCE_FLOWUNIT_TEST_H_

#include "modelbox/base/status.h"
#include "test/mock/minimodelbox/mockflow.h"

class DLEngineInferenceFlowUnitTest {
 public:
  DLEngineInferenceFlowUnitTest(const std::string &device_type);

  modelbox::Status SetUp(const std::string &infer_flowunit_name);

  void Run(const std::string &name);

  void TearDown();

 private:
  std::string device_type_;
  std::shared_ptr<modelbox::MockFlow> flow_ =
      std::make_shared<modelbox::MockFlow>();

  std::string test_driver_dir_ = TEST_DRIVER_DIR;
  std::string test_data_dir_ = TEST_DATA_DIR;
  std::string infer_flowunit_name_;
};

#endif  // MODELBOX_DLENGINE_INFERENCE_FLOWUNIT_TEST_H_
