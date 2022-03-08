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

#ifndef MODELBOX_DRIVER_TEST_TENSORFLOW_INFERENCE_MOCK_H_
#define MODELBOX_DRIVER_TEST_TENSORFLOW_INFERENCE_MOCK_H_

#include "driver_flow_test.h"
#include "modelbox/base/status.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace tensorflow_inference {

static std::set<std::string> SUPPORT_TF_VERSION = {"1.13.1", "1.15.0",
                                                   "2.6.0-dev20210809"};

modelbox::Status AddMockFlowUnit(
    std::shared_ptr<modelbox::DriverFlowTest> &flow);

modelbox::Status ReplaceVersion(const std::string &src, const std::string &dest,
                                const std::string &version);

std::string GetTFVersion();
};  // namespace tensorflow_inference

#endif  // MODELBOX_DRIVER_TEST_TENSORFLOW_INFERENCE_MOCK_H_