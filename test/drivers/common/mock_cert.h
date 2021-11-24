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


#ifndef MODELBOX_DRIVER_TEST_CERT_MOCK_H_
#define MODELBOX_DRIVER_TEST_CERT_MOCK_H_

#include <string>

#include "modelbox/base/status.h"
#include "driver_flow_test.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {

/**
 * @brief generate cert for mock
 */
Status GenerateCert(std::string* enPass, std::string* ekRootKey,
                    const std::string& private_key,
                    const std::string& public_key);

/**
 * @brief generate cert for mock
 */
Status GenerateCert(const std::string& private_key,
                    const std::string& public_key);
};  // namespace modelbox

#endif  // MODELBOX_DRIVER_TEST_CERT_MOCK_H_