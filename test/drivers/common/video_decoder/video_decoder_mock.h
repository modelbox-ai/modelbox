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


#ifndef MODELBOX_DRIVER_TEST_VIDEO_DECODER_MOCK_H_
#define MODELBOX_DRIVER_TEST_VIDEO_DECODER_MOCK_H_

#include <string>

#include "modelbox/base/status.h"
#include "driver_flow_test.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace videodecoder {
modelbox::Status AddMockFlowUnit(std::shared_ptr<modelbox::MockFlow>& flow,
                               bool is_stream = false);

std::string GetTomlConfig(const std::string& device,
                          const std::string& pix_fmt);
};  // namespace videodecoder

#endif  // MODELBOX_DRIVER_TEST_VIDEO_DECODER_MOCK_H_