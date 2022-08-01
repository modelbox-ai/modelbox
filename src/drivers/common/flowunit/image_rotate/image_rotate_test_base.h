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

#ifndef MODELBOX_FLOWUNIT_IMAGE_ROTATE_TEST_BASE_H_
#define MODELBOX_FLOWUNIT_IMAGE_ROTATE_TEST_BASE_H_

#include <securec.h>

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class ImageRotateFlowUnitTest : public testing::Test {
 public:
  ImageRotateFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

  std::vector<int32_t> test_rotate_angle_{90, 180, 270};

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_IMAGE_ROTATE_TEST_BASE_H_