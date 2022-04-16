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

#include "modelbox/stream.h"

#include <functional>
#include <future>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"

namespace modelbox {
class StreamTest : public testing::Test {};

TEST_F(StreamTest, DataMetaTest) {
  DataMeta data_meta;
  auto value = std::make_shared<int32_t>();
  *value = 123;
  data_meta.SetMeta("test", value);
  auto result = std::static_pointer_cast<int32_t>(data_meta.GetMeta("test"));
  ASSERT_EQ(result, value);
  EXPECT_EQ(*result, *value);
  EXPECT_EQ(data_meta.GetMetas().size(), 1);

  DataMeta data_meta2(data_meta);
  auto result2 = std::static_pointer_cast<int32_t>(data_meta2.GetMeta("test"));
  ASSERT_EQ(result2, value);
  EXPECT_EQ(*result2, *value);
  EXPECT_EQ(data_meta.GetMetas().size(), 1);
}

TEST_F(StreamTest, StreamOrderTest) {
  auto order = std::make_shared<StreamOrder2>();
  auto order2 = order->Copy();
  order2->Expand(0);
  auto order3 = order2->Copy();
  order3->Expand(1);

  auto order4 = order->Copy();
  order4->Expand(1);
  EXPECT_TRUE(*order3 < *order4);

  auto order5 = order3->Copy();
  order5->Collapse();
  EXPECT_FALSE(*order5 < *order2);
  EXPECT_FALSE(*order2 < *order5);
}

TEST_F(StreamTest, StreamTest) {
  Stream stream(nullptr);
  EXPECT_FALSE(stream.ReachEnd());
  stream.SetMaxBufferCount(3);
  stream.IncreaseBufferCount();
  EXPECT_FALSE(stream.ReachEnd());
  stream.IncreaseBufferCount();
  stream.IncreaseBufferCount();
  EXPECT_TRUE(stream.ReachEnd());
  EXPECT_EQ(stream.GetBufferCount(), 3);
}

}  // namespace modelbox