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


#include <fstream>
#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "common/video_decoder/video_decoder_mock.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {
class VideoDecoderFlowUnitTest : public testing::Test {
 public:
  VideoDecoderFlowUnitTest() = default;

 protected:
  void SetUp() override{};

  void TearDown() override{};

 public:
  std::shared_ptr<MockFlow> flow_;

  void StartFlow(std::string& toml_content, uint64_t millisecond);
};

void VideoDecoderFlowUnitTest::StartFlow(std::string& toml_content,
                                         const uint64_t millisecond) {
  flow_ = std::make_shared<MockFlow>();
  auto ret = videodecoder::AddMockFlowUnit(flow_);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  ret = flow_->BuildAndRun("VideoDecoder", toml_content, millisecond);
  EXPECT_EQ(ret, STATUS_SUCCESS);
}

TEST_F(VideoDecoderFlowUnitTest, cpuDecoderNv12Test) {
  auto toml_content = videodecoder::GetTomlConfig("cpu", "nv12");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderFlowUnitTest, cpuDecoderRgbTest) {
  auto toml_content = videodecoder::GetTomlConfig("cpu", "rgb");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderFlowUnitTest, cpuDecoderBgrTest) {
  auto toml_content = videodecoder::GetTomlConfig("cpu", "bgr");
  StartFlow(toml_content, 5 * 1000);
}

}  // namespace modelbox