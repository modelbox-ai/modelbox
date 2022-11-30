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

#include <functional>
#include <future>
#include <thread>

#include "common/video_decoder/video_decoder_mock.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "modelbox/device/rockchip/rockchip_api.h"

namespace modelbox {
class VideoDecoderRockchipFlowUnitTest : public testing::Test {
 public:
  VideoDecoderRockchipFlowUnitTest()
      : flow_(std::make_shared<MockFlow>()),
        jpeg_decode_(std::make_shared<modelbox::MppJpegDecode>()) {}

 protected:
  void SetUp() override {
    auto ret = jpeg_decode_->Init();
    if (ret != modelbox::STATUS_OK) {
      MBLOG_INFO << "no rockchip device, skip test suit";
      GTEST_SKIP();
    }
  }

  void TearDown() override {}

 public:
  std::shared_ptr<MockFlow> flow_;
  std::shared_ptr<modelbox::MppJpegDecode> jpeg_decode_;

  void StartFlow(const std::string& toml_content, uint64_t millisecond);
};

void VideoDecoderRockchipFlowUnitTest::StartFlow(
    const std::string& toml_content, const uint64_t millisecond) {
  auto ret = videodecoder::AddMockFlowUnit(flow_);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  ret = flow_->BuildAndRun("decoder", toml_content, millisecond);
  EXPECT_EQ(ret, STATUS_SUCCESS);
}

TEST_F(VideoDecoderRockchipFlowUnitTest, rockchipDecoderNv12Test) {
  auto toml_content = videodecoder::GetTomlConfig("rockchip", "nv12");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderRockchipFlowUnitTest, rockchipDecoderRgbTest) {
  auto toml_content = videodecoder::GetTomlConfig("rockchip", "rgb");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderRockchipFlowUnitTest, rockchipDecoderBgrTest) {
  auto toml_content = videodecoder::GetTomlConfig("rockchip", "bgr");
  StartFlow(toml_content, 5 * 1000);
}

}  // namespace modelbox