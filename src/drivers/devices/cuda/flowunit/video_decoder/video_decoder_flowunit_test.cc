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


#include <functional>
#include <future>
#include <thread>
#include <cuda_runtime.h>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "common/video_decoder/video_decoder_mock.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {
class VideoDecoderCudaFlowUnitTest : public testing::Test {
 public:
  VideoDecoderCudaFlowUnitTest() {}

 protected:
  virtual void SetUp(){
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }
  };

  virtual void TearDown(){};

 public:
  std::shared_ptr<MockFlow> flow_;

  void StartFlow(const std::string& toml_content, const uint64_t millisecond);
};

void VideoDecoderCudaFlowUnitTest::StartFlow(const std::string& toml_content,
                                             const uint64_t millisecond) {
  flow_ = std::make_shared<MockFlow>();
  auto ret = videodecoder::AddMockFlowUnit(flow_);
  EXPECT_EQ(ret, STATUS_SUCCESS);
  ret = flow_->BuildAndRun("decoder", toml_content, millisecond);
  EXPECT_EQ(ret, STATUS_SUCCESS);
}

TEST_F(VideoDecoderCudaFlowUnitTest, cudaDecoderNv12Test) {
  auto toml_content = videodecoder::GetTomlConfig("cuda", "nv12");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderCudaFlowUnitTest, cudaDecoderRgbTest) {
  auto toml_content = videodecoder::GetTomlConfig("cuda", "rgb");
  StartFlow(toml_content, 5 * 1000);
}

TEST_F(VideoDecoderCudaFlowUnitTest, cudaDecoderBgrTest) {
  auto toml_content = videodecoder::GetTomlConfig("cuda", "bgr");
  StartFlow(toml_content, 5 * 1000);
}

}  // namespace modelbox