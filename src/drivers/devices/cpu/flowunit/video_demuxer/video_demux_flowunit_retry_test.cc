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

#include "common/video_decoder/video_decoder_mock.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "securec.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class VideoDemuxerFlowUnitRetryTest : public testing::Test {
 public:
  VideoDemuxerFlowUnitRetryTest()
      : driver_flow_(std::make_shared<MockFlow>()) {}

  std::shared_ptr<MockFlow> GetDriverFlow() { return driver_flow_; };
  std::shared_ptr<MockFlow> RunDriverFlow();
  modelbox::Status SendDataSourceCfg(const std::string &data_source_cfg,
                                     const std::string &source_type);

 protected:
  void SetUp() override{};

  void TearDown() override{};

  std::string GetRtspTomlConfig();

  modelbox::Status StartFlow(std::string &toml_content, uint64_t millisecond);

 private:
  std::shared_ptr<MockFlow> driver_flow_;
  std::shared_ptr<Flow> flow_;
};

modelbox::Status VideoDemuxerFlowUnitRetryTest::StartFlow(
    std::string &toml_content, const uint64_t millisecond) {
  driver_flow_ = std::make_shared<MockFlow>();
  auto ret = videodecoder::AddMockFlowUnit(driver_flow_, true);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  driver_flow_->BuildAndRun("VideoDecoder", toml_content, -1);
  std::string source_type = "url";
  std::string data_source_cfg = R"({
        "url": "rtsp://192.168.59.29:10054/live/k14XeNAIR",
        "url_type": "stream"
  })";
  flow_ = driver_flow_->GetFlow();
  SendDataSourceCfg(data_source_cfg, source_type);
  return flow_->Wait(millisecond);
}

TEST_F(VideoDemuxerFlowUnitRetryTest, RtspInputTest) {
  auto toml_content = GetRtspTomlConfig();
  auto ret = StartFlow(toml_content, 10 * 1000);
  EXPECT_EQ(ret, modelbox::STATUS_TIMEDOUT);
}

std::string VideoDemuxerFlowUnitRetryTest::GetRtspTomlConfig() {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string dest_url = "rtmp://192.168.59.29:10035/live/iEunZv0IR?sign=mPu7WDASRz";
  std::string toml_content =
      R"(
      [log]
      level = "INFO"
      [driver]
      skip-default = true
      dir=[")" +
      test_lib_dir + "\"]\n    " +
      R"([graph]
      thread-num = 16
      max-thread-num = 100
      graphconf = '''digraph demo {
            input[type=input, device=cpu, deviceid=0]
            data_source_parser[type=flowunit, flowunit=data_source_parser, device=cpu, deviceid=0, retry_interval_ms = 1000, obs_retry_interval_ms = 3000,url_retry_interval_ms = 1000, label="<data_uri>", plugin_dir=")" +
      test_lib_dir + R"("] 
            videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>", queue_size = 16]
            videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16]  
            // videodecoder[type=flowunit, flowunit=video_decoder, device=cuda, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16]  
            // videodecoder[type=flowunit, flowunit=video_decoder, device=ascend, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=nv12, queue_size = 16]  
            videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, queue_size = 16, deviceid=0, default_dest_url=")" +
      dest_url + R"(
            ", format=flv, encoder=libx264 ]
            input -> data_source_parser:in_data
            data_source_parser:out_video_url -> videodemuxer:in_video_url
            videodemuxer:out_video_packet -> videodecoder:in_video_packet
            videodecoder:out_video_frame -> videoencoder:in_video_frame
          }'''
      format = "graphviz"
    )";

  return toml_content;
}

modelbox::Status VideoDemuxerFlowUnitRetryTest::SendDataSourceCfg(
    const std::string &data_source_cfg, const std::string &source_type) {
  auto ext_data = flow_->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({data_source_cfg.size()});
  auto buffer = buffer_list->At(0);
  memcpy_s(buffer->MutableData(), buffer->GetBytes(), data_source_cfg.data(),
           data_source_cfg.size());
  buffer->Set("source_type", source_type);
  ext_data->Send("input", buffer_list);
  ext_data->Close();
  for (size_t i = 0; i < 5; ++i) {
    // should continue reconnect
    std::this_thread::sleep_for(std::chrono::seconds(1));
    OutputBufferList output;
    auto ret = ext_data->Recv(output, 100);
    EXPECT_EQ(ret, STATUS_TIMEDOUT);
    if (ret != STATUS_TIMEDOUT) {
      return STATUS_FAULT;
    }
  }
  // stop reconnect
  ext_data->Shutdown();
  Status final_state = STATUS_OK;
  for (size_t i = 0; i < 5; ++i) {
    // should stop reconnect, session will close
    std::this_thread::sleep_for(std::chrono::seconds(1));
    OutputBufferList output;
    auto ret = ext_data->Recv(output, 100);
    if (ret == STATUS_INVALID) {
      final_state = ret;
      break;
    }
  }
  EXPECT_EQ(final_state, STATUS_INVALID);
  return modelbox::STATUS_OK;
}

}  // namespace modelbox