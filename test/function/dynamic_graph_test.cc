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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/modelbox/data_handler.h"
#include "include/modelbox/modelbox_engine.h"
#include "mock_driver_ctl.h"

namespace modelbox {
class DynamicGraphTest : public testing::Test {
 public:
  DynamicGraphTest() {}

 protected:
  std::shared_ptr<ModelBoxEngine> modelbox_engine;
  virtual void SetUp(){

  };

  virtual void TearDown(){

  };
};

std::shared_ptr<ModelBoxEngine> Createmodelbox_engine() {
  auto modelbox_engine = std::make_shared<ModelBoxEngine>();
  return modelbox_engine;
}

TEST_F(DynamicGraphTest, DataHandlerTest) {
  auto data_handler0 = std::make_shared<DataHandler>(BUFFERLIST_NODE);
  auto data_handler = std::make_shared<DataHandler>(BUFFERLIST_NODE);
  std::map<std::string, std::shared_ptr<DataHandler>> data;
  auto status = data_handler->SetDataHandler(data);
  EXPECT_TRUE(status == STATUS_FAULT);
}

TEST_F(DynamicGraphTest, StreamTest) {
  modelbox_engine = std::make_shared<ModelBoxEngine>();
  auto builder = std::make_shared<ConfigurationBuilder>();
  auto config = builder->Build();
  config->SetProperty("graph.queue_size", "32");
  config->SetProperty("graph.queue_size_external", "1000");
  config->SetProperty("graph.batch_size", "16");
  config->SetProperty("drivers.skip-default", "true");
  config->SetProperty("drivers.dir", TEST_DRIVER_DIR);
  modelbox_engine->Init(config);
  auto input_stream = modelbox_engine->CreateInput({"input"});
  auto source_url =
      std::string(TEST_ASSETS) + "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4";
  input_stream->SetMeta("source_url", source_url);
  std::map<std::string, std::string> demuxer_config;

  demuxer_config.emplace("device", "cpu");
  demuxer_config.emplace("deviceid", "0");

  auto video_demuxer_output =
      modelbox_engine->Execute("video_demuxer", demuxer_config, input_stream);
  auto buffer = video_demuxer_output->GetData();
  video_demuxer_output->Close();
  modelbox_engine->ShutDown();
  EXPECT_NE(buffer, nullptr);
}

TEST_F(DynamicGraphTest, VideoReEncodeTest) {
  std::shared_ptr<ModelBoxEngine> modelbox_engine = Createmodelbox_engine();

  auto builder = std::make_shared<ConfigurationBuilder>();
  auto config = builder->Build();
  config->SetProperty("graph.queue_size", "32");
  config->SetProperty("graph.queue_size_external", "1000");
  config->SetProperty("graph.batch_size", "16");
  config->SetProperty("drivers.skip-default", "true");
  config->SetProperty("drivers.dir", TEST_DRIVER_DIR);
  Status status = modelbox_engine->Init(config);
  EXPECT_EQ(status, STATUS_SUCCESS);
  if (status != STATUS_SUCCESS) {
    MBLOG_ERROR << "failed init modelbox_engine";
    return;
  }

  auto stream = modelbox_engine->CreateInput({"input1"});
  std::string path =
      std::string(TEST_ASSETS) + "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4";
  stream->SetMeta("source_url", path);
  stream->Close();

  auto encoder_input_stream = modelbox_engine->CreateInput({"input2"});

  std::map<std::string, std::string> demuxer_config = {{"deviceid", "0"},
                                                       {"device", "cpu"}};

  std::map<std::string, std::string> decoder_config;
  {
    decoder_config.emplace("device", "cpu");
    decoder_config.emplace("deviceid", "0");
    decoder_config.emplace("pix_fmt", "nv12");
  }
  std::map<std::string, std::string> encoder_config;
  {
    encoder_config.emplace("device", "cpu");
    encoder_config.emplace("deviceid", "0");
    encoder_config.emplace("queue_size", "1");
    encoder_config.emplace("format", "mp4");
    encoder_config.emplace("default_dest_url", "/tmp/ters.mp4");
    encoder_config.emplace("encoder", "libx264");
  }

  auto video_demuxer_output =
      modelbox_engine->Execute("video_demuxer", demuxer_config, stream);
  auto video_decoder_output = modelbox_engine->Execute(
      "video_decoder", decoder_config, video_demuxer_output);

  modelbox_engine->Execute("video_encoder", encoder_config,
                           encoder_input_stream);
  std::shared_ptr<DataHandler> buffer = nullptr;
  int frame_num = 0;
  while ((buffer = video_decoder_output->GetData()) != nullptr) {
    encoder_input_stream->PushData(buffer, "input2");
    frame_num++;
  }
  EXPECT_TRUE(frame_num > 1);
  encoder_input_stream->Close();
  EXPECT_EQ(status, STATUS_SUCCESS);
}

}  // namespace modelbox
