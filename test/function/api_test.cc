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

#include <opencv2/opencv.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/data_handler.h"
#include "modelbox/external_data_simple.h"
#include "modelbox/flow.h"
#include "modelbox/flow_graph_desc.h"
#include "securec.h"

namespace modelbox {
class FlowGraphTest : public testing::Test {
 public:
  FlowGraphTest() {}

 protected:
  std::shared_ptr<Flow> flow;
  virtual void SetUp(){

  };

  virtual void TearDown(){

  };
};

std::shared_ptr<FlowGraphDesc> CreateGraphDesc() {
  auto graph_desc = std::make_shared<FlowGraphDesc>();
  return graph_desc;
}

TEST_F(FlowGraphTest, StreamTest) {
  auto graph_desc = CreateGraphDesc();
  auto builder = std::make_shared<ConfigurationBuilder>();
  auto config = builder->Build();
  config->SetProperty("graph.queue_size", "32");
  config->SetProperty("graph.queue_size_external", "1000");
  config->SetProperty("graph.batch_size", "16");
  config->SetProperty("drivers.skip-default", "true");
  config->SetProperty("drivers.dir", TEST_DRIVER_DIR);
  graph_desc->Init(config);
  auto source_url =
      std::string(TEST_ASSETS) + "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4";

  std::map<std::string, std::string> demuxer_config;

  demuxer_config.emplace("device", "cpu");
  demuxer_config.emplace("deviceid", "0");

  auto video_demuxer = graph_desc->AddNode("video_demuxer", {}, nullptr);
  auto input = graph_desc->BindInput(video_demuxer);
  auto output = graph_desc->BindOutput(video_demuxer);

  auto flow = std::make_shared<Flow>();
  flow->Init(graph_desc);
  flow->Build();
  flow->RunAsync();
  auto data_map = flow->CreateExternalDataMap();
  auto data_simple = std::make_shared<ExternalDataSimple>(data_map);
  data_simple->PushData(input->GetNodeName(), source_url.data(),
                        source_url.size());

  std::shared_ptr<void> data = nullptr;
  size_t data_len = 0;
  auto status =
      data_simple->GetResult(output->GetNodeName(), data, data_len, 1000);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_GT(data_len, 1000);
}
TEST_F(FlowGraphTest, ResizeTest) {
  auto graph_desc = CreateGraphDesc();
  auto builder = std::make_shared<ConfigurationBuilder>();
  auto config = builder->Build();
  config->SetProperty("drivers.dir", TEST_DRIVER_DIR);
  graph_desc->Init(config);

  auto resize_output = graph_desc->AddNode(
      "resize", {{"width", "256"}, {"height", "256"}}, nullptr);

  auto callback = [](std::shared_ptr<DataContext> data_context) -> Status {
    auto input = data_context->Input("In_1");
    auto output = data_context->Output("Out_1");
    int data_size = input->At(0)->GetBytes();
    auto buffer = input->At(0);
    output->Build({(unsigned long)data_size});
    memcpy_s(output->At(0)->MutableData(), data_size, buffer->ConstData(),
             data_size);
    return STATUS_SUCCESS;
  };

  auto data_handler = std::make_shared<NodeDesc>();
  data_handler->SetNodeDesc(
      {{"In_1", resize_output->GetNodeDesc("out_image")}});
  auto process_output =
      graph_desc->AddNode(callback, {"In_1"}, {"Out_1"}, data_handler);
  auto input = graph_desc->BindInput(resize_output);
  auto output = graph_desc->BindOutput(process_output);

  auto flow = std::make_shared<Flow>();
  flow->Init(graph_desc);
  flow->Build();
  flow->RunAsync();
  auto data_map = flow->CreateExternalDataMap();
  auto data_simple = std::make_shared<ExternalDataSimple>(data_map);
  std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";
  cv::Mat gimg_data = cv::imread(gimg_path.c_str());

  long unsigned int gcols = gimg_data.cols;
  long unsigned int grows = gimg_data.rows;
  long unsigned int gchannels = gimg_data.channels();
  long unsigned int data_size = gimg_data.cols * gimg_data.rows * 3;

  auto bufferlist = data_map->CreateBufferList();
  bufferlist->Build({data_size});
  auto buffer = bufferlist->At(0);
  auto buffer_data = buffer->MutableData();
  if (data_size > 0) {
    memcpy_s(buffer_data, data_size, gimg_data.data, data_size);
  }
  buffer->Set("width", (int)gcols);
  buffer->Set("height", (int)grows);
  buffer->Set("channel", (int)gchannels);
  auto status = data_map->Send(input->GetNodeName(), bufferlist);

  std::shared_ptr<void> data;
  size_t data_len = 0;
  status = data_simple->GetResult(output->GetNodeName(), data, data_len);
  cv::Mat img_data(cv::Size(256, 256), CV_8UC3);
  memcpy_s(img_data.data, img_data.total() * img_data.elemSize(), data.get(),
           data_len);
  EXPECT_EQ(status, STATUS_SUCCESS);
}

}  // namespace modelbox
