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
 protected:
  void SetUp() override {
    graph_desc_ = std::make_shared<FlowGraphDesc>();
    graph_desc_->SetQueueSize(32);
    graph_desc_->SetBatchSize(8);
    graph_desc_->SetSkipDefaultDrivers(true);
    graph_desc_->SetDriversDir({TEST_DRIVER_DIR});
  }

  std::shared_ptr<FlowGraphDesc> graph_desc_;
};

TEST_F(FlowGraphTest, AddNodeTest) {
  auto source_url =
      std::string(TEST_ASSETS) + "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4";

  auto input = graph_desc_->AddInput("input1");
  auto video_demuxer = graph_desc_->AddNode("video_demuxer", "cpu", input);
  graph_desc_->AddOutput("output1", video_demuxer);

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init(graph_desc_);
  ASSERT_EQ(ret, STATUS_OK);

  flow->StartRun();
  auto data_map = flow->CreateExternalDataMap();
  auto data_simple = std::make_shared<ExternalDataSimple>(data_map);
  data_simple->PushData("input1", source_url.data(), source_url.size());

  std::shared_ptr<void> data = nullptr;
  size_t data_len = 0;
  auto status = data_simple->GetResult("output1", data, data_len, 1000);
  EXPECT_EQ(status, STATUS_SUCCESS);
  EXPECT_GT(data_len, 1000);
}

TEST_F(FlowGraphTest, AddFuncTest) {
  auto input1 = graph_desc_->AddInput("input1");
  auto process_func =
      [](const std::shared_ptr<DataContext> &data_context) -> Status {
    auto input = data_context->Input("in_1");
    const auto *in_data = (const uint8_t *)(input->ConstBufferData(0));

    auto output = data_context->Output("out_1");
    auto buffer = input->At(0);
    output->Build({buffer->GetBytes()});
    auto *data_ptr = (uint8_t *)(output->MutableBufferData(0));
    for (uint8_t i = 0; i < 10; ++i) {
      data_ptr[i] = in_data[i] + 1;
    }
    output->At(0)->Set("test_meta", "test_meta");
    return STATUS_SUCCESS;
  };
  auto func_node =
      graph_desc_->AddFunction(process_func, {"in_1"}, {"out_1"}, input1);
  graph_desc_->AddOutput("output1", func_node);

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init(graph_desc_);
  ASSERT_EQ(ret, STATUS_OK);

  flow->StartRun();
  auto stream_io = flow->CreateStreamIO();
  auto buffer = stream_io->CreateBuffer();
  buffer->Build(10);
  auto *buffer_data = (uint8_t *)(buffer->MutableData());
  for (uint8_t i = 0; i < 10; ++i) {
    buffer_data[i] = i;
  }
  stream_io->Send("input1", buffer);

  std::shared_ptr<Buffer> out_buffer;
  stream_io->Recv("output1", out_buffer);
  std::string meta;
  out_buffer->Get("test_meta", meta);
  EXPECT_EQ(meta, "test_meta");
  const auto *data = (const uint8_t *)(out_buffer->ConstData());
  for (uint8_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data[i], i + 1);
  }
}

class CustomFlowUnit : public FlowUnit {
 public:
  Status Process(std::shared_ptr<DataContext> data_ctx) override {
    auto in = data_ctx->Input("in1");
    EXPECT_EQ(in->Size(), 1);
    auto in_buffer = in->Front();
    auto out = data_ctx->Output("out1");
    out->Build({1});
    (*((uint8_t *)out->MutableData())) =
        (*((const uint8_t *)in_buffer->ConstData()));
    return STATUS_OK;
  }
};

class CustomFlowUnitBuilder : public FlowUnitBuilder {
 public:
  void Probe(std::shared_ptr<FlowUnitDesc> &desc) override {
    desc->SetFlowUnitType("cpu");
    desc->SetFlowUnitName("custom_flowunit");
    desc->AddFlowUnitInput({"in1"});
    desc->AddFlowUnitOutput({"out1"});
  }

  std::shared_ptr<FlowUnit> Build() override {
    return std::make_shared<CustomFlowUnit>();
  }
};

TEST_F(FlowGraphTest, RegisterFlowUnitTest) {
  auto input1 = graph_desc_->AddInput("input1");
  auto node1 = graph_desc_->AddNode("custom_flowunit", "cpu", input1);
  graph_desc_->AddOutput("output1", node1);

  auto flow = std::make_shared<Flow>();
  flow->RegisterFlowUnit(std::make_shared<CustomFlowUnitBuilder>());
  auto ret = flow->Init(graph_desc_);
  ASSERT_EQ(ret, STATUS_OK);

  ret = flow->StartRun();
  ASSERT_EQ(ret, STATUS_OK);

  auto stream_io = flow->CreateStreamIO();
  auto buffer = stream_io->CreateBuffer();
  buffer->Build(1);
  auto *buffer_ptr = (uint8_t *)(buffer->MutableData());
  (*buffer_ptr) = 123;
  stream_io->Send("input1", buffer);

  auto out_buffer = stream_io->Recv("output1", 0);
  ASSERT_NE(out_buffer, nullptr);
  const auto *ptr = (const uint8_t *)out_buffer->ConstData();
  EXPECT_EQ((*ptr), 123);
}

}  // namespace modelbox
