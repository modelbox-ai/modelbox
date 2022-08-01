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


#include <dsmi_common_interface.h>

#include <fstream>
#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class DvppVideoDecoderFlowUnitTest : public testing::Test {
 public:
  DvppVideoDecoderFlowUnitTest() = default;

 protected:
  void SetUp() override {
    // Test ascend runtime
    int32_t count = 0;
    auto dsmi_ret = dsmi_get_device_count(&count);
    if (dsmi_ret != 0) {
      MBLOG_INFO << "no ascend device, skip test suit";
      GTEST_SKIP();
    }
  };

  void TearDown() override{};

 public:
  std::shared_ptr<MockFlow> flow_;

  void StartFlow(const std::string& graph, uint64_t millisecond);

 private:
  Status AddMockFlowUnit();
};

void DvppVideoDecoderFlowUnitTest::StartFlow(const std::string& graph,
                                             const uint64_t millisecond) {
  flow_ = std::make_shared<MockFlow>();
  auto ret = AddMockFlowUnit();
  EXPECT_EQ(ret, STATUS_SUCCESS);
  flow_->Init(false);

  flow_->BuildAndRun("DvppVideoDecoder", graph, millisecond);
}

std::string GetGraphToml(const std::string& device,
                         const std::string& pix_fmt) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;
  std::string read_frame = "read_frame_acl";
  std::string start_unit = "start_unit_acl";

  std::string toml_content =
      R"(
      [log]
      level = "DEBUG"
      [driver]
      skip-default = true
      dir=[")" +
      test_lib_dir + "\",\"" + test_data_dir + "\"]\n    " +
      R"([graph]
      thread-num = 16
      max-thread-num = 100
      graphconf = '''digraph demo {
            start_unit_acl[type=flowunit, flowunit=start_unit_acl, device=cpu, deviceid=0, label="<stream_meta>"]
            videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>", queue_size = 16]
            videodecoder[type=flowunit, flowunit=video_decoder, device=)" +
      device +
      R"(, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=)" +
      pix_fmt +
      ", queue_size = 16]\n            "
      R"(read_frame_acl[type=flowunit, flowunit=read_frame_acl, device=cpu, deviceid=0, label="<frame_info>", queue_size = 16]
            start_unit_acl:stream_meta -> videodemuxer:in_video_url
            videodemuxer:out_video_packet -> videodecoder:in_video_packet
            videodecoder:out_video_frame -> read_frame_acl:frame_info
          }'''
      format = "graphviz"
    )";
  return toml_content;
}

Status DvppVideoDecoderFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("start_unit_acl", {}, {"stream_meta"});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetStreamSameCount(true);
    auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                         std::shared_ptr<MockFlowUnit> mock_flowunit) {
      for (uint32_t i = 0; i < 16; i++) {
        auto ext_data = mock_flowunit->CreateExternalData();
        if (!ext_data) {
          MBLOG_ERROR << "can not get external data.";
          return STATUS_FAULT;
        }

        auto source_url = std::string(TEST_ASSETS) +
                          "/video/avc1_5s_480x320_24fps_yuv420_8bit.mp4";

        auto output_buf = ext_data->CreateBufferList();
        modelbox::TensorList output_tensor_list(output_buf);
        output_tensor_list.BuildFromHost<unsigned char>(
            {1, {source_url.size() + 1}}, (void*)source_url.data(),
            source_url.size() + 1);

        auto data_meta = std::make_shared<DataMeta>();
        data_meta->SetMeta("source_url",
                           std::make_shared<std::string>(source_url));
        ext_data->SetOutputMeta(data_meta);

        auto status = ext_data->Send(output_buf);
        if (!status) {
          MBLOG_ERROR << "external data send buffer list failed:" << status;
          return STATUS_FAULT;
        }

        status = ext_data->Close();
        if (!status) {
          MBLOG_ERROR << "external data close failed:" << status;
          return STATUS_FAULT;
        }
      }

      return modelbox::STATUS_SUCCESS;
    };

    auto process_func =
        [=](std::shared_ptr<DataContext> data_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto output_buf = data_ctx->Output("stream_meta");
      std::vector<size_t> shape(1, 1);
      output_buf->Build(shape);
      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    mock_funcitons->RegisterOpenFunc(open_func);
    flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                           TEST_DRIVER_DIR);
  }

  {
    auto mock_desc = GenerateFlowunitDesc("read_frame_acl", {"frame_info"}, {});
    mock_desc->SetFlowType(STREAM);
    auto data_pre_func =
        [=](std::shared_ptr<DataContext> data_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto index_counter = std::make_shared<int64_t>(0);
      data_ctx->SetPrivate("index", index_counter);
      return modelbox::STATUS_OK;
    };

    auto process_func =
        [=](std::shared_ptr<DataContext> op_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto index_counter =
          std::static_pointer_cast<int64_t>(op_ctx->GetPrivate("index"));

      auto frame_buffer_list = op_ctx->Input("frame_info");
      EXPECT_NE(frame_buffer_list, nullptr);
      for (size_t i = 0; i < frame_buffer_list->Size(); ++i) {
        auto frame_buffer = frame_buffer_list->At(i);
        if (frame_buffer->GetBytes() == 0) {
          continue;
        }

        int64_t index;
        int32_t width;
        int32_t height;
        int32_t rate_num;
        int32_t rate_den;
        bool eos;
        frame_buffer->Get("index", index);
        frame_buffer->Get("width", width);
        frame_buffer->Get("height", height);
        frame_buffer->Get("rate_num", rate_num);
        frame_buffer->Get("rate_den", rate_den);
        frame_buffer->Get("eos", eos);
        EXPECT_EQ(index, *index_counter);
        *index_counter = *index_counter + 1;
        EXPECT_EQ(width, 480);
        EXPECT_EQ(height, 320);
        EXPECT_EQ(rate_num, 24);
        EXPECT_EQ(rate_den, 1);
        EXPECT_FALSE(eos);
      }
      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    mock_funcitons->RegisterDataPreFunc(data_pre_func);
    flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                           TEST_DRIVER_DIR);
  }

  return STATUS_SUCCESS;
}

TEST_F(DvppVideoDecoderFlowUnitTest, ascendDecoderRgbTest) {
  auto toml_content = GetGraphToml("ascend", "nv12");
  StartFlow(toml_content, 10 * 1000);
}

}  // namespace modelbox