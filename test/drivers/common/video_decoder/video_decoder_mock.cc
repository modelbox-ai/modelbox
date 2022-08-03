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

#include "video_decoder_mock.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace videodecoder {

static modelbox::Status StartFlowUnitOpenFunc(
    const std::shared_ptr<modelbox::Configuration>& flow_option,
    const std::shared_ptr<modelbox::MockFlowUnit>& mock_flowunit) {
  for (uint32_t i = 0; i < 2; i++) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
      return modelbox::STATUS_FAULT;
    }

    auto source_url = std::string();
    if (i == 0) {
      source_url = std::string(TEST_ASSETS) +
                   "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4";
    } else {
      source_url = std::string(TEST_ASSETS) +
                   "/video/avc1_5s_480x320_24fps_yuv420_8bit.mp4";
    }

    auto output_buf = ext_data->CreateBufferList();
    output_buf->BuildFromHost({source_url.size()}, (void*)source_url.data(),
                              source_url.size());
    if (i == 0) {
      // Test demuxer url in output meta
      auto data_meta = std::make_shared<modelbox::DataMeta>();
      data_meta->SetMeta("source_url",
                         std::make_shared<std::string>(source_url));
      ext_data->SetOutputMeta(data_meta);
    } else {
      // Test demuxer url in output buffer
    }

    auto status = ext_data->Send(output_buf);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
      return modelbox::STATUS_FAULT;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

static void AddStartFlowUnit(std::shared_ptr<modelbox::MockFlow>& flow) {
  auto mock_desc =
      modelbox::GenerateFlowunitDesc("start_unit", {}, {"stream_meta"});
  mock_desc->SetFlowType(modelbox::STREAM);
  mock_desc->SetStreamSameCount(true);
  auto process_func =
      [=](const std::shared_ptr<modelbox::DataContext>& data_ctx,
          const std::shared_ptr<modelbox::MockFlowUnit>& mock_flowunit) {
        auto output_buffers = data_ctx->Output("stream_meta");
        auto input_buffers = data_ctx->External();
        for (const auto& buffer : *input_buffers) {
          output_buffers->PushBack(buffer);
        }
        return modelbox::STATUS_OK;
      };
  auto mock_functions = std::make_shared<modelbox::MockFunctionCollection>();
  mock_functions->RegisterOpenFunc(StartFlowUnitOpenFunc);
  mock_functions->RegisterProcessFunc(process_func);
  flow->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                        TEST_DRIVER_DIR);
}

static void CheckVideoFrame(
    const std::shared_ptr<modelbox::Buffer>& frame_buffer,
    const std::shared_ptr<int64_t>& index_counter) {
  int64_t index = 0;
  int32_t width = 0;
  int32_t height = 0;
  int32_t rate_num = 0;
  int32_t rate_den = 0;
  int64_t duration = 0;
  bool eos = false;
  int64_t timestamp;
  frame_buffer->Get("index", index);
  frame_buffer->Get("width", width);
  frame_buffer->Get("height", height);
  frame_buffer->Get("rate_num", rate_num);
  frame_buffer->Get("rate_den", rate_den);
  frame_buffer->Get("duration", duration);
  frame_buffer->Get("eos", eos);
  frame_buffer->Get("timestamp", timestamp);

  EXPECT_EQ(index, *index_counter);
  *index_counter = *index_counter + 1;
  EXPECT_EQ(width, 480);
  EXPECT_EQ(height, 320);
  EXPECT_EQ(rate_num, 24);
  EXPECT_EQ(rate_den, 1);
  if (index < 119) {
    EXPECT_FALSE(eos);
  } else {
    EXPECT_TRUE(eos);
  }

  EXPECT_EQ(duration, 5);
  if (index == 0) {
    EXPECT_EQ(timestamp, 0);
  } else if (index == 119) {
    EXPECT_EQ(timestamp, 4958);
  }
}

static void AddReadFrameFlowUnit(std::shared_ptr<modelbox::MockFlow>& flow,
                                 bool is_stream) {
  auto mock_desc =
      modelbox::GenerateFlowunitDesc("read_frame", {"frame_info"}, {});
  mock_desc->SetFlowType(modelbox::STREAM);
  auto data_pre_func =
      [&](const std::shared_ptr<modelbox::DataContext>& data_ctx,
          const std::shared_ptr<modelbox::MockFlowUnit>& mock_flowunit) {
        MBLOG_INFO << "read_frame DataPre";
        auto index_counter = std::make_shared<int64_t>(0);
        data_ctx->SetPrivate("index", index_counter);
        return modelbox::STATUS_OK;
      };
  auto process_func =
      [=](const std::shared_ptr<modelbox::DataContext>& op_ctx,
          const std::shared_ptr<modelbox::MockFlowUnit>& mock_flowunit) {
        auto index_counter =
            std::static_pointer_cast<int64_t>(op_ctx->GetPrivate("index"));

        auto frame_buffer_list = op_ctx->Input("frame_info");
        EXPECT_NE(frame_buffer_list, nullptr);
        if (is_stream) {
          return modelbox::STATUS_OK;
        }
        for (size_t i = 0; i < frame_buffer_list->Size(); ++i) {
          auto frame_buffer = frame_buffer_list->At(i);
          if (frame_buffer->GetBytes() == 0) {
            continue;
          }

          CheckVideoFrame(frame_buffer, index_counter);
        }

        return modelbox::STATUS_OK;
      };

  auto mock_functions = std::make_shared<modelbox::MockFunctionCollection>();
  mock_functions->RegisterDataPreFunc(data_pre_func);
  mock_functions->RegisterProcessFunc(process_func);
  flow->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                        TEST_DRIVER_DIR);
}

modelbox::Status AddMockFlowUnit(std::shared_ptr<modelbox::MockFlow>& flow,
                                 bool is_stream) {
  AddStartFlowUnit(flow);
  AddReadFrameFlowUnit(flow, is_stream);
  return modelbox::STATUS_SUCCESS;
}

std::string GetTomlConfig(const std::string& device,
                          const std::string& pix_fmt) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;
  std::string toml_content =
      R"(
      [driver]
      skip-default = true
      dir=[")" +
      test_lib_dir + "\",\"" + test_data_dir + "\"]\n    " +
      R"([graph]
      graphconf = '''digraph demo {
            start_unit[type=flowunit, flowunit=start_unit, device=cpu, deviceid=0, label="<stream_meta>"]
            videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
            videodecoder[type=flowunit, flowunit=video_decoder, device=)" +
      device +
      R"(, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=)" +
      pix_fmt + R"(]
            read_frame[type=flowunit, flowunit=read_frame, device=cpu, deviceid=0, label="<frame_info>"]
            start_unit:stream_meta -> videodemuxer:in_video_url
            videodemuxer:out_video_packet -> videodecoder:in_video_packet
            videodecoder:out_video_frame -> read_frame:frame_info
          }'''
      format = "graphviz"
    )";
  return toml_content;
}
}  // namespace videodecoder