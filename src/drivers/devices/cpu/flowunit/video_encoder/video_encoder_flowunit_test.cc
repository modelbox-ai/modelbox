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
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class VideoEncoderFlowUnitTest : public testing::Test {
 public:
  VideoEncoderFlowUnitTest() = default;

 protected:
  void SetUp() override{};

  void TearDown() override{};

 public:
  std::shared_ptr<MockFlow> flow_;

  void StartFlow(std::string& toml_content, uint64_t millisecond);

 private:
  Status AddMockFlowUnit();
};

void VideoEncoderFlowUnitTest::StartFlow(std::string& toml_content,
                                         const uint64_t millisecond) {
  flow_ = std::make_shared<MockFlow>();
  auto ret = AddMockFlowUnit();
  EXPECT_EQ(ret, STATUS_SUCCESS);

  ret = flow_->BuildAndRun("VideoEncoder", toml_content, millisecond);
  EXPECT_EQ(ret, STATUS_STOP);
}

Status VideoEncoderFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("encoder_start_unit", {}, {"stream_meta"});
    auto open_func =
        [=](const std::shared_ptr<modelbox::Configuration>& flow_option,
            std::shared_ptr<MockFlowUnit> mock_flowunit) {
          auto ext_data = mock_flowunit->CreateExternalData();
          EXPECT_NE(ext_data, nullptr);
          auto buffer_list = ext_data->CreateBufferList();
          buffer_list->Build({1});
          auto status = ext_data->Send(buffer_list);
          EXPECT_EQ(status, STATUS_SUCCESS);
          status = ext_data->Close();
          EXPECT_EQ(status, STATUS_SUCCESS);
          return modelbox::STATUS_OK;
        };
    auto data_pre_func = [&](std::shared_ptr<DataContext> data_ctx,
                             std::shared_ptr<MockFlowUnit> mock_flowunit) {
      MBLOG_INFO << "stream_meta  "
                 << "DataPre";
      auto test_meta = std::make_shared<std::string>("test");
      auto data_meta = std::make_shared<DataMeta>();
      data_meta->SetMeta("test", test_meta);
      data_ctx->SetOutputMeta("stream_meta", data_meta);
      return modelbox::STATUS_OK;
    };
    auto process_func = [=](std::shared_ptr<DataContext> data_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
      auto output_buf = data_ctx->Output("stream_meta");
      std::vector<size_t> shape(1, 1);
      output_buf->Build(shape);

      return modelbox::STATUS_OK;
    };
    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterOpenFunc(open_func);
    mock_functions->RegisterDataPreFunc(data_pre_func);
    mock_functions->RegisterProcessFunc(process_func);
    flow_->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                           TEST_DRIVER_DIR);
  }
  {
    auto mock_desc = GenerateFlowunitDesc("encoder_image_produce",
                                          {"stream_meta"}, {"frame_info"});
    mock_desc->SetOutputType(EXPAND);
    auto process_func = [=](std::shared_ptr<DataContext> data_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
      std::string img_path;
      static int64_t frame_index = 0;
      if ((frame_index / 24) % 2 == 0) {
        img_path =
            std::string(TEST_ASSETS) + "/video/rgb_460800_480x320_a.data";
      } else {
        img_path =
            std::string(TEST_ASSETS) + "/video/rgb_460800_480x320_b.data";
      }

      std::ifstream img_file(img_path);
      if (!img_file.is_open()) {
        MBLOG_ERROR << "Open failed, path " << img_path;
        return STATUS_FAULT;
      }

      size_t file_size = 460800;
      auto output_buff_list = data_ctx->Output("frame_info");
      std::vector<size_t> shape(1, file_size);
      output_buff_list->Build(shape);
      auto output_buff = output_buff_list->At(0);
      auto* ptr = (char*)output_buff->MutableData();
      img_file.read(ptr, file_size);
      output_buff->Set("width", 480);
      output_buff->Set("height", 320);
      output_buff->Set("rate_num", 24);
      output_buff->Set("rate_den", 1);
      output_buff->Set("pix_fmt", std::string("rgb"));
      output_buff->Set("index", frame_index);

      if (frame_index == 1339) {  // 60S
        return modelbox::STATUS_STOP;
      }

      ++frame_index;
      auto event = std::make_shared<FlowUnitEvent>();
      data_ctx->SendEvent(event);
      return modelbox::STATUS_CONTINUE;
    };
    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterProcessFunc(process_func);
    flow_->AddFlowUnitDesc(mock_desc, mock_functions->GenerateCreateFunc(),
                           TEST_DRIVER_DIR);
  }

  return STATUS_SUCCESS;
}

TEST_F(VideoEncoderFlowUnitTest, InitUnit) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;
  auto ret = system("nc localhost 554 -z");
  if (errno != 0 || ret != 0) {
    GTEST_SKIP();
  }

  std::string dest_url = "rtsp://localhost/test_" + std::to_string(rand());
  std::string toml_content = R"(
      [driver]
      skip-default = true
      dir=[")" + test_lib_dir +
                             "\",\"" + test_data_dir + "\"]\n    " +
                             R"([graph]
      graphconf = '''digraph demo {
            encoder_start_unit[type=flowunit, flowunit=encoder_start_unit, device=cpu, deviceid=0, label="<stream_meta>"]
            encoder_image_produce[type=flowunit, flowunit=encoder_image_produce, device=cpu, deviceid=0, label="<stream_meta> | <frame_info>"]
            videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size_frame_info=16, default_dest_url=")" +
                             dest_url + R"("]
            encoder_start_unit:stream_meta -> encoder_image_produce:stream_meta
            encoder_image_produce:frame_info -> videoencoder:in_video_frame
          }'''
      format = "graphviz"
    )";
  StartFlow(toml_content, 1000 * 1000);
}

}  // namespace modelbox