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


#include "car_flow.h"

#include <securec.h>

#include <functional>
#include <future>
#include <random>
#include <thread>
#include <cuda_runtime.h>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;

namespace modelbox {

class CarFlowTest : public testing::Test {
 public:
  CarFlowTest() : car_flow_(std::make_shared<CarFlow>()) {}

  std::shared_ptr<CarFlow> GetCarFlow();

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  }

  virtual void TearDown() { car_flow_->Clear(); };

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<CarFlow> car_flow_;
};

std::shared_ptr<CarFlow> CarFlowTest::GetCarFlow() { return car_flow_; }

Status CarFlowTest::AddMockFlowUnit() {
  auto ctl_ = car_flow_->GetMockFlowCtl();

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("start_unit");
    desc_flowunit.SetDescription("start unit in test");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-start_unit.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("start_unit");
    mock_flowunit_desc->AddFlowUnitOutput(
        modelbox::FlowUnitOutput("stream_meta"));
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              auto spt = mock_flowunit_wp.lock();
              for (uint32_t i = 0; i < 1; i++) {
                auto ext_data = spt->CreateExternalData();
                if (!ext_data) {
                  MBLOG_ERROR << "can not get external data.";
                }
                auto source_url = std::string();
                if (i == 0) {
                  source_url = std::string(TEST_ASSETS) +
                               "/car_detection/test_video.mp4";
                } else {
                  source_url = std::string(TEST_ASSETS) +
                               "/car_detection/test_video.mp4";
                }

                auto output_buf = ext_data->CreateBufferList();
                modelbox::TensorList output_tensor_list(output_buf);
                output_tensor_list.BuildFromHost<unsigned char>(
                    {1, {source_url.size() + 1}}, (void*)source_url.data(),
                    source_url.size() + 1);

                auto status = ext_data->Send(output_buf);
                if (!status) {
                  MBLOG_ERROR << "external data send buffer list failed:"
                              << status;
                }

                status = ext_data->Close();
                if (!status) {
                  MBLOG_ERROR << "external data close failed:" << status;
                }
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              auto output_buf = data_ctx->Output("stream_meta");
              std::vector<size_t> shape(1, 1);
              output_buf->Build(shape);

              auto external = data_ctx->External();
              auto source_url = std::make_shared<std::string>(
                  (char*)(*external)[0]->ConstData());

              auto data_meta = std::make_shared<DataMeta>();
              data_meta->SetMeta("source_url", source_url);

              data_ctx->SetOutputMeta("stream_meta", data_meta);
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("start_unit", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("color_transpose");
    desc_flowunit.SetDescription("the test in 1 out 0");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-color_transpose.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("color_transpose");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("in_image"));
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image"));
    mock_flowunit_desc->SetFlowType(NORMAL);
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "color_transpose "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "color_transpose "
                         << "DataPost";
              return modelbox::STATUS_STOP;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> op_ctx) {
              MBLOG_INFO << "test color_transpose process";
              auto input_buf = op_ctx->Input("in_image");
              auto output_buf = op_ctx->Output("out_image");

              std::vector<size_t> shape_vector;
              for (size_t i = 0; i < input_buf->Size(); ++i) {
                shape_vector.push_back(input_buf->At(i)->GetBytes());
              }

              output_buf->Build(shape_vector);
              int32_t width, height, channel;
              std::string pix_fmt;
              modelbox::ModelBoxDataType type = MODELBOX_TYPE_INVALID;
              input_buf->At(0)->Get("width", width);
              input_buf->At(0)->Get("height", height);
              input_buf->At(0)->Get("channel", channel);
              input_buf->At(0)->Get("pix_fmt", pix_fmt);
              input_buf->At(0)->Get("type", type);
              size_t elem_size = width * height;

              auto input_data =
                  static_cast<const u_char*>(input_buf->ConstData());
              auto output_data =
                  static_cast<u_char*>(output_buf->MutableData());
              for (size_t i = 0; i < (size_t)channel; ++i) {
                for (size_t j = 0; j < elem_size; ++j) {
                  output_data[i * elem_size + j] = input_data[j * channel + i];
                }
              }

              output_buf->Set("width", width);
              output_buf->Set("height", height);
              output_buf->Set("channel", channel);
              output_buf->Set("pix_fmt", pix_fmt);
              output_buf->Set("type", type);

              MBLOG_DEBUG << "color_transpose process data finish";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("color_transpose", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("free");
    desc_flowunit.SetDescription("free in test");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cpu-free.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("free");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("in_data"));
    mock_flowunit_desc->SetFlowType(NORMAL);
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    static std::atomic<int64_t> run_count(0);
    static std::atomic<bool> is_print(false);

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              auto input_buf = data_ctx->Input("in_data");

              static auto begin_time = GetTickCount();
              static std::atomic<uint64_t> print_time{GetTickCount()};

              run_count += input_buf->Size();

              auto end_time = GetTickCount();
              if (end_time - print_time > 1000) {
                auto expected = false;
                if (is_print.compare_exchange_weak(expected, true)) {
                  MBLOG_INFO << "Average throughput: "
                             << (run_count * 1000) / (end_time - begin_time)
                             << "/s";
                  is_print = false;
                  print_time = GetTickCount();
                }
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("free", "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cuda");
    desc_flowunit.SetName("free");
    desc_flowunit.SetDescription("free in test");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_DRIVER_DIR) + "/libmodelbox-unit-cuda-free.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("free");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("in_data"));
    mock_flowunit_desc->SetFlowType(NORMAL);
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    static std::atomic<int64_t> run_count(0);
    static std::atomic<bool> is_print(false);

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPre";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_meta  "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              auto input_buf = data_ctx->Input("in_data");

              static auto begin_time = GetTickCount();
              static std::atomic<uint64_t> print_time{GetTickCount()};

              run_count += input_buf->Size();

              auto end_time = GetTickCount();
              if (end_time - print_time > 1000) {
                auto expected = false;
                if (is_print.compare_exchange_weak(expected, true)) {
                  MBLOG_INFO << "Average throughput: "
                             << (run_count * 1000) / (end_time - begin_time)
                             << "/s";
                  is_print = false;
                  print_time = GetTickCount();
                }
              }

              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit("free", "cuda", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_SUCCESS;
}

TEST_F(CarFlowTest, DISABLED_CarDetection) {
  MBLOG_INFO << "car detection get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;
  std::string toml_content = R"(
    [log]
    level = "DEBUG"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_data_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {

          video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, label="<out_video_url>", source_url="test_video.mp4"]                                           
          videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
          videodecoder[type=flowunit, flowunit=video_decoder, device=cuda, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16]
          cv_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <out_image>", width=800, height=480, method="inter_nearest", batch_size=5, queue_size = 16]
          color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", queue_size = 16]
          normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<in_data> | <out_data>", normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16]
          day_inference[type=flowunit, flowunit=day_inference, device=cuda, deviceid=0, label="<data> | <layer15_conv> | <layer22_conv>", queue_size = 16, batch_size = 1]
          yolobox[type=flowunit, flowunit=yolobox, device=cpu, deviceid=0, label="<layer15_conv> | <layer22_conv> | <Out_1>", queue_size = 16, batch_size = 1]
          draw_bbox[type=flowunit, flowunit=draw_bbox, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>", queue_size = 16]
          videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size=16, default_dest_url="rtsp://localhost/test", encoder="mpeg4"]

          video_input:out_video_url -> videodemuxer:in_video_url
          videodemuxer:out_video_packet -> videodecoder:in_video_packet
          videodecoder:out_video_frame -> cv_resize:in_image
          cv_resize:out_image -> color_transpose: in_image
          color_transpose: out_image -> normalize: in_data
          normalize: out_data -> day_inference:data
          day_inference:layer15_conv -> yolobox: layer15_conv
          day_inference:layer22_conv -> yolobox: layer22_conv
          yolobox: Out_1 -> draw_bbox: in_region
          videodecoder:out_video_frame -> draw_bbox: in_image
          draw_bbox:out_image -> videoencoder: in_video_frame
        }'''
    format = "graphviz"
  )";
  std::string config_file_path = std::string(TEST_DATA_DIR) + "/test.toml";
  struct stat buffer;
  if (stat(config_file_path.c_str(), &buffer) == 0) {
    remove(config_file_path.c_str());
  }
  std::ofstream ofs(config_file_path);
  EXPECT_TRUE(ofs.is_open());
  ofs.write(toml_content.data(), toml_content.size());
  ofs.flush();
  ofs.close();
  Defer {
    auto rmret = remove(config_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  };

  auto car_flow = GetCarFlow();
  auto ret = car_flow->Init(config_file_path);
  EXPECT_EQ(ret, STATUS_OK);

  ret = car_flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  car_flow->Run();
  car_flow->Wait(10000 * 1000);
}

}  // namespace modelbox
