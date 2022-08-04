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

#include <cuda_runtime.h>
#include <securec.h>

#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class NppiCropFlowUnitTest : public testing::Test {
 public:
  NppiCropFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
      MBLOG_INFO << "no cuda device, skip test suit";
      GTEST_SKIP();
    }

    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> NppiCropFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status NppiCropFlowUnitTest::AddMockFlowUnit() {
  typedef struct RoiBox {
    int32_t x, y, width, height;
  } RoiBox;

  {
    auto mock_desc = GenerateFlowunitDesc("c3r_test_0_1_nppi_crop", {},
                                          {"Out_img", "Out_box"});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(16);
    auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                         const std::shared_ptr<MockFlowUnit>& mock_flowunit) {
      std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
      mock_flowunit_wp = mock_flowunit;
      auto spt = mock_flowunit_wp.lock();
      auto ext_data = spt->CreateExternalData();
      if (!ext_data) {
        MBLOG_ERROR << "can not get external data.";
      }

      auto buffer_list = ext_data->CreateBufferList();
      buffer_list->Build({10 * sizeof(int)});
      auto* data = (int*)buffer_list->MutableData();
      for (size_t i = 0; i < 10; i++) {
        data[i] = i;
      }

      auto status = ext_data->Send(buffer_list);
      if (!status) {
        MBLOG_ERROR << "external data send buffer list failed:" << status;
        return status;
      }

      status = ext_data->Close();
      if (!status) {
        MBLOG_ERROR << "external data close failed:" << status;
        return status;
      }

      return modelbox::STATUS_OK;
    };

    auto process_func =
        [=](const std::shared_ptr<DataContext>& data_ctx,
            const std::shared_ptr<MockFlowUnit>& mock_flowunit) -> Status {
      MBLOG_INFO << "test_0_1_nppi_crop process";

      auto output_img_bufs = data_ctx->Output("Out_img");

      uint32_t batch_size = 10;

      std::string img_path = std::string(TEST_ASSETS) + "/test.jpg";
      cv::Mat img_data = cv::imread(img_path);
      MBLOG_INFO << "image col " << img_data.cols << "  row " << img_data.rows
                 << " channel " << img_data.channels();
      std::vector<size_t> img_shape_vector(
          batch_size, img_data.total() * img_data.elemSize());

      output_img_bufs->Build(img_shape_vector);

      for (size_t i = 0; i < batch_size; ++i) {
        std::string img_path = std::string(TEST_ASSETS) + "/test.jpg";
        cv::Mat img_data = cv::imread(img_path);
        int32_t cols = img_data.cols;
        int32_t rows = img_data.rows;
        int32_t channels = img_data.channels();
        output_img_bufs->At(i)->Set("width", cols);
        output_img_bufs->At(i)->Set("height", rows);
        output_img_bufs->At(i)->Set("channel", channels);
        auto* output_img_data =
            static_cast<uchar*>(output_img_bufs->MutableBufferData(i));
        memcpy_s(output_img_data, output_img_bufs->At(i)->GetBytes(),
                 img_data.data, img_data.total() * img_data.elemSize());
      }

      auto output_box_bufs = data_ctx->Output("Out_box");

      std::vector<size_t> box_shape_vector(batch_size, sizeof(RoiBox));

      output_box_bufs->Build(box_shape_vector);

      for (size_t i = 0; i < 5; ++i) {
        auto* output_box1_data = output_box_bufs->MutableBufferData(2 * i);
        std::shared_ptr<RoiBox> bbox1 = std::make_shared<RoiBox>();
        bbox1->width = 100;
        bbox1->height = 110;
        bbox1->x = 30;
        bbox1->y = 100;
        memcpy_s(output_box1_data, sizeof(RoiBox), bbox1.get(), sizeof(RoiBox));

        auto* output_box2_data = output_box_bufs->MutableBufferData(2 * i + 1);
        std::shared_ptr<RoiBox> bbox2 = std::make_shared<RoiBox>();
        bbox2->width = 50;
        bbox2->height = 90;
        bbox2->x = 60;
        bbox2->y = 130;
        memcpy_s(output_box2_data, sizeof(RoiBox), bbox2.get(), sizeof(RoiBox));
      }
      MBLOG_INFO << "nppi_test_0_1 gen data finish";

      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterOpenFunc(open_func);
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  {
    auto mock_desc =
        GenerateFlowunitDesc("c3r_test_1_0_nppi_crop", {"In_img"}, {});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(16);
    auto process_func =
        [=](const std::shared_ptr<DataContext>& op_ctx,
            const std::shared_ptr<MockFlowUnit>& mock_flowunit) -> Status {
      MBLOG_INFO << "c3r_test_1_0_nppi_crop process";

      auto input_buf = op_ctx->Input("In_img");
      if (input_buf->Size() <= 0) {
        auto errMsg =
            "input images size is " + std::to_string(input_buf->Size());
        MBLOG_ERROR << errMsg;
      }

      for (size_t i = 0; i < input_buf->Size(); ++i) {
        int32_t width = 0;
        int32_t height = 0;
        int32_t channels = 0;

        bool exists = false;

        exists = input_buf->At(i)->Get("width", width);
        if (!exists) {
          MBLOG_ERROR << "meta don't have key width";
        }
        exists = input_buf->At(i)->Get("height", height);
        if (!exists) {
          MBLOG_ERROR << "meta don't have key height";
        }
        exists = input_buf->At(i)->Get("channel", channels);
        if (!exists) {
          MBLOG_ERROR << "meta don't have key channel";
        }

        const auto* input_data =
            static_cast<const uchar*>(input_buf->ConstBufferData(i));

        cv::Mat img_data(cv::Size(width, height), CV_8UC3);
        memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                 input_data, input_buf->At(i)->GetBytes());
        std::string name =
            std::string(TEST_DATA_DIR) + "/test" + std::to_string(i) + ".jpg";
        cv::imwrite(name, img_data);
      }

      MBLOG_INFO << "c3r_test_1_0_nppi_crop process data finish";

      return modelbox::STATUS_STOP;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }
  return STATUS_OK;
}

TEST_F(NppiCropFlowUnitTest, TestC3r) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          c3r_test_0_1_nppi_crop[type=flowunit, flowunit=c3r_test_0_1_nppi_crop,
          device=cpu, deviceid=0, label="<Out_img> | <Out_box>",batch_size=10]
          nppi_crop[type=flowunit, flowunit=crop, label="<in_image> | <in_region> | <out_image>",batch_size=10]
          c3r_test_1_0_nppi_crop[type=flowunit, flowunit=c3r_test_1_0_nppi_crop,
          device=cpu, deviceid=0, label="<In_img>",batch_size=10]

          c3r_test_0_1_nppi_crop:Out_img -> nppi_crop:in_image
          c3r_test_0_1_nppi_crop:Out_box -> nppi_crop:in_region
          nppi_crop:out_image -> c3r_test_1_0_nppi_crop:In_img
        }'''
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("TestC3r", toml_content, 3 * 1000);
  EXPECT_EQ(ret, STATUS_STOP);

  std::vector<std::string> filePath;
  ListFiles(std::string(TEST_DATA_DIR), "*", &filePath);
  for (auto& elem : filePath) {
    MBLOG_INFO << "filePath: " << elem;
  }

  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 2; j++) {
      std::string expected_file_path = std::string(TEST_ASSETS) +
                                       "/crop_result_" + std::to_string(j) +
                                       ".jpg";
      cv::Mat expected_img = cv::imread(expected_file_path);

      std::string crop_result_file_path = std::string(TEST_DATA_DIR) + "/test" +
                                          std::to_string(2 * i + j) + ".jpg";
      cv::Mat crop_result_img = cv::imread(crop_result_file_path);

      int result_data_size =
          crop_result_img.total() * crop_result_img.elemSize();
      int expected_data_size = expected_img.total() * expected_img.elemSize();
      EXPECT_EQ(result_data_size, expected_data_size);

      int ret =
          memcmp(crop_result_img.data, expected_img.data, result_data_size);
      EXPECT_EQ(ret, 0);

      auto rmret = remove(crop_result_file_path.c_str());
      EXPECT_EQ(rmret, 0);
    }
  }
}

}  // namespace modelbox