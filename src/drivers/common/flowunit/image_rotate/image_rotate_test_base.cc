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

#include "image_rotate_test_base.h"

#include <opencv2/opencv.hpp>

namespace modelbox {

std::shared_ptr<MockFlow> ImageRotateFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status ImageRotateFlowUnitTest::AddMockFlowUnit() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();

  {
    auto mock_desc = GenerateFlowunitDesc("test_0_1_rotate", {}, {"out_1"});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(16);
    auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                         const std::shared_ptr<MockFlowUnit>& mock_flowunit) {
      auto ext_data = mock_flowunit->CreateExternalData();
      std::string gimg_path = std::string(TEST_ASSETS) + "/test.jpg";

      auto output_buf = ext_data->CreateBufferList();
      modelbox::TensorList output_tensor_list(output_buf);
      output_tensor_list.BuildFromHost<uchar>({1, {gimg_path.size() + 1}},
                                              (void*)gimg_path.data(),
                                              gimg_path.size() + 1);

      auto status = ext_data->Send(output_buf);
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
      MBLOG_INFO << "test_0_1_rotate process";

      auto external = data_ctx->External();
      std::string input_path = std::string((char*)(*external)[0]->ConstData());
      cv::Mat input_img = cv::imread(input_path);

      MBLOG_INFO << "gimage col " << input_img.cols << "  grow "
                 << input_img.rows << " gchannel:" << input_img.channels();

      auto output_bufs = data_ctx->Output("out_1");

      for (int &i : test_rotate_angle_) {
        auto output_buffer =
            std::make_shared<modelbox::Buffer>(mock_flowunit->GetBindDevice());
        output_buffer->Build(input_img.total() * input_img.elemSize());
        auto *output_data = static_cast<uchar *>(output_buffer->MutableData());
        auto ret =
            memcpy_s(output_data, output_buffer->GetBytes(), input_img.data,
                     input_img.total() * input_img.elemSize());
        if (ret != EOK) {
          MBLOG_ERROR << "Cpu memcpy failed, ret " << ret;
          return modelbox::STATUS_FAULT;
        }
        output_buffer->Set("width", (int32_t)input_img.cols);
        output_buffer->Set("height", (int32_t)input_img.rows);
        output_buffer->Set("layout", std::string("hwc"));
        output_buffer->Set("type", ModelBoxDataType::MODELBOX_UINT8);

        output_buffer->Set("rotate_angle", i);
        output_bufs->PushBack(output_buffer);
      }

      return modelbox::STATUS_OK;
    };

    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterOpenFunc(open_func);
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  {
    auto mock_desc = GenerateFlowunitDesc("test_1_0_rotate", {"in_origin", "in_rotate"}, {});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(16);

    auto process_func =
        [=](const std::shared_ptr<DataContext>& data_ctx,
            const std::shared_ptr<MockFlowUnit>& mock_flowunit) -> Status {
      MBLOG_INFO << "test_1_0_rotate process";
      auto origin_buf = data_ctx->Input("in_origin");
      auto rotate_buf = data_ctx->Input("in_rotate");
      int32_t width = 0;
      int32_t height = 0;
      int32_t channels = 0;
      int32_t rotate_angle = 0;

      for (size_t i = 0; i < rotate_buf->Size(); ++i) {
        rotate_buf->At(i)->Get("width", width);
        rotate_buf->At(i)->Get("height", height);
        rotate_buf->At(i)->Get("channel", channels);
        origin_buf->At(i)->Get("rotate_angle", rotate_angle);
        const auto *input_data =
            static_cast<const uchar *>(rotate_buf->ConstBufferData(i));

        cv::Mat img_data(cv::Size(width, height), CV_8UC3);
        auto ret =
            memcpy_s(img_data.data, img_data.total() * img_data.elemSize(),
                     input_data, rotate_buf->At(i)->GetBytes());
        if (ret != EOK) {
          MBLOG_ERROR << "Cpu memcpy failed, ret " << ret;
          return modelbox::STATUS_FAULT;
        }

        MBLOG_INFO << "output image col " << img_data.cols << "  row "
                   << img_data.rows << " channel:" << img_data.channels();

        std::string name = std::string(TEST_DATA_DIR) + "/rotate_result_" +
                           std::to_string(rotate_angle) + ".jpg";

        cv::imwrite(name, img_data);
      }

      return modelbox::STATUS_STOP;
    };

    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }
  return STATUS_OK;
}

}  // namespace modelbox