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

#include <mock_modelbox.h>
#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"

using ::testing::_;

namespace modelbox {
class ResizeFlowUnitTest : public testing::Test {
 public:
  ResizeFlowUnitTest() : mock_modelbox_(std::make_shared<MockModelBox>()) {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown() { mock_modelbox_->Stop(); };
  std::shared_ptr<MockModelBox> GetMockModelbox() { return mock_modelbox_; }

 private:
  std::shared_ptr<MockModelBox> mock_modelbox_;
};

TEST_F(ResizeFlowUnitTest, TestCase1) {
  /*create graph config , build and run, "input1" and "output2" are virtual
    nodes used to send or receive buffer.
    you can add "input2" or "input3" when there are multiple inputs*/
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
            [log]
            level="DEBUG"
            [driver]
            skip-default=false
            dir=[")" + test_lib_dir +
                             "\"]\n    " +
                             R"([graph]
            graphconf = '''digraph demo {                                                                            
                input1[type=input]
                resize_test[type=flowunit, flowunit=resize_test, device=cpu, deviceid=0, label="<in_1> | <out_1>", image_width=128, image_height=128,batch_size=5]
                output1[type=output]                                
                input1 -> resize_test:in_1 
                resize_test:out_1 -> output1                                                                      
                }'''
            format = "graphviz"
        )";
  auto mock_modelbox = GetMockModelbox();
  auto ret = mock_modelbox->BuildAndRun("graph_name", toml_content, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  /*create buffer list and fill parmeters if you want*/
  auto ext_data = mock_modelbox->GetFlow()->CreateExternalDataMap();
  EXPECT_NE(ext_data, nullptr);
  auto buffer_list = ext_data->CreateBufferList();
  EXPECT_NE(buffer_list, nullptr);
  auto img = cv::imread(std::string(TEST_ASSETS) + "/test.jpg");
  buffer_list->Build({img.total() * img.elemSize()});
  auto buffer = buffer_list->At(0);
  buffer->Set("width", img.cols);
  buffer->Set("height", img.rows);
  buffer->Set("width_stride", img.cols * 3);
  buffer->Set("height_stride", img.rows);
  buffer->Set("pix_fmt", std::string("bgr"));
  memcpy(buffer->MutableData(), img.data, img.total() * img.elemSize());

  /*send buffer list to port "input1" ,and then transmit to next flowunit*/
  auto status = ext_data->Send("input1", buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  status = ext_data->Shutdown();
  EXPECT_EQ(status, STATUS_OK);

  /*wait for output buffer list for port "output1" */
  std::vector<std::shared_ptr<BufferList>> output_buffer_lists =
      mock_modelbox->GetOutputBufferList(ext_data, "output1");

  /*check out whether the results meet expectations*/
  EXPECT_EQ(output_buffer_lists.size(), 1);
  auto output_buffer_list = output_buffer_lists[0];
  EXPECT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  int32_t width = 0;
  int32_t height = 0;
  auto exists = output_buffer->Get("width", width);
  EXPECT_EQ(exists, true);
  exists = output_buffer->Get("height", height);
  EXPECT_EQ(exists, true);
  void *img_data = const_cast<void *>(output_buffer->ConstData());
  cv::Mat out_img(cv::Size(width, height), CV_8UC3, img_data);
  // cv::imwrite(std::string(TEST_ASSETS) + "/result.jpg", out_img);
}

TEST_F(ResizeFlowUnitTest, TestCase2) {
  /*create graph config , build and run, "input1" and "output2" are virtual
   * nodes used to send or receive buffer*/
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string toml_content = R"(
            [log]
            level="INFO"
            [driver]
            skip-default=false
            dir=[")" + test_lib_dir +
                             "\"]\n    " +
                             R"([graph]
                graphconf = '''digraph demo {                                                                            
                    input1[type=input]                                          
                    videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0]
                    videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, pix_fmt=rgb, queue_size = 16, batch_size=5]
                    resize_test[type=flowunit, flowunit=resize_test, device=cpu, deviceid=0, label="<in_1> | <out_1>", image_width=128, image_height=128, batch_size=5]
                    output1[type=output]               
                    input1 -> videodemuxer:in_video_url
                    videodemuxer:out_video_packet -> videodecoder:in_video_packet
                    videodecoder:out_video_frame -> resize_test:in_1                 
                    resize_test:out_1 -> output1                                                                      
                    }'''
                format = "graphviz"
        )";

  auto mock_modelbox = GetMockModelbox();
  auto ret = mock_modelbox->BuildAndRun("graph_name", toml_content, 10);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  /*create buffer list and fill parmeters if you want*/
  auto ext_data = mock_modelbox->GetFlow()->CreateExternalDataMap();
  EXPECT_NE(ext_data, nullptr);
  auto buffer_list = ext_data->CreateBufferList();
  EXPECT_NE(ext_data, nullptr);
  auto source_url = std::string(TEST_ASSETS) + "/test.mp4";
  buffer_list->Build({source_url.size() + 1});
  auto buffer = buffer_list->At(0);
  memcpy(buffer->MutableData(), source_url.data(), source_url.size() + 1);
  buffer->Set("source_url", source_url);
  auto data_meta = std::make_shared<modelbox::DataMeta>();
  data_meta->SetMeta("source_url", std::make_shared<std::string>(source_url));
  ext_data->SetOutputMeta("input1", data_meta);

  /*send buffer list to port "input1" ,and then transmit to next flowunit*/
  auto status = ext_data->Send("input1", buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  status = ext_data->Shutdown();
  EXPECT_EQ(status, STATUS_OK);

  /*wait for output buffer list for port "output1" */
  std::vector<std::shared_ptr<BufferList>> output_buffer_lists =
      mock_modelbox->GetOutputBufferList(ext_data, "output1");

  /*check out whether the results meet expectations*/
  uint32_t count = 1;
  for (auto &output_buffer_list : output_buffer_lists) {
    for (size_t i = 0; i < output_buffer_list->Size(); i++) {
      int32_t width = 0;
      int32_t height = 0;
      auto output_buffer = output_buffer_list->At(i);
      auto exists = output_buffer->Get("width", width);
      EXPECT_EQ(exists, true);
      exists = output_buffer->Get("height", height);
      EXPECT_EQ(exists, true);
      void *img_data = const_cast<void *>(output_buffer->ConstData());
      cv::Mat out_img(cv::Size(width, height), CV_8UC3, img_data);
      // cv::imwrite(std::string(TEST_ASSETS) + "/" + std::to_string(count) +
      // ".jpg", out_img);
      count++;
    }
  }
}

}  // namespace modelbox