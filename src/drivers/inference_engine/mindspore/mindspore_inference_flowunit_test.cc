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

#include "mindspore_inference_flowunit_test.h"

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {

Status InferenceMindSporeFlowUnitTest::Init() {
  auto ret = AddMockFlowUnit();
  return ret;
}

Status InferenceMindSporeFlowUnitTest::Run(const std::string &name,
                                           const std::string &graph) {
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun(name, graph);
  return ret;
}

Status InferenceMindSporeFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("prepare_ms_infer_data", {}, {"out1", "out2"});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(2);
    auto open_func = [=](const std::shared_ptr<Configuration> &opts,
                         const std::shared_ptr<MockFlowUnit> &mock_flowunit) {
      auto ext_data = mock_flowunit->CreateExternalData();
      if (!ext_data) {
        MBLOG_ERROR << "can not get external data.";
      }

      auto buffer_list = ext_data->CreateBufferList();
      buffer_list->Build({10});

      auto status = ext_data->Send(buffer_list);
      if (!status) {
        MBLOG_ERROR << "external data send buffer list failed:" << status;
      }

      status = ext_data->Close();
      if (!status) {
        MBLOG_ERROR << "external data close failed:" << status;
      }

      return STATUS_OK;
    };

    auto process_func =
        [=](const std::shared_ptr<DataContext> &op_ctx,
            const std::shared_ptr<MockFlowUnit> &mock_flowunit) {
          MBLOG_INFO << "prepare_ms_infer_data "
                     << "Process";
          auto output_buf_1 = op_ctx->Output("out1");
          auto output_buf_2 = op_ctx->Output("out2");
          const size_t len = 2;
          std::vector<size_t> shape_vector(2, len * sizeof(float));
          ModelBoxDataType type = MODELBOX_FLOAT;

          output_buf_1->Build(shape_vector);
          output_buf_1->Set("type", type);
          std::vector<size_t> shape{len, 2};
          output_buf_1->Set("shape", shape);
          auto *dev_data1 = (float *)(output_buf_1->MutableData());
          MBLOG_INFO << "output_buf_1.size: " << output_buf_1->Size();
          float val = 1.0;
          for (size_t i = 0; i < output_buf_1->Size(); ++i) {
            for (size_t j = 0; j < len; ++j) {
              dev_data1[i * len + j] = val;
              val += 1.0;
            }
          }

          output_buf_2->Build(shape_vector);
          output_buf_2->Set("type", type);
          output_buf_2->Set("shape", shape);
          auto *dev_data2 = (float *)(output_buf_2->MutableData());
          val = 2.0;
          for (size_t i = 0; i < output_buf_2->Size(); ++i) {
            for (size_t j = 0; j < len; ++j) {
              dev_data2[i * len + j] = val;
              val += 1.0;
            }
          }

          return STATUS_OK;
        };

    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterOpenFunc(open_func);
    mock_functions->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_functions->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  {
    auto mock_desc = GenerateFlowunitDesc("check_ms_infer_result", {"in"}, {});
    mock_desc->SetFlowType(STREAM);
    mock_desc->SetMaxBatchSize(2);
    auto data_post_func =
        [=](const std::shared_ptr<DataContext> &op_ctx,
            const std::shared_ptr<MockFlowUnit> &mock_flowunit) {
          MBLOG_INFO << "check_ms_infer_result "
                     << "DataPost";
          return STATUS_STOP;
        };

    auto process_func =
        [=](const std::shared_ptr<DataContext> &op_ctx,
            const std::shared_ptr<MockFlowUnit> &mock_flowunit) {
          std::shared_ptr<BufferList> input_bufs = op_ctx->Input("in");
          EXPECT_EQ(input_bufs->Size(), 2);
          std::vector<int64_t> input_shape;
          auto result = input_bufs->At(0)->Get("shape", input_shape);
          EXPECT_TRUE(result);
          EXPECT_EQ(input_shape.size(), 2);
          EXPECT_EQ(input_shape[0], 2);
          EXPECT_EQ(input_shape[1], 2);

          const auto *ptr = (const float *)input_bufs->ConstData();
          float val = 3.0;
          for (size_t i = 0; i < 4; ++i) {
            EXPECT_TRUE((std::abs(ptr[i]) - val) < 1e-7);
            val += 2.0;
          }

          return STATUS_OK;
        };

    auto mock_functions = std::make_shared<MockFunctionCollection>();
    mock_functions->RegisterDataPostFunc(data_post_func);
    mock_functions->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_functions->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  return STATUS_OK;
}

std::shared_ptr<MockFlow> InferenceMindSporeFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}
}  // namespace modelbox
