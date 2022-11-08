/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#include "dlengine_inference_flowunit_test.h"

using namespace modelbox;

DLEngineInferenceFlowUnitTest::DLEngineInferenceFlowUnitTest(
    const std::string &device_type)
    : device_type_(device_type) {}

Status DLEngineInferenceFlowUnitTest::SetUp(
    const std::string &infer_flowunit_name) {
  infer_flowunit_name_ = infer_flowunit_name;
  auto ret = flow_->Init();
  if (!ret) {
    return ret;
  }

  return STATUS_OK;
}

void DLEngineInferenceFlowUnitTest::Run(const std::string &name) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_driver_dir_ +
                             "\",\"" + test_data_dir_ + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input1[type=input]
          input2[type=input]
          dlengine_inference[type=flowunit, flowunit=")" +
                             infer_flowunit_name_ + R"(", device=)" +
                             device_type_ + R"(, deviceid=0, batch_size=2]
          output[type=output, device=cpu]

          input1 -> dlengine_inference:in1
          input2 -> dlengine_inference:in2
          dlengine_inference:out -> output
        }'''
    format = "graphviz"
    )";

  auto ret = flow_->BuildAndRun(name, toml_content, -1);
  ASSERT_EQ(ret, STATUS_OK);

  auto extern_data = flow_->GetFlow()->CreateExternalDataMap();
  // prepare input
  auto in_buffer_list = extern_data->CreateBufferList();
  const size_t tensor_size = 3 * 16 * 16;
  in_buffer_list->Build({tensor_size * sizeof(float)});
  auto in_buffer = in_buffer_list->At(0);
  auto in_ptr = (float *)(in_buffer->MutableData());
  for (size_t i = 0; i < tensor_size; ++i) {
    in_ptr[i] = 1;
  }
  // send input
  extern_data->Send("input1", in_buffer_list);
  extern_data->Send("input2", in_buffer_list);
  // recv output
  OutputBufferList output_buffer_list_map;
  ret = extern_data->Recv(output_buffer_list_map);
  ASSERT_EQ(ret, STATUS_OK);

  auto output_buffer_list = output_buffer_list_map["output"];
  ASSERT_NE(output_buffer_list, nullptr);
  ASSERT_EQ(output_buffer_list->Size(), 1);

  auto output_buffer = output_buffer_list->At(0);
  // check output
  auto out_ptr = (const float *)(output_buffer->ConstData());
  ASSERT_NE(out_ptr, nullptr);

  for (size_t i = 0; i < tensor_size; ++i) {
    ASSERT_EQ(out_ptr[i], 2);
  }

  ModelBoxDataType data_type;
  std::vector<size_t> shape;
  auto b_ret = output_buffer->Get("type", data_type);
  ASSERT_TRUE(b_ret);
  b_ret = output_buffer->Get("shape", shape);
  ASSERT_TRUE(b_ret);
  ASSERT_EQ(data_type, ModelBoxDataType::MODELBOX_FLOAT);
  ASSERT_EQ(shape, std::vector<size_t>({1, 3, 16, 16}));
  // wait end
  extern_data->Close();
  flow_->GetFlow()->Wait(5000);
}
