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

#include <securec.h>

#include <nlohmann/json.hpp>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/crypto.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class Base64DecoderFlowUnitTest : public testing::Test {
 public:
  Base64DecoderFlowUnitTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}

 protected:
  void SetUp() override {}

  void TearDown() override { driver_flow_ = nullptr; };

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> Base64DecoderFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(Base64DecoderFlowUnitTest, DecodeTest) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          base64_decoder[type=flowunit, flowunit=base64_decoder, device=cpu, deviceid=0, batch_size=3]
          input -> base64_decoder:in_data
          base64_decoder:out_data -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("DecodeTest", toml_content, -1);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  MBLOG_INFO << toml_content;

  std::string test_text = "this is base64 decoder test text";
  std::vector<unsigned char> in_text(test_text.begin(), test_text.end());
  std::string base64_text;
  EXPECT_TRUE(modelbox::Base64Encode(in_text, &base64_text));

  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto in_buffer_list = extern_data->CreateBufferList();
  in_buffer_list->Build({base64_text.size()});
  auto in_buffer = in_buffer_list->At(0);
  auto e_ret = memcpy_s(in_buffer->MutableData(), in_buffer->GetBytes(),
                        base64_text.c_str(), base64_text.size());
  EXPECT_EQ(e_ret, EOK);

  auto status = extern_data->Send("input", in_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  // check output
  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  ASSERT_EQ(output_buffer->GetBytes(), test_text.size());

  std::shared_ptr<unsigned char> out_buf(
      new (std::nothrow) unsigned char[output_buffer->GetBytes()],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_buf.get(), output_buffer->GetBytes(), 0,
                   output_buffer->GetBytes());
  EXPECT_EQ(e_ret, EOK);

  e_ret = memcpy_s(out_buf.get(), output_buffer->GetBytes(),
                   output_buffer->ConstData(), output_buffer->GetBytes());
  EXPECT_EQ(e_ret, EOK);

  // cmp memory
  EXPECT_EQ(memcmp(out_buf.get(), test_text.c_str(), output_buffer->GetBytes()),
            0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(Base64DecoderFlowUnitTest, JsonDecodeTest) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\"]\n" +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output[type=output]
          base64_decoder[type=flowunit, flowunit=base64_decoder, device=cpu, deviceid=0, batch_size=3, data_format=json, key=data_base64]
          input -> base64_decoder:in_data
          base64_decoder:out_data -> output
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("DecodeTest", toml_content, -1);
  EXPECT_EQ(ret, STATUS_SUCCESS);

  MBLOG_INFO << toml_content;

  std::string test_text = "this is base64 decoder test text";
  std::vector<unsigned char> in_text(test_text.begin(), test_text.end());
  std::string base64_text;
  EXPECT_TRUE(modelbox::Base64Encode(in_text, &base64_text));

  nlohmann::json base64_data_json;
  base64_data_json["data_base64"] = base64_text;
  std::string base64_data_json_str = base64_data_json.dump();

  auto extern_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto in_buffer_list = extern_data->CreateBufferList();
  in_buffer_list->Build({base64_data_json_str.size()});
  auto in_buffer = in_buffer_list->At(0);
  auto e_ret =
      memcpy_s(in_buffer->MutableData(), in_buffer->GetBytes(),
               base64_data_json_str.c_str(), base64_data_json_str.size());
  EXPECT_EQ(e_ret, EOK);

  auto status = extern_data->Send("input", in_buffer_list);
  EXPECT_EQ(status, STATUS_OK);

  // check output
  OutputBufferList map_buffer_list;
  status = extern_data->Recv(map_buffer_list);
  EXPECT_EQ(status, STATUS_OK);
  auto output_buffer_list = map_buffer_list["output"];
  ASSERT_EQ(output_buffer_list->Size(), 1);
  auto output_buffer = output_buffer_list->At(0);
  ASSERT_EQ(output_buffer->GetBytes(), test_text.size());

  std::shared_ptr<unsigned char> out_buf(
      new (std::nothrow) unsigned char[output_buffer->GetBytes()],
      std::default_delete<unsigned char[]>());
  e_ret = memset_s(out_buf.get(), output_buffer->GetBytes(), 0,
                   output_buffer->GetBytes());
  EXPECT_EQ(e_ret, EOK);

  e_ret = memcpy_s(out_buf.get(), output_buffer->GetBytes(),
                   output_buffer->ConstData(), output_buffer->GetBytes());
  EXPECT_EQ(e_ret, EOK);

  // cmp memory
  EXPECT_EQ(memcmp(out_buf.get(), test_text.c_str(), output_buffer->GetBytes()),
            0);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox