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

#define _TURN_OFF_PLATFORM_STRING
#include "modelbox/drivers/common/file_requester.h"

#include <securec.h>

#include <cstring>

#include "cpprest/http_client.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {

class ModelBoxFileRequesterTest : public testing::Test {
 public:
  ModelBoxFileRequesterTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  }

  virtual void TearDown() { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

Status ModelBoxFileRequesterTest::AddMockFlowUnit() { return STATUS_OK; }

class MockFileHandler : public modelbox::FileGetHandler {
 public:
  virtual ~MockFileHandler() = default;

  std::string msg{"Hello, world"};
  modelbox::Status Get(unsigned char *buff, size_t size, off_t off) {
    if (off > (off_t)msg.length()) {
      return modelbox::STATUS_INVALID;
    }
    memcpy_s(buff, size, msg.c_str(), size);
    return modelbox::STATUS_OK;
  }
  int GetFileSize() { return msg.length(); }
};

TEST_F(ModelBoxFileRequesterTest, Register) {
  std::shared_ptr<modelbox::MockFileHandler> mock_handler =
      std::make_shared<modelbox::MockFileHandler>();
  std::string mock_uri =
      std::string("/mock/") + std::string("test_double_register");
  auto uri = DEFAULT_FILE_REQUEST_URI + mock_uri;
  auto ret = modelbox::FileRequester::GetInstance()->RegisterUrlHandler(
      mock_uri, mock_handler);
  EXPECT_EQ(ret, modelbox::STATUS_OK);
  ret = modelbox::FileRequester::GetInstance()->RegisterUrlHandler(
      mock_uri, mock_handler);
  EXPECT_EQ(ret, modelbox::STATUS_EXIST);
  ret = modelbox::FileRequester::GetInstance()->DeregisterUrl(mock_uri);
  EXPECT_EQ(ret, modelbox::STATUS_OK);
  ret = modelbox::FileRequester::GetInstance()->DeregisterUrl(mock_uri);
  EXPECT_EQ(ret, modelbox::STATUS_NOTFOUND);
}

TEST_F(ModelBoxFileRequesterTest, Request) {
  std::shared_ptr<modelbox::MockFileHandler> mock_handler =
      std::make_shared<modelbox::MockFileHandler>();
  std::string mock_uri = std::string("/mock/") + std::string("test_request");
  auto uri = DEFAULT_FILE_REQUEST_URI + mock_uri;
  auto ret = modelbox::FileRequester::GetInstance()->RegisterUrlHandler(
      mock_uri, mock_handler);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  web::http::client::http_client client(
      web::http::uri_builder(DEFAULT_FILE_REQUEST_URI).to_uri());
  web::http::http_headers headers_get;
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("*/*"));
  headers_get.add(_XPLATSTR("Range"), _XPLATSTR("bytes=0-"));
  web::http::http_request msg_get;
  msg_get.set_method(web::http::methods::GET);
  msg_get.set_request_uri(_XPLATSTR(mock_uri));
  msg_get.headers() = headers_get;
  msg_get.headers().set_content_type("application/json");
  std::string msg{"Hello, world"};
  try {
    web::http::http_response resp_get = client.request(msg_get).get();
    resp_get.headers().set_content_type("application/json");
    resp_get.content_ready().wait();
    if (resp_get.status_code() == web::http::status_codes::OK) {
      auto body = resp_get.content_ready().get().extract_utf8string(true).get();
      EXPECT_EQ(msg, body);
    } else {
      EXPECT_EQ("", resp_get.extract_string().get());
    }
    MBLOG_INFO << "get response status codes: " << resp_get.status_code();
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }

  ret = modelbox::FileRequester::GetInstance()->DeregisterUrl(mock_uri);
  EXPECT_EQ(ret, modelbox::STATUS_OK);
}

}  // namespace modelbox