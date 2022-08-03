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

#include <securec.h>

#include "common/mock_cert.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#define _TURN_OFF_PLATFORM_STRING
#include "cpprest/http_client.h"
#include "test/mock/minimodelbox/mockflow.h"

#define REQUEST_URL "https://localhost:56789"

namespace modelbox {
class HttpServerAsyncFlowUnitTest : public testing::Test {
 public:
  HttpServerAsyncFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> HttpServerAsyncFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status HttpServerAsyncFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("httpserver_async_post_unit", {"In_1"}, {});
    auto process_func =
        [=](const std::shared_ptr<DataContext>& op_ctx,
            const std::shared_ptr<MockFlowUnit>& mock_flowunit) -> Status {
      auto input_buf = op_ctx->Input("In_1");
      std::string request_url;
      input_buf->At(0)->Get("endpoint", request_url);
      EXPECT_EQ(REQUEST_URL, request_url);
      auto* input_data = (char*)input_buf->ConstBufferData(0);
      std::string request_body(input_data, input_buf->At(0)->GetBytes());
      std::string method;
      input_buf->At(0)->Get("method", method);
      std::string uri;
      input_buf->At(0)->Get("uri", uri);
      if (method == "PUT") {
        auto putvalue = web::json::value::object();
        putvalue["param"] = web::json::value::string(
            "{\"image_id\":0,\"algorithm\":\"face_detection\",\"alg_"
            "threshold\":12.0}");
        putvalue["image"] =
            web::json::value::string("image base 64 data string put");
        std::string body_put = putvalue.serialize();
        EXPECT_EQ(body_put, request_body);
        EXPECT_EQ("/restdemo_put", uri);
      } else if (method == "POST") {
        auto postvalue = web::json::value::object();
        postvalue["param"] = web::json::value::string(
            "{\"image_id\":100,\"algorithm\":\"vehicle_detection\","
            "\"detect_threshold\":0.5}");
        postvalue["image"] =
            web::json::value::string("image base 64 data string post");
        std::string body_post = postvalue.serialize();
        EXPECT_EQ(body_post, request_body);
        EXPECT_EQ("/restdemo_post", uri);
      } else if (method == "GET") {
        std::string body_get;
        EXPECT_EQ(body_get, request_body);
        EXPECT_EQ("/restdemo_get", uri);
      } else if (method == "DELETE") {
        auto delvalue = web::json::value::array();
        delvalue[0] = web::json::value::string("image_id");
        std::string body_del = delvalue.serialize();
        EXPECT_EQ(body_del, request_body);
        EXPECT_EQ("/restdemo_del", uri);
      } else {
        MBLOG_ERROR << "unsupport method";
      }
      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  return STATUS_OK;
}

void PutRequestAsync(
    const web::http::uri& uri,
    const web::http::client::http_client_config& client_config) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_put;

  headers_put.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_put.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_put;
  msg_put.set_method(web::http::methods::PUT);
  msg_put.set_request_uri(_XPLATSTR("/restdemo_put"));
  msg_put.headers() = headers_put;
  auto putvalue = web::json::value::object();
  putvalue["param"] = web::json::value::string(
      R"({"image_id":0,"algorithm":"face_detection","alg_threshold":12.0})");
  putvalue["image"] = web::json::value::string("image base 64 data string put");
  msg_put.set_body(putvalue);

  try {
    web::http::http_response resp_put = client.request(msg_put).get();
    MBLOG_INFO << "put response status codes: " << resp_put.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
    return;
  }
}

void PostRequestAsync(
    const web::http::uri& uri,
    const web::http::client::http_client_config& client_config) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_post;

  headers_post.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_post.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_post;
  msg_post.set_method(web::http::methods::POST);
  msg_post.set_request_uri(_XPLATSTR("/restdemo_post"));
  msg_post.headers() = headers_post;
  auto postvalue = web::json::value::object();
  postvalue["param"] = web::json::value::string(
      "{\"image_id\":100,\"algorithm\":\"vehicle_detection\",\"detect_"
      "threshold\":0.5}");
  postvalue["image"] =
      web::json::value::string("image base 64 data string post");
  msg_post.set_body(postvalue);

  try {
    web::http::http_response resp_post = client.request(msg_post).get();
    MBLOG_INFO << "post response status codes: " << resp_post.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
    return;
  }
}

void GetRequestAsync(
    const web::http::uri& uri,
    const web::http::client::http_client_config& client_config) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_get;

  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_get;
  msg_get.set_method(web::http::methods::GET);
  msg_get.set_request_uri(_XPLATSTR("/restdemo_get"));
  msg_get.headers() = headers_get;

  try {
    web::http::http_response resp_get = client.request(msg_get).get();
    MBLOG_INFO << "get response status codes: " << resp_get.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
    return;
  }
}

void DelRequestAsync(
    const web::http::uri& uri,
    const web::http::client::http_client_config& client_config) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_del;

  headers_del.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_del.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_del;
  msg_del.set_method(web::http::methods::DEL);
  msg_del.set_request_uri(_XPLATSTR("/restdemo_del"));
  msg_del.headers() = headers_del;
  auto delvalue = web::json::value::array();
  delvalue[0] = web::json::value::string("image_id");
  msg_del.set_body(delvalue);

  try {
    web::http::http_response resp_del = client.request(msg_del).get();
    MBLOG_INFO << "del response status codes: " << resp_del.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
    return;
  }
}

TEST_F(HttpServerAsyncFlowUnitTest, InitUnit) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;

  std::string cert_file_path = std::string(TEST_DATA_DIR) + "/certificate.pem";
  std::string key_file_path = std::string(TEST_DATA_DIR) + "/private_key.pem";
  std::string encrypt_passwd;
  std::string passwd_key;

  ASSERT_EQ(
      GenerateCert(&encrypt_passwd, &passwd_key, key_file_path, cert_file_path),
      STATUS_OK);

  Defer {
    remove(key_file_path.c_str());
    remove(cert_file_path.c_str());
  };

  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {                                                                          
          httpserver_async[type=flowunit, flowunit=httpserver_async, device=cpu, deviceid=0, label="<Out_1>", endpoint=")" +
                             std::string(REQUEST_URL) + R"(", cert=")" +
                             cert_file_path + R"(", key=")" + key_file_path +
                             R"(", passwd=")" + encrypt_passwd +
                             R"(", key_pass=")" + passwd_key +
                             R"(", max_requests=10]
          httpserver_async_post_unit[type=flowunit, flowunit=httpserver_async_post_unit, device=cpu, deviceid=0, label="<In_1>"]                       
          httpserver_async:out_request_info -> httpserver_async_post_unit:In_1                                                                     
        }'''
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("InitUnit", toml_content, -1);

  web::http::uri uri = web::http::uri(_XPLATSTR(REQUEST_URL));
  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(60));
  client_config.set_ssl_context_callback([&](boost::asio::ssl::context& ctx) {
    ctx.load_verify_file(cert_file_path);
  });

  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back(PutRequestAsync, uri, client_config);
    threads.emplace_back(DelRequestAsync, uri, client_config);
    threads.emplace_back(PostRequestAsync, uri, client_config);
    threads.emplace_back(GetRequestAsync, uri, client_config);
  }
  for (auto& th : threads) {
    th.join();
  }
}

}  // namespace modelbox