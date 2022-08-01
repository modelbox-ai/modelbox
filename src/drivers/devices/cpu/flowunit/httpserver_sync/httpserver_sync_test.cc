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

#include <errno.h>
#include <securec.h>

#include "common/mock_cert.h"
#include "driver_flow_test.h"
#include "modelbox/base/crypto.h"
#define _TURN_OFF_PLATFORM_STRING
#include "cpprest/http_client.h"
#include "test/mock/minimodelbox/mockflow.h"

constexpr const char* REQUEST_URL_HTTPS = "https://localhost:54321";
constexpr const char* REQUEST_URL_HTTP = "http://localhost:54321";

namespace modelbox {
class HttpServerSyncFlowUnitTest : public testing::Test {
 public:
  HttpServerSyncFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}

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

std::shared_ptr<MockFlow> HttpServerSyncFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

Status HttpServerSyncFlowUnitTest::AddMockFlowUnit() {
  {
    auto mock_desc =
        GenerateFlowunitDesc("receive_post_unit", {"In_1"}, {"Out_1"});
    auto process_func =
        [=](std::shared_ptr<DataContext> op_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto input_buf = op_ctx->Input("In_1");
      auto output_buf = op_ctx->Output("Out_1");

      std::string request_url;
      input_buf->At(0)->Get("endpoint", request_url);
      EXPECT_EQ(REQUEST_URL_HTTPS, request_url);

      const auto* input_data =
          static_cast<const char*>(input_buf->ConstBufferData(0));
      std::string request_body(input_data, input_buf->At(0)->GetBytes());
      const utf8string& response_body = "response_body: " + request_body;
      auto size = response_body.size();
      std::vector<std::size_t> shape = {size};
      output_buf->Build(shape);
      memcpy_s(output_buf->MutableBufferData(0), size, response_body.data(),
               size);

      std::string uri;
      input_buf->At(0)->Get("uri", uri);
      std::string method;
      input_buf->At(0)->Get("method", method);

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
        MBLOG_ERROR << "unsupported method";
      }
      return modelbox::STATUS_OK;
    };
    auto mock_funcitons = std::make_shared<MockFunctionCollection>();
    mock_funcitons->RegisterProcessFunc(process_func);
    driver_flow_->AddFlowUnitDesc(
        mock_desc, mock_funcitons->GenerateCreateFunc(), TEST_DRIVER_DIR);
  }

  {
    auto mock_desc =
        GenerateFlowunitDesc("receive_health_post_unit", {"In_1"}, {"Out_1"});
    auto process_func =
        [=](std::shared_ptr<DataContext> op_ctx,
            std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
      auto input_buf = op_ctx->Input("In_1");
      auto output_buf = op_ctx->Output("Out_1");

      std::string request_url;
      input_buf->At(0)->Get("endpoint", request_url);
      EXPECT_EQ(REQUEST_URL_HTTP, request_url);

      const auto* input_data =
          static_cast<const char*>(input_buf->ConstBufferData(0));
      std::string request_body(input_data, input_buf->At(0)->GetBytes());
      const utf8string& response_body = "response_body: " + request_body;
      auto size = response_body.size();
      std::vector<std::size_t> shape = {size};
      output_buf->Build(shape);
      memcpy_s(output_buf->MutableBufferData(0), size, response_body.data(),
               size);

      std::string uri;
      input_buf->At(0)->Get("uri", uri);
      std::string method;
      input_buf->At(0)->Get("method", method);

      std::string health_uri{"/health"};
      std::string body;
      if (method == "PUT") {
        EXPECT_EQ(body, request_body);
        EXPECT_EQ(health_uri, uri);
      } else if (method == "POST") {
        EXPECT_EQ(body, request_body);
        EXPECT_EQ(health_uri, uri);
      } else if (method == "GET") {
        EXPECT_EQ(body, request_body);
        EXPECT_EQ(health_uri, uri);
      } else if (method == "DELETE") {
        EXPECT_EQ(body, request_body);
        EXPECT_EQ(health_uri, uri);
      } else {
        MBLOG_ERROR << "unsupported method";
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

void PutRequestSync(web::http::uri uri,
                    web::http::client::http_client_config client_config,
                    const std::string& request_uri) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_put;
  headers_put.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_put.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_put;
  msg_put.set_method(web::http::methods::PUT);
  msg_put.set_request_uri(_XPLATSTR(request_uri));
  msg_put.headers() = headers_put;
  auto putvalue = web::json::value::object();
  putvalue["param"] = web::json::value::string(
      R"({"image_id":0,"algorithm":"face_detection","alg_threshold":12.0})");
  putvalue["image"] = web::json::value::string("image base 64 data string put");
  msg_put.set_body(putvalue);
  try {
    web::http::http_response resp_put = client.request(msg_put).get();
    if (resp_put.status_code() == web::http::status_codes::OK) {
      EXPECT_EQ("response_body: " + putvalue.serialize(),
                resp_put.extract_string().get());
    } else {
      EXPECT_EQ("", resp_put.extract_string().get());
    }
    MBLOG_INFO << "put response status codes: " << resp_put.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }
}

void PostRequestSync(web::http::uri uri,
                     web::http::client::http_client_config client_config,
                     const std::string& request_uri) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_post;
  headers_post.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_post.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_post;
  msg_post.set_method(web::http::methods::POST);
  msg_post.set_request_uri(_XPLATSTR(request_uri));
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
    if (resp_post.status_code() == web::http::status_codes::OK) {
      EXPECT_EQ("response_body: " + postvalue.serialize(),
                resp_post.extract_string().get());
    } else {
      EXPECT_EQ("", resp_post.extract_string().get());
    }
    MBLOG_INFO << "post response status codes: " << resp_post.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }
}

void GetRequestSync(web::http::uri uri,
                    web::http::client::http_client_config client_config,
                    const std::string& request_uri) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_get;
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_get;
  msg_get.set_method(web::http::methods::GET);
  msg_get.set_request_uri(_XPLATSTR(request_uri));
  msg_get.headers() = headers_get;
  try {
    web::http::http_response resp_get = client.request(msg_get).get();
    if (resp_get.status_code() == web::http::status_codes::OK) {
      EXPECT_EQ("response_body: ", resp_get.extract_string().get());
    } else {
      EXPECT_EQ("", resp_get.extract_string().get());
    }
    MBLOG_INFO << "get response status codes: " << resp_get.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }
}

void HealthCheckRequesSync(web::http::uri uri,
                           web::http::client::http_client_config client_config,
                           const std::string& request_uri,
                           const web::http::method& method) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_get;
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_get.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg;
  msg.set_method(method);
  msg.set_request_uri(_XPLATSTR(request_uri));
  msg.headers() = headers_get;
  auto value = web::json::value::object();
  value["status"] = web::json::value(200);
  value["message"] =
      web::json::value::string("success");
  msg.set_body(value);
  try {
    web::http::http_response resp_get = client.request(msg).get();
    EXPECT_EQ(resp_get.status_code(), web::http::status_codes::OK);
    EXPECT_EQ(value.serialize(), resp_get.extract_string().get());
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }
}

void DelRequestSync(web::http::uri uri,
                    web::http::client::http_client_config client_config,
                    const std::string& request_uri) {
  web::http::client::http_client client(web::http::uri_builder(uri).to_uri(),
                                        client_config);
  web::http::http_headers headers_del;
  headers_del.add(_XPLATSTR("Accept"), _XPLATSTR("application/json"));
  headers_del.add(_XPLATSTR("Accept"), _XPLATSTR("text/plain"));
  web::http::http_request msg_del;
  msg_del.set_method(web::http::methods::DEL);
  msg_del.set_request_uri(_XPLATSTR(request_uri));
  msg_del.headers() = headers_del;
  auto delvalue = web::json::value::array();
  delvalue[0] = web::json::value::string("image_id");
  msg_del.set_body(delvalue);
  try {
    web::http::http_response resp_del = client.request(msg_del).get();
    if (resp_del.status_code() == web::http::status_codes::OK) {
      EXPECT_EQ("response_body: " + delvalue.serialize(),
                resp_del.extract_string().get());
    } else {
      EXPECT_EQ("", resp_del.extract_string().get());
    }
    MBLOG_INFO << "del response status codes: " << resp_del.status_code();
  } catch (std::exception const& e) {
    MBLOG_ERROR << e.what();
    ASSERT_TRUE(false);
  }
}

TEST_F(HttpServerSyncFlowUnitTest, InitUnit) {
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

  std::string toml_content =
      R"(
    [driver]
    skip-default=true
    dir=[")" +
      test_lib_dir + "\"]\n    " +
      R"([graph]
    graphconf = '''digraph demo {                                                                          
          httpserver_sync_receive[type=flowunit, flowunit=httpserver_sync_receive, device=cpu, deviceid=0, label="<out_request_info>", endpoint=")" +
      std::string(REQUEST_URL_HTTPS) + R"(", cert=")" + cert_file_path +
      R"(", key=")" + key_file_path + R"(", passwd=")" + encrypt_passwd +
      R"(", key_pass=")" + passwd_key +
      R"(", max_requests=1000, time_out_ms=5000, keepalive_timeout_sec=10]
          receive_post_unit[type=flowunit, flowunit=receive_post_unit, device=cpu, deviceid=0, label="<In_1> | <Out_1>"] 
          httpserver_sync_reply[type=flowunit, flowunit=httpserver_sync_reply, device=cpu, deviceid=0, label="<In_1>"]        
          httpserver_sync_receive:out_request_info -> receive_post_unit:In_1   
          receive_post_unit:Out_1 -> httpserver_sync_reply:in_reply_info                                                      
        }'''
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("InitUnit", toml_content, -1);

  web::http::uri uri = web::http::uri(_XPLATSTR(REQUEST_URL_HTTPS));
  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(60));
  client_config.set_ssl_context_callback([&](boost::asio::ssl::context& ctx) {
    ctx.load_verify_file(cert_file_path);
  });

  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back(PutRequestSync, uri, client_config, "/restdemo_put");
    threads.emplace_back(DelRequestSync, uri, client_config, "/restdemo_del");
    threads.emplace_back(PostRequestSync, uri, client_config, "/restdemo_post");
    threads.emplace_back(GetRequestSync, uri, client_config, "/restdemo_get");
  }
  for (auto& th : threads) {
    th.join();
  }
}

TEST_F(HttpServerSyncFlowUnitTest, HealthCheck) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content =
      R"(
    [driver]
    skip-default=true
    dir=[")" +
      test_lib_dir + "\"]\n    " +
      R"([graph]
    graphconf = '''digraph demo {                                                                          
          httpserver_sync_receive[type=flowunit, flowunit=httpserver_sync_receive, device=cpu, deviceid=0, label="<out_request_info>", endpoint=")" +
      std::string(REQUEST_URL_HTTP) +
      R"(", max_requests=10, time_out_ms=5000, keepalive_timeout_sec=10]
          receive_post_unit[type=flowunit, flowunit=receive_health_post_unit, device=cpu, deviceid=0, label="<In_1> | <Out_1>"] 
          httpserver_sync_reply[type=flowunit, flowunit=httpserver_sync_reply, device=cpu, deviceid=0, label="<In_1>"]        
          httpserver_sync_receive:out_request_info -> receive_post_unit:In_1   
          receive_post_unit:Out_1 -> httpserver_sync_reply:in_reply_info                                                      
        }'''
    format = "graphviz"
  )";

  MBLOG_INFO << toml_content;
  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("InitUnit", toml_content, -1);

  web::http::uri uri = web::http::uri(_XPLATSTR(REQUEST_URL_HTTP));
  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(60));

  std::vector<std::thread> threads;
  std::string health_uri = "/health";
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back(HealthCheckRequesSync, uri, client_config, health_uri,
                         web::http::methods::GET);
    threads.emplace_back(HealthCheckRequesSync, uri, client_config, health_uri,
                         web::http::methods::PUT);
    threads.emplace_back(HealthCheckRequesSync, uri, client_config, health_uri,
                         web::http::methods::POST);
    threads.emplace_back(HealthCheckRequesSync, uri, client_config, health_uri,
                         web::http::methods::DEL);
  }
  for (auto& th : threads) {
    th.join();
  }
}

}  // namespace modelbox