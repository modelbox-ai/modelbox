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


#include "output_broker_flowunit.h"

#include <securec.h>

#include <functional>
#include <future>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "common/mock_cert.h"
#define _TURN_OFF_PLATFORM_STRING
#include "cpprest/http_listener.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/iam_auth.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
class OutputBrokerFlowUnitTest : public testing::Test {
 public:
  OutputBrokerFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}
  void PreparationToGetCert();
  modelbox::Status HandleFunc(web::http::http_request request);

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();
  std::shared_ptr<MockFlow> RunDriverFlow();
  modelbox::Status SendOutputData(std::shared_ptr<MockFlow> &driver_flow,
                                const std::string &output_data,
                                const std::string &output_broker_names,
                                const std::string &output_broker_cfg);

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> OutputBrokerFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<MockFlow> OutputBrokerFlowUnitTest::RunDriverFlow() {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input]
          output_broker[type=flowunit, flowunit=output_broker, device=cpu, deviceid=0, label="<in_output_info>", retry_count_limit="2", retry_interval_base_ms="100", retry_interval_increment_ms="100", retry_interval_limit_ms="200"]
 
          input -> output_broker:in_output_info
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  driver_flow->BuildAndRun("InitUnit", toml_content, -1);
  return driver_flow;
}

modelbox::Status OutputBrokerFlowUnitTest::SendOutputData(
    std::shared_ptr<MockFlow> &driver_flow, const std::string &output_data,
    const std::string &output_broker_names,
    const std::string &output_broker_cfg) {
  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({output_data.size(), output_data.size()});

  auto buffer = buffer_list->At(0);
  memcpy_s(buffer->MutableData(), buffer->GetBytes(), output_data.data(),
           output_data.size());
  buffer->Set("msg_name", std::string("webhook_msg"));
  buffer->Set("output_broker_names", output_broker_names);

  buffer = buffer_list->At(1);
  memcpy_s(buffer->MutableData(), buffer->GetBytes(), output_data.data(),
           output_data.size());
  buffer->Set("msg_name", std::string("webhook_msg2"));
  buffer->Set("output_broker_names", output_broker_names);

  auto config = ext_data->GetSessionConfig();
  config->SetProperty("flowunit.output_broker.config", output_broker_cfg);
  ext_data->Send("input", buffer_list);
  ext_data->Shutdown();
  return modelbox::STATUS_OK;
}

Status OutputBrokerFlowUnitTest::AddMockFlowUnit() { return STATUS_OK; }

TEST_F(OutputBrokerFlowUnitTest, InitUnit) {
  auto driver_flow = RunDriverFlow();

  std::string output_data = "output data";
  std::string output_broker_cfg = R"({
    "brokers": [
      {
        "type" : "dis",
        "name" : "dis1",
        "cfg": "xxx"
      },
      {
        "type" : "dis",
        "name" : "dis2",
        "cfg" : "xxx"
      },
      {
        "type" : "webhook",
        "name" : "webhook",
        "cfg" : "xxx"
      }
    ]
  })";
  auto ret = SendOutputData(driver_flow, output_data, "dis1|webhook",
                            output_broker_cfg);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(OutputBrokerFlowUnitTest, DisOutputTest) {
  // This test would be skipped, if no auth info is provided.
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }

  auto driver_flow = RunDriverFlow();

  std::string output_data(0.8 * 1024 * 1024, 'a');
  std::string disEndPoint(config_file->GetString("output_broker.obsEndPoint"));
  std::string region(config_file->GetString("output_broker.region"));
  std::string streamName(config_file->GetString("output_broker.streamName"));
  std::string projectId(config_file->GetString("output_broker.projectId"));
  std::string domainName(config_file->GetString("output_broker.domainName"));
  std::string xroleName(config_file->GetString("output_broker.xroleName"));
  std::string output_broker_cfg = R"({
    "brokers": [
      {
        "type" : "dis",
        "name" : "dis1",
        "cfg": "{\"disEndPoint\" : \")" + disEndPoint + R"(\", \"region\" : \")" + region + R"(\", \"streamName\" : \")" + streamName + R"(\",\"projectId\" : \")" + projectId + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
     },
      {
        "type" : "dis",
        "name" : "dis2",
        "cfg": "{\"disEndPoint\" : \")" + disEndPoint + R"(\", \"region\" : \")" + region + R"(\", \"streamName\" : \")" + streamName + R"(\",\"projectId\" : \")" + projectId + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}"  
      }
    ]
  })";
  PreparationToGetCert();
  auto ret =
      SendOutputData(driver_flow, output_data, "dis1|dis2", output_broker_cfg);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

modelbox::Status OutputBrokerFlowUnitTest::HandleFunc(
    web::http::http_request request) {
  utility::string_t request_body = request.extract_string().get();
  EXPECT_EQ("{\"output data webhook\":\"a\"}", request_body);
  utility::string_t resp_body = "OK";
  request.reply(web::http::status_codes::OK, resp_body);

  return modelbox::STATUS_OK;
}

TEST_F(OutputBrokerFlowUnitTest, WebhookOutputTest) {
  auto driver_flow = RunDriverFlow();

  std::string output_data = R"({"output data webhook":"a"})";
  std::string output_broker_cfg = R"({
    "brokers": [
      {
        "type" : "webhook",
        "name" : "webhook1",
        "cfg" : "{\"url\" : \"https://localhost:54321\", \"headers\" : {\"header1\" : \"test1\",\"header2\" : \"test2\"}}" 
      }
    ]
  })";
  auto ret =
      SendOutputData(driver_flow, output_data, "webhook1", output_broker_cfg);

  std::string request_url = "https://localhost:54321";
  std::shared_ptr<web::http::experimental::listener::http_listener> listener;

  web::http::experimental::listener::http_listener_config server_config;
  server_config.set_timeout(std::chrono::seconds(60));
  std::string cert = std::string(TEST_DATA_DIR) + "/certificate.pem";
  std::string key = std::string(TEST_DATA_DIR) + "/private_key_nopass.pem";

  ASSERT_EQ(GenerateCert(key, cert), STATUS_OK);

  Defer {
    remove(key.c_str());
    remove(cert.c_str());
  };

  if (cert.length() > 0 && key.length() > 0) {
    server_config.set_ssl_context_callback(
        [cert, key](boost::asio::ssl::context &ctx) {
          ctx.set_options(boost::asio::ssl::context::default_workarounds);
          modelbox::HardeningSSL(ctx.native_handle());
          ctx.use_certificate_file(
              cert, boost::asio::ssl::context_base::file_format::pem);
          ctx.use_private_key_file(key, boost::asio::ssl::context::pem);
        });
  }

  listener = std::make_shared<web::http::experimental::listener::http_listener>(
      request_url, server_config);

  listener->support(web::http::methods::POST,
                    [this](const web::http::http_request &request) {
                      this->HandleFunc(request);
                    });

  try {
    listener->open().wait();
    MBLOG_INFO << "start to listen ";
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
  }

  driver_flow->GetFlow()->Wait(3 * 1000);
}

void OutputBrokerFlowUnitTest::PreparationToGetCert() {
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  std::string ak(config_file->GetString("base.ak"));
  std::string sk(config_file->GetString("base.sk"));
  std::string domain_id(config_file->GetString("base.domain_id"));
  std::string project_id(config_file->GetString("base.project_id"));
  std::string iam_host(config_file->GetString("base.iam_host"));

  modelbox::IAMAuth::GetInstance()->SetIAMHostAddress(iam_host);

  if (modelbox::STATUS_OK != modelbox::IAMAuth::GetInstance()->SetConsigneeInfo(
                               ak, sk, domain_id, project_id)) {
    MBLOG_ERROR << "set Consignee failed";
    return;
  }
}

}  // namespace modelbox