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


#include "data_source_parser_flowunit.h"

#include <securec.h>

#include <functional>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "common/mock_cert.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iam_auth.h"
#define _TURN_OFF_PLATFORM_STRING
#include "cpprest/http_listener.h"
#include "test/mock/minimodelbox/mockflow.h"

#define CHECK_SOURCE_OUTPUT_URL "check_data_source_url_parser_output"
#define CHECK_SOURCE_OUTPUT_VIS "check_data_source_vis_parser_output"
#define CHECK_SOURCE_OUTPUT_RESTFUL "check_data_source_restful_parser_output"

#define RESTFUL_URL "https://localhost:54321"

namespace modelbox {

class DataSourceParserFlowUnitTest : public testing::Test {
 public:
  DataSourceParserFlowUnitTest() : driver_flow_(std::make_shared<MockFlow>()) {}
  void PreparationToGetCert();
  modelbox::Status HandleFunc(const web::http::http_request &request);
  void MockRestfulServer(std::shared_ptr<MockFlow> &driver_flow);

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    cert_ = std::string(TEST_DATA_DIR) + "/certificate.pem";
    key_ = std::string(TEST_DATA_DIR) + "/private_key_nopass.pem";

    ASSERT_EQ(GenerateCert(key_, cert_), STATUS_OK);
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override {
    driver_flow_ = nullptr;
    remove(key_.c_str());
    remove(cert_.c_str());
  };
  std::shared_ptr<MockFlow> GetDriverFlow();
  std::shared_ptr<MockFlow> RunDriverFlow(
      const std::string &mock_flowunit_name);
  modelbox::Status SendDataSourceCfg(std::shared_ptr<MockFlow> &driver_flow,
                                   const std::string &data_source_cfg,
                                   const std::string &source_type);

  void GetMockKey(std::string &key, std::string &cert) {
    key = key_;
    cert = cert_;
  }

 private:
  Status AddMockFlowUnit();
  Status AddMockUrl();
  Status AddMockVis();
  Status AddMockRestful();
  std::string key_;
  std::string cert_;
  std::shared_ptr<MockFlow> driver_flow_;
};

std::shared_ptr<MockFlow> DataSourceParserFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<MockFlow> DataSourceParserFlowUnitTest::RunDriverFlow(
    const std::string &mock_flowunit_name) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input, device=cpu, deviceid=0]
          data_source_parser[type=flowunit, flowunit=data_source_parser, device=cpu, deviceid=0, label="<data_uri>", plugin_dir=")" +
                             test_lib_dir + R"("]
          )" + mock_flowunit_name +
                             R"([type=flowunit, flowunit=)" +
                             mock_flowunit_name +
                             R"(, device=cpu, deviceid=0, label="<data_uri>"]
          input -> data_source_parser:in_data
          data_source_parser:out_video_url -> )" +
                             mock_flowunit_name + R"(:stream_meta
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun(mock_flowunit_name, toml_content, -1);

  return driver_flow;
}

modelbox::Status DataSourceParserFlowUnitTest::SendDataSourceCfg(
    std::shared_ptr<MockFlow> &driver_flow, const std::string &data_source_cfg,
    const std::string &source_type) {
  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();
  buffer_list->Build({data_source_cfg.size()});
  auto buffer = buffer_list->At(0);
  memcpy_s(buffer->MutableData(), buffer->GetBytes(), data_source_cfg.data(),
           data_source_cfg.size());
  buffer->Set("source_type", source_type);
  ext_data->Send("input", buffer_list);
  ext_data->Shutdown();
  return modelbox::STATUS_OK;
}

Status DataSourceParserFlowUnitTest::AddMockFlowUnit() {
  AddMockUrl();
  AddMockVis();
  AddMockRestful();
  return modelbox::STATUS_OK;
}

Status DataSourceParserFlowUnitTest::AddMockUrl() {
  auto mock_desc =
      GenerateFlowunitDesc(CHECK_SOURCE_OUTPUT_URL, {"stream_meta"}, {});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](const std::shared_ptr<DataContext> &data_ctx,
          const std::shared_ptr<MockFlowUnit> &mock_flowunit) -> Status {
    auto stream_meta = data_ctx->GetInputMeta("stream_meta");
    EXPECT_NE(stream_meta, nullptr);
    if (!stream_meta) {
      return modelbox::STATUS_SUCCESS;
    }

    auto source_url = std::static_pointer_cast<std::string>(
        stream_meta->GetMeta("source_url"));
    EXPECT_NE(source_url, nullptr);
    if (source_url != nullptr) {
      EXPECT_FALSE((*source_url).empty());
      EXPECT_EQ(
          (*source_url)
              .substr((*source_url).rfind(':'),
                      (*source_url).rfind('.') - (*source_url).rfind(':') + 1),
          "://ip/path/test.");
    }
    return modelbox::STATUS_SUCCESS;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
  return STATUS_OK;
}

Status DataSourceParserFlowUnitTest::AddMockVis() {
  auto mock_desc =
      GenerateFlowunitDesc(CHECK_SOURCE_OUTPUT_VIS, {"stream_meta"}, {});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](const std::shared_ptr<DataContext> &data_ctx,
          const std::shared_ptr<MockFlowUnit> &mock_flowunit) -> Status {
    auto stream_meta = data_ctx->GetInputMeta("stream_meta");
    EXPECT_NE(stream_meta, nullptr);
    if (!stream_meta) {
      return modelbox::STATUS_SUCCESS;
    }

    auto source_url = std::static_pointer_cast<std::string>(
        stream_meta->GetMeta("source_url"));

    EXPECT_EQ(*source_url, "https://test.com");

    return modelbox::STATUS_SUCCESS;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
  return STATUS_OK;
}

Status DataSourceParserFlowUnitTest::AddMockRestful() {
  auto mock_desc =
      GenerateFlowunitDesc(CHECK_SOURCE_OUTPUT_RESTFUL, {"stream_meta"}, {});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](const std::shared_ptr<DataContext> &data_ctx,
          const std::shared_ptr<MockFlowUnit> &mock_flowunit) -> Status {
    auto stream_meta = data_ctx->GetInputMeta("stream_meta");
    EXPECT_NE(stream_meta, nullptr);
    if (!stream_meta) {
      return modelbox::STATUS_SUCCESS;
    }

    auto source_url = std::static_pointer_cast<std::string>(
        stream_meta->GetMeta("source_url"));

    EXPECT_EQ(*source_url, "rtsp://admin:password@127.0.0.0:808/2");

    return modelbox::STATUS_SUCCESS;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  driver_flow_->AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(),
                                TEST_DRIVER_DIR);
  return STATUS_OK;
}

TEST_F(DataSourceParserFlowUnitTest, UrlInputTest) {
  auto driver_flow = RunDriverFlow(CHECK_SOURCE_OUTPUT_URL);

  std::string source_type = "url";
  std::string data_source_cfg_file = R"({
        "url": "https://ip/path/test.avi",
        "url_type": "file"
  })";
  auto ret = SendDataSourceCfg(driver_flow, data_source_cfg_file, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  std::string data_source_cfg_rtsp = R"({
        "url": "rtsp://ip/path/test.sdp",
        "url_type": "stream"
  })";
  ret = SendDataSourceCfg(driver_flow, data_source_cfg_rtsp, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

TEST_F(DataSourceParserFlowUnitTest, VisInputTest) {
  // This test would be skipped, if no auth info is provided.
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }

  auto driver_flow = RunDriverFlow(CHECK_SOURCE_OUTPUT_VIS);

  std::string source_type = "vis";
  std::string visEndPoint(config_file->GetString("data_source.visEndPoint"));
  std::string projectId(config_file->GetString("base.project_id"));
  std::string streamName(config_file->GetString("data_source.streamName"));
  std::string domainName(config_file->GetString("data_source.domainName"));
  std::string xroleName(config_file->GetString("data_source.xroleName"));

  std::string data_source_cfg = R"({
        "visEndPoint":")" + visEndPoint +
                                R"(\",
        "projectId":")" + projectId +
                                R"(\", 
        "streamName":")" + streamName +
                                R"(\",
        "domainName":")" + domainName +
                                R"(\",
        "xroleName":")" + xroleName +
                                R"(\",
        "certificate": true
  })";

  PreparationToGetCert();

  auto ret = SendDataSourceCfg(driver_flow, data_source_cfg, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

void DataSourceParserFlowUnitTest::PreparationToGetCert() {
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

TEST_F(DataSourceParserFlowUnitTest, CredentialTest) {
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }
  modelbox::AgencyInfo agency_info;
  agency_info.xrole_name = "admin";
  agency_info.user_domain_name = "user";
  modelbox::UserAgencyCredential user_credential;

  modelbox::Status code =
      modelbox::IAMAuth::GetInstance()->GetUserAgencyProjectCredential(
          user_credential, agency_info);
  if (modelbox::STATUS_OK != code) {
    MBLOG_ERROR << "failed get user project credential";
  }
  EXPECT_EQ(code, modelbox::STATUS_OK);
}

TEST_F(DataSourceParserFlowUnitTest, TokenTest) {
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }
  modelbox::AgencyInfo agency_info;
  agency_info.xrole_name = "admin";
  agency_info.user_domain_name = "user";
  modelbox::UserAgencyToken user_token;
  ProjectInfo project_info;
  project_info.project_name = "cn";

  modelbox::Status code =
      modelbox::IAMAuth::GetInstance()->GetUserAgencyProjectToken(
          user_token, agency_info, project_info);
  if (modelbox::STATUS_OK != code) {
    MBLOG_ERROR << "failed get user project token";
  }
  EXPECT_EQ(code, modelbox::STATUS_OK);
}

modelbox::Status DataSourceParserFlowUnitTest::HandleFunc(
    const web::http::http_request &request) {
  utility::string_t uri = request.request_uri().to_string();
  utility::string_t decode_uri = web::uri::decode(uri);
  std::vector<std::string> uri_vec;
  uri_vec = modelbox::StringSplit(uri, '?');
  std::string file_path = uri_vec[0];
  if (file_path != "/test/get!@*&=Rtsp") {
    utility::string_t resp_body = "Data Not Found";
    request.reply(web::http::status_codes::NotFound, resp_body);
  }
  std::string params = uri_vec[1];
  std::vector<std::string> params_vec;
  params_vec = modelbox::StringSplit(params, '&');
  std::vector<std::string> param_vec;
  std::vector<std::string> param_value;
  for (const auto &i : params_vec) {
    param_vec = modelbox::StringSplit(i, '=');
    param_value.push_back(web::uri::decode(param_vec[1]));
  }

  if (param_value[0] == "1!@#$%^&*()_2" && param_value[1] == "two") {
    utility::string_t resp_body =
        R"({"data" : {"url1" : {"url2" : "rtsp://admin:password@127.0.0.0:808/2"}}})";
    request.reply(web::http::status_codes::OK, resp_body);
  } else {
    utility::string_t resp_body = "Data Not Found";
    request.reply(web::http::status_codes::NotFound, resp_body);
  }

  return modelbox::STATUS_OK;
}

void DataSourceParserFlowUnitTest::MockRestfulServer(
    std::shared_ptr<MockFlow> &driver_flow) {
  std::string request_url = RESTFUL_URL;
  std::shared_ptr<web::http::experimental::listener::http_listener> listener;

  web::http::experimental::listener::http_listener_config server_config;
  server_config.set_timeout(std::chrono::seconds(60));

  std::string cert;
  std::string key;

  GetMockKey(key, cert);

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
  listener->support(web::http::methods::GET,
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

TEST_F(DataSourceParserFlowUnitTest, RestfulInputTest) {
  auto driver_flow = RunDriverFlow(CHECK_SOURCE_OUTPUT_RESTFUL);
  std::string source_type = "restful";
  std::string data_source_cfg_with_params = R"({
        "request_url":")" + std::string(RESTFUL_URL) +
                                            "/test/get!@*&=Rtsp" + R"(",
        "params":[{"param_key":"id","param_value":"1!@#$%^&*()_2"},{"param_key":"name","param_value":"two"}],
        "response_url_position":"data/url1/url2",
        "headers":{"header1":"test1","header2":"test2"}
  })";
  auto ret =
      SendDataSourceCfg(driver_flow, data_source_cfg_with_params, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  std::string data_source_cfg_no_params =
      R"({
        "request_url":")" +
      std::string(RESTFUL_URL) +
      "/test/get!@*&=Rtsp?id=1%21%40%23%24%25%5E%26%2A%28%29_2&name=two" +
      R"(",
        "response_url_position":"data/url1/url2"
  })";
  ret = SendDataSourceCfg(driver_flow, data_source_cfg_no_params, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  MockRestfulServer(driver_flow);
}

}  // namespace modelbox