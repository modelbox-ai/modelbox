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

#include <functional>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iam_auth.h"

#define CHECK_SOURCE_OUTPUT_OBS "check_data_source_obs_parser_output"

using ::testing::_;

namespace modelbox {

class DataSourceObsParserPluginTest : public testing::Test {
 public:
  DataSourceObsParserPluginTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}
  void PreparationToGetCert();

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  virtual void TearDown() { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();
  std::shared_ptr<DriverFlowTest> RunDriverFlow(
      const std::string mock_flowunit_name);
  modelbox::Status SendDataSourceCfg(std::shared_ptr<DriverFlowTest> &driver_flow,
                                   const std::string &data_source_cfg,
                                   const std::string &source_type);

 private:
  Status AddMockFlowUnit();
  Status AddMockObs();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> DataSourceObsParserPluginTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<DriverFlowTest> DataSourceObsParserPluginTest::RunDriverFlow(
    const std::string mock_flowunit_name) {
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
          data_source_parser:stream_meta -> )" +
                             mock_flowunit_name + R"(:stream_meta
        }'''
    format = "graphviz"
  )";

  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun(mock_flowunit_name, toml_content, -1);

  return driver_flow;
}

modelbox::Status DataSourceObsParserPluginTest::SendDataSourceCfg(
    std::shared_ptr<DriverFlowTest> &driver_flow,
    const std::string &data_source_cfg, const std::string &source_type) {
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

Status DataSourceObsParserPluginTest::AddMockFlowUnit() {
  AddMockObs();
  return modelbox::STATUS_OK;
}

Status DataSourceObsParserPluginTest::AddMockObs() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName(CHECK_SOURCE_OUTPUT_OBS);
    desc_flowunit.SetDescription(CHECK_SOURCE_OUTPUT_OBS);
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit = std::string(TEST_DRIVER_DIR) +
                                     "/libmodelbox-unit-cpu-" +
                                     CHECK_SOURCE_OUTPUT_OBS + ".so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName(CHECK_SOURCE_OUTPUT_OBS);
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("stream_meta"));
    mock_flowunit_desc->SetFlowType(modelbox::FlowType::STREAM);
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;

    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration> &flow_option) {
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, DataPre(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
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
                EXPECT_EQ((*source_url).substr((*source_url).rfind('_') + 1),
                          "nv-codec-headers.tar.gz");
              }

              return modelbox::STATUS_SUCCESS;
            }));

    EXPECT_CALL(*mock_flowunit, DataPost(_))
        .WillRepeatedly(
            testing::Invoke([&](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_info "
                         << "DataPost";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit,
                Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
        .WillRepeatedly(
            testing::Invoke([=](std::shared_ptr<DataContext> data_ctx) {
              MBLOG_INFO << "stream_info "
                         << "Process";
              return modelbox::STATUS_OK;
            }));

    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      return modelbox::STATUS_OK;
    }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl_->AddMockDriverFlowUnit(CHECK_SOURCE_OUTPUT_OBS, "cpu", desc_flowunit,
                                std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}

TEST_F(DataSourceObsParserPluginTest, ObsInputTest) {
  // This test would be skipped, if no auth info is provided.
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }

  auto driver_flow = RunDriverFlow(CHECK_SOURCE_OUTPUT_OBS);
  std::string source_type = "obs";

  // construct DATA SOURCE CFG
  std::string obsEndPoint(config_file->GetString("data_source.obsEndPoint"));
  std::string bucket(config_file->GetString("data_source.bucket"));
  std::string path(config_file->GetString("data_source.path"));
  std::string domainName(config_file->GetString("data_source.domainName"));
  std::string xroleName(config_file->GetString("data_source.xroleName"));

  std::string data_source_cfg = R"({
        "obsEndPoint":")" + obsEndPoint +
                                R"(",
        "bucket":")" + bucket + R"(",
        "path":")" + path + R"(",
        "taskid":"test_task_id",
        "domainName":")" + domainName +
                                R"(",
        "xroleName":")" + xroleName +
                                R"("
  })";

  PreparationToGetCert();
  auto ret = SendDataSourceCfg(driver_flow, data_source_cfg, source_type);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

void DataSourceObsParserPluginTest::PreparationToGetCert() {
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));

  std::string ak(config_file->GetString("base.ak").c_str());
  std::string sk(config_file->GetString("base.sk").c_str());
  std::string domain_id(config_file->GetString("base.domain_id").c_str());
  std::string project_id(config_file->GetString("base.project_id").c_str());
  std::string iam_host(config_file->GetString("base.iam_host").c_str());

  modelbox::IAMAuth::GetInstance()->SetIAMHostAddress(iam_host);
  if (modelbox::STATUS_OK != modelbox::IAMAuth::GetInstance()->SetConsigneeInfo(
                               ak, sk, domain_id, project_id)) {
    MBLOG_ERROR << "set Consignee failed";
    return;
  }
}

}  // namespace modelbox