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
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iam_auth.h"

using ::testing::_;

using ObsOutputDataPack = struct _OutputDataPack {
  std::string output_data;
  std::string output_file_name;
  std::string output_broker_names;
};

namespace modelbox {
class OutputBrokerObsPluginTest : public testing::Test {
 public:
  OutputBrokerObsPluginTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}
  void PreparationToGetCert();

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  virtual void TearDown() { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow();
  std::shared_ptr<DriverFlowTest> RunDriverFlow();
  modelbox::Status SendOutputDataObs(
      std::shared_ptr<DriverFlowTest> &driver_flow,
      std::vector<std::shared_ptr<ObsOutputDataPack>> &output_data_pack_vec,
      const std::string &output_broker_cfg);

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> OutputBrokerObsPluginTest::GetDriverFlow() {
  return driver_flow_;
}

std::shared_ptr<DriverFlowTest> OutputBrokerObsPluginTest::RunDriverFlow() {
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
  auto ret = driver_flow->BuildAndRun("null", toml_content, -1);

  return driver_flow;
}

modelbox::Status OutputBrokerObsPluginTest::SendOutputDataObs(
    std::shared_ptr<DriverFlowTest> &driver_flow,
    std::vector<std::shared_ptr<ObsOutputDataPack>> &output_data_pack_vec,
    const std::string &output_broker_cfg) {
  auto ext_data = driver_flow->GetFlow()->CreateExternalDataMap();
  auto buffer_list = ext_data->CreateBufferList();

  std::vector<size_t> tmp_sizes;
  for (auto &pack : output_data_pack_vec) {
    tmp_sizes.push_back(pack->output_data.size());
  }

  buffer_list->Build(tmp_sizes);
  for (size_t i = 0; i < output_data_pack_vec.size(); ++i) {
    auto buffer = buffer_list->At(i);
    auto data_pack = output_data_pack_vec[i];
    memcpy_s(buffer->MutableData(), buffer->GetBytes(),
             data_pack->output_data.data(), data_pack->output_data.size());
    buffer->Set("output_broker_names", data_pack->output_broker_names);
    buffer->Set("output_file_name", data_pack->output_file_name);
  }

  auto config = ext_data->GetSessionConfig();
  config->SetProperty("flowunit.output_broker.config", output_broker_cfg);
  ext_data->Send("input", buffer_list);
  ext_data->Shutdown();
  return modelbox::STATUS_OK;
}

Status OutputBrokerObsPluginTest::AddMockFlowUnit() { return STATUS_OK; }

TEST_F(OutputBrokerObsPluginTest, ObsOutputTest) {
  // This test would be skipped, if no auth info is provided.
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }

  auto driver_flow = RunDriverFlow();
  std::vector<std::shared_ptr<ObsOutputDataPack>> output_data_pack_vec;
  std::shared_ptr<ObsOutputDataPack> output_data_pack =
      std::make_shared<ObsOutputDataPack>();
  output_data_pack->output_broker_names = "obs1|obs2";
  output_data_pack->output_file_name = "text";
  output_data_pack->output_data = "output data text to obs1 & obs2.";
  output_data_pack_vec.push_back(output_data_pack);
  output_data_pack = std::make_shared<ObsOutputDataPack>();
  output_data_pack->output_broker_names = "obs2|obs3";
  output_data_pack->output_file_name = "frame";
  output_data_pack->output_data = "output data text to obs2 & obs3.";
  output_data_pack_vec.push_back(output_data_pack);

  // construct OUTPUT BROKER CFG
  std::string obsEndPoint(config_file->GetString("output_broker.obsEndPoint"));
  std::string bucket(config_file->GetString("output_broker.bucket"));
  std::string path1(config_file->GetString("output_broker.path1"));
  std::string path2(config_file->GetString("output_broker.path2"));
  std::string path3(config_file->GetString("output_broker.path3"));
  std::string domainName(config_file->GetString("output_broker.domainName"));
  std::string xroleName(config_file->GetString("output_broker.xroleName"));

  std::string output_broker_cfg = R"({ 
    "brokers": [
      {
        "type" : "obs",
        "name" : "obs1",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path1 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      },
      {
        "type" : "obs",
        "name" : "obs2",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path2 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      },
      {
        "type" : "obs",
        "name" : "obs3",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path3 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      }
    ]
  })";

  PreparationToGetCert();
  auto ret =
      SendOutputDataObs(driver_flow, output_data_pack_vec, output_broker_cfg);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(1 * 1000);
}


TEST_F(OutputBrokerObsPluginTest, ObsOutputTestWithNoFileName) {
  // This test would be skipped, if no auth info is provided.
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> config_file =
      conf_builder->Build(TEST_ASSETS + std::string("/auth/auth_info.toml"));
  if (config_file == nullptr || config_file->GetString("base.ak").empty()) {
    GTEST_SKIP();
  }

  auto driver_flow = RunDriverFlow();
  std::vector<std::shared_ptr<ObsOutputDataPack>> output_data_pack_vec;
  std::shared_ptr<ObsOutputDataPack> output_data_pack =
      std::make_shared<ObsOutputDataPack>();
  output_data_pack->output_broker_names = "obs1|obs2";
  output_data_pack->output_file_name = "";
  output_data_pack->output_data = "output data text to obs1 & obs2.";
  output_data_pack_vec.push_back(output_data_pack);
  output_data_pack = std::make_shared<ObsOutputDataPack>();
  output_data_pack->output_broker_names = "obs2|obs3";
  output_data_pack->output_file_name = "";
  output_data_pack->output_data = "output data text to obs2 & obs3.";
  output_data_pack_vec.push_back(output_data_pack);

  // construct OUTPUT BROKER CFG
  std::string obsEndPoint(config_file->GetString("output_broker.obsEndPoint"));
  std::string bucket(config_file->GetString("output_broker.bucket"));
  std::string path1(config_file->GetString("output_broker.path1"));
  std::string path2(config_file->GetString("output_broker.path2"));
  std::string path3(config_file->GetString("output_broker.path3"));
  std::string domainName(config_file->GetString("output_broker.domainName"));
  std::string xroleName(config_file->GetString("output_broker.xroleName"));

  std::string output_broker_cfg = R"({ 
    "brokers": [
      {
        "type" : "obs",
        "name" : "obs1",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path1 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      },
      {
        "type" : "obs",
        "name" : "obs2",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path2 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      },
      {
        "type" : "obs",
        "name" : "obs3",
        "cfg": "{\"obsEndPoint\" : \")" + obsEndPoint + R"(\", \"bucket\" : \")" + bucket + R"(\", \"path\" : \")" + path3 + R"(\",\"domainName\" :\")" + domainName + R"(\",\"xroleName\" : \")" + xroleName + R"(\"}" 
      }
    ]
  })";

  PreparationToGetCert();
  auto ret =
      SendOutputDataObs(driver_flow, output_data_pack_vec, output_broker_cfg);
  EXPECT_EQ(ret, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(1 * 1000);
}

void OutputBrokerObsPluginTest::PreparationToGetCert() {
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