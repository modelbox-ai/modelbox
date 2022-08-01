#include <securec.h>

#include <functional>
#include <thread>

#include "data_source_parser_flowunit.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "vcn_restful_client.h"
#include "vcn_restful_wrapper_mock_test.h"

#define CHECK_SOURCE_OUTPUT_VCN_RESTFUL \
  "check_data_source_vcn_restful_parser_output"

using ::testing::_;

namespace modelbox {
class DataSourceVcnRestfulParserPluginTest : public testing::Test {
 public:
  DataSourceVcnRestfulParserPluginTest()
      : driver_flow_(std::make_shared<DriverFlowTest>()) {}
  void MockRestfulServer(std::shared_ptr<DriverFlowTest> &driver_flow);

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  void TearDown() override { driver_flow_->Clear(); };
  std::shared_ptr<DriverFlowTest> GetDriverFlow() { return driver_flow_; }
  std::shared_ptr<DriverFlowTest> RunDriverFlow(std::string mock_flowunit_name);
  modelbox::Status SendDataSourceCfg(
      std::shared_ptr<DriverFlowTest> &driver_flow,
      const std::string &data_source_cfg, const std::string &source_type);

 private:
  Status AddMockFlowUnit();
  Status AddMockVcn();
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest>
DataSourceVcnRestfulParserPluginTest::RunDriverFlow(
    const std::string mock_flowunit_name) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
          input[type=input, device=cpu, deviceid=0]
          data_source_parser[type=flowunit, flowunit=data_source_parser, device=cpu, deviceid=0, vcn_keep_alive_interval_sec=2, label="<data_uri>", plugin_dir=")" +
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
  if (modelbox::STATUS_OK != ret) {
    std::string msg = "Failed to build and run, reason:" + ret.Errormsg();
    MBLOG_ERROR << msg;
  }

  return driver_flow_;
}

modelbox::Status DataSourceVcnRestfulParserPluginTest::SendDataSourceCfg(
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
  std::this_thread::sleep_for(std::chrono::seconds(5));
  ext_data->Shutdown();
  return modelbox::STATUS_OK;
}

Status DataSourceVcnRestfulParserPluginTest::AddMockFlowUnit() {
  AddMockVcn();
  return modelbox::STATUS_OK;
}

Status DataSourceVcnRestfulParserPluginTest::AddMockVcn() {
  auto ctl_ = driver_flow_->GetMockFlowCtl();

  {
    MockFlowUnitDriverDesc desc_flowunit;
    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName(CHECK_SOURCE_OUTPUT_VCN_RESTFUL);
    desc_flowunit.SetDescription(CHECK_SOURCE_OUTPUT_VCN_RESTFUL);
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit = std::string(TEST_DRIVER_DIR) +
                                     "/libmodelbox-unit-cpu-" +
                                     CHECK_SOURCE_OUTPUT_VCN_RESTFUL + ".so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName(CHECK_SOURCE_OUTPUT_VCN_RESTFUL);
    mock_flowunit_desc->AddFlowUnitInput(
        modelbox::FlowUnitInput("stream_meta"));
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
                EXPECT_EQ(*source_url, "https://www.Hello_World.com");
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
    ctl_->AddMockDriverFlowUnit(CHECK_SOURCE_OUTPUT_VCN_RESTFUL, "cpu",
                                desc_flowunit, std::string(TEST_DRIVER_DIR));
  }

  return STATUS_OK;
}

TEST_F(DataSourceVcnRestfulParserPluginTest, VcnInputTest) {
  auto driver_flow = RunDriverFlow(CHECK_SOURCE_OUTPUT_VCN_RESTFUL);
  auto vcn_client = modelbox::VcnRestfulClient::GetInstance(600);
  auto wrapper_mock =
      std::shared_ptr<VcnRestfulWrapper>(new VcnRestfulWrapperMock());
  vcn_client->SetRestfulWrapper(wrapper_mock);

  std::string source_type = "vcn_restful";
  std::string data_source_cfg_1 = R"({
        "userName": "user",
        "password":"password",
        "ip":"192.168.1.1",
        "port":"666",
        "cameraCode":"01234567890123456789#01234567890123456789012345678901",
        "streamType":1
  })";

  std::string data_source_cfg_2 = R"({
        "userName": "user",
        "password":"password",
        "ip":"192.168.1.1",
        "port":"666",
        "cameraCode":"01234567890123456789#01234567890123456789012345678901",
        "streamType":1
  })";

  auto ret_1 = SendDataSourceCfg(driver_flow, data_source_cfg_1, source_type);
  auto ret_2 = SendDataSourceCfg(driver_flow, data_source_cfg_2, source_type);

  EXPECT_EQ(ret_1, modelbox::STATUS_OK);
  EXPECT_EQ(ret_2, modelbox::STATUS_OK);

  driver_flow->GetFlow()->Wait(3 * 1000);
}

}  // namespace modelbox