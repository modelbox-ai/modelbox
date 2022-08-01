#include <fstream>
#include <string>

#include "modelbox/base/log.h"
#include "modelbox/graph.h"
#include "modelbox/session_context.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"
#include "yolobox_flowunit.h"

using ::testing::_;
namespace modelbox {
class CommonYoloboxFlowUintTest : public testing::Test {
 public:
  CommonYoloboxFlowUintTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  void SetUp() override {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);

    const std::string src_toml = test_data_dir + "/" + test_toml_file;
    yolobox_dir = test_data_dir + "/yolobox/";
    dest_toml = yolobox_dir + test_toml_file;
    mkdir(yolobox_dir.c_str(), 0700);
    CopyFile(src_toml, dest_toml, 0);
  }

  void TearDown() override {
    auto ret = remove(dest_toml.c_str());
    EXPECT_EQ(ret, 0);
    ret = remove(yolobox_dir.c_str());
    EXPECT_EQ(ret, 0);
    driver_flow_ = nullptr;
  }

  std::shared_ptr<MockFlow> GetDriverFlow() { return driver_flow_; }

  const std::string driver_lib_dir = TEST_DRIVER_DIR;
  const std::string test_lib_dir = TEST_LIB_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;
  const std::string test_toml_file = "modelbox.test.yolobox.toml";
  std::string yolobox_dir;
  std::string dest_toml;

  void ReadFile(const char *path, char *buf, int len);

 private:
  Status AddMockFlowUnit() { return STATUS_OK; }
  std::shared_ptr<MockFlow> driver_flow_;
};

void CommonYoloboxFlowUintTest::ReadFile(const char *path, char *buf, int len) {
  std::ifstream fd(path, std::ios::binary);
  fd.read(buf, len);
  fd.close();
}

TEST_F(CommonYoloboxFlowUintTest, Process) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\",\"" +
                             test_data_dir + "\",\"" + driver_lib_dir +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input]
          input2[type=input]
          output1[type=output]
          test_yolobox[type=flowunit, flowunit=test_yolobox, device=cpu, deviceid=0, label="<layer15_conv> | <layer22_conv> | <Out_1>"]
          input1 ->test_yolobox:in_1
          input2 ->test_yolobox:in_2
          test_yolobox:out_1->output1
        }'''
    format = "graphviz"
  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("InitUnit", toml_content, 10);
  auto flow = driver_flow->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();
    auto layer15_conv = ext_data->CreateBufferList();
    layer15_conv->Build({36000});
    layer15_conv->Set("shape", std::vector<size_t>({24, 15, 25}));
    auto *data = (char *)layer15_conv->MutableData();
    ReadFile(TEST_ASSETS "/yolobox/data_36000_0", data, 36000);
    auto status = ext_data->Send("input1", layer15_conv);
    EXPECT_EQ(status, STATUS_OK);

    auto layer22_conv = ext_data->CreateBufferList();
    layer22_conv->Build({144000});
    layer22_conv->Set("shape", std::vector<size_t>({24, 30, 50}));
    data = (char *)layer22_conv->MutableData();
    ReadFile(TEST_ASSETS "/yolobox/data_144000_0", data, 144000);
    status = ext_data->Send("input2", layer22_conv);
    EXPECT_EQ(status, STATUS_OK);

    status = ext_data->Shutdown();
    EXPECT_EQ(status, STATUS_OK);

    OutputBufferList map_buffer_list;

    status = ext_data->Recv(map_buffer_list);
    EXPECT_EQ(status, STATUS_OK);

    auto buffer_list = map_buffer_list["output1"];
    for (size_t batch_idx = 0; batch_idx < buffer_list->Size(); ++batch_idx) {
      auto bbox_count =
          buffer_list->At(batch_idx)->GetBytes() / sizeof(BoundingBox);
      const auto *boxes = static_cast<const BoundingBox *>(
          buffer_list->At(batch_idx)->ConstData());
      std::vector<std::vector<float>> result{
          {741.81, 0.0, 219.01, 229.03, 0.998},
          {1067.44, 1.78, 130.7, 79.52, 0.950},
          {16.24, 815.40, 395.60, 263.49, 0.779}};
      for (size_t bbox_idx = 0; bbox_idx < bbox_count; ++bbox_idx) {
        MBLOG_INFO << " batch_idx:" << batch_idx << " bbox_idx:" << bbox_idx
                   << " [" << boxes[bbox_idx].x_ << " " << boxes[bbox_idx].y_
                   << " " << boxes[bbox_idx].w_ << " " << boxes[bbox_idx].h_
                   << "]"
                   << " score:" << boxes[bbox_idx].score_
                   << ", category:" << boxes[bbox_idx].category_;
        const float w_scale = 1920 / 800.0F;
        const float h_scale = 1080 / 480.0F;
        EXPECT_NEAR(boxes[bbox_idx].x_ * w_scale, result[bbox_idx][0], 0.1);
        EXPECT_NEAR(boxes[bbox_idx].y_ * h_scale, result[bbox_idx][1], 0.1);
        EXPECT_NEAR(boxes[bbox_idx].w_ * w_scale, result[bbox_idx][2], 0.1);
        EXPECT_NEAR(boxes[bbox_idx].h_ * h_scale, result[bbox_idx][3], 0.1);
        EXPECT_NEAR(boxes[bbox_idx].score_, result[bbox_idx][4], 0.01);
      }
    }
  }

  EXPECT_EQ(flow->Wait(3 * 1000), STATUS_TIMEDOUT);
}

}  // namespace modelbox
