
#include "driver_flow_test.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

namespace modelbox {

DriverFlowTest::DriverFlowTest() : flow_(std::make_shared<Flow>()) {}

DriverFlowTest::~DriverFlowTest() {
  flow_ = nullptr;
  ctl_ = nullptr;
}

void DriverFlowTest::Clear() { flow_ = nullptr; }

Status DriverFlowTest::InitFlow(const std::string &name,
                                const std::string &graph) {
  ctl_ = GetMockFlowCtl();

  modelbox::DriverDesc desc;
  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl_->AddMockDriverDevice("cpu", desc, std::string(TEST_DRIVER_DIR));

  desc.SetClass("DRIVER-GRAPHCONF");
  desc.SetType("GRAPHVIZ");
  desc.SetName("GRAPHCONF-GRAPHVIZ");
  desc.SetDescription("graph config parse graphviz");
  desc.SetVersion("0.1.0");
  std::string file_path_graph =
      std::string(TEST_DRIVER_DIR) + "/libmodelbox-graphconf-graphviz.so";
  desc.SetFilePath(file_path_graph);

  ctl_->AddMockDriverGraphConf("graphviz", "", desc,
                               std::string(TEST_DRIVER_DIR));
  auto status = flow_->Init(name, graph);
  return status;
}

Status DriverFlowTest::BuildAndRun(const std::string &name,
                                   const std::string &graph, int timeout) {
  auto ret = InitFlow(name, graph);
  if (!ret) {
    return ret;
  }

  ret = flow_->Build();
  if (!ret) {
    return ret;
  }

  ret = flow_->RunAsync();
  if (!ret) {
    return ret;
  }

  if (timeout < 0) {
    return ret;
  }

  Status retval;
  flow_->Wait(timeout, &retval);
  return retval;
}

std::shared_ptr<MockDriverCtl> DriverFlowTest::GetMockFlowCtl() {
  if (ctl_ == nullptr) {
    ctl_ = std::make_shared<MockDriverCtl>();
  }
  return ctl_;
}

std::shared_ptr<Flow> DriverFlowTest::GetFlow() { return flow_; }

}  // namespace modelbox