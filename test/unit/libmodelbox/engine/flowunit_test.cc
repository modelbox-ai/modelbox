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

#include "modelbox/flowunit.h"

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/config.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"
#include "virtualdriver_python.h"

using ::testing::_;

namespace modelbox {

class FlowUnitTest : public testing::Test {
 public:
  FlowUnitTest() {}

 protected:
  MockDriverCtl ctl;

  virtual void SetUp() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    modelbox::DriverDesc desc;
    MockFlowUnitDriverDesc desc_flowunit;

    desc.SetClass("DRIVER-DEVICE");
    desc.SetType("cpu");
    desc.SetName("device-driver-cpu");
    desc.SetDescription("the cpu device");
    desc.SetVersion("8.9.2");
    std::string file_path_device =
        std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
    desc.SetFilePath(file_path_device);
    ctl.AddMockDriverDevice("cpu", desc);
    auto status_drivers_add = drivers->Add(file_path_device);
    EXPECT_EQ(status_drivers_add, STATUS_OK);

    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    Status status_device_init = device_mgr->InitDeviceFactory(drivers);
    EXPECT_EQ(status_device_init, STATUS_OK);

    auto cpu_factory = device_mgr->GetDeviceFactoryList().begin();
    auto mockdevice_factory =
        std::dynamic_pointer_cast<MockDeviceFactory>(cpu_factory->second);
    EXPECT_CALL(*mockdevice_factory, DeviceProbe())
        .WillRepeatedly(testing::Invoke([&]() {
          std::map<std::string, std::shared_ptr<DeviceDesc>> tmp_map;
          std::shared_ptr<DeviceDesc> device_desc =
              std::make_shared<DeviceDesc>();
          device_desc->SetDeviceId("0");
          device_desc->SetDeviceDesc("test desc");
          device_desc->SetDeviceMemory("8Gi");
          device_desc->SetDeviceVersion("xxxx");
          device_desc->SetDeviceType("CPU");
          tmp_map.insert(std::make_pair("0", device_desc));
          return tmp_map;
        }));

    Status status_device_probe = device_mgr->DeviceProbe();
    EXPECT_EQ(status_device_probe, STATUS_OK);

    EXPECT_CALL(*mockdevice_factory, CreateDevice(_))
        .WillRepeatedly(testing::Invoke([&](const std::string& device_id) {
          return std::make_shared<MockDevice>();
        }));

    desc_flowunit.SetClass("DRIVER-FLOWUNIT");
    desc_flowunit.SetType("cpu");
    desc_flowunit.SetName("httpserver");
    desc_flowunit.SetDescription("the cpu httpserver");
    desc_flowunit.SetVersion("1.0.0");
    std::string file_path_flowunit =
        std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-httpserver.so";
    desc_flowunit.SetFilePath(file_path_flowunit);
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    auto device = device_mgr->CreateDevice("cpu", "0");
    EXPECT_NE(device, nullptr);
    device->SetMemQuota(10240);

    auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
    mock_flowunit_desc->SetFlowUnitName("httpserver");
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("input"));
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput("output"));
    mock_flowunit_desc->AddFlowUnitOption(modelbox::FlowUnitOption(
        "ip", "string", true, "127.0.0.1", "input ip"));
    mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
    mock_flowunit->SetBindDevice(device);
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp;
    mock_flowunit_wp = mock_flowunit;
    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(
            testing::Invoke([=](const std::shared_ptr<Configuration>& opts) {
              if (auto spt = mock_flowunit_wp.lock()) {
                auto device = spt->GetBindDevice();
              }
              return modelbox::STATUS_OK;
            }));
    desc_flowunit.SetMockFlowUnit(mock_flowunit);
    ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

    status_drivers_add = drivers->Add(file_path_flowunit);
    EXPECT_EQ(status_drivers_add, STATUS_OK);
  };

  virtual void TearDown() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    std::shared_ptr<FlowUnitManager> flowunit_mgr =
        FlowUnitManager::GetInstance();
    flowunit_mgr->Clear();
    device_mgr->Clear();
    drivers->Clear();
  };
};

TEST_F(FlowUnitTest, InitFlowUnitFactory) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  std::shared_ptr<FlowUnitManager> flowunit_mgr =
      FlowUnitManager::GetInstance();

  flowunit_mgr->InitFlowUnitFactory(drivers);
  auto factory_list = flowunit_mgr->GetFlowUnitFactoryList();
  for (auto iter = factory_list.begin(); iter != factory_list.end(); iter++) {
    EXPECT_EQ(iter->first.first, "cpu");
    EXPECT_EQ(iter->first.second, "httpserver");
    EXPECT_NE(iter->second, nullptr);
  }
}

TEST_F(FlowUnitTest, Probe) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  std::shared_ptr<FlowUnitManager> flowunit_mgr =
      FlowUnitManager::GetInstance();

  Status status1 = flowunit_mgr->InitFlowUnitFactory(drivers);
  Status status2 = flowunit_mgr->FlowUnitProbe();

  EXPECT_EQ(status1, STATUS_OK);
  EXPECT_EQ(status2, STATUS_OK);

  auto desc_list = flowunit_mgr->GetFlowUnitDescList();
  auto iter1 = desc_list.find("cpu");
  EXPECT_EQ(iter1->first, "cpu");

  auto iter2 = iter1->second.find("httpserver");
  EXPECT_EQ(iter2->first, "httpserver");
  auto flowunit_desc = iter2->second;
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  std::vector<FlowUnitInput> input_list = flowunit_desc->GetFlowUnitInput();
  std::vector<FlowUnitOutput> output_list = flowunit_desc->GetFlowUnitOutput();
  std::vector<FlowUnitOption> option_list = flowunit_desc->GetFlowUnitOption();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");
  EXPECT_EQ(option_list[0].GetOptionName(), "ip");
  EXPECT_EQ(option_list[0].GetOptionType(), "string");
  EXPECT_EQ(option_list[0].IsRequire(), true);
  EXPECT_EQ(option_list[0].GetOptionDefault(), "127.0.0.1");
  EXPECT_EQ(option_list[0].GetOptionDesc(), "input ip");
}

TEST_F(FlowUnitTest, CreateFlowUnit) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  auto device_mgr = DeviceManager::GetInstance();
  device_mgr->Initialize(drivers, configbuilder.Build());
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  flowunit_mgr->Initialize(drivers, device_mgr, configbuilder.Build());

  auto flowunit = flowunit_mgr->CreateFlowUnit("httpserver");
  EXPECT_EQ(flowunit[0]->GetBindDevice()->GetDeviceManager()->GetDrivers(),
            drivers);

  EXPECT_EQ(flowunit.size(), 1);
  auto flowunit_desc = flowunit[0]->GetFlowUnitDesc();
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  std::vector<FlowUnitInput> input_list = flowunit_desc->GetFlowUnitInput();
  std::vector<FlowUnitOutput> output_list = flowunit_desc->GetFlowUnitOutput();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");

  auto flowunit_device = flowunit[0]->GetBindDevice();
  auto device_desc = flowunit_device->GetDeviceDesc();
  EXPECT_EQ(device_desc->GetDeviceDesc(), "test desc");
  EXPECT_EQ(device_desc->GetDeviceId(), "0");
  EXPECT_EQ(device_desc->GetDeviceMemory(), "8Gi");
  EXPECT_EQ(device_desc->GetDeviceVersion(), "xxxx");
  EXPECT_EQ(device_desc->GetDeviceType(), "CPU");
  auto config = configbuilder.Build();
  EXPECT_EQ(flowunit[0]->Open(config), STATUS_OK);

  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit_desc = flowunit[0]->GetFlowUnitDesc();
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  input_list = flowunit_desc->GetFlowUnitInput();
  output_list = flowunit_desc->GetFlowUnitOutput();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");
  flowunit_device = flowunit[0]->GetBindDevice();
  device_desc = flowunit_device->GetDeviceDesc();
  EXPECT_EQ(device_desc->GetDeviceDesc(), "test desc");
  EXPECT_EQ(device_desc->GetDeviceId(), "0");
  EXPECT_EQ(device_desc->GetDeviceMemory(), "8Gi");
  EXPECT_EQ(device_desc->GetDeviceVersion(), "xxxx");
  EXPECT_EQ(device_desc->GetDeviceType(), "CPU");
  EXPECT_EQ(flowunit[0]->Open(config), STATUS_OK);

  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu", "0");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit_desc = flowunit[0]->GetFlowUnitDesc();
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  input_list = flowunit_desc->GetFlowUnitInput();
  output_list = flowunit_desc->GetFlowUnitOutput();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");
  flowunit_device = flowunit[0]->GetBindDevice();
  device_desc = flowunit_device->GetDeviceDesc();
  EXPECT_EQ(device_desc->GetDeviceDesc(), "test desc");
  EXPECT_EQ(device_desc->GetDeviceId(), "0");
  EXPECT_EQ(device_desc->GetDeviceMemory(), "8Gi");
  EXPECT_EQ(device_desc->GetDeviceVersion(), "xxxx");
  EXPECT_EQ(device_desc->GetDeviceType(), "CPU");
  EXPECT_EQ(flowunit[0]->Open(config), STATUS_OK);

  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit_desc = flowunit[0]->GetFlowUnitDesc();
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  EXPECT_EQ(flowunit[0]->GetBindDevice()->GetDeviceID(), "0");
}

TEST_F(FlowUnitTest, CreateFlowUnitFail) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  auto device_mgr = DeviceManager::GetInstance();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  ConfigurationBuilder configbuilder;

  flowunit_mgr->Initialize(drivers, device_mgr, configbuilder.Build());

  auto flowunit = flowunit_mgr->CreateFlowUnit("test");
  EXPECT_EQ(flowunit.size(), 0);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cuda");
  EXPECT_EQ(flowunit.size(), 0);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu", "1");
  EXPECT_EQ(flowunit.size(), 0);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0", "1");
  EXPECT_EQ(flowunit.size(), 0);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0,1");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0~1");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0,1;cuda");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0~1;cuda");
  EXPECT_EQ(flowunit.size(), 1);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0:1;cuda");
  EXPECT_EQ(flowunit.size(), 0);
  flowunit = flowunit_mgr->CreateFlowUnit("httpserver", "cpu:0;1;cuda");
  EXPECT_EQ(flowunit.size(), 1);
}

TEST_F(FlowUnitTest, GetFlowUnitDesc) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  auto device_mgr = DeviceManager::GetInstance();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  flowunit_mgr->Initialize(drivers, device_mgr, configbuilder.Build());

  auto flowunit_desc = flowunit_mgr->GetFlowUnitDesc("cpu", "httpserver");
  EXPECT_TRUE(flowunit_desc != nullptr);
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  auto input_list = flowunit_desc->GetFlowUnitInput();
  auto output_list = flowunit_desc->GetFlowUnitOutput();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");

  flowunit_desc = flowunit_mgr->GetFlowUnitDesc("cuda", "httpserver");
  EXPECT_TRUE(flowunit_desc == nullptr);
}

TEST_F(FlowUnitTest, GetAllFlowUnitDesc) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  auto device_mgr = DeviceManager::GetInstance();
  auto flowunit_mgr = FlowUnitManager::GetInstance();
  flowunit_mgr->Initialize(drivers, device_mgr, configbuilder.Build());

  auto flowunit_vec = flowunit_mgr->GetAllFlowUnitDesc();
  EXPECT_EQ(flowunit_vec.size(), 1);
  auto flowunit_desc = flowunit_vec[0];
  EXPECT_EQ(flowunit_desc->GetFlowUnitName(), "httpserver");
  auto input_list = flowunit_desc->GetFlowUnitInput();
  auto output_list = flowunit_desc->GetFlowUnitOutput();
  std::vector<FlowUnitOption> option_list = flowunit_desc->GetFlowUnitOption();
  auto driver_desc = flowunit_desc->GetDriverDesc();
  EXPECT_EQ(input_list[0].GetPortName(), "input");
  EXPECT_EQ(output_list[0].GetPortName(), "output");
  EXPECT_EQ(option_list[0].GetOptionName(), "ip");
  EXPECT_EQ(option_list[0].GetOptionType(), "string");
  EXPECT_EQ(option_list[0].IsRequire(), true);
  EXPECT_EQ(option_list[0].GetOptionDefault(), "127.0.0.1");
  EXPECT_EQ(option_list[0].GetOptionDesc(), "input ip");
  EXPECT_EQ(driver_desc->GetName(), "httpserver");
  EXPECT_EQ(driver_desc->GetType(), "cpu");
  EXPECT_EQ(driver_desc->GetClass(), "DRIVER-FLOWUNIT");
  EXPECT_EQ(driver_desc->GetDescription(), "the cpu httpserver");
  EXPECT_EQ(driver_desc->GetVersion(), "1.0.0");
}

TEST_F(FlowUnitTest, FlowUnitDescCheckGroupType) {
  FlowUnitDesc flow_desc;
  flow_desc.SetFlowUnitGroupType("input");
  EXPECT_TRUE(flow_desc.GetGroupType().empty());

  flow_desc.SetFlowUnitGroupType("Input $@#@");
  EXPECT_TRUE(flow_desc.GetGroupType().empty());

  flow_desc.SetFlowUnitGroupType("Input/http/reply");
  EXPECT_TRUE(flow_desc.GetGroupType().empty());

  flow_desc.SetFlowUnitGroupType("Input");
  EXPECT_EQ(flow_desc.GetGroupType(), "Input");

  flow_desc.SetFlowUnitGroupType("Input321");
  EXPECT_EQ(flow_desc.GetGroupType(), "Input321");

  flow_desc.SetFlowUnitGroupType("Input_321");
  EXPECT_EQ(flow_desc.GetGroupType(), "Input_321");

  flow_desc.SetFlowUnitGroupType("Input/http");
  EXPECT_EQ(flow_desc.GetGroupType(), "Input/http");
}

class VirtualFlowUnitTest : public testing::Test {
 public:
  VirtualFlowUnitTest(){};

  virtual void SetUp() {
    std::string misc_python_src_path = std::string(PYTHON_PATH);
    misc_python_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-python.so";
    CopyFile(misc_python_src_path, misc_python_dest_path, 0, true);

    std::string virtual_python_src_path = std::string(VIRTUAL_PYTHON_PATH);
    virtual_python_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-virtualdriver-python.so";
    CopyFile(virtual_python_src_path, virtual_python_dest_path, 0, true);
  };

  virtual void TearDown() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
    std::shared_ptr<FlowUnitManager> flowunit_mgr =
        FlowUnitManager::GetInstance();
    flowunit_mgr->Clear();
    device_mgr->Clear();
    drivers->Clear();

    remove(misc_python_dest_path.c_str());
    remove(virtual_python_dest_path.c_str());
  };

  MockDriverCtl ctl;

 private:
  std::string misc_python_dest_path;
  std::string virtual_python_dest_path;
};

TEST_F(VirtualFlowUnitTest, VirtualTest) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  configbuilder.AddProperty(DRIVER_DIR, std::string(TEST_ASSETS));
  configbuilder.AddProperty(DRIVER_SKIP_DEFAULT, "true");
  std::shared_ptr<Configuration> config = configbuilder.Build();
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;

  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Initialize(config);
  EXPECT_TRUE(result);
  result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-device-*");
  EXPECT_TRUE(result);

  std::string file_misc_python =
      std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-python.so";
  result = drivers->Add(file_misc_python);
  result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-virtualdriver-python.so");
  drivers->VirtualDriverScan();
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  device_mgr->Initialize(drivers, config);

  auto flowunit_mgr = FlowUnitManager::GetInstance();
  flowunit_mgr->Initialize(drivers, device_mgr, config);
  auto flowunit_python =
      flowunit_mgr->CreateFlowUnit("httpserver_python", "cpu");
  auto desc_python = flowunit_python[0]->GetFlowUnitDesc();
  EXPECT_EQ(desc_python->GetFlowUnitName(), "httpserver_python");
  auto input = desc_python->GetFlowUnitInput();
  auto output = desc_python->GetFlowUnitOutput();
  EXPECT_EQ(input[0].GetPortName(), "image");
  EXPECT_EQ(input[0].GetDeviceType(), "cpu");
  EXPECT_EQ(input[1].GetPortName(), "anchor");
  EXPECT_EQ(output[0].GetPortName(), "output");
  EXPECT_EQ(output[0].GetDeviceType(), "cpu");
  EXPECT_EQ(desc_python->GetConditionType(), NONE);
  EXPECT_EQ(desc_python->GetOutputType(), ORIGIN);
}

}  // namespace modelbox