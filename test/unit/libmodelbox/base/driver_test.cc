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

#include "modelbox/base/driver.h"

#include <dlfcn.h>
#include <poll.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "flowunit_mockflowunit.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/config.h"
#include "modelbox/base/driver_utils.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace modelbox {

class DriverTest : public testing::Test {
 public:
  DriverTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    drivers->Clear();
  };
};
static std::string CalCode(const std::vector<std::string> &dirs);

std::string CalCode(const std::vector<std::string> &dirs) {
  int64_t check_sum = 0;
  for (const auto &dir : dirs) {
    std::vector<std::string> drivers_list;
    std::string filter = "libmodelbox-*.so*";
    struct stat s;
    lstat(dir.c_str(), &s);

    if (!S_ISDIR(s.st_mode)) {
      check_sum += s.st_mtim.tv_sec;
      continue;
    }

    Status status = ListFiles(dir, filter, &drivers_list);
    if (drivers_list.size() == 0) {
      continue;
    }

    for (auto &driver_file : drivers_list) {
      struct stat buf;
      auto ret = lstat(driver_file.c_str(), &buf);
      if (ret) {
        continue;
      }

      if (S_ISLNK(buf.st_mode)) {
        continue;
      }
      check_sum += buf.st_mtim.tv_sec;
    }
  }

  return GenerateKey(check_sum);
}

TEST_F(DriverTest, Factory) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  std::string file_path_flowunit =
      std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-httpserver.so";
  desc_flowunit.SetFilePath(file_path_flowunit);
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("ascend");
  desc.SetName("device-driver-ascend");
  desc.SetDescription("the ascend device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-ascend.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("ascend", desc);

  bool result = drivers->Add(file_path_device);
  EXPECT_TRUE(result);
  result = drivers->Add(file_path_flowunit);
  EXPECT_TRUE(result);
  std::vector<std::shared_ptr<Driver>> driver_list =
      drivers->GetAllDriverList();
  EXPECT_EQ(driver_list.size(), 2);
  std::shared_ptr<DriverDesc> desc_fu = driver_list[1]->GetDriverDesc();
  EXPECT_EQ(desc_fu->GetFilePath(), file_path_flowunit);

  std::shared_ptr<FlowUnitFactory> flowunit_factory =
      std::dynamic_pointer_cast<FlowUnitFactory>(
          driver_list[1]->CreateFactory());
  std::shared_ptr<FlowUnit> flowunit =
      flowunit_factory->CreateFlowUnit(desc_fu->GetName(), desc_fu->GetType());
  EXPECT_NE(flowunit.get(), nullptr);

  std::shared_ptr<DriverDesc> desc_de = driver_list[0]->GetDriverDesc();
  EXPECT_EQ(desc_de->GetFilePath(), file_path_device);

  std::shared_ptr<DeviceFactory> device_factory =
      std::dynamic_pointer_cast<DeviceFactory>(driver_list[0]->CreateFactory());
  std::shared_ptr<Device> device =
      device_factory->CreateDevice(desc_de->GetName());
  EXPECT_NE(device.get(), nullptr);
}

TEST_F(DriverTest, ScanFail) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libaifolw-*");
  EXPECT_FALSE(result);
  std::vector<std::shared_ptr<Driver>> driver_list =
      drivers->GetAllDriverList();
  EXPECT_EQ(driver_list.size(), 0);

  result = drivers->Scan(
      std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-httpserver.so", "");
  EXPECT_TRUE(result);
}

TEST_F(DriverTest, ScanSuccess) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cuda");
  desc_flowunit.SetName("resize");
  desc_flowunit.SetDescription("A resize flowunit on GPU");
  desc_flowunit.SetVersion("0.1.2");
  ctl.AddMockDriverFlowUnit("resize", "cuda", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("ascend");
  desc_flowunit.SetName("inference");
  desc_flowunit.SetDescription("A inference flowunit on NPU");
  desc_flowunit.SetVersion("2.0.1");
  ctl.AddMockDriverFlowUnit("inference", "cpu", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("ascend");
  desc.SetName("device-driver-ascend");
  desc.SetDescription("the ascend device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("ascend", desc);

  desc.SetClass("driver-device");
  desc.SetType("cuda");
  desc.SetName("device-driver-cuda");
  desc.SetDescription("the gpu device");
  desc.SetVersion("7.0.0");
  ctl.AddMockDriverDevice("cuda", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-unit-*");
  std::vector<std::shared_ptr<Driver>> driver_list1 =
      drivers->GetAllDriverList();
  EXPECT_TRUE(result);
  EXPECT_EQ(driver_list1.size(), 3);
  for (auto &driver : driver_list1) {
    std::shared_ptr<DriverDesc> desc_unit = driver->GetDriverDesc();
    EXPECT_EQ(desc_unit->GetClass(), "driver-flowunit");
    if (desc_unit->GetName() == "httpserver") {
      EXPECT_EQ(desc_unit->GetType(), "cpu");
      EXPECT_EQ(desc_unit->GetDescription(), "A httpserver flowunit on CPU");
      EXPECT_EQ(desc_unit->GetVersion(), "1.0.0");
    } else if (desc_unit->GetName() == "resize") {
      EXPECT_EQ(desc_unit->GetType(), "cuda");
      EXPECT_EQ(desc_unit->GetDescription(), "A resize flowunit on GPU");
      EXPECT_EQ(desc_unit->GetVersion(), "0.1.2");
    } else {
      EXPECT_EQ(desc_unit->GetType(), "ascend");
      EXPECT_EQ(desc_unit->GetDescription(), "A inference flowunit on NPU");
      EXPECT_EQ(desc_unit->GetVersion(), "2.0.1");
    }
  }
}

TEST_F(DriverTest, Add) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("ascend");
  desc.SetName("device-driver-ascend");
  desc.SetDescription("the ascend device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("ascend", desc);

  std::string file_unit =
      std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-httpserver.so";
  std::string file_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-ascend.so";
  auto result = drivers->Add(file_unit);
  EXPECT_TRUE(result);
  result = drivers->Add(file_device);
  EXPECT_TRUE(result);

  std::vector<std::shared_ptr<Driver>> drivers_list =
      drivers->GetAllDriverList();
  EXPECT_EQ(drivers_list.size(), 2);
  std::shared_ptr<DriverDesc> desc_unit = drivers_list[0]->GetDriverDesc();
  EXPECT_EQ(desc_unit->GetClass(), "driver-flowunit");
  EXPECT_EQ(desc_unit->GetType(), "cpu");
  EXPECT_EQ(desc_unit->GetName(), "httpserver");
  EXPECT_EQ(desc_unit->GetDescription(), "A httpserver flowunit on CPU");
  EXPECT_EQ(desc_unit->GetVersion(), "1.0.0");

  std::shared_ptr<DriverDesc> desc_device = drivers_list[1]->GetDriverDesc();
  EXPECT_EQ(desc_device->GetClass(), "driver-device");
  EXPECT_EQ(desc_device->GetType(), "ascend");
  EXPECT_EQ(desc_device->GetName(), "device-driver-ascend");
  EXPECT_EQ(desc_device->GetDescription(), "the ascend device");
  EXPECT_EQ(desc_device->GetVersion(), "8.9.2");

  result = drivers->Add(file_unit);
  EXPECT_EQ(result, STATUS_EXIST);
}

TEST_F(DriverTest, GetDriverListByClass) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("resize");
  desc_flowunit.SetDescription("A resize flowunit on cpu");
  desc_flowunit.SetVersion("0.1.2");
  ctl.AddMockDriverFlowUnit("resize", "cpu", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("ascend");
  desc.SetName("device-driver-ascend");
  desc.SetDescription("the ascend device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("ascend", desc);

  desc.SetClass("driver-device");
  desc.SetType("cuda");
  desc.SetName("device-driver-cuda");
  desc.SetDescription("the gpu device");
  desc.SetVersion("7.0.0");
  ctl.AddMockDriverDevice("cuda", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "*");
  EXPECT_TRUE(result);

  std::vector<std::shared_ptr<Driver>> device_list =
      drivers->GetDriverListByClass("driver-device");
  EXPECT_EQ(device_list.size(), 2);
  for (auto &device_item : device_list) {
    auto desc_device = device_item->GetDriverDesc();
    EXPECT_EQ(desc_device->GetClass(), "driver-device");
  }

  std::vector<std::shared_ptr<Driver>> flowunit_list =
      drivers->GetDriverListByClass("driver-flowunit");
  EXPECT_EQ(device_list.size(), 2);
  for (auto &flowunit_item : flowunit_list) {
    auto desc_flowunit = flowunit_item->GetDriverDesc();
    EXPECT_EQ(desc_flowunit->GetClass(), "driver-flowunit");
  }
}

TEST_F(DriverTest, GetDriverTypeList) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("resize");
  desc_flowunit.SetDescription("A resize flowunit on cpu");
  desc_flowunit.SetVersion("0.1.2");
  ctl.AddMockDriverFlowUnit("resize", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("ascend");
  desc_flowunit.SetName("inference");
  desc_flowunit.SetDescription("A inference flowunit on NPU");
  desc_flowunit.SetVersion("2.0.1");
  ctl.AddMockDriverFlowUnit("inference", "ascend", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::vector<std::string> type_list =
      drivers->GetDriverTypeList("driver-flowunit");
  EXPECT_EQ(type_list.size(), 2);

  bool found = false;
  for (auto it = type_list.begin(); it != type_list.end(); it++) {
    if ((*it) == "cuda") {
      found = true;
      break;
    }
  }
  EXPECT_FALSE(found);
  EXPECT_EQ(*find(type_list.begin(), type_list.end(), "cpu"), "cpu");
  EXPECT_EQ(*find(type_list.begin(), type_list.end(), "ascend"), "ascend");
}

TEST_F(DriverTest, GetDriverClassList) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("resize");
  desc_flowunit.SetDescription("A resize flowunit on cpu");
  desc_flowunit.SetVersion("0.1.2");
  ctl.AddMockDriverFlowUnit("resize", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("ascend");
  desc_flowunit.SetName("inference");
  desc_flowunit.SetDescription("A inference flowunit on NPU");
  desc_flowunit.SetVersion("2.0.1");
  ctl.AddMockDriverFlowUnit("inference", "cpu", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::vector<std::string> class_list = drivers->GetDriverClassList();
  EXPECT_EQ(class_list.size(), 2);

  bool found = false;
  for (auto it = class_list.begin(); it != class_list.end(); it++) {
    if ((*it) == "test") {
      found = true;
      break;
    }
  }
  EXPECT_FALSE(found);
  EXPECT_EQ(*find(class_list.begin(), class_list.end(), "driver-flowunit"),
            "driver-flowunit");
  EXPECT_EQ(*find(class_list.begin(), class_list.end(), "driver-device"),
            "driver-device");
}

TEST_F(DriverTest, GetDriverNameList) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("resize");
  desc_flowunit.SetDescription("A resize flowunit on cpu");
  desc_flowunit.SetVersion("0.1.2");
  ctl.AddMockDriverFlowUnit("resize", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("ascend");
  desc_flowunit.SetName("inference");
  desc_flowunit.SetDescription("A inference flowunit on NPU");
  desc_flowunit.SetVersion("2.0.1");
  ctl.AddMockDriverFlowUnit("inference", "ascend", desc_flowunit);

  desc.SetClass("driver-device");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  ctl.AddMockDriverDevice("cpu", desc);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::vector<std::string> name_list =
      drivers->GetDriverNameList("driver-flowunit", "cpu");
  EXPECT_EQ(name_list.size(), 2);

  bool found = false;
  for (auto it = name_list.begin(); it != name_list.end(); it++) {
    if ((*it) == "test") {
      found = true;
      break;
    }
  }
  EXPECT_FALSE(found);
  EXPECT_EQ(*find(name_list.begin(), name_list.end(), "httpserver"),
            "httpserver");
  EXPECT_EQ(*find(name_list.begin(), name_list.end(), "resize"), "resize");
}

TEST_F(DriverTest, GetDriver) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.0.0");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::shared_ptr<Driver> driver_success =
      drivers->GetDriver("driver-flowunit", "cpu", "httpserver");
  std::shared_ptr<Driver> driver_fail =
      drivers->GetDriver("driver-flowunit", "cuda", "httpserver");
  std::shared_ptr<Driver> driver_version =
      drivers->GetDriver("driver-flowunit", "cpu", "httpserver", "1.0.0");

  std::shared_ptr<DriverDesc> desc_success = driver_success->GetDriverDesc();
  std::shared_ptr<DriverDesc> desc_version = driver_version->GetDriverDesc();
  EXPECT_EQ(desc_success->GetDescription(), "A httpserver flowunit on CPU");
  EXPECT_EQ(desc_success->GetVersion(), "1.0.0");
  EXPECT_EQ(driver_fail.get(), nullptr);
  EXPECT_EQ(desc_version->GetDescription(), "A httpserver flowunit on CPU");
}

TEST_F(DriverTest, VersionTest) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.1.1");
  ctl.AddMockDriverFlowUnit("httpserver111", "cpu", desc_flowunit);

  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1.2.0");
  ctl.AddMockDriverFlowUnit("httpserver120", "cpu", desc_flowunit);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::shared_ptr<Driver> driver_120 =
      drivers->GetDriver("driver-flowunit", "cpu", "httpserver");
  std::shared_ptr<Driver> driver_111 =
      drivers->GetDriver("driver-flowunit", "cpu", "httpserver", "1.1.1");
  std::shared_ptr<DriverDesc> desc_120 = driver_120->GetDriverDesc();
  std::shared_ptr<DriverDesc> desc_111 = driver_111->GetDriverDesc();
  EXPECT_EQ(desc_120->GetVersion(), "1.2.0");
  EXPECT_EQ(desc_111->GetVersion(), "1.1.1");
}

TEST_F(DriverTest, SetVersionFailTest) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  MockDriverCtl ctl;
  modelbox::DriverDesc desc;
  MockFlowUnitDriverDesc desc_flowunit;
  desc_flowunit.SetClass("driver-flowunit");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName("httpserver");
  desc_flowunit.SetDescription("A httpserver flowunit on CPU");
  desc_flowunit.SetVersion("1111");
  desc_flowunit.SetVersion("1.1");
  desc_flowunit.SetVersion("a.b.c");
  ctl.AddMockDriverFlowUnit("httpserver", "cpu", desc_flowunit);

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-*");
  EXPECT_TRUE(result);

  std::shared_ptr<Driver> driver_version =
      drivers->GetDriver("driver-flowunit", "cpu", "httpserver");

  std::shared_ptr<DriverDesc> desc_version = driver_version->GetDriverDesc();
  EXPECT_EQ(desc_version->GetVersion(), "");
}

TEST_F(DriverTest, DoubleScan) {
  ConfigurationBuilder builder;
  builder.AddProperty(DRIVER_SKIP_DEFAULT, "false");
  std::shared_ptr<Configuration> config = builder.Build();
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  drivers->Initialize(config);
  struct stat buffer;
  if (stat(DEFAULT_SCAN_INFO, &buffer) != -1) {
    EXPECT_EQ(remove(DEFAULT_SCAN_INFO), 0);
  }
  auto status = drivers->Scan();
  EXPECT_EQ(stat(DEFAULT_SCAN_INFO, &buffer), 0);
  auto driver_nums = drivers->GetAllDriverList().size();
  drivers->Clear();
  drivers->Initialize(config);

  std::ifstream ifs(DEFAULT_SCAN_INFO);
  nlohmann::json dump_json;
  ifs >> dump_json;
  std::string read_check_node = dump_json["check_code"];

  std::vector<std::string> dirs{MODELBOX_DEFAULT_DRIVER_PATH};
  auto check_code = CalCode(dirs);
  MBLOG_INFO << "check_code: " << check_code << ", read_check_node: " << read_check_node;
  EXPECT_EQ(check_code, read_check_node);
  status = drivers->Scan();
  auto second_driver_nums = drivers->GetAllDriverList().size();
  EXPECT_EQ(driver_nums, second_driver_nums);
  EXPECT_EQ(status, STATUS_OK);
}

class VirtualDriverTest : public testing::Test {
 public:
  VirtualDriverTest() {}

 protected:
  virtual void SetUp() {
    std::string cpu_python_src_path = std::string(PYTHON_PATH);
    cpu_python_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-python.so";
    CopyFile(cpu_python_src_path, cpu_python_dest_path, 0, true);

    std::string cpu_inference_src_path = std::string(INFERENCE_PATH);
    cpu_inference_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-inference.so";
    CopyFile(cpu_python_src_path, cpu_inference_dest_path, 0, true);

    std::string virtual_python_src_path = std::string(VIRTUAL_PYTHON_PATH);
    virtual_python_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-virtualdriver-python.so";
    CopyFile(virtual_python_src_path, virtual_python_dest_path, 0, true);

    std::string virtual_inference_src_path =
        std::string(VIRTUAL_INFERENCE_PATH);
    virtual_inference_dest_path =
        std::string(TEST_LIB_DIR) + "/libmodelbox-virtualdriver-inference.so";
    CopyFile(virtual_inference_src_path, virtual_inference_dest_path, 0, true);
  };

  virtual void TearDown() {
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    drivers->Clear();
    remove(cpu_python_dest_path.c_str());
    remove(virtual_python_dest_path.c_str());
    remove(cpu_inference_dest_path.c_str());
    remove(virtual_inference_dest_path.c_str());
  };

 private:
  std::string cpu_python_dest_path, cpu_inference_dest_path;
  std::string virtual_python_dest_path, virtual_inference_dest_path;
};

TEST_F(VirtualDriverTest, VirtualDriver) {
  ConfigurationBuilder builder;
  builder.AddProperty(DRIVER_DIR, TEST_ASSETS);
  builder.AddProperty(DRIVER_SKIP_DEFAULT, "true");
  std::shared_ptr<Configuration> config = builder.Build();
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  modelbox::DriverDesc desc;
  MockDriverCtl ctl;
  MockFlowUnitDriverDesc desc_flowunit;

  desc.SetClass("driver-device");
  desc.SetType("ascend");
  desc.SetName("device-driver-ascend");
  desc.SetDescription("the ascend device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-ascend.so";
  desc.SetFilePath(file_path_device);
  ctl.AddMockDriverDevice("ascend", desc);
  bool result = drivers->Initialize(config);
  EXPECT_TRUE(result);

  result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-device-*");
  EXPECT_TRUE(result);
  result = drivers->Add(PYTHON_PATH);
  EXPECT_TRUE(result);
  if (access(INFERENCE_PATH, R_OK) == 0) {
    result = drivers->Add(INFERENCE_PATH);
    EXPECT_TRUE(result);
  }
  result = drivers->Scan(TEST_LIB_DIR, "libmodelbox-virtualdriver-*.so");
  drivers->VirtualDriverScan();
  EXPECT_TRUE(result);
  std::shared_ptr<Driver> driver_python = drivers->GetDriver(
      "DRIVER-FLOWUNIT", "cpu", "httpserver_python", "1.1.1");
  std::shared_ptr<DriverDesc> desc_python_test = driver_python->GetDriverDesc();
  EXPECT_EQ(desc_python_test->GetClass(), "DRIVER-FLOWUNIT");
  EXPECT_EQ(desc_python_test->GetType(), "cpu");
  EXPECT_EQ(desc_python_test->GetName(), "httpserver_python");
  EXPECT_EQ(desc_python_test->GetVersion(), "1.1.1");
  std::string file_path_python =
      std::string(TEST_ASSETS) + "/resize_cpu/virtual_python_test.toml";
  EXPECT_EQ(desc_python_test->GetFilePath(), file_path_python);

  std::shared_ptr<Driver> driver_inference =
      drivers->GetDriver("DRIVER-FLOWUNIT", "cpu", "inference", "1.1.2");
  std::shared_ptr<DriverDesc> desc_inference_test =
      driver_inference->GetDriverDesc();
  EXPECT_EQ(desc_inference_test->GetClass(), "DRIVER-FLOWUNIT");
  EXPECT_EQ(desc_inference_test->GetType(), "cpu");
  EXPECT_EQ(desc_inference_test->GetName(), "inference");
  EXPECT_EQ(desc_inference_test->GetVersion(), "1.1.2");
  std::string file_path_inference =
      std::string(TEST_ASSETS) + "/tensorflow_cpu/virtual_model_test.toml";
  EXPECT_EQ(desc_inference_test->GetFilePath(), file_path_inference);
}

}  // namespace modelbox