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


#include <pybind11/embed.h>

#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace modelbox {
class PythonFlowUnitTest : public testing::Test {
 public:
  PythonFlowUnitTest() : driver_flow_(std::make_shared<DriverFlowTest>()) {}

  std::shared_ptr<DriverFlowTest> GetDriverFlow();

 protected:
  virtual void SetUp() {}

  virtual void TearDown() { driver_flow_->Clear(); };

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS;

 private:
  std::shared_ptr<DriverFlowTest> driver_flow_;
};

std::shared_ptr<DriverFlowTest> PythonFlowUnitTest::GetDriverFlow() {
  return driver_flow_;
}

TEST_F(PythonFlowUnitTest, Init) {
  auto op_dir = test_data_dir + "/python_op";
  std::string toml_content = R"(
    [driver]
    dir=[")" + test_lib_dir + "\",\"" +
                             op_dir + "\"]\n    " +
                             R"(
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                            
    python_image[type=flowunit, flowunit=python_image, device=cpu, deviceid=0, label="<image_out/out_1>", batch_size = 10]   
    python_resize[type=flowunit, flowunit=python_resize, device=cpu, deviceid=0, label="<resize_in> | <resize_out>"]   
    python_brightness[type=flowunit, flowunit=python_brightness, device=cpu, deviceid=0, label="<brightness_in> | <brightness_out>", brightness = 0.1]  
    python_show[type=flowunit, flowunit=python_show, device=cpu, deviceid=0, label="<show_in>", is_save = true]    
    python_image:"image_out/out_1" -> python_resize:resize_in
    python_resize:resize_out -> python_brightness:brightness_in
    python_brightness:brightness_out -> python_show:show_in                                                                                              
}}'''
format = "graphviz"

  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("PythonFlowUnit", toml_content, 0);
  EXPECT_EQ(ret, STATUS_STOP);
}

TEST_F(PythonFlowUnitTest, StatusCount) {
  auto op_dir = test_data_dir + "/python_op";
  std::string toml_content = R"(
    [driver]
    dir=[")" + test_lib_dir + "\",\"" +
                             op_dir + "\"]\n    " +
                             R"(
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                            
    python_image[type=flowunit, flowunit=python_image, device=cpu, deviceid=0, batch_size = 10]   
    python_resize[type=flowunit, flowunit=python_resize, device=cpu, deviceid=0]   
    python_brightness[type=flowunit, flowunit=python_brightness, device=cpu, deviceid=0, brightness = 0.1]  
    python_show[type=flowunit, flowunit=python_show, device=cpu, deviceid=0, is_save = true]    
    python_image:image_out -> python_resize:resize_in
    python_resize:resize_out -> python_brightness:brightness_in
    python_brightness:brightness_out -> python_show:show_in                                                                                              
}}'''
format = "graphviz"

  )";
  auto driver_flow = GetDriverFlow();
  auto ret = driver_flow->BuildAndRun("PythonFlowUnit", toml_content, -1);
  EXPECT_NE(ret, STATUS_OK);

  {
    py::gil_scoped_acquire interpreter_guard{};
    py::object python_status;
    try {
      python_status = py::module::import("_flowunit").attr("Status");
    } catch (const std::exception& ex) {
      MBLOG_ERROR << "import _flowunit.Status failed:" << ex.what();
      EXPECT_TRUE(false);
    }

    try {
      auto obj = python_status(STATUS_LASTFLAG - 1);
    } catch (const std::exception& ex) {
      MBLOG_ERROR << "init _flowunit.Status failed:" << ex.what();
      EXPECT_TRUE(false);
    }

    try {
      auto obj = python_status(STATUS_LASTFLAG);
      EXPECT_TRUE(false);
    } catch (const std::exception& ex) {
      MBLOG_ERROR << "init _flowunit.Status failed:" << ex.what();
      EXPECT_TRUE(true);
    }
  }
}

}  // namespace modelbox