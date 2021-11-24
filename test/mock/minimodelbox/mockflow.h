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

#ifndef MODELBOX_MOCKFLOW_H_
#define MODELBOX_MOCKFLOW_H_

#include <iostream>
#include <string>

#include "mock_driver_ctl.h"

namespace modelbox {

std::shared_ptr<FlowUnitDesc> GenerateFlowunitDesc(
    std::string name, std::set<std::string> inputs,
    std::set<std::string> outputs);
class MockFlow {
 public:
  MockFlow() { ctl_ = std::make_shared<MockDriverCtl>(); };
  virtual ~MockFlow() { Destroy(); };

  bool Init(bool with_default_flowunit = true);
  void Destroy();
  Status BuildAndRun(const std::string &name, const std::string &graph,
                     int timeout = 15 * 1000);
  std::shared_ptr<Flow> GetFlow();
  std::shared_ptr<Device> GetDevice() {
    auto device_mgr = DeviceManager::GetInstance();
    auto device = device_mgr->GetDevice("cpu", "0");
    if (device == nullptr) {
      MBLOG_ERROR << "create device failed, " << StatusError;
      return nullptr;
    }
    return device;
  }
  std::shared_ptr<MockDriverCtl> GetMockFlowCtl();

  void AddFlowUnitDesc(std::shared_ptr<FlowUnitDesc> flow_desc,
                       std::function<std::shared_ptr<modelbox::FlowUnit>(
                           const std::string &, const std::string &)>
                           create_func,
                       std::string lib_path = TEST_LIB_DIR);

 public:
  void Register_Test_0_1_Flowunit();
  void Register_Test_1_0_Flowunit();
  void Register_Test_0_1_Batch_Flowunit();
  void Register_Test_1_0_Batch_Flowunit();
  void Register_Test_0_1_Batch_Thread_Flowunit();
  void Register_Test_1_0_Batch_Thread_Flowunit();
  void Register_Test_0_2_Flowunit();
  void Register_Test_2_0_Flowunit();
  void Register_Test_Orgin_0_2_Flowunit();
  void Register_Listen_Flowunit();
  void Register_ExternData_Flowunit();
  void Register_Test_2_inputs_2_outputs_Flowunit();
  void Register_Condition_Flowunit();
  void Register_Loop_Flowunit();
  void Register_Loop_End_Flowunit();
  void Register_Half_Condition_Flowunit();
  void Register_Normal_Condition_Flowunit();
  void Register_Expand_Normal_Flowunit();
  void Register_Collapse_Normal_Flowunit();
  void Register_Stream_Add_Flowunit();
  void Register_Add_Flowunit();
  void Register_Wrong_Add_Flowunit();
  void Register_Wrong_Add_2_Flowunit();
  void Register_Scatter_Flowunit();
  void Register_Garther_Flowunit();
  void Register_Garther_Gen_Flowunit();
  void Register_Print_Flowunit();
  void Register_Check_Print_Flowunit();
  void Register_Dynamic_Config_Flowunit();
  void Register_Dynamic_Get_Config_Flowunit();
  void Register_Dynamic_Get_Config_Other_Flowunit();
  void Register_Stream_Info_Flowunit();
  void Register_Stream_Normal_Info_Flowunit();
  void Register_Stream_Normal_Info_2_Flowunit();
  void Register_Stream_Start_Flowunit();
  void Register_Normal_Expand_Start_Flowunit();
  void Register_Stream_Tail_Filter_Flowunit();
  void Register_Stream_Mid_Flowunit();
  void Register_Stream_End_Flowunit();
  void Register_Add_1_Flowunit();
  void Register_Iflow_Add_1_Flowunit();
  void Register_Add_1_And_Error_Flowunit();
  void Register_Test_Condition_Flowunit();
  void Register_Get_Priority_Flowunit();
  void Register_Error_Start_Flowunit();
  void Register_Error_Start_Normal_Flowunit();
  void Register_Error_End_Flowunit();
  void Register_Error_End_Normal_Flowunit();
  void Register_Normal_Start_Flowunit();
  void Register_Expand_Datapre_Error_Flowunit();
  void Register_Normal_Expand_Process_Error_Flowunit();
  void Register_Expand_Process_Error_Flowunit();
  void Register_Normal_Expand_Process_Flowunit();
  void Register_HttpServer_Flowunit();
  void Register_Expand_Process_Flowunit();
  void Register_Simple_Pass_Flowunit();
  void Register_Stream_Simple_Pass_Flowunit();
  void Register_Simple_Error_Flowunit();
  void Register_Stream_Datapre_Error_Flowunit();
  void Register_Stream_In_Process_Error_Flowunit();
  void Register_Stream_Process_Error_Flowunit();
  void Register_Stream_Process_Flowunit();
  void Register_Collapse_Recieve_Error_Flowunit();
  void Register_Normal_Collapse_Recieve_Error_Flowunit();
  void Register_Collapse_Datagrouppre_Error_Flowunit();
  void Register_Normal_Collapse_Datagrouppre_Error_Flowunit();
  void Register_Normal_Collapse_Process_Error_Flowunit();
  void Register_Normal_Collapse_Process_Flowunit();
  void Register_Collapse_Datapre_Error_Flowunit();
  void Register_Collapse_Process_Error_Flowunit();
  void Register_Collapse_Process_Flowunit();
  void Register_Virtual_Stream_Start_Flowunit();
  void Register_Virtual_Stream_Mid_Flowunit();
  void Register_Virtual_Stream_End_Flowunit();
  void Register_Virtual_Expand_Flowunit();
  void Register_Virtual_Stream_Flowunit();
  void Register_Tensorlist_Test_1_Flowunit();
  void Register_Tensorlist_Test_2_Flowunit();
  void Register_Check_Tensorlist_Test_1_Flowunit();
  void Register_Check_Tensorlist_Test_2_Flowunit();
  void Register_Statistic_Test_Flowunit();
  void Register_Slow_Flowunit();

  Status InitFlow(const std::string &name, const std::string &graph);

  std::shared_ptr<MockDriverCtl> ctl_;
  std::shared_ptr<Flow> flow_;
};

class MockFunctionCollection
    : public std::enable_shared_from_this<MockFunctionCollection> {
 public:
  MockFunctionCollection(){};
  virtual ~MockFunctionCollection(){};

  void RegisterOpenFunc(
      std::function<Status(const std::shared_ptr<Configuration> &,
                           std::shared_ptr<MockFlowUnit>)>
          open_func) {
    open_func_ = open_func;
  };
  void RegisterCloseFunc(
      std::function<Status(std::shared_ptr<MockFlowUnit>)> close_func) {
    close_func_ = close_func;
  };
  void RegisterDataGroupPreFunc(
      std::function<Status(std::shared_ptr<DataContext> data_ctx,
                           std::shared_ptr<MockFlowUnit>)>
          data_group_pre_func) {
    data_group_pre_func_ = data_group_pre_func;
  };
  void RegisterDataPreFunc(
      std::function<Status(std::shared_ptr<DataContext> data_ctx,
                           std::shared_ptr<MockFlowUnit>)>
          data_pre_func) {
    data_pre_func_ = data_pre_func;
  };
  void RegisterProcessFunc(
      std::function<Status(std::shared_ptr<DataContext> data_ctx,
                           std::shared_ptr<MockFlowUnit>)>
          process_func) {
    process_func_ = process_func;
  };
  void RegisterDataPostFunc(
      std::function<Status(std::shared_ptr<DataContext> data_ctx,
                           std::shared_ptr<MockFlowUnit>)>
          data_post_func) {
    data_post_func_ = data_post_func;
  };
  void RegisterDataGroupPostFunc(
      std::function<Status(std::shared_ptr<DataContext> data_ctx,
                           std::shared_ptr<MockFlowUnit>)>
          data_group_post_func) {
    data_group_post_func_ = data_group_post_func;
  };

  std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string &,
                                                    const std::string &)>
  GenerateCreateFunc(bool need_sequence = false);

 private:
  std::function<Status(const std::shared_ptr<Configuration> &,
                       std::shared_ptr<MockFlowUnit>)>
      open_func_;
  std::function<Status(std::shared_ptr<MockFlowUnit>)> close_func_;
  std::function<Status(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit>)>
      data_group_pre_func_;
  std::function<Status(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit>)>
      data_pre_func_;
  std::function<Status(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit>)>
      process_func_;
  std::function<Status(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit>)>
      data_post_func_;
  std::function<Status(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit>)>
      data_group_post_func_;
};

}  // namespace modelbox
#endif  // MODELBOX_MOCKFLOW_H_
