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

#include "mockflow.h"

#include <sstream>

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/data_context.h"
#include "modelbox/session_context.h"

using ::testing::_;
namespace modelbox {

std::shared_ptr<FlowUnitDesc> GenerateFlowunitDesc(
    std::string name, std::set<std::string> inputs,
    std::set<std::string> outputs) {
  auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
  mock_flowunit_desc->SetFlowUnitName(name);
  for (auto input : inputs) {
    mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput(input));
  }
  for (auto output : outputs) {
    mock_flowunit_desc->AddFlowUnitOutput(modelbox::FlowUnitOutput(output));
  }
  return mock_flowunit_desc;
}

std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string&,
                                                  const std::string&)>
MockFunctionCollection::GenerateCreateFunc(bool need_sequence) {
  auto function_collections = shared_from_this();
  auto fu_create_func = [=](const std::string& unitname,
                            const std::string& unittype) {
    auto mock_flowunit = std::make_shared<MockFlowUnit>();
    std::weak_ptr<MockFlowUnit> mock_flowunit_wp = mock_flowunit;
    auto hold = function_collections;
    EXPECT_CALL(*mock_flowunit, Open(_))
        .WillRepeatedly(testing::Invoke(
            [=](const std::shared_ptr<modelbox::Configuration>& flow_option) {
              auto mock_flowunit_lock = mock_flowunit_wp.lock();
              MBLOG_DEBUG << unitname << " Open";
              if (open_func_ && mock_flowunit_lock != nullptr) {
                return open_func_(flow_option, mock_flowunit_lock);
              }
              return modelbox::STATUS_OK;
            }));
    EXPECT_CALL(*mock_flowunit, Close()).WillRepeatedly(testing::Invoke([=]() {
      auto mock_flowunit_lock = mock_flowunit_wp.lock();
      MBLOG_DEBUG << unitname << " Close";
      if (close_func_ && mock_flowunit_lock != nullptr) {
        return close_func_(mock_flowunit_lock);
      }
      return modelbox::STATUS_OK;
    }));
    if (need_sequence) {
      ON_CALL(
          *mock_flowunit,
          DataGroupPre(testing::An<std::shared_ptr<modelbox::DataContext>>()))
          .WillByDefault(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataGroupPre";
                if (data_group_pre_func_ && mock_flowunit_lock != nullptr) {
                  return data_group_pre_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      ON_CALL(*mock_flowunit,
              DataPre(testing::An<std::shared_ptr<modelbox::DataContext>>()))
          .WillByDefault(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataPre";
                if (data_pre_func_ && mock_flowunit_lock != nullptr) {
                  return data_pre_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      ON_CALL(*mock_flowunit,
              Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
          .WillByDefault(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " Process";
                if (process_func_ && mock_flowunit_lock != nullptr) {
                  return process_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      ON_CALL(*mock_flowunit,
              DataPost(testing::An<std::shared_ptr<modelbox::DataContext>>()))
          .WillByDefault(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataPost";
                if (data_post_func_ && mock_flowunit_lock != nullptr) {
                  return data_post_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      ON_CALL(
          *mock_flowunit,
          DataGroupPost(testing::An<std::shared_ptr<modelbox::DataContext>>()))
          .WillByDefault(
              testing::Invoke([=](std::shared_ptr<DataContext> ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataGroupPost";
                if (data_group_post_func_ && mock_flowunit_lock != nullptr) {
                  return data_group_post_func_(ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));
    } else {
      EXPECT_CALL(*mock_flowunit, DataGroupPre(_))
          .WillRepeatedly(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataGroupPre";
                if (data_group_pre_func_ && mock_flowunit_lock != nullptr) {
                  return data_group_pre_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      EXPECT_CALL(*mock_flowunit, DataPre(_))
          .WillRepeatedly(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataPre";
                if (data_pre_func_ && mock_flowunit_lock != nullptr) {
                  return data_pre_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      EXPECT_CALL(*mock_flowunit, Process(_))
          .WillRepeatedly(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " Process";
                if (process_func_ && mock_flowunit_lock != nullptr) {
                  return process_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      EXPECT_CALL(*mock_flowunit, DataPost(_))
          .WillRepeatedly(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataPost";
                if (data_post_func_ && mock_flowunit_lock != nullptr) {
                  return data_post_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));

      EXPECT_CALL(*mock_flowunit, DataGroupPost(_))
          .WillRepeatedly(testing::Invoke(
              [=](std::shared_ptr<DataContext> data_ctx) -> Status {
                auto mock_flowunit_lock = mock_flowunit_wp.lock();
                MBLOG_DEBUG << unitname << " DataGroupPost";
                if (data_group_post_func_ && mock_flowunit_lock != nullptr) {
                  return data_group_post_func_(data_ctx, mock_flowunit_lock);
                }
                return STATUS_OK;
              }));
    }

    return mock_flowunit;
  };
  return fu_create_func;
}

void MockFlow::AddFlowUnitDesc(
    std::shared_ptr<FlowUnitDesc> flow_desc,
    std::function<std::shared_ptr<modelbox::FlowUnit>(const std::string& name,
                                                      const std::string& type)>
        create_func,
    std::string lib_path) {
  MockFlowUnitDriverDesc desc_flowunit;
  auto name = flow_desc->GetFlowUnitName();
  desc_flowunit.SetClass("DRIVER-FLOWUNIT");
  desc_flowunit.SetType("cpu");
  desc_flowunit.SetName(name);
  desc_flowunit.SetDescription(name);
  desc_flowunit.SetVersion("1.0.0");
  std::string file_path_flowunit =
      lib_path + "/libmodelbox-unit-cpu-" + name + "so";
  desc_flowunit.SetFilePath(file_path_flowunit);
  desc_flowunit.SetMockFlowUnit(create_func, flow_desc);
  ctl_->AddMockDriverFlowUnit(name, "cpu", desc_flowunit, lib_path);
}

void MockFlow::Register_Test_0_2_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_0_2", {}, {"Out_1", "Out_2"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();

    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto session_ctx = ext_data->GetSessionContext();
    auto session_content = std::make_shared<int>(1111);
    session_ctx->SetPrivate("session", session_content);

    if (!session_ctx) {
      MBLOG_ERROR << "can not get session.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({10 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    for (size_t i = 0; i < 10; i++) {
      data[i] = i;
    }

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];

    auto external = data_ctx->External();
    auto external_data_1 = (int*)(*external)[0]->ConstData();
    auto bytes = external->GetBytes();

    auto output_buf_1 = data_ctx->Output("Out_1");
    auto output_buf_2 = data_ctx->Output("Out_2");

    std::vector<size_t> data_1_shape({bytes});
    output_buf_1->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_buf_1->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_1[i] = external_data_1[i];
    }

    std::vector<size_t> data_2_shape({bytes});
    output_buf_2->Build({data_2_shape});
    auto dev_data_2 = (int*)(output_buf_2->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_2[i] = external_data_1[i] + 10;
    }

    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_0_1_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_0_1", {}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();

    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto session_ctx = ext_data->GetSessionContext();
    auto session_content = std::make_shared<int>(1111);
    session_ctx->SetPrivate("session", session_content);

    if (!session_ctx) {
      MBLOG_ERROR << "can not get session.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({10 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    for (size_t i = 0; i < 10; i++) {
      data[i] = i;
    }

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];

    auto external = data_ctx->External();
    auto external_data_1 = (int*)(*external)[0]->ConstData();
    auto bytes = external->GetBytes();

    auto output_buf_1 = data_ctx->Output("Out_1");

    std::vector<size_t> data_1_shape({bytes});
    output_buf_1->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_buf_1->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_1[i] = external_data_1[i];
    }

    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_1_0_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_1_0", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    int ending;
    bool flag = input_bufs_1->At(0)->Get("ending", ending);
    if (flag) {
      MBLOG_INFO << ending;
    }
    MBLOG_INFO << *((int*)input_bufs_1->ConstData());

    return modelbox::STATUS_OK;
  };

  auto data_post_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_STOP;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_0_1_Batch_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_0_1_batch", {}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();

    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto session_ctx = ext_data->GetSessionContext();
    auto session_content = std::make_shared<int>(1111);
    session_ctx->SetPrivate("session", session_content);

    if (!session_ctx) {
      MBLOG_ERROR << "can not get session.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    std::vector<size_t> buffer_shape(10, sizeof(int));
    buffer_list->Build(buffer_shape);
    for (size_t i = 0; i < 10; i++) {
      auto data = (int*)buffer_list->At(i)->MutableData();
      *data = i;
    }

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];

    auto external = data_ctx->External();
    auto bytes = external->GetBytes();

    auto output_buf_1 = data_ctx->Output("Out_1");

    std::vector<size_t> data_1_shape(bytes / sizeof(int), sizeof(int));
    output_buf_1->Build(data_1_shape);
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      auto dev_data_1 = (int*)(output_buf_1->At(i)->MutableData());
      *dev_data_1 = *((int*)(external->At(i)->ConstData()));
    }

    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_0_1_Batch_Thread_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_0_1_batch_thread", {}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();

  static std::atomic<bool> is_closed(false);
  static std::shared_ptr<std::thread> listener_thread = nullptr;
  static int32_t interval_time = 5 * 1000;

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    interval_time = opts->GetInt32("interval_time", 5000);
    std::packaged_task<void()> task([=]() {
      while (!is_closed) {
        if (interval_time > 0) {
          usleep(interval_time);
        }
        auto ext_data = mock_flowunit->CreateExternalData();

        if (!ext_data) {
          MBLOG_ERROR << "can not get external data.";
        }

        auto session_ctx = ext_data->GetSessionContext();
        auto session_content = std::make_shared<int>(1111);
        session_ctx->SetPrivate("session", session_content);

        if (!session_ctx) {
          MBLOG_ERROR << "can not get session.";
        }

        auto buffer_list = ext_data->CreateBufferList();
        std::vector<size_t> buffer_shape(10, sizeof(int));
        buffer_list->Build(buffer_shape);
        for (size_t i = 0; i < 10; i++) {
          auto data = (int*)buffer_list->At(i)->MutableData();
          *data = i;
        }

        auto status = ext_data->Send(buffer_list);
        if (!status) {
          MBLOG_ERROR << "external data send buffer list failed:" << status;
        }

        status = ext_data->Close();
        if (!status) {
          MBLOG_ERROR << "external data close failed:" << status;
        }
      }
    });

    is_closed = false;
    listener_thread = std::make_shared<std::thread>(std::move(task));
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];

    auto external = data_ctx->External();
    auto bytes = external->GetBytes();

    auto output_buf_1 = data_ctx->Output("Out_1");

    std::vector<size_t> data_1_shape(bytes / sizeof(int), sizeof(int));
    output_buf_1->Build(data_1_shape);
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      auto dev_data_1 = (int*)(output_buf_1->At(i)->MutableData());
      *dev_data_1 = *((int*)(external->At(i)->ConstData()));
    }

    return modelbox::STATUS_OK;
  };

  auto close_func = [=](std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (listener_thread && listener_thread->joinable()) {
      is_closed = true;
      listener_thread->join();
      listener_thread = nullptr;
    }
    return modelbox::STATUS_OK;
  };

  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterCloseFunc(close_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_1_0_Batch_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_1_0_batch", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    int ending;
    bool flag = input_bufs_1->At(0)->Get("ending", ending);
    if (flag) {
      MBLOG_INFO << ending;
    }
    MBLOG_INFO << *((int*)input_bufs_1->ConstData());

    return modelbox::STATUS_OK;
  };

  auto data_post_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_STOP;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_1_0_Batch_Thread_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_1_0_batch_thread", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    int ending;
    bool flag = input_bufs_1->At(0)->Get("ending", ending);
    if (flag) {
      MBLOG_INFO << ending;
    }
    MBLOG_INFO << *((int*)input_bufs_1->ConstData());

    run_count += input_bufs_1->Size();

    return modelbox::STATUS_OK;
  };

  auto data_post_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (MAX_COUNT < run_count) {
      MBLOG_DEBUG << "check reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_2_0_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_2_0", {"In_1", "In_2"}, {});
  mock_desc->SetFlowType(STREAM);
  auto data_post_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_STOP;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_Orgin_0_2_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("test_orgin_0_2", {}, {"Out_1", "Out_2"});
  mock_desc->SetFlowType(STREAM);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto external = data_ctx->External();
    auto external_data_1 = (int*)(*external)[0]->ConstData();
    auto bytes = external->GetBytes();

    auto output_buf_1 = data_ctx->Output("Out_1");
    auto output_buf_2 = data_ctx->Output("Out_2");

    std::vector<size_t> data_1_shape(10, 4);
    output_buf_1->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_buf_1->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_1[i] = external_data_1[i];
    }

    std::vector<size_t> data_2_shape(10, 4);
    output_buf_2->Build(data_2_shape);
    auto dev_data_2 = (int*)(output_buf_2->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_2[i] = external_data_1[i] + 10;
    }

    return modelbox::STATUS_OK;
  };

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({10 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    for (size_t i = 0; i < 10; i++) {
      data[i] = i;
    }

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterOpenFunc(open_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Loop_End_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("loop_end", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(NORMAL);
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");

    auto device = mock_flowunit->GetBindDevice();

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0] * 2;

      output_bufs_1->PushBack(buffer_ptr);
    }

    for (size_t i = 0; i < output_bufs_1->Size(); ++i) {
      int ending;
      input_bufs_1->At(i)->Get("ending", ending);
      output_bufs_1->At(i)->Set("ending", ending);
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Listen_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("listen", {}, {"Out_1", "Out_2"});

  static std::atomic<bool> is_closed(false);
  static std::shared_ptr<std::thread> listener_thread = nullptr;
  static int32_t interval_time = 5 * 1000;

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    interval_time = opts->GetInt32("interval_time", 5000);
    std::packaged_task<void()> task([=]() {
      while (!is_closed) {
        if (interval_time > 0) {
          usleep(interval_time);
        }

        auto ext_data = mock_flowunit->CreateExternalData();
        if (!ext_data) {
          MBLOG_ERROR << "can not get external data.";
          continue;
        }

        auto session = ext_data->GetSessionContext();
        if (!session) {
          MBLOG_ERROR << "can not get session.";
          continue;
        }

        auto buffer_list = ext_data->CreateBufferList();
        TensorList ext_tl(buffer_list);

        constexpr int BUFF_SIZE = 10;
        ext_tl.Build<int>({BUFF_SIZE, {1}});
        auto dev_data = ext_tl.MutableData<int>();
        for (size_t i = 0; i < BUFF_SIZE; ++i) {
          dev_data[i] = i;
        }

        auto status = ext_data->Send(buffer_list);
        if (!status) {
          MBLOG_ERROR << "external data send buffer list failed:" << status;
          continue;
        }

        status = ext_data->Close();
        if (!status) {
          MBLOG_ERROR << "external data close failed:" << status;
          continue;
        }

        MBLOG_DEBUG << "listen send event.";
      }
    });

    is_closed = false;
    listener_thread = std::make_shared<std::thread>(std::move(task));
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> op_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_buf_1 = op_ctx->Output("Out_1");
    auto output_buf_2 = op_ctx->Output("Out_2");
    auto ext_buf = op_ctx->External();

    TensorList output_tl_1(output_buf_1);
    TensorList output_tl_2(output_buf_2);
    TensorList ext_tl_1(ext_buf);

    output_tl_1.Build<int>(ext_tl_1.GetShape());
    output_tl_2.Build<int>(ext_tl_1.GetShape());

    const auto dev_data = ext_tl_1.ConstData<int>();
    auto out_data_1 = output_tl_1.MutableData<int>();
    auto out_data_2 = output_tl_2.MutableData<int>();
    for (size_t i = 0; i < ext_tl_1.Size(); ++i) {
      out_data_1[i] = dev_data[i];
      out_data_2[i] = dev_data[i] + 10;
    }

    return modelbox::STATUS_OK;
  };

  auto close_func = [=](std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (listener_thread && listener_thread->joinable()) {
      is_closed = true;
      listener_thread->join();
      listener_thread = nullptr;
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterCloseFunc(close_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_ExternData_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("extern_data", {}, {"Out_1"});

  static std::atomic<bool> is_closed(false);
  static std::shared_ptr<std::thread> listener_thread = nullptr;
  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    std::packaged_task<void()> task([=]() {
      while (!is_closed) {
        usleep(5 * 1000);
        auto ext_data = mock_flowunit->CreateExternalData();
        if (!ext_data) {
          MBLOG_ERROR << "can not get external data.";
          continue;
        }

        auto session = ext_data->GetSessionContext();
        if (!session) {
          MBLOG_ERROR << "can not get session.";
          continue;
        }

        auto buffer_list = ext_data->CreateBufferList();
        buffer_list->Build({1, 10 * sizeof(int)});
        auto data = (int*)buffer_list->MutableData();
        for (size_t i = 0; i < 10; i++) {
          data[i] = i;
        }

        auto status = ext_data->Send(buffer_list);
        if (!status) {
          MBLOG_ERROR << "external data send buffer list failed:" << status;
          continue;
        }

        status = ext_data->Close();
        if (!status) {
          MBLOG_ERROR << "external data close failed:" << status;
          continue;
        }

        MBLOG_DEBUG << "listen send event.";
      }
    });

    is_closed = false;
    listener_thread = std::make_shared<std::thread>(std::move(task));
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> op_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_buf_1 = op_ctx->Output("Out_1");
    auto ext_buf = op_ctx->External();

    for (auto& buffer : *ext_buf) {
      output_buf_1->PushBack(buffer);
    }

    MBLOG_DEBUG << "test_0_2 gen data";

    return modelbox::STATUS_OK;
  };

  auto close_func = [=](std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (listener_thread && listener_thread->joinable()) {
      is_closed = true;
      listener_thread->join();
      listener_thread = nullptr;
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterCloseFunc(close_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_2_inputs_2_outputs_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("test_2_inputs_2_outputs",
                                        {"In_1", "In_2"}, {"Out_1", "Out_2"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Loop_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("loop", {"In_1"}, {"Out_1", "Out_2"});
  mock_desc->SetLoopType(LOOP);
  mock_desc->SetFlowType(NORMAL);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    int ending;

    auto output_bufs_1 = ctx->Output("Out_1");
    auto output_bufs_2 = ctx->Output("Out_2");

    auto device = mock_flowunit->GetBindDevice();
    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      bool flag = input_bufs_1->At(0)->Get("ending", ending);
      if (!flag) {
        ending = 0;
      }
      auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0] + 1;
      if (ending == 9) {
        output_bufs_2->PushBack(buffer_ptr);
      } else {
        output_bufs_1->PushBack(buffer_ptr);
      }
    }
    ending++;
    for (size_t i = 0; i < output_bufs_1->Size(); ++i) {
      output_bufs_1->At(i)->Set("ending", ending);
    }

    for (size_t i = 0; i < output_bufs_2->Size(); ++i) {
      output_bufs_2->At(i)->Set("ending", ending);
    }
    return modelbox::STATUS_OK;
  };

  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Condition_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("condition", {"In_1"}, {"Out_1", "Out_2"});
  mock_desc->SetConditionType(IF_ELSE);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");
    auto output_bufs_2 = ctx->Output("Out_2");

    auto device = mock_flowunit->GetBindDevice();

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0];
      if (input_data[0] % 2 == 0) {
        output_bufs_1->PushBack(buffer_ptr);
      } else {
        output_bufs_2->PushBack(buffer_ptr);
      }
    }
    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Half_Condition_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("half-condition", {"In_1"}, {"Out_1", "Out_2"});
  mock_desc->SetConditionType(IF_ELSE);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");
    auto output_bufs_2 = ctx->Output("Out_2");

    auto device = mock_flowunit->GetBindDevice();

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0];
      if (input_data[0] >= 5) {
        output_bufs_1->PushBack(buffer_ptr);
      } else {
        output_bufs_2->PushBack(buffer_ptr);
      }
    }
    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Normal_Condition_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("normal-condition", {"In_1"}, {"Out_1", "Out_2"});
  mock_desc->SetConditionType(IF_ELSE);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");
    auto output_bufs_2 = ctx->Output("Out_2");

    auto device = mock_flowunit->GetBindDevice();

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0];
      if (input_data[0] >= 5) {
        output_bufs_1->PushBack(buffer_ptr);
      } else {
        output_bufs_2->PushBack(buffer_ptr);
      }
    }
    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Expand_Normal_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("expand_normal", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");

    auto input_data = (int*)(*input_bufs_1)[0]->ConstData();
    std::vector<size_t> data_shape(5, 4);
    output_bufs_1->Build(data_shape);
    auto output_data = (int*)output_bufs_1->MutableData();
    for (uint32_t j = 0; j < 5; j++) {
      output_data[j] = input_data[0] + j;
    }

    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Collapse_Normal_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("collapse_normal", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");
    std::vector<size_t> data_shape(1, 4);
    output_bufs_1->Build(data_shape);
    auto output_data = (int*)output_bufs_1->MutableData();
    auto input_data = (int*)input_bufs_1->ConstData();
    output_data[0] = 0;
    for (uint32_t j = 0; j < 5; j++) {
      output_data[0] += input_data[j];
    }

    return modelbox::STATUS_OK;
  };
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

Status Add_Funciton(std::shared_ptr<DataContext> ctx,
                    std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto input_bufs_1 = ctx->Input("In_1");
  auto input_bufs_2 = ctx->Input("In_2");
  auto output_bufs = ctx->Output("Out_1");

  if (input_bufs_1->Size() <= 0 || input_bufs_2->Size() <= 0) {
    return STATUS_FAULT;
  }

  std::vector<size_t> shape(input_bufs_1->Size(),
                            (*input_bufs_1)[0]->GetBytes());
  output_bufs->Build(shape);
  for (size_t i = 0; i < input_bufs_1->Size(); ++i) {
    auto input_data_1 = (int*)(*input_bufs_1)[i]->ConstData();
    auto input_data_2 = (int*)(*input_bufs_2)[i]->ConstData();
    auto output_data = (int*)(*output_bufs)[i]->MutableData();
    auto data_size = (*input_bufs_1)[i]->GetBytes() / sizeof(int);
    for (size_t j = 0; j < data_size; ++j) {
      output_data[j] = input_data_1[j] + input_data_2[j];
    }
  }
  return modelbox::STATUS_OK;
}

void MockFlow::Register_Stream_Add_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_add", {"In_1", "In_2"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetStreamSameCount(true);

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Add_Funciton);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Add_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("add", {"In_1", "In_2"}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Add_Funciton);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

Status Wrong_Add_Funciton(std::shared_ptr<DataContext> ctx,
                          std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto input_bufs_1 = ctx->Input("In_1");
  auto input_bufs_2 = ctx->Input("In_2");
  auto output_bufs = ctx->Output("Out_1");

  if (input_bufs_1->Size() <= 0 || input_bufs_2->Size() <= 0) {
    return STATUS_FAULT;
  }

  std::vector<size_t> shape(input_bufs_1->Size() + 2,
                            (*input_bufs_1)[0]->GetBytes());
  output_bufs->Build(shape);
  for (size_t i = 0; i < input_bufs_1->Size(); ++i) {
    auto input_data_1 = (int*)(*input_bufs_1)[i]->ConstData();
    auto input_data_2 = (int*)(*input_bufs_2)[i]->ConstData();
    auto output_data = (int*)(*output_bufs)[i]->MutableData();
    auto data_size = (*input_bufs_1)[i]->GetBytes() / sizeof(int);
    for (size_t j = 0; j < data_size; ++j) {
      output_data[j] = input_data_1[j] + input_data_2[j];
    }
  }
  return modelbox::STATUS_OK;
}

void MockFlow::Register_Wrong_Add_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("wrong_add", {"In_1", "In_2"}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Wrong_Add_Funciton);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Wrong_Add_2_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("wrong_add_2", {"In_1", "In_2"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetStreamSameCount(true);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Wrong_Add_Funciton);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Scatter_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("scatter", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(EXPAND);

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = ctx->Input("In_1");
    auto output_bufs = ctx->Output("Out_1");

    if (input_bufs->Size() != 1) {
      return STATUS_FAULT;
    }

    auto size_byte = (*input_bufs)[0]->GetBytes() / sizeof(int);
    auto input_data_1 = (int*)(*input_bufs)[0]->ConstData();
    std::vector<size_t> output_shape(size_byte, 1 * sizeof(int));
    output_bufs->Build(output_shape);
    auto output_data_2 = (int*)(output_bufs->MutableData());
    for (uint32_t i = 0; i < size_byte; i++) {
      output_data_2[i] = input_data_1[i];
    }
    return modelbox::STATUS_OK;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

std::function<Status(std::shared_ptr<DataContext> data_ctx,
                     std::shared_ptr<MockFlowUnit>)>
Generate_Garther_function(int32_t i) {
  return [=](std::shared_ptr<DataContext> ctx,
             std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = ctx->Input("In_1");
    auto output_bufs = ctx->Output("Out_1");

    uint32_t total_size = 0;
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      total_size += (*input_bufs)[i]->GetBytes();
    }
    std::vector<size_t> output_shape(i, total_size);
    output_bufs->Build(output_shape);
    auto out_data = (int*)(output_bufs->MutableData());

    size_t z = 0;
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto size_byte = (*input_bufs)[i]->GetBytes() / sizeof(int);
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      for (uint32_t j = 0; j < size_byte; j++) {
        out_data[z] = input_data[j];
        z++;
      }
    }
    return modelbox::STATUS_OK;
  };
}

void MockFlow::Register_Garther_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("garther", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Generate_Garther_function(1));
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Garther_Gen_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("garther_gen_more", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Generate_Garther_function(2));
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Print_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("print", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_bufs = ctx->Input("In_1");

    std::stringstream ostr;
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      auto data_size = (*input_bufs)[i]->GetBytes() / sizeof(int);
      for (size_t j = 0; j < data_size; ++j) {
        ostr << input_data[j] << " ";
      }
    }

    MBLOG_DEBUG << ostr.str();

    if (MAX_COUNT < run_count++) {
      MBLOG_DEBUG << "print reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterOpenFunc(open_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Check_Print_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("check_print", {"IN1", "IN2", "IN3"}, {});

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;
  static std::atomic<bool> is_print(false);

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("IN1");
    auto input_bufs_2 = ctx->Input("IN2");
    auto input_bufs_3 = ctx->Input("IN3");

    if (input_bufs_1->Size() == 0 || input_bufs_2->Size() == 0 ||
        input_bufs_3->Size() == 0 ||
        (input_bufs_1->Size() != input_bufs_2->Size()) ||
        (input_bufs_2->Size() != input_bufs_3->Size())) {
      return modelbox::STATUS_SUCCESS;
    }

    for (size_t i = 0; i < input_bufs_1->Size(); ++i) {
      const auto in_data_1 = (int*)input_bufs_1->ConstBufferData(i);
      const auto in_data_2 = (int*)input_bufs_2->ConstBufferData(i);
      const auto in_data_3 = (int*)input_bufs_3->ConstBufferData(i);
      if (in_data_3[0] != in_data_1[0] + in_data_2[0]) {
        return STATUS_SHUTDOWN;
      }
    }

    static auto begin_time = GetTickCount();
    static std::atomic<uint64_t> print_time{GetTickCount()};

    run_count += input_bufs_1->Size();
    if (MAX_COUNT < run_count) {
      MBLOG_DEBUG << "check reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }

    auto end_time = GetTickCount();
    if (end_time - print_time > 1000) {
      auto expected = false;
      if (is_print.compare_exchange_weak(expected, true)) {
        MBLOG_INFO << "Average throughput: "
                   << (run_count * 1000) / (end_time - begin_time) << "/s";
        is_print = false;
        print_time = GetTickCount();
      }
    }

    return modelbox::STATUS_SUCCESS;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterOpenFunc(open_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Dynamic_Config_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("dynamic_config", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }
    auto config = ext_data->GetSessionConfig();
    config->SetProperty("nodes.test", "nodes.test");
    config->SetProperty("flowunit.dynamic_get_config.test",
                        "flowunit.dynamic_get_config.test");
    config->SetProperty("node.dynamic_get_config_1.test",
                        "node.dynamic_get_config_1.test");

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({3 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    data[0] = 0;
    data[1] = 15;
    data[2] = 3;

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "listen send event.";
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto external = data_ctx->External();
    auto external_data_1 = (int*)(*external)[0]->ConstData();
    auto bytes = external->GetBytes();

    std::vector<size_t> data_1_shape({bytes});
    output_bufs->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_bufs->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_1[i] = external_data_1[i];
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterOpenFunc(open_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Dynamic_Get_Config_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("dynamic_get_config", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    auto output_bufs_1 = data_ctx->Output("Out_1");

    auto device = mock_flowunit->GetBindDevice();
    auto config = data_ctx->GetSessionConfig();
    auto test = config->GetProperty("test", std::string(""));

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_buffer = (*input_bufs_1)[i];
      auto input_data = (int*)input_buffer->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      buffer_ptr->Set("test", test);
      output_data[0] = input_data[0];
      output_bufs_1->PushBack(buffer_ptr);
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Dynamic_Get_Config_Other_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("dynamic_get_config_other", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    auto output_bufs_1 = data_ctx->Output("Out_1");

    auto device = mock_flowunit->GetBindDevice();

    auto config = data_ctx->GetSessionConfig();
    auto test = config->GetProperty("test", std::string(""));

    for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
      auto input_buffer = (*input_bufs_1)[i];
      auto input_data = (int*)input_buffer->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      buffer_ptr->Set("test", test);
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[0];
      output_bufs_1->PushBack(buffer_ptr);
    }
    return modelbox::STATUS_OK;
    ;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

std::function<Status(const std::shared_ptr<Configuration>&,
                     std::shared_ptr<MockFlowUnit>)>
Genrate_Stream_Open(uint32_t i) {
  return [=](const std::shared_ptr<Configuration>& opts,
             std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto session_ctx = ext_data->GetSessionContext();
    auto session_content = std::make_shared<int>(1111);
    session_ctx->SetPrivate("session", session_content);

    if (!session_ctx) {
      MBLOG_ERROR << "can not get session.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({3 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    data[0] = 0;
    data[1] = i;
    data[2] = 3;

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "listen send event.";
    return modelbox::STATUS_OK;
  };
}

Status Stream_Process(std::shared_ptr<DataContext> data_ctx,
                      std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto session_ctx = data_ctx->GetSessionContext();
  auto session_content = (int*)(session_ctx->GetPrivate("session").get());
  MBLOG_INFO << "session_content is " << session_content[0];

  auto output_bufs = data_ctx->Output("Out_1");
  auto external = data_ctx->External();
  auto external_data_1 = (int*)(*external)[0]->ConstData();
  auto bytes = external->GetBytes();

  std::vector<size_t> data_1_shape({bytes});
  output_bufs->Build(data_1_shape);
  auto dev_data_1 = (int*)(output_bufs->MutableData());
  for (size_t i = 0; i < bytes / sizeof(int); ++i) {
    dev_data_1[i] = external_data_1[i];
  }
  return modelbox::STATUS_OK;
}

Status Stream_DataPre(std::shared_ptr<DataContext> data_ctx,
                      std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto output_meta = std::make_shared<DataMeta>();
  auto magic_num = std::make_shared<int>(3343);
  output_meta->SetMeta("magic_num", magic_num);
  data_ctx->SetOutputMeta("Out_1", output_meta);
  return modelbox::STATUS_OK;
}

Status Stream_DataPost(std::shared_ptr<DataContext> data_ctx,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
  return modelbox::STATUS_OK;
}

void MockFlow::Register_Stream_Info_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_info", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Stream_Process);
  mock_funcitons->RegisterDataPreFunc(Stream_DataPre);
  mock_funcitons->RegisterOpenFunc(Genrate_Stream_Open(15));
  mock_funcitons->RegisterDataPostFunc(Stream_DataPost);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Normal_Info_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_normal_info", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Stream_Process);
  mock_funcitons->RegisterDataPreFunc(Stream_DataPre);
  mock_funcitons->RegisterOpenFunc(Genrate_Stream_Open(25));
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Normal_Info_2_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_normal_info_2", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({8, 8});
    auto data = (int*)buffer_list->MutableData();
    data[0] = 0;
    data[1] = 5;

    data[2] = 0;
    data[3] = 10;

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "listen send event.";
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto external = data_ctx->External();
    auto external_data_1 = (int*)(*external)[0]->ConstData();
    auto bytes = external->GetBytes();

    std::vector<size_t> data_1_shape(2, 8);
    output_bufs->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_bufs->MutableData());
    for (size_t i = 0; i < bytes / sizeof(int); ++i) {
      dev_data_1[i] = external_data_1[i];
    }

    return modelbox::STATUS_OK;
  };

  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPreFunc(Stream_DataPre);
  mock_funcitons->RegisterOpenFunc(open_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Start_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_start", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(EXPAND);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto now_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("now_index")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    std::vector<size_t> shape(5, sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < 5; ++i) {
      auto output_data = (int*)(*output_bufs)[i]->MutableData();
      output_data[0] = now_index + i;
    }
    now_index = now_index + 5;
    auto now_index_content = std::make_shared<int>(now_index);
    data_ctx->SetPrivate("now_index", now_index_content);
    if (now_index + 5 <= end_index) {
      auto event = std::make_shared<FlowUnitEvent>();
      data_ctx->SendEvent(event);
      return modelbox::STATUS_CONTINUE;
    } else {
      return modelbox::STATUS_OK;
    }
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    MBLOG_INFO << "session_content is " << session_content[0];

    auto input_bufs = data_ctx->Input("In_1");
    auto input_data = (int*)(*input_bufs)[0]->ConstData();
    auto start_index = input_data[0];
    auto end_index = input_data[1];
    auto interval = input_data[2];
    auto start_index_content = std::make_shared<int>(start_index);
    data_ctx->SetPrivate("now_index", start_index_content);
    auto end_index_content = std::make_shared<int>(end_index);
    data_ctx->SetPrivate("end_index", end_index_content);
    auto interval_content = std::make_shared<int>(interval);
    auto output_meta = std::make_shared<DataMeta>();
    output_meta->SetMeta("start_index", start_index_content);
    output_meta->SetMeta("end_index", end_index_content);
    output_meta->SetMeta("interval", interval_content);
    data_ctx->SetOutputMeta("Out_1", output_meta);
    return modelbox::STATUS_OK;
  };

  auto data_post_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_OK;
  };

  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Normal_Expand_Start_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("normal_expand_start", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto event = data_ctx->Event();
    if (event == nullptr) {
      auto input_bufs = data_ctx->Input("In_1");
      auto input_data = (int*)(*input_bufs)[0]->ConstData();
      auto start_index = input_data[0];
      auto end_index = input_data[1];
      auto start_index_content = std::make_shared<int>(start_index);
      data_ctx->SetPrivate("now_index", start_index_content);
      auto end_index_content = std::make_shared<int>(end_index);
      data_ctx->SetPrivate("end_index", end_index_content);
    }

    auto output_bufs = data_ctx->Output("Out_1");
    auto now_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("now_index")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    std::vector<size_t> shape(5, sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < 5; ++i) {
      auto output_data = (int*)(*output_bufs)[i]->MutableData();
      output_data[0] = now_index + i;
    }
    now_index = now_index + 5;
    auto now_index_content = std::make_shared<int>(now_index);
    data_ctx->SetPrivate("now_index", now_index_content);
    if (now_index + 5 <= end_index) {
      auto event = std::make_shared<FlowUnitEvent>();
      data_ctx->SendEvent(event);
      return modelbox::STATUS_CONTINUE;
    } else {
      return modelbox::STATUS_OK;
    }
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Tail_Filter_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_tail_filter", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");

    auto device = mock_flowunit->GetBindDevice();

    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      if (input_data[0] < 10) {
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = input_data[0];
        output_bufs->PushBack(buffer_ptr);
      }

      if (input_data[0] == end_index - 1) {
        return modelbox::STATUS_OK;
      }
    }
    return modelbox::STATUS_OK;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto meta = data_ctx->GetInputMeta("In_1");
    auto start_index =
        *(std::static_pointer_cast<int>(meta->GetMeta("start_index")).get());
    auto end_index =
        *(std::static_pointer_cast<int>(meta->GetMeta("end_index")).get());
    data_ctx->SetPrivate("end_index", std::make_shared<int>(end_index));
    auto output_meta = std::make_shared<DataMeta>();
    output_meta->SetMeta("start_index", std::make_shared<int>(start_index));
    output_meta->SetMeta("end_index", std::make_shared<int>(end_index));
    data_ctx->SetOutputMeta("Out_1", output_meta);

    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Mid_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_mid", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");

    auto device = mock_flowunit->GetBindDevice();

    auto interval = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("interval")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      if (input_data[0] % interval == 0) {
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = input_data[0];
        output_bufs->PushBack(buffer_ptr);
      }

      if (input_data[0] == end_index - 1) {
        return modelbox::STATUS_OK;
      }
    }
    return modelbox::STATUS_CONTINUE;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto session_ctx = data_ctx->GetSessionContext();
    auto session_content = (int*)(session_ctx->GetPrivate("session").get());
    auto meta = data_ctx->GetInputMeta("In_1");

    std::shared_ptr<Device> device;
    auto interval =
        *(std::static_pointer_cast<int>(meta->GetMeta("interval")).get());
    auto end_index =
        *(std::static_pointer_cast<int>(meta->GetMeta("end_index")).get());
    data_ctx->SetPrivate("interval", std::make_shared<int>(interval));
    data_ctx->SetPrivate("end_index", std::make_shared<int>(end_index));

    MBLOG_INFO << "session_content is " << session_content[0];
    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_End_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_end", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(COLLAPSE);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = data_ctx->Input("In_1");
    auto data_meta = data_ctx->GetInputGroupMeta("In_1");
    if (data_meta != nullptr) {
      auto magic_num =
          *(std::static_pointer_cast<int>(data_meta->GetMeta("magic_num")));
      MBLOG_INFO << "Data Process magic_num " << magic_num;
    }

    if (data_ctx->GetPrivate("total_count") == nullptr) {
      auto total_count = std::make_shared<int>(0);
      data_ctx->SetPrivate("total_count", total_count);
    }
    auto total_count =
        *(std::static_pointer_cast<int>(data_ctx->GetPrivate("total_count"))
              .get());

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      total_count += input_data[0];
      if (input_data[0] == 12) {
        auto output_bufs = data_ctx->Output("Out_1");
        auto device = mock_flowunit->GetBindDevice();
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = total_count;
        output_bufs->PushBack(buffer_ptr);
        return modelbox::STATUS_OK;
      }
    }
    auto new_total_count = std::make_shared<int>(total_count);
    data_ctx->SetPrivate("total_count", new_total_count);

    return modelbox::STATUS_OK;
  };

  auto data_group_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    MBLOG_INFO << "stream_end "
               << "DataGroupPre";
    auto session_ctx = data_ctx->GetSessionContext();
    if (session_ctx->GetPrivate("session") != nullptr) {
      auto session_content = (int*)(session_ctx->GetPrivate("session").get());
      MBLOG_INFO << "session_content is " << session_content[0];
    }
    auto data_meta = data_ctx->GetInputGroupMeta("In_1");
    if (data_meta != nullptr) {
      auto magic_num =
          *(std::static_pointer_cast<int>(data_meta->GetMeta("magic_num")));
      MBLOG_INFO << "DataGroupPre magic_num " << magic_num;
    }

    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataGroupPreFunc(data_group_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Add_1_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("add_1", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_bufs = ctx->Input("In_1");
    auto output_bufs = ctx->Output("Out_1");

    if (input_bufs->Size() <= 0) {
      return STATUS_FAULT;
    }

    std::vector<size_t> shape(input_bufs->Size(), 1 * sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)input_bufs->At(i)->ConstData();
      auto output_data = (int*)output_bufs->At(i)->MutableData();
      *output_data = *input_data + 1;
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Iflow_Add_1_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("iflow_add_1", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = data_ctx->Input("In_1");
    auto output_bufs = data_ctx->Output("Out_1");

    if (input_bufs->Size() <= 0) {
      return STATUS_FAULT;
    }

    std::vector<size_t> shape(input_bufs->Size(), 1 * sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)((*input_bufs)[i]->ConstData());
      auto output_data = (int*)output_bufs->At(i)->MutableData();
      *output_data = *input_data + 1;
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Add_1_And_Error_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("add_1_and_error", {"In_1"}, {"Out_1"});

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_bufs = ctx->Input("In_1");
    auto output_bufs = ctx->Output("Out_1");

    std::vector<size_t> shape(input_bufs->Size(), 1 * sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)input_bufs->At(i)->ConstData();
      if (*input_data == 10) {
        return modelbox::STATUS_INVALID;
      }

      auto output_data = (int*)output_bufs->At(i)->MutableData();
      *output_data = *input_data + 1;
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Test_Condition_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("test_condition", {"In_1"}, {"Out_1", "Out_2"});
  mock_desc->SetConditionType(IF_ELSE);
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = ctx->Input("In_1");
    auto output_bufs_1 = ctx->Output("Out_1");
    auto output_bufs_2 = ctx->Output("Out_2");

    auto device = mock_flowunit->GetBindDevice();
    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)input_bufs->At(i)->ConstData();
      auto buffer = std::make_shared<Buffer>(device);
      buffer->Build(1 * sizeof(int));
      auto output_data = (int*)buffer->MutableData();
      *output_data = *input_data;

      if (*input_data == 10) {
        return STATUS_INVALID;
      }

      if (*input_data % 2 == 0) {
        output_bufs_1->PushBack(buffer);
      } else {
        output_bufs_2->PushBack(buffer);
      }
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Get_Priority_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("get_priority", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Error_Start_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("error_start", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({1 * sizeof(int)});
    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }
    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "error start";
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Error_Start_Normal_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("error_start_normal", {}, {"Out_1"});
  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({1 * sizeof(int)});
    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }
    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "error start";
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    MBLOG_INFO << "error start process.";
    return modelbox::STATUS_INVALID;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Normal_Start_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("normal_start", {}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto open_func = [=](const std::shared_ptr<Configuration>& opts,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    auto ext_data = mock_flowunit->CreateExternalData();
    if (!ext_data) {
      MBLOG_ERROR << "can not get external data.";
    }

    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({2 * sizeof(int)});
    auto data = (int*)buffer_list->MutableData();
    data[0] = 0;
    data[1] = 16;

    auto status = ext_data->Send(buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    MBLOG_INFO << "expand start";
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto external = data_ctx->External();
    auto external_data_1 = (uint32_t*)(*external)[0]->ConstData();

    auto start = external_data_1[0];
    auto end = external_data_1[1];

    std::vector<size_t> data_1_shape(4, {(end - start) * sizeof(uint32_t) / 4});
    output_bufs->Build(data_1_shape);
    auto dev_data_1 = (int*)(output_bufs->MutableData());
    for (size_t i = 0; i < (end - start); ++i) {
      dev_data_1[i] = i;
    }

    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Expand_Datapre_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("expand_datapre_error", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(EXPAND);

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

Status Expand_Process_Error(std::shared_ptr<DataContext> data_ctx,
                            std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto input_bufs_1 = data_ctx->Input("In_1");
  auto output_bufs_1 = data_ctx->Output("Out_1");
  auto device = mock_flowunit->GetBindDevice();
  auto input_data = (int*)(input_bufs_1->At(0)->ConstData());
  auto input_count = input_bufs_1->At(0)->GetBytes() / sizeof(int);

  if (input_data[0] == 4) {
    return modelbox::STATUS_INVALID;
  } else {
    for (uint32_t i = 0; i < input_count; i++) {
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      auto output_data = (int*)buffer_ptr->MutableData();
      output_data[0] = input_data[i];
      output_bufs_1->PushBack(buffer_ptr);
    }
    return modelbox::STATUS_OK;
  }
}

void MockFlow::Register_Normal_Expand_Process_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("normal_expand_process_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Expand_Process_Error);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Expand_Process_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("expand_process_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Expand_Process_Error);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

Status Expand_Process(std::shared_ptr<DataContext> data_ctx,
                      std::shared_ptr<MockFlowUnit> mock_flowunit) {
  if (data_ctx->HasError()) {
    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto buffer_ptr = std::make_shared<Buffer>();
    buffer_ptr->SetError(data_ctx->GetError());
    output_bufs_1->PushBack(buffer_ptr);
    return modelbox::STATUS_OK;
  }
  auto input_bufs_1 = data_ctx->Input("In_1");
  auto output_bufs_1 = data_ctx->Output("Out_1");
  auto device = mock_flowunit->GetBindDevice();

  auto input_data = (int*)(input_bufs_1->At(0)->ConstData());
  auto input_count = input_bufs_1->At(0)->GetBytes() / sizeof(int);

  for (uint32_t i = 0; i < input_count; i++) {
    auto buffer_ptr = std::make_shared<Buffer>(device);
    buffer_ptr->Build(1 * sizeof(int));
    auto output_data = (int*)buffer_ptr->MutableData();
    output_data[0] = input_data[i];
    output_bufs_1->PushBack(buffer_ptr);
  }
  return modelbox::STATUS_OK;
}

void MockFlow::Register_Normal_Expand_Process_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("normal_expand_process", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Expand_Process);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Expand_Process_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("expand_process", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Expand_Process);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

Status Simple_Pass(std::shared_ptr<DataContext> data_ctx,
                   std::shared_ptr<MockFlowUnit> mock_flowunit) {
  if (data_ctx->HasError()) {
    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto device = mock_flowunit->GetBindDevice();
    auto buffer_ptr = std::make_shared<Buffer>(device);
    buffer_ptr->Build(1 * sizeof(int));
    buffer_ptr->SetError(data_ctx->GetError());
    output_bufs_1->PushBack(buffer_ptr);
    return modelbox::STATUS_OK;
  }
  auto input_bufs_1 = data_ctx->Input("In_1");
  auto output_bufs_1 = data_ctx->Output("Out_1");
  auto device = mock_flowunit->GetBindDevice();
  for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
    auto input_buffer = (*input_bufs_1)[i];
    if (input_buffer->HasError()) {
      auto error = input_buffer->GetError();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      buffer_ptr->Build(1 * sizeof(int));
      buffer_ptr->SetError(input_buffer->GetError());
      output_bufs_1->PushBack(buffer_ptr);
    } else {
      output_bufs_1->PushBack(input_buffer);
    }
  }
  return modelbox::STATUS_OK;
}

void MockFlow::Register_Simple_Pass_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("simple_pass", {"In_1"}, {"Out_1"});
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Simple_Pass);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_HttpServer_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("httpserver", {"IN1"}, {"OUT1"});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Stream_Simple_Pass_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_simple_pass", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetStreamSameCount(false);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Simple_Pass);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Simple_Error_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("simple_error", {"In_1"}, {"Out_1"});
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = data_ctx->Input("In_1");
    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto device = mock_flowunit->GetBindDevice();
    auto input_data = (int*)(*input_bufs_1)[0]->ConstData();
    if (input_data[0] < 2) {
      return modelbox::STATUS_INVALID;
    } else {
      for (uint32_t i = 0; i < input_bufs_1->Size(); i++) {
        auto input_data = (int*)(*input_bufs_1)[i]->ConstData();
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = input_data[0];
        output_bufs_1->PushBack(buffer_ptr);
      }
      return modelbox::STATUS_OK;
    }
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Datapre_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_datapre_error", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_In_Process_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_in_process_error", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto start_index_content = std::make_shared<int>(0);
    data_ctx->SetPrivate("error_index", start_index_content);
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto error_index =
        *(std::static_pointer_cast<int>(data_ctx->GetPrivate("error_index"))
              .get());
    error_index++;
    data_ctx->SetPrivate("error_index", std::make_shared<int>(error_index));
    if (error_index < 2) {
      auto output_bufs_1 = data_ctx->Output("Out_1");
      auto device = mock_flowunit->GetBindDevice();
      for (int i = 0; i < 5; i++) {
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = 0;
        output_bufs_1->PushBack(buffer_ptr);
      }

      return modelbox::STATUS_OK;

    } else {
      return modelbox::STATUS_INVALID;
    }
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Stream_Process_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("stream_process_error", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Error_End_Normal_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("error_end_normal", {"In_1"}, {});
  mock_desc->SetExceptionVisible(true);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "error_end process" << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Error_End_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("error_end", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetExceptionVisible(true);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "error_end process" << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Stream_Process_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("stream_process", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "stream_process process "
                 << data_ctx->GetError()->GetDesc();
    }

    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto device = mock_flowunit->GetBindDevice();
    auto buffer_ptr = std::make_shared<Buffer>(device);
    buffer_ptr->Build(1 * sizeof(int));
    auto output_data = (int*)buffer_ptr->MutableData();
    output_data[0] = 0;
    output_bufs_1->PushBack(buffer_ptr);
    return modelbox::STATUS_OK;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "stream_process DataPre "
                 << data_ctx->GetError()->GetDesc();
    }

    return modelbox::STATUS_OK;
  };

  auto data_post_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "stream_process DataPost "
                 << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  mock_funcitons->RegisterDataPostFunc(data_post_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Collapse_Recieve_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("collapse_recieve_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  mock_desc->SetFlowType(STREAM);

  auto data_group_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error DataGroupPre recive error"
                 << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error DataPre recive error"
                 << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error Process recive error"
                 << data_ctx->GetError()->GetDesc();
    } else {
      auto input_bufs_1 = data_ctx->Input("In_1");
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataGroupPreFunc(data_group_pre_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Normal_Collapse_Recieve_Error_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("normal_collapse_recieve_error",
                                        {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);

  auto data_group_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error DataGroupPre recive error"
                 << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error DataPre recive error"
                 << data_ctx->GetError()->GetDesc();
    }
    return modelbox::STATUS_OK;
  };

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    if (data_ctx->HasError()) {
      MBLOG_INFO << "collapse_recieve_error Process recive error"
                 << data_ctx->GetError()->GetDesc();
    } else {
      auto input_bufs_1 = data_ctx->Input("In_1");
    }

    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataGroupPreFunc(data_group_pre_func);
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Collapse_Datagrouppre_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("collapse_datagrouppre_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  mock_desc->SetFlowType(STREAM);

  auto data_group_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataGroupPreFunc(data_group_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Normal_Collapse_Datagrouppre_Error_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("normal_collapse_datapre_error",
                                        {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Normal_Collapse_Process_Error_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("normal_collapse_process_error",
                                        {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Collapse_Datapre_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("collapse_datapre_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  mock_desc->SetFlowType(STREAM);

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Collapse_Process_Error_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("collapse_process_error", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  mock_desc->SetFlowType(STREAM);

  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    return modelbox::STATUS_INVALID;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

Status Collapse_Process(std::shared_ptr<DataContext> data_ctx,
                        std::shared_ptr<MockFlowUnit> mock_flowunit) {
  auto device = mock_flowunit->GetBindDevice();
  if (data_ctx->HasError()) {
    auto error = data_ctx->GetError();
    MBLOG_INFO << "normal_collapse_process error" << error->GetDesc();
    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto buffer_ptr = std::make_shared<Buffer>(device);
    buffer_ptr->Build(1 * sizeof(int));
    auto output_data = (int*)buffer_ptr->MutableData();
    output_data[0] = 0;
    output_bufs_1->PushBack(buffer_ptr);
  } else {
    auto input_bufs = data_ctx->Input("In_1");
    auto output_bufs_1 = data_ctx->Output("Out_1");
    auto buffer_ptr = std::make_shared<Buffer>(device);
    buffer_ptr->Build(1 * sizeof(int));
    auto output_data = (int*)buffer_ptr->MutableData();
    output_data[0] = 0;

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      auto buffer_ptr = std::make_shared<Buffer>(device);
      output_data[0] += input_data[0];
    }
    output_bufs_1->PushBack(buffer_ptr);
  }

  return modelbox::STATUS_OK;
}

void MockFlow::Register_Normal_Collapse_Process_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("normal_collapse_process", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Collapse_Process);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Collapse_Process_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("collapse_process", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(COLLAPSE);
  mock_desc->SetCollapseAll(true);
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(Collapse_Process);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc(true));
}

void MockFlow::Register_Virtual_Stream_Start_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("virtual_stream_start", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  mock_desc->SetOutputType(EXPAND);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");
    auto event = data_ctx->Event();

    if (event == nullptr) {
      auto output_meta = std::make_shared<DataMeta>();
      auto input_data = (int*)(*input_bufs)[0]->ConstData();
      auto start_index = input_data[0];
      auto end_index = input_data[1];
      auto interval = input_data[2];
      auto start_index_content = std::make_shared<int>(start_index);
      data_ctx->SetPrivate("now_index", start_index_content);
      auto end_index_content = std::make_shared<int>(end_index);
      data_ctx->SetPrivate("end_index", end_index_content);
      auto interval_content = std::make_shared<int>(interval);
      output_meta->SetMeta("start_index", start_index_content);
      output_meta->SetMeta("end_index", end_index_content);
      output_meta->SetMeta("interval", interval_content);
      data_ctx->SetOutputMeta("Out_1", output_meta);
    }

    auto now_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("now_index")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    std::vector<size_t> shape(5, sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < 5; ++i) {
      auto output_data = (int*)(*output_bufs)[i]->MutableData();
      output_data[0] = now_index + i;
    }
    now_index = now_index + 5;
    auto now_index_content = std::make_shared<int>(now_index);
    data_ctx->SetPrivate("now_index", now_index_content);
    if (now_index + 5 <= end_index) {
      auto event = std::make_shared<FlowUnitEvent>();
      data_ctx->SendEvent(event);
      return modelbox::STATUS_CONTINUE;
    } else {
      return modelbox::STATUS_OK;
    }
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Virtual_Stream_Mid_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("virtual_stream_mid", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");

    auto device = mock_flowunit->GetBindDevice();
    auto interval = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("interval")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    for (size_t i = 0; i < input_bufs->Size(); ++i) {
      auto input_data = (int*)(*input_bufs)[i]->ConstData();
      if (input_data[0] % interval == 0) {
        auto buffer_ptr = std::make_shared<Buffer>(device);
        buffer_ptr->Build(1 * sizeof(int));
        auto output_data = (int*)buffer_ptr->MutableData();
        output_data[0] = input_data[0];
        output_bufs->PushBack(buffer_ptr);
      }

      if (input_data[0] == end_index - 1) {
        return modelbox::STATUS_OK;
      }
    }
    return modelbox::STATUS_CONTINUE;
  };

  auto data_pre_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto meta = data_ctx->GetInputMeta("In_1");
    std::shared_ptr<Device> device;
    auto interval =
        *(std::static_pointer_cast<int>(meta->GetMeta("interval")).get());
    auto end_index =
        *(std::static_pointer_cast<int>(meta->GetMeta("end_index")).get());
    data_ctx->SetPrivate("interval", std::make_shared<int>(interval));
    data_ctx->SetPrivate("end_index", std::make_shared<int>(end_index));
    return modelbox::STATUS_OK;
  };
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterDataPreFunc(data_pre_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Virtual_Stream_End_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("virtual_stream_end", {"In_1"}, {});
  mock_desc->SetFlowType(STREAM);
  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Virtual_Expand_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("virtual_expand", {"In_1"}, {"Out_1"});
  mock_desc->SetOutputType(EXPAND);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");
    auto event = data_ctx->Event();
    auto input_data = (int*)(*input_bufs)[0]->ConstData();
    std::vector<size_t> shape(1, 3 * sizeof(int));
    output_bufs->Build(shape);
    auto output_data = (int*)(*output_bufs)[0]->MutableData();
    for (size_t i = 0; i < 3; ++i) {
      output_data[i] = input_data[i];
    }
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Virtual_Stream_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("virtual_stream", {"In_1"}, {"Out_1"});
  mock_desc->SetFlowType(STREAM);
  auto process_func =
      [=](std::shared_ptr<DataContext> data_ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto output_bufs = data_ctx->Output("Out_1");
    auto input_bufs = data_ctx->Input("In_1");
    auto event = data_ctx->Event();

    if (event == nullptr) {
      auto output_meta = std::make_shared<DataMeta>();
      auto input_data = (int*)(*input_bufs)[0]->ConstData();
      auto start_index = input_data[0];
      auto end_index = input_data[1];
      auto interval = input_data[2];
      auto start_index_content = std::make_shared<int>(start_index);
      data_ctx->SetPrivate("now_index", start_index_content);
      auto end_index_content = std::make_shared<int>(end_index);
      data_ctx->SetPrivate("end_index", end_index_content);
      auto interval_content = std::make_shared<int>(interval);
      output_meta->SetMeta("start_index", start_index_content);
      output_meta->SetMeta("end_index", end_index_content);
      output_meta->SetMeta("interval", interval_content);
      data_ctx->SetOutputMeta("Out_1", output_meta);
    }

    auto now_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("now_index")).get());
    auto end_index = *(
        std::static_pointer_cast<int>(data_ctx->GetPrivate("end_index")).get());

    std::vector<size_t> shape(5, sizeof(int));
    output_bufs->Build(shape);
    for (size_t i = 0; i < 5; ++i) {
      auto output_data = (int*)(*output_bufs)[i]->MutableData();
      output_data[0] = now_index + i;
    }
    now_index = now_index + 5;
    auto now_index_content = std::make_shared<int>(now_index);
    data_ctx->SetPrivate("now_index", now_index_content);
    if (now_index + 5 <= end_index) {
      auto event = std::make_shared<FlowUnitEvent>();
      data_ctx->SendEvent(event);
      return modelbox::STATUS_CONTINUE;
    } else {
      return modelbox::STATUS_OK;
    }
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Tensorlist_Test_1_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("tensorlist_test_1", {"IN1"}, {"OUT1"});
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs = ctx->Input("IN1");
    auto output_bufs = ctx->Output("OUT1");

    TensorList in_tl(input_bufs);
    TensorList out_tl(output_bufs);

    if (in_tl.Size() == 0) {
      return modelbox::STATUS_FAULT;
    }

    out_tl.Build<int>(in_tl.GetShape());
    for (size_t i = 0; i < in_tl.Size(); ++i) {
      auto tensor = in_tl[i];
      const auto in_data = in_tl.ConstBufferData<int>(i);
      auto out_data = out_tl.MutableBufferData<int>(i);
      out_data[0] = in_data[0] + 10;
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Tensorlist_Test_2_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("tensorlist_test_2", {"IN1", "IN2"}, {"OUT1"});
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto input_bufs_1 = ctx->Input("IN1");
    auto input_bufs_2 = ctx->Input("IN2");
    auto output_bufs_1 = ctx->Output("OUT1");

    TensorList in_tl_1(input_bufs_1);
    TensorList in_tl_2(input_bufs_2);
    TensorList out_tl_1(output_bufs_1);

    if (in_tl_1.Size() == 0 || in_tl_2.Size() == 0 ||
        (in_tl_1.Size() != in_tl_2.Size())) {
      return modelbox::STATUS_FAULT;
    }

    out_tl_1.Build<int>(in_tl_1.GetShape());
    for (size_t i = 0; i < in_tl_1.Size(); ++i) {
      const auto in_data_1 = in_tl_1.ConstBufferData<int>(i);
      const auto in_data_2 = in_tl_2.ConstBufferData<int>(i);
      auto out_data_1 = out_tl_1.MutableBufferData<int>(i);
      out_data_1[0] = in_data_1[0] + in_data_2[0];
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Check_Tensorlist_Test_1_Flowunit() {
  auto mock_desc =
      GenerateFlowunitDesc("check_tensorlist_test_1", {"IN1", "IN2"}, {});

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_tl_1 = ctx->Input("IN1");
    const auto input_tl_2 = ctx->Input("IN2");

    TensorList in_tl_1(input_tl_1);
    TensorList in_tl_2(input_tl_2);

    if (in_tl_1.Size() == 0 || in_tl_1.Size() != in_tl_2.Size()) {
      return modelbox::STATUS_FAULT;
    }

    for (size_t i = 0; i < in_tl_1.Size(); ++i) {
      const auto in_data_1 = in_tl_1.ConstBufferData<int>(i);
      const auto in_data_2 = in_tl_2.ConstBufferData<int>(i);
      if (in_data_2[0] != in_data_1[0]) {
        return modelbox::STATUS_FAULT;
      }
    }

    if (MAX_COUNT < run_count++) {
      MBLOG_DEBUG << "check reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Slow_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("slow", {"IN1", "IN2"}, {});

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_tl_1 = ctx->Input("IN1");
    const auto input_tl_2 = ctx->Input("IN2");

    TensorList in_tl_1(input_tl_1);
    TensorList in_tl_2(input_tl_2);

    if (in_tl_1.Size() == 0 || in_tl_1.Size() != in_tl_2.Size()) {
      return modelbox::STATUS_FAULT;
    }

    MBLOG_INFO << "slow flow get data";
    sleep(5);
    MBLOG_INFO << "slow flow unit sleep 3s, run_count:" << run_count;

    for (size_t i = 0; i < in_tl_1.Size(); ++i) {
      const auto in_data_1 = in_tl_1.ConstBufferData<int>(i);
      const auto in_data_2 = in_tl_2.ConstBufferData<int>(i);
      if (in_data_2[0] != in_data_1[0]) {
        return modelbox::STATUS_FAULT;
      }
    }

    if (MAX_COUNT < run_count++) {
      MBLOG_DEBUG << "check reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Check_Tensorlist_Test_2_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("check_tensorlist_test_2",
                                        {"IN1", "IN2", "IN3"}, {});

  static std::atomic<int64_t> run_count(0);
  static int64_t MAX_COUNT = 0;
  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    const auto input_tl_1 = ctx->Input("IN1");
    const auto input_tl_2 = ctx->Input("IN2");
    const auto input_tl_3 = ctx->Input("IN3");

    TensorList in_tl_1(input_tl_1);
    TensorList in_tl_2(input_tl_2);
    TensorList in_tl_3(input_tl_3);

    if (in_tl_1.Size() == 0 || in_tl_2.Size() == 0 || in_tl_3.Size() == 0 ||
        in_tl_1.Size() != in_tl_2.Size() || in_tl_2.Size() != in_tl_3.Size()) {
      return modelbox::STATUS_FAULT;
    }

    for (size_t i = 0; i < in_tl_1.Size(); ++i) {
      const auto in_data_1 = in_tl_1.ConstBufferData<int>(i);
      const auto in_data_2 = in_tl_2.ConstBufferData<int>(i);
      const auto in_data_3 = in_tl_3.ConstBufferData<int>(i);
      if (in_data_3[0] != (in_data_1[0] + in_data_2[0])) {
        return modelbox::STATUS_FAULT;
      }
    }

    if (MAX_COUNT < run_count++) {
      MBLOG_DEBUG << "check reach max running times, should stop.";
      return modelbox::STATUS_STOP;
    }

    return modelbox::STATUS_SUCCESS;
  };

  auto open_func = [=](const std::shared_ptr<Configuration>& flow_option,
                       std::shared_ptr<MockFlowUnit> mock_flowunit) {
    MAX_COUNT = flow_option->GetInt64("max_count", 50);
    run_count = 0;
    return modelbox::STATUS_OK;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterOpenFunc(open_func);
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

void MockFlow::Register_Statistic_Test_Flowunit() {
  auto mock_desc = GenerateFlowunitDesc("statistic_test", {"IN1"}, {"OUT1"});

  auto process_func =
      [=](std::shared_ptr<DataContext> ctx,
          std::shared_ptr<MockFlowUnit> mock_flowunit) -> Status {
    auto stats = ctx->GetStatistics();
    EXPECT_NE(stats, nullptr);
    if (stats == nullptr) {
      return modelbox::STATUS_FAULT;
    }

    int32_t test_val = 1;
    auto test_stats = stats->AddItem("test_key", test_val);
    EXPECT_NE(test_stats, nullptr);
    if (test_stats == nullptr) {
      return modelbox::STATUS_FAULT;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    test_stats->SetValue(test_val);  // notify cooldown test
    test_stats->SetValue(test_val);
    test_stats->SetValue(test_val);
    auto output = ctx->Output("OUT1");
    output->Build({1});
    return modelbox::STATUS_SUCCESS;
  };

  auto mock_funcitons = std::make_shared<MockFunctionCollection>();
  mock_funcitons->RegisterProcessFunc(process_func);
  AddFlowUnitDesc(mock_desc, mock_funcitons->GenerateCreateFunc());
}

bool MockFlow::Init(bool with_default_flowunit) {
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  ConfigurationBuilder configbuilder;
  auto config = configbuilder.Build();

  // generate cpu driver
  modelbox::DriverDesc desc;
  desc.SetClass("DRIVER-DEVICE");
  desc.SetType("cpu");
  desc.SetName("device-driver-cpu");
  desc.SetDescription("the cpu device");
  desc.SetVersion("8.9.2");
  std::string file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-device-cpu.so";
  desc.SetFilePath(file_path_device);
  ctl_->AddMockDriverDevice("cpu", desc);
  drivers->Add(file_path_device);

  // generate graphmanager
  desc.SetClass("DRIVER-GRAPHCONF");
  desc.SetType("GRAPHVIZ");
  desc.SetName("GRAPHCONF-GRAPHVIZ");
  desc.SetDescription("graph config parse graphviz");
  desc.SetVersion("0.1.0");
  file_path_device =
      std::string(TEST_LIB_DIR) + "/libmodelbox-graphconf-graphviz.so";
  desc.SetFilePath(file_path_device);
  ctl_->AddMockDriverGraphConf("graphviz", "", desc);
  drivers->Add(file_path_device);

  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  Status status1 = device_mgr->Initialize(drivers, config);

  auto device = device_mgr->CreateDevice("cpu", "0");
  if (device == nullptr) {
    MBLOG_ERROR << "create device failed, " << StatusError;
    return false;
  }
  device->SetMemQuota(102400000);
  if (with_default_flowunit) {
    Register_Test_0_1_Flowunit();
    Register_Test_0_1_Batch_Thread_Flowunit();
    Register_Test_0_1_Batch_Flowunit();
    Register_Test_0_2_Flowunit();
    Register_Test_1_0_Flowunit();
    Register_Test_1_0_Batch_Flowunit();
    Register_Test_1_0_Batch_Thread_Flowunit();
    Register_Test_2_0_Flowunit();
    Register_Test_Orgin_0_2_Flowunit();
    Register_Listen_Flowunit();
    Register_ExternData_Flowunit();
    Register_Test_2_inputs_2_outputs_Flowunit();
    Register_Condition_Flowunit();
    Register_Loop_Flowunit();
    Register_Loop_End_Flowunit();
    Register_Half_Condition_Flowunit();
    Register_Normal_Condition_Flowunit();
    Register_Expand_Normal_Flowunit();
    Register_Collapse_Normal_Flowunit();
    Register_Stream_Add_Flowunit();
    Register_Add_Flowunit();
    Register_Wrong_Add_Flowunit();
    Register_Wrong_Add_2_Flowunit();
    Register_Scatter_Flowunit();
    Register_Garther_Flowunit();
    Register_Garther_Gen_Flowunit();
    Register_Print_Flowunit();
    Register_Check_Print_Flowunit();
    Register_Dynamic_Config_Flowunit();
    Register_Dynamic_Get_Config_Flowunit();
    Register_Dynamic_Get_Config_Other_Flowunit();
    Register_Stream_Info_Flowunit();
    Register_Stream_Normal_Info_Flowunit();
    Register_Stream_Normal_Info_2_Flowunit();
    Register_Stream_Start_Flowunit();
    Register_Normal_Expand_Start_Flowunit();
    Register_Stream_Tail_Filter_Flowunit();
    Register_Stream_Mid_Flowunit();
    Register_Stream_End_Flowunit();
    Register_Add_1_Flowunit();
    Register_Iflow_Add_1_Flowunit();
    Register_Add_1_And_Error_Flowunit();
    Register_Test_Condition_Flowunit();
    Register_Get_Priority_Flowunit();
    Register_Error_Start_Flowunit();
    Register_Error_Start_Normal_Flowunit();
    Register_Error_End_Flowunit();
    Register_Error_End_Normal_Flowunit();
    Register_Normal_Start_Flowunit();
    Register_Expand_Datapre_Error_Flowunit();
    Register_Normal_Expand_Process_Error_Flowunit();
    Register_Expand_Process_Error_Flowunit();
    Register_Normal_Expand_Process_Flowunit();
    Register_Expand_Process_Flowunit();
    Register_Simple_Pass_Flowunit();
    Register_Stream_Simple_Pass_Flowunit();
    Register_Simple_Error_Flowunit();
    Register_Stream_Datapre_Error_Flowunit();
    Register_Stream_In_Process_Error_Flowunit();
    Register_Stream_Process_Error_Flowunit();
    Register_Stream_Process_Flowunit();
    Register_Collapse_Recieve_Error_Flowunit();
    Register_Normal_Collapse_Recieve_Error_Flowunit();
    Register_Collapse_Datagrouppre_Error_Flowunit();
    Register_Normal_Collapse_Datagrouppre_Error_Flowunit();
    Register_Normal_Collapse_Process_Error_Flowunit();
    Register_Normal_Collapse_Process_Flowunit();
    Register_Collapse_Datapre_Error_Flowunit();
    Register_Collapse_Process_Error_Flowunit();
    Register_Collapse_Process_Flowunit();
    Register_Virtual_Stream_Start_Flowunit();
    Register_Virtual_Stream_Mid_Flowunit();
    Register_Virtual_Stream_End_Flowunit();
    Register_Virtual_Expand_Flowunit();
    Register_Virtual_Stream_Flowunit();
    Register_Tensorlist_Test_1_Flowunit();
    Register_Tensorlist_Test_2_Flowunit();
    Register_Check_Tensorlist_Test_1_Flowunit();
    Register_Check_Tensorlist_Test_2_Flowunit();
    Register_Slow_Flowunit();
    Register_Statistic_Test_Flowunit();
    Register_HttpServer_Flowunit();
  }

  bool result = drivers->Scan(TEST_LIB_DIR, "/libmodelbox-unit-*");

  std::shared_ptr<FlowUnitManager> flowunit_mgr =
      FlowUnitManager::GetInstance();
  result = flowunit_mgr->Initialize(drivers, device_mgr, config);

  return result;
};

Status MockFlow::InitFlow(const std::string& name, const std::string& graph) {
  flow_ = std::make_shared<Flow>();
  return flow_->Init(name, graph);
  ;
}

Status MockFlow::BuildAndRun(const std::string& name, const std::string& graph,
                             int timeout) {
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

std::shared_ptr<MockDriverCtl> MockFlow::GetMockFlowCtl() { return ctl_; }

std::shared_ptr<Flow> MockFlow::GetFlow() { return flow_; }

void MockFlow::Destroy() {
  std::shared_ptr<FlowUnitManager> flowunit_mgr =
      FlowUnitManager::GetInstance();
  flowunit_mgr->Clear();
  flowunit_mgr = nullptr;
  std::shared_ptr<DeviceManager> device_mgr = DeviceManager::GetInstance();
  device_mgr->Clear();
  device_mgr = nullptr;
  std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
  drivers->Clear();
  drivers = nullptr;
  ctl_ = nullptr;
  flow_ = nullptr;
};

}  // namespace modelbox
