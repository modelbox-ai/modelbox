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

#include "modelbox/virtual_node.h"

#include <fstream>
#include <string>

#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "graph_conf_mockgraphconf/graph_conf_mockgraphconf.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/log.h"
#include "modelbox/data_context.h"
#include "modelbox/graph.h"
#include "modelbox/session_context.h"

using ::testing::_;
namespace modelbox {
class VirtualNodeTest : public testing::Test {
 public:
  VirtualNodeTest() = default;

 protected:
  std::shared_ptr<MockFlow> mock_flow_;
  void SetUp() override {
    old_level_ = ModelBoxLogger.GetLogger()->GetLogLevel();
    std::shared_ptr<Drivers> drivers = Drivers::GetInstance();
    mock_flow_ = std::make_shared<MockFlow>();
    mock_flow_->Init();
    {
      // flowunit add10
      {
        MockFlowUnitDriverDesc desc_flowunit;
        desc_flowunit.SetClass("DRIVER-FLOWUNIT");
        desc_flowunit.SetType("cpu");
        desc_flowunit.SetName("add10");
        desc_flowunit.SetDescription("the int add10 function");
        desc_flowunit.SetVersion("1.0.0");
        std::string file_path_flowunit =
            std::string(TEST_LIB_DIR) + "/libmodelbox-unit-cpu-add10.so";
        desc_flowunit.SetFilePath(file_path_flowunit);
        auto mock_flowunit = std::make_shared<MockFlowUnit>();
        auto mock_flowunit_desc = std::make_shared<FlowUnitDesc>();
        mock_flowunit_desc->SetFlowType(NORMAL);
        mock_flowunit_desc->SetFlowUnitName("add10");
        mock_flowunit_desc->AddFlowUnitInput(modelbox::FlowUnitInput("In_1"));
        mock_flowunit_desc->AddFlowUnitOutput(
            modelbox::FlowUnitOutput("Out_1"));
        mock_flowunit->SetFlowUnitDesc(mock_flowunit_desc);
        EXPECT_CALL(*mock_flowunit, Open(_))
            .WillRepeatedly(testing::Invoke(
                [&](const std::shared_ptr<modelbox::Configuration>&
                        flow_option) {
                  MBLOG_INFO << "add Open";
                  return modelbox::STATUS_OK;
                }));

        EXPECT_CALL(*mock_flowunit, DataPre(_))
            .WillRepeatedly(testing::Invoke(
                [&](const std::shared_ptr<DataContext> &data_ctx) {
                  MBLOG_INFO << "add DataPre";
                  return modelbox::STATUS_OK;
                }));

        EXPECT_CALL(*mock_flowunit, DataPost(_))
            .WillRepeatedly(testing::Invoke(
                [&](const std::shared_ptr<DataContext> &data_ctx) {
                  MBLOG_INFO << "add DataPost";
                  return modelbox::STATUS_OK;
                }));

        EXPECT_CALL(
            *mock_flowunit,
            Process(testing::An<std::shared_ptr<modelbox::DataContext>>()))
            .WillRepeatedly(testing::Invoke(
                [=](const std::shared_ptr<DataContext> &data_ctx) -> Status {
                  MBLOG_INFO << "add Process";
                  const auto input_bufs_1 = data_ctx->Input("In_1");
                  auto output_bufs = data_ctx->Output("Out_1");

                  std::vector<size_t> shape;
                  for (size_t i = 0; i < input_bufs_1->Size(); ++i) {
                    auto input_size_1 = (*input_bufs_1)[i]->GetBytes();
                    shape.emplace_back(input_size_1);
                  }
                  output_bufs->Build(shape);

                  for (size_t i = 0; i < shape.size(); ++i) {
                    auto *input_data_1 = (int *)(*input_bufs_1)[i]->ConstData();
                    auto *output_data = (int *)(*output_bufs)[i]->MutableData();
                    auto data_size = shape[i] / sizeof(int);
                    for (size_t j = 0; j < data_size; ++j) {
                      output_data[j] = input_data_1[j] + 10;
                      MBLOG_DEBUG << input_data_1[j] << " + " << 10 << " = "
                                  << output_data[j];
                    }
                  }

                  return modelbox::STATUS_OK;
                }));
        EXPECT_CALL(*mock_flowunit, Close())
            .WillRepeatedly(testing::Invoke([&]() {
              MBLOG_INFO << "add Close";
              return modelbox::STATUS_OK;
            }));
        desc_flowunit.SetMockFlowUnit(mock_flowunit);
        ctl_.AddMockDriverFlowUnit("add10", "cpu", desc_flowunit);
      }
    }
    drivers->Scan(TEST_LIB_DIR, "/libmodelbox-unit-*");
  }

  void TearDown() override {
    ModelBoxLogger.GetLogger()->SetLogLevel(old_level_);
  }

  const std::string test_lib_dir = TEST_LIB_DIR;
  const std::string test_data_dir = TEST_DATA_DIR;

 private:
  LogLevel old_level_;
  MockDriverCtl ctl_;
};

TEST_F(VirtualNodeTest, VirtualNode_ONE_INPUT) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          add[type=flowunit, flowunit=add10, device=cpu, deviceid=0, label="<In_1> |<In_2> | <Out_1>"]
          
          input1 ->add:In_1
          add:Out_1->output1

        }'''
    format = "graphviz"
  )";

  auto ret = mock_flow_->BuildAndRun("VirtualNode_ONE_INPUT", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  // data 1
  {
    auto ext_data = flow->CreateExternalDataMap();
    int len = 10;
    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({len * sizeof(int)});
    auto *data = (int *)buffer_list->MutableData();
    std::string dataStr;
    for (auto i = 0; i < len; ++i) {
      data[i] = i;
      dataStr += std::to_string(data[i]) + ",";
    }
    MBLOG_INFO << "in: " << dataStr;

    auto status = ext_data->Send("input1", buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    OutputBufferList map_buffer_list;
    ext_data->Recv(map_buffer_list);

    for (const auto &buffer_list_iter : map_buffer_list) {
      auto name = buffer_list_iter.first;
      auto buffer_list = buffer_list_iter.second;
      auto buffer_size = buffer_list->Size();

      std::string dataStr;
      for (size_t i = 0; i < buffer_size; ++i) {
        auto *data = (int *)buffer_list->At(i)->ConstData();
        auto data_size = buffer_list->At(i)->GetBytes();
        for (size_t j = 0; j < data_size / sizeof(int); ++j) {
          dataStr += std::to_string(data[j]) + ",";
        }
      }
      MBLOG_INFO << name << " " << dataStr;
    }
  }

  // data 2
  {
    auto ext_data = flow->CreateExternalDataMap();

    int len = 10;
    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({len * sizeof(int)});
    auto *data = (int *)buffer_list->MutableData();
    std::string dataStr;
    for (auto i = 0; i < len; ++i) {
      data[i] = i + 100;
      dataStr += std::to_string(data[i]) + ",";
    }
    MBLOG_INFO << "in: " << dataStr;

    auto status = ext_data->Send("input1", buffer_list);
    if (!status) {
      MBLOG_ERROR << "external data send buffer list failed:" << status;
    }

    status = ext_data->Close();
    if (!status) {
      MBLOG_ERROR << "external data close failed:" << status;
    }

    OutputBufferList map_buffer_list;
    ext_data->Recv(map_buffer_list);

    for (const auto &buffer_list_iter : map_buffer_list) {
      auto name = buffer_list_iter.first;
      auto buffer_list = buffer_list_iter.second;
      auto buffer_size = buffer_list->Size();

      std::string dataStr;
      for (size_t i = 0; i < buffer_size; ++i) {
        auto *data = (int *)buffer_list->At(i)->ConstData();
        auto data_size = buffer_list->At(i)->GetBytes();
        for (size_t j = 0; j < data_size / sizeof(int); ++j) {
          EXPECT_EQ(data[j], 110 + j);
          dataStr += std::to_string(data[j]) + ",";
        }
      }
      MBLOG_INFO << name << " " << dataStr;
    }
  }

  flow->Wait(3 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_MULTI_INPUT) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          input2[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          add[type=flowunit, flowunit=add, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>"]
          
          input1 ->add:In_1
          input2 ->add:In_2
          add:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret =
      mock_flow_->BuildAndRun("VirtualNode_MULTI_INPUT", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto sess_ctx = ext_data->GetSessionContext();
    sess_ctx->SetPrivate("test", std::make_shared<int64_t>(1111));
    sess_ctx = nullptr;
    int len = 10;
    auto buffer_list = ext_data->CreateBufferList();
    buffer_list->Build({len * sizeof(int)});
    auto *data = (int *)buffer_list->MutableData();
    std::string dataStr;
    for (auto i = 0; i < len; ++i) {
      data[i] = i;
      dataStr += std::to_string(data[i]) + ",";
    }
    MBLOG_INFO << "in: " << dataStr;

    auto status = ext_data->Send("input1", buffer_list);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Send("input2", buffer_list);
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list_1;

    status = ext_data->Recv(map_buffer_list_1);
    EXPECT_EQ(status, STATUS_SUCCESS);

    for (const auto &buffer_list_iter : map_buffer_list_1) {
      auto name = buffer_list_iter.first;
      auto buffer_list = buffer_list_iter.second;
      auto buffer_size = buffer_list->Size();

      std::string dataStr;
      for (size_t i = 0; i < buffer_size; ++i) {
        auto *data = (int *)buffer_list->At(i)->ConstData();
        auto data_size = buffer_list->At(i)->GetBytes();
        for (size_t j = 0; j < data_size / sizeof(int); ++j) {
          EXPECT_EQ(data[j], 2 * j);
          dataStr += std::to_string(data[j]) + ",";
        }
      }
      MBLOG_INFO << name << " " << dataStr;
    }

    status = ext_data->Send("input1", buffer_list);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Send("input2", buffer_list);

    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list_2;
    status = ext_data->Recv(map_buffer_list_2);
    EXPECT_EQ(status, STATUS_SUCCESS);

    for (const auto &buffer_list_iter : map_buffer_list_2) {
      auto name = buffer_list_iter.first;
      auto buffer_list = buffer_list_iter.second;
      auto buffer_size = buffer_list->Size();

      std::string dataStr;
      for (size_t i = 0; i < buffer_size; ++i) {
        auto *data = (int *)buffer_list->At(i)->ConstData();
        auto data_size = buffer_list->At(i)->GetBytes();
        for (size_t j = 0; j < data_size / sizeof(int); ++j) {
          EXPECT_EQ(data[j], 2 * j);
          dataStr += std::to_string(data[j]) + ",";
        }
      }
      MBLOG_INFO << name << " " << dataStr;
    }

    OutputBufferList map_buffer_list_3;
    status = ext_data->Recv(map_buffer_list_3);
    EXPECT_TRUE(map_buffer_list_3.empty());
    EXPECT_EQ(status, STATUS_EOF);
  }

  flow->Wait(3 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_NO_OUTPUT) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_end[type=flowunit, flowunit=virtual_stream_end, device=cpu, deviceid=0, label="<In_1>"]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->stream_end:In_1

        }'''
    format = "graphviz"
  )";

  auto ret = mock_flow_->BuildAndRun("VirtualNode_NO_OUTPUT", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto output_buf = ext_data->CreateBufferList();
    output_buf->Build({3 * sizeof(int)});
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 25000;
    data[2] = 3;

    auto status = ext_data->Send("input1", output_buf);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list;

    status = ext_data->Recv(map_buffer_list);
    EXPECT_EQ(status, STATUS_EOF);
  }

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto output_buf = ext_data->CreateBufferList();
    output_buf->Build({3 * sizeof(int)});
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 25000;
    data[2] = 3;

    auto status = ext_data->Send("input1", output_buf);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list;
    status = ext_data->Recv(map_buffer_list);
    EXPECT_EQ(status, STATUS_EOF);
    auto error = ext_data->GetLastError();
    EXPECT_EQ(error, nullptr);
  }
  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Stop) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret = mock_flow_->BuildAndRun("VirtualNode_Stop", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto output_buf = ext_data->CreateBufferList();
    output_buf->Build({3 * sizeof(int)});
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 25000;
    data[2] = 3;

    auto status = ext_data->Send("input1", output_buf);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list;

    uint32_t i = 0;
    while (true) {
      auto status = ext_data->Recv(map_buffer_list);
      if (i == 5) {
        ext_data->Shutdown();
      }
      if (status != STATUS_SUCCESS) {
        EXPECT_EQ(status, STATUS_INVALID);
        auto error = ext_data->GetLastError();
        EXPECT_EQ(error->GetDesc(), "EOF");
        break;
      }
      i++;
    }
  }
  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Stop_2) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0]
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=stream_simple_pass,
          device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream, device=cpu,
          deviceid=0, label="<In_1> | <Out_1>"]

          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret = mock_flow_->BuildAndRun("VirtualNode_Stop_2", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto output_buf = ext_data->CreateBufferList();
    std::vector<size_t> shape(1, 3 * sizeof(int));
    output_buf->Build(shape);
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 25000;
    data[2] = 3;

    auto status = ext_data->Send("input1", output_buf);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    OutputBufferList map_buffer_list;

    uint32_t i = 0;
    while (true) {
      auto status = ext_data->Recv(map_buffer_list);
      if (i == 200) {
        ext_data->Shutdown();
      }
      if (status != STATUS_SUCCESS) {
        EXPECT_EQ(status, STATUS_INVALID);
        auto error = ext_data->GetLastError();
        EXPECT_EQ(error->GetDesc(), "EOF");
        break;
      }
      i++;
    }
  }
  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Stop_3) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>", batch_size=5]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret = mock_flow_->BuildAndRun("VirtualNode_Select", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data_1 = flow->CreateExternalDataMap();

    auto output_buf_1 = ext_data_1->CreateBufferList();
    output_buf_1->Build({3 * sizeof(int)});
    auto *data_1 = (int *)output_buf_1->MutableData();
    data_1[0] = 0;
    data_1[1] = 25000;
    data_1[2] = 3;

    auto status = ext_data_1->Send("input1", output_buf_1);
    EXPECT_EQ(status, STATUS_SUCCESS);

    auto selector = std::make_shared<ExternalDataSelect>();
    selector->RegisterExternalData(ext_data_1);

    int recv_count = 0;
    while (true) {
      std::list<std::shared_ptr<ExternalDataMap>> external_list;
      auto select_status = selector->SelectExternalData(
          external_list, std::chrono::milliseconds(3000));
      if (select_status == STATUS_TIMEDOUT) {
        break;
      }

      for (const auto &external : external_list) {
        OutputBufferList map_buffer_list;
        external->Recv(map_buffer_list);

        if (external == ext_data_1) {
          recv_count++;
          if (recv_count >= 50) {
            ext_data_1->Close();
          }
          if (recv_count >= 100) {
            ext_data_1->Shutdown();
          }
        }
      }
    }

    OutputBufferList map_buffer_list;
    auto last_status_1 = ext_data_1->Recv(map_buffer_list);
    EXPECT_EQ(last_status_1, STATUS_INVALID);
  }

  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Select) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>", batch_size=5]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret = mock_flow_->BuildAndRun("VirtualNode_Select", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data_1 = flow->CreateExternalDataMap();

    auto output_buf_1 = ext_data_1->CreateBufferList();
    output_buf_1->Build({3 * sizeof(int)});
    auto *data_1 = (int *)output_buf_1->MutableData();
    data_1[0] = 0;
    data_1[1] = 25000;
    data_1[2] = 3;

    auto status = ext_data_1->Send("input1", output_buf_1);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data_1->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    auto ext_data_2 = flow->CreateExternalDataMap();

    auto output_buf_2 = ext_data_2->CreateBufferList();
    output_buf_2->Build({3 * sizeof(int)});
    auto *data_2 = (int *)output_buf_2->MutableData();
    data_2[0] = 0;
    data_2[1] = 25000;
    data_2[2] = 3;

    status = ext_data_2->Send("input1", output_buf_2);
    EXPECT_EQ(status, STATUS_SUCCESS);

    auto selector = std::make_shared<ExternalDataSelect>();
    selector->RegisterExternalData(ext_data_1);
    selector->RegisterExternalData(ext_data_2);

    int size = 0;
    int recv_count = 0;

    while (true) {
      std::list<std::shared_ptr<ExternalDataMap>> external_list;
      auto select_status = selector->SelectExternalData(
          external_list, std::chrono::milliseconds(3000));
      if (select_status == STATUS_TIMEDOUT) {
        break;
      }

      for (const auto &external : external_list) {
        OutputBufferList map_buffer_list;
        auto status = external->Recv(map_buffer_list);
        if (status == STATUS_SUCCESS && external == ext_data_1) {
          size += map_buffer_list["output1"]->Size();
        }

        if (external == ext_data_2) {
          recv_count++;
          if (recv_count >= 10) {
            ext_data_2->Shutdown();
          }
        }
      }
    }
    EXPECT_EQ(size, 8334);

    OutputBufferList map_buffer_list;
    auto last_status_1 = ext_data_1->Recv(map_buffer_list);
    EXPECT_EQ(last_status_1, STATUS_EOF);
    auto last_status_2 = ext_data_2->Recv(map_buffer_list);
    EXPECT_EQ(last_status_2, STATUS_INVALID);
  }

  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Select_Timeout) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret =
      mock_flow_->BuildAndRun("VirtualNode_Select_Timeout", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data = flow->CreateExternalDataMap();

    auto output_buf = ext_data->CreateBufferList();
    output_buf->Build({3 * sizeof(int)});
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 25000;
    data[2] = 3;

    auto selector = std::make_shared<ExternalDataSelect>();
    selector->RegisterExternalData(ext_data);
    std::list<std::shared_ptr<ExternalDataMap>> external_list;
    auto select_status = selector->SelectExternalData(
        external_list, std::chrono::milliseconds(100));
    EXPECT_EQ(select_status, STATUS_TIMEDOUT);
  }

  {
    auto selector = std::make_shared<ExternalDataSelect>();
    std::list<std::shared_ptr<ExternalDataMap>> external_list;
    auto select_status = selector->SelectExternalData(
        external_list, std::chrono::milliseconds(100));
    EXPECT_EQ(select_status, STATUS_TIMEDOUT);
  }

  flow->Wait(5 * 1000);
}

TEST_F(VirtualNodeTest, VirtualNode_Muliti_Output) {
  std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                             "\"]\n    " +
                             R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output,output_type=unmatch ,device=cpu, deviceid=0]
          output2[type=output,output_type=unmatch, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_start:Out_1 ->output2
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
  auto ret =
      mock_flow_->BuildAndRun("VirtualNode_Muliti_Output", toml_content, -1);
  auto flow = mock_flow_->GetFlow();

  {
    auto ext_data_1 = flow->CreateExternalDataMap();

    auto output_buf_1 = ext_data_1->CreateBufferList();
    output_buf_1->Build({3 * sizeof(int)});
    auto *data_1 = (int *)output_buf_1->MutableData();
    data_1[0] = 0;
    data_1[1] = 25000;
    data_1[2] = 3;

    auto status = ext_data_1->Send("input1", output_buf_1);
    EXPECT_EQ(status, STATUS_SUCCESS);

    status = ext_data_1->Close();
    EXPECT_EQ(status, STATUS_SUCCESS);

    uint32_t size_1 = 0;
    uint32_t size_2 = 0;
    while (true) {
      OutputBufferList map_buffer_list;
      auto status = ext_data_1->Recv(map_buffer_list);
      if (map_buffer_list["output1"] != nullptr) {
        size_1 += map_buffer_list["output1"]->Size();
      }
      if (map_buffer_list["output2"] != nullptr) {
        size_2 += map_buffer_list["output2"]->Size();
      }
      if (status == STATUS_EOF) {
        break;
      }
    }
    EXPECT_EQ(size_1, 8334);
    EXPECT_EQ(size_2, 25000);
  }
  flow->Wait(3 * 1000);
}

}  // namespace modelbox
