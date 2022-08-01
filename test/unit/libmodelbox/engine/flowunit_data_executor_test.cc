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

#include <modelbox/flowunit_data_executor.h>
#include <modelbox/node.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace modelbox {

class ExecutorMockMemory : public DeviceMemory {
 public:
  ExecutorMockMemory(const std::shared_ptr<Device> &device,
                     const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                     std::shared_ptr<void> device_mem_ptr, size_t size)
      : DeviceMemory(device, mem_mgr, device_mem_ptr, size) {}
};

class ExecutorMockMemMgr : public DeviceMemoryManager {
 public:
  ExecutorMockMemMgr() : DeviceMemoryManager("0") { SetMemQuota(1024 * 1024); }

  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) override {
    return std::make_shared<ExecutorMockMemory>(device, shared_from_this(),
                                                mem_ptr, size);
  }

  void *Malloc(size_t size, uint32_t mem_flags) override {
    return new (std::nothrow) uint8_t[size];
  }

  void Free(void *mem_ptr, uint32_t mem_flags) override {
    delete[](uint8_t *) mem_ptr;
  }

  Status Copy(void *dest, size_t dest_size, const void *src_buffer,
              size_t src_size, DeviceMemoryCopyKind kind) override {
    return STATUS_OK;
  }

  Status DeviceMemoryCopy(const std::shared_ptr<DeviceMemory> &dest_memory,
                          size_t dest_offset,
                          const std::shared_ptr<const DeviceMemory> &src_memory,
                          size_t src_offset, size_t src_size,
                          DeviceMemoryCopyKind copy_kind =
                              DeviceMemoryCopyKind::FromHost) override {
    return STATUS_OK;
  }

  Status GetDeviceMemUsage(size_t *free, size_t *total) const override {
    return STATUS_OK;
  }
};

class ExecutorMockDevice : public Device {
 public:
  ExecutorMockDevice() : Device(std::make_shared<ExecutorMockMemMgr>()) {}

  std::string GetDeviceID() const override { return "0"; }
};

class ExecutorMockFlowUnit : public FlowUnit {
 public:
  Status Open(const std::shared_ptr<Configuration> &config) override {
    return STATUS_OK;
  }

  Status Close() override { return STATUS_OK; }

  MOCK_METHOD1(Process, Status(std::shared_ptr<DataContext> data_ctx));
};

class ExecutorMockDataContext : public FlowUnitDataContext {
 public:
  ExecutorMockDataContext(Node *node)
      : FlowUnitDataContext(node, nullptr, nullptr) {}

  void MockInput(std::shared_ptr<Device> device, size_t port_num,
                 size_t port_data_size) {
    cur_input_valid_data_.clear();
    for (size_t port_idx = 0; port_idx < port_num; ++port_idx) {
      auto &port_data = cur_input_valid_data_[std::to_string(port_idx)];
      for (size_t data_idx = 0; data_idx < port_data_size; ++data_idx) {
        auto mem = device->MemAlloc(10);
        port_data.push_back(std::make_shared<Buffer>(mem));
      }
    }
  }

 protected:
  void UpdateProcessState() override{};
};

class ExecutorTestConfig {
 public:
  size_t device_count{5};
  size_t input_port_count{1};
  size_t input_data_count{1};
  size_t output_port_count{1};
  size_t process_call_times{2};
  std::function<Status(std::shared_ptr<DataContext>)> fu_process;
  size_t ctx_count{10};
  size_t batch_size{4};
  bool need_contiguous{false};
  FlowType node_flow_type{NORMAL};
  FlowOutputType node_output_type{ORIGIN};
  ConditionType node_condition_type{ConditionType::NONE};
  std::function<void(FUExecContextList &ctx_list)> before_process;
  Status expect_process_ret{STATUS_SUCCESS};
  std::function<void(FUExecContextList &ctx_list)> after_process;
};

class FlowUnitExecutorTest : public testing::Test {
 public:
  std::vector<std::shared_ptr<Device>> CreateDevices(size_t count) {
    std::vector<std::shared_ptr<Device>> devices;
    devices.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      devices.push_back(std::make_shared<ExecutorMockDevice>());
    }

    return devices;
  }

  std::vector<std::shared_ptr<FlowUnit>> CreateFlowUnits(
      const std::vector<std::shared_ptr<Device>> &devices) {
    std::vector<std::shared_ptr<FlowUnit>> flowunits;
    flowunits.reserve(devices.size());
    for (const auto &device : devices) {
      auto fu = std::make_shared<ExecutorMockFlowUnit>();
      fu->SetBindDevice(device);
      flowunits.push_back(fu);
    }

    return flowunits;
  }

  std::list<std::shared_ptr<FlowUnitExecContext>> CreateExecCtxs(
      size_t ctx_count, Node *node,
      const std::vector<std::shared_ptr<FlowUnit>> &flowunits) {
    std::list<std::shared_ptr<FlowUnitExecContext>> exec_ctx_list;
    for (size_t i = 0; i < ctx_count; ++i) {
      auto data_ctx = std::make_shared<ExecutorMockDataContext>(node);
      auto exec_ctx = std::make_shared<FlowUnitExecContext>(data_ctx);
      exec_ctx->SetFlowUnit(flowunits[i % flowunits.size()]);
      exec_ctx_list.push_back(exec_ctx);
    }

    return exec_ctx_list;
  }

  void MockInput(const std::vector<std::shared_ptr<Device>> &devices,
                 std::list<std::shared_ptr<FlowUnitExecContext>> &exec_ctx_list,
                 size_t port_num, size_t port_data_size) {
    size_t i = 0;
    for (auto &exec_ctx : exec_ctx_list) {
      const auto &device = devices[i % devices.size()];
      ++i;
      std::dynamic_pointer_cast<ExecutorMockDataContext>(exec_ctx->GetDataCtx())
          ->MockInput(device, port_num, port_data_size);
    }
  }

  void ExecutorTest(const ExecutorTestConfig &cfg) {
    MBLOG_INFO << "Flow type " << cfg.node_flow_type;
    auto devices = CreateDevices(cfg.device_count);
    auto flowunits = CreateFlowUnits(devices);
    for (auto &flowunit : flowunits) {
      auto mock_fu = std::dynamic_pointer_cast<ExecutorMockFlowUnit>(flowunit);
      auto desc = mock_fu->GetFlowUnitDesc();
      for (size_t i = 0; i < cfg.input_port_count; ++i) {
        FlowUnitInput input_port(std::to_string(i), "cpu");
        input_port.SetDevice(mock_fu->GetBindDevice());
        desc->AddFlowUnitInput(input_port);
      }

      for (size_t i = 0; i < cfg.output_port_count; ++i) {
        desc->AddFlowUnitOutput({std::to_string(i), "cpu"});
      }

      EXPECT_CALL(*mock_fu, Process(testing::_))
          .Times(cfg.process_call_times)
          .WillRepeatedly(testing::Invoke(cfg.fu_process));
    }

    auto node = std::make_shared<Node>();
    node->SetName("test_node");
    node->SetFlowType(cfg.node_flow_type);
    node->SetOutputType(cfg.node_output_type);
    node->SetConditionType(cfg.node_condition_type);
    node->SetInputContiguous(cfg.need_contiguous);
    std::set<std::string> input_names;
    std::set<std::string> output_names;
    for (size_t i = 0; i < cfg.input_port_count; ++i) {
      input_names.insert(std::to_string(i));
    }
    for (size_t i = 0; i < cfg.output_port_count; ++i) {
      output_names.insert(std::to_string(i));
    }
    ConfigurationBuilder builder;
    node->Init(input_names, output_names, builder.Build());
    auto ctx_list = CreateExecCtxs(cfg.ctx_count, node.get(), flowunits);
    if (cfg.input_port_count > 0) {
      MockInput(devices, ctx_list, cfg.input_port_count, cfg.input_data_count);
    }

    if (cfg.before_process) {
      cfg.before_process(ctx_list);
    }

    FlowUnitDataExecutor executor(node, cfg.batch_size);
    executor.SetNeedCheckOutput(true);
    auto ret = executor.Process(ctx_list);
    ASSERT_EQ(ret, cfg.expect_process_ret);
    if (cfg.after_process) {
      cfg.after_process(ctx_list);
    }
  }

  void TestDataPreparePerf(
      const std::vector<std::shared_ptr<Device>> &devices,
      const std::vector<std::shared_ptr<FlowUnit>> &flowunits,
      std::shared_ptr<Node> node, bool is_stream) {
    auto ctx_list = CreateExecCtxs(320, node.get(), flowunits);
    MockInput(devices, ctx_list, 4, 32);
    FlowUnitExecDataView data_view(ctx_list);
    auto start = std::chrono::steady_clock::now();
    data_view.LoadInputFromExecCtx(true, is_stream, 8, false);
    auto end = std::chrono::steady_clock::now();
    auto cost =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    MBLOG_ERROR << "Prepare view, type:" << (is_stream ? "stream" : "normal")
                << " device:8, stream_per_device:40, port_num:4, "
                   "data_per_stream:32, batch_size:8, cost:"
                << cost << " ms";
  }

  void TestWriteBackPerf(
      const std::vector<std::shared_ptr<Device>> &devices,
      const std::vector<std::shared_ptr<FlowUnit>> &flowunits,
      std::shared_ptr<Node> node, bool is_stream) {
    auto ctx_list = CreateExecCtxs(320, node.get(), flowunits);
    MockInput(devices, ctx_list, 4, 32);
    FlowUnitExecDataView data_view(ctx_list);
    data_view.LoadInputFromExecCtx(true, is_stream, 8, false);
    auto data_flowunits = data_view.GetFlowUnits();
    for (auto *flowunit : data_flowunits) {
      auto batched_exec_data_ctx_list =
          data_view.GetFlowUnitProcessData(flowunit);
      for (auto &batch_data_ctx : batched_exec_data_ctx_list) {
        for (auto &data_ctx : batch_data_ctx) {
          auto outputs = data_ctx->Output();
          for (auto &port_item : *outputs) {
            std::vector<size_t> shape(32, 10);
            port_item.second->Build(shape);
          }
        }
      }
    }
    auto start = std::chrono::steady_clock::now();
    data_view.SaveOutputToExecCtx();
    auto end = std::chrono::steady_clock::now();
    auto cost =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    MBLOG_ERROR << "WriteBack, type:" << (is_stream ? "stream" : "normal")
                << " device:8, stream_per_device:40, port_num:2, "
                   "data_per_stream:32, batch_size:8, cost:"
                << cost << " ms";
  }

  std::vector<FlowType> flow_types_ = {NORMAL, STREAM};

 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(FlowUnitExecutorTest, EventInputTest) {
  ExecutorTestConfig cfg;
  cfg.input_port_count = 0;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto inputs = data_ctx->Input();
    EXPECT_EQ(inputs->size(), 0);
    auto outputs = data_ctx->Output();
    EXPECT_EQ(outputs->size(), 1);
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({1});
    auto *ptr = (uint8_t *)output->At(0)->MutableData();
    auto val =
        std::static_pointer_cast<uint8_t>(data_ctx->GetPrivate("test_val"));
    ptr[0] = *val;
    return STATUS_OK;
  };
  cfg.before_process = [](FUExecContextList &ctx_list) {
    uint8_t index = 0;
    for (auto &ctx : ctx_list) {
      auto test_val = std::make_shared<uint8_t>(index);
      ++index;
      ctx->GetDataCtx()->SetPrivate("test_val", test_val);
    }
  };
  cfg.after_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto outputs = data_ctx->Output();
      auto val =
          std::static_pointer_cast<uint8_t>(data_ctx->GetPrivate("test_val"));
      EXPECT_EQ(outputs->size(), 1);
      for (auto &out_item : *outputs) {
        const auto &port_name = out_item.first;
        auto &port_data = out_item.second;
        EXPECT_EQ(port_name, "0");
        ASSERT_NE(port_data, nullptr);
        EXPECT_EQ(port_data->Size(), 1);
        auto buffer = port_data->At(0);
        ASSERT_NE(buffer, nullptr);
        auto *ptr = (uint8_t *)(buffer->MutableData());
        EXPECT_EQ(*ptr, *val);
      }
    }
  };
  cfg.node_flow_type = NORMAL;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, ExpandTest) {
  ExecutorTestConfig cfg;
  cfg.output_port_count = 2;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto inputs = data_ctx->Input();
    EXPECT_EQ(inputs->size(), 1);
    auto outputs = data_ctx->Output();
    EXPECT_EQ(outputs->size(), 2);
    auto input = data_ctx->Input("0");
    EXPECT_NE(input, nullptr);
    EXPECT_EQ(input->Size(), 1);
    auto output1 = data_ctx->Output("0");
    EXPECT_NE(output1, nullptr);
    auto output2 = data_ctx->Output("1");
    EXPECT_NE(output2, nullptr);
    output1->Build({1, 1});
    auto *ptr = (uint8_t *)output1->At(0)->MutableData();
    ptr[0] = 1;
    ptr = (uint8_t *)output1->At(1)->MutableData();
    ptr[0] = 2;
    output2->Build({1, 1});
    ptr = (uint8_t *)output2->At(0)->MutableData();
    ptr[0] = 3;
    ptr = (uint8_t *)output2->At(1)->MutableData();
    ptr[0] = 4;
    return STATUS_OK;
  };
  cfg.node_output_type = EXPAND;
  cfg.input_data_count = 1;
  cfg.batch_size = 1;
  cfg.after_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto outputs = data_ctx->Output();
      uint8_t val = 1;
      EXPECT_EQ(outputs->size(), 2);
      for (size_t port_idx = 0; port_idx < outputs->size(); ++port_idx) {
        auto &buffer_list = outputs->at(std::to_string(port_idx));
        ASSERT_EQ(buffer_list->Size(), 2);
        auto *ptr = (uint8_t *)(buffer_list->At(0)->ConstData());
        EXPECT_EQ(*ptr, val);
        ++val;
        ptr = (uint8_t *)(buffer_list->At(1)->ConstData());
        EXPECT_EQ(*ptr, val);
        ++val;
      }
    }
  };
  cfg.node_flow_type = NORMAL;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, CollapseTest) {
  ExecutorTestConfig cfg;
  cfg.input_port_count = 2;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto inputs = data_ctx->Input();
    EXPECT_EQ(inputs->size(), 2);
    auto outputs = data_ctx->Output();
    EXPECT_EQ(outputs->size(), 1);
    auto input = data_ctx->Input("0");
    EXPECT_NE(input, nullptr);
    EXPECT_EQ(input->Size(), 4);
    auto input2 = data_ctx->Input("1");
    EXPECT_NE(input2, nullptr);
    EXPECT_EQ(input2->Size(), 4);
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({1});
    auto *ptr = (uint8_t *)output->At(0)->MutableData();
    ptr[0] = 1;
    return STATUS_OK;
  };
  cfg.node_output_type = COLLAPSE;
  cfg.input_data_count = 4;
  cfg.batch_size = 2;
  cfg.after_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto outputs = data_ctx->Output();
      auto &buffer_list = outputs->at("0");
      ASSERT_EQ(buffer_list->Size(), 1);
      auto *ptr = (uint8_t *)(buffer_list->At(0)->ConstData());
      EXPECT_EQ(*ptr, 1);
    }
  };
  cfg.node_flow_type = NORMAL;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, OriginErrorTest) {
  ExecutorTestConfig cfg;
  cfg.input_port_count = 2;
  cfg.output_port_count = 2;
  cfg.process_call_times = 1;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({1});
    output = data_ctx->Output("1");
    EXPECT_NE(output, nullptr);
    output->Build({1, 1, 1, 1});
    return STATUS_OK;
  };
  cfg.ctx_count = 5;
  cfg.input_data_count = 4;
  cfg.batch_size = 6;
  cfg.expect_process_ret = STATUS_FAULT;
  cfg.node_flow_type = NORMAL;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, OriginError2Test) {
  ExecutorTestConfig cfg;
  cfg.input_port_count = 2;
  cfg.output_port_count = 2;
  cfg.process_call_times = 1;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({1, 1, 1});
    output = data_ctx->Output("1");
    EXPECT_NE(output, nullptr);
    output->Build({1, 1, 1});
    return STATUS_OK;
  };
  cfg.ctx_count = 5;
  cfg.input_data_count = 4;
  cfg.batch_size = 6;
  cfg.node_flow_type = NORMAL;
  cfg.expect_process_ret = STATUS_FAULT;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.expect_process_ret = STATUS_SUCCESS;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  cfg.expect_process_ret = STATUS_FAULT;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  cfg.expect_process_ret = STATUS_SUCCESS;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, OriginTest) {
  ExecutorTestConfig cfg;
  cfg.input_port_count = 2;
  cfg.output_port_count = 2;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    for (size_t port_idx = 0; port_idx < 2; ++port_idx) {
      auto port_name = std::to_string(port_idx);
      auto input = data_ctx->Input(port_name);
      EXPECT_NE(input, nullptr);
      auto output = data_ctx->Output(port_name);
      EXPECT_NE(output, nullptr);
      for (auto &buffer : *input) {
        output->PushBack(buffer);
      }
    }

    return STATUS_OK;
  };
  cfg.input_data_count = 5;
  cfg.batch_size = 2;
  cfg.before_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto inputs = data_ctx->Input();
      auto ctx_id = std::to_string((uintptr_t)ctx.get());
      for (size_t port_idx = 0; port_idx < 2; ++port_idx) {
        auto port_name = std::to_string(port_idx);
        auto input = data_ctx->Input(port_name);
        for (size_t buffer_idx = 0; buffer_idx < input->Size(); ++buffer_idx) {
          auto buffer_id = std::to_string(buffer_idx);
          auto buffer = input->At(buffer_idx);
          buffer->Set("input_id", ctx_id + port_name + buffer_id);
        }
      }
    }
  };
  cfg.after_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto ctx_id = std::to_string((uintptr_t)ctx.get());
      for (size_t port_idx = 0; port_idx < 2; ++port_idx) {
        auto port_name = std::to_string(port_idx);
        auto output = data_ctx->Output(port_name);
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->Size(), 5);
        for (size_t buffer_idx = 0; buffer_idx < output->Size(); ++buffer_idx) {
          auto buffer_id = std::to_string(buffer_idx);
          auto buffer = output->At(buffer_idx);
          std::string input_id;
          buffer->Get("input_id", input_id);
          EXPECT_EQ(input_id, ctx_id + port_name + buffer_id);
        }
      }
    }
  };
  cfg.node_flow_type = NORMAL;
  cfg.process_call_times = 5;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.process_call_times = 6;
  ExecutorTest(cfg);
  cfg.node_flow_type = NORMAL;
  cfg.need_contiguous = false;
  cfg.process_call_times = 5;
  ExecutorTest(cfg);
  cfg.node_flow_type = STREAM;
  cfg.need_contiguous = false;
  cfg.process_call_times = 6;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, IfElseTest) {
  ExecutorTestConfig cfg;
  cfg.process_call_times = 4;
  cfg.input_port_count = 2;
  cfg.output_port_count = 2;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({10});
    return STATUS_OK;
  };
  cfg.input_data_count = 2;
  cfg.batch_size = 2;
  cfg.node_condition_type = ConditionType::IF_ELSE;
  cfg.after_process = [](FUExecContextList &ctx_list) {
    for (auto &ctx : ctx_list) {
      auto data_ctx = ctx->GetDataCtx();
      auto outputs = data_ctx->Output();
      auto buffer_list = outputs->at("0");
      EXPECT_EQ(buffer_list->Size(), 2);
      for (auto &buffer : *buffer_list) {
        EXPECT_NE(buffer, nullptr);
      }
      buffer_list = outputs->at("1");
      EXPECT_EQ(buffer_list->Size(), 2);
      for (auto &buffer : *buffer_list) {
        EXPECT_EQ(buffer, nullptr);
      }
    }
  };
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, IfElseErrorTest) {
  ExecutorTestConfig cfg;
  cfg.process_call_times = 4;
  cfg.input_port_count = 2;
  cfg.output_port_count = 2;
  cfg.fu_process = [](std::shared_ptr<DataContext> data_ctx) -> Status {
    auto output = data_ctx->Output("0");
    EXPECT_NE(output, nullptr);
    output->Build({10, 10, 10});
    output = data_ctx->Output("1");
    EXPECT_NE(output, nullptr);
    output->Build({10});
    return STATUS_OK;
  };
  cfg.input_data_count = 2;
  cfg.batch_size = 2;
  cfg.node_condition_type = ConditionType::IF_ELSE;
  cfg.expect_process_ret = STATUS_FAULT;
  ExecutorTest(cfg);
}

TEST_F(FlowUnitExecutorTest, DataViewPerfTest) {
  /**
   * case 8 device, 40 stream per device, 32 data per stream, batch is 8
   */
  auto devices = CreateDevices(8);
  auto flowunits = CreateFlowUnits(devices);
  for (auto &flowunit : flowunits) {
    auto mock_fu = std::dynamic_pointer_cast<ExecutorMockFlowUnit>(flowunit);
    auto desc = mock_fu->GetFlowUnitDesc();
    for (size_t i = 0; i < 4; ++i) {
      FlowUnitInput in{std::to_string(i), "cpu"};
      in.SetDevice(flowunit->GetBindDevice());
      desc->AddFlowUnitInput(in);
    }
    desc->AddFlowUnitOutput({"0", "cpu"});
    desc->AddFlowUnitOutput({"1", "cpu"});
    EXPECT_CALL(*mock_fu, Process(testing::_))
        .WillRepeatedly(testing::Invoke(
            [](std::shared_ptr<DataContext> data_ctx) -> Status {
              return STATUS_OK;
            }));
  }
  auto node = std::make_shared<Node>();
  TestDataPreparePerf(devices, flowunits, node, false);
  TestDataPreparePerf(devices, flowunits, node, true);
  TestWriteBackPerf(devices, flowunits, node, false);
  TestWriteBackPerf(devices, flowunits, node, true);
}

}  // namespace modelbox