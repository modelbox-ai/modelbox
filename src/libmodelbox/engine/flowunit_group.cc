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

#include "modelbox/flowunit_group.h"

#include "modelbox/error.h"
#include "modelbox/flowunit_balancer.h"
#include "modelbox/flowunit_data_executor.h"
#include "modelbox/node.h"

namespace modelbox {

void FlowUnitGroup::InitTrace() {
  if (profiler_ == nullptr) {
    return;
  }

  auto trace = profiler_->GetTrace();
  if (trace == nullptr) {
    return;
  }

  auto node = node_.lock();
  if (node == nullptr) {
    MBLOG_WARN << "node is nullptr for flownit " << unit_name_
               << ", init trace failed";
    return;
  }

  flowunit_trace_ = trace->FlowUnit(node->GetName());
  if (flowunit_trace_ == nullptr) {
    MBLOG_WARN << "create trace for node " << node->GetName() << " failed";
  }
}

uint32_t FlowUnitGroup::GetBatchSize() const { return batch_size_; }

std::shared_ptr<TraceSlice> FlowUnitGroup::StartTrace(
    FUExecContextList &exec_ctx_list) {
  std::call_once(trace_init_flag_, &FlowUnitGroup::InitTrace, this);

  if (flowunit_trace_ == nullptr) {
    return nullptr;
  }

  auto total_input_count = std::accumulate(
      exec_ctx_list.begin(), exec_ctx_list.end(), (size_t)0,
      [](size_t sum, std::shared_ptr<FlowUnitExecContext> &exec_ctx) {
        const auto &data_ctx = exec_ctx->GetDataCtx();
        auto inputs = data_ctx->GetInputs();
        if (inputs.empty()) {
          // this is event
          return sum + 1;
        }

        auto input_count = inputs.begin()->second.size();
        return sum + input_count;
      });

  auto slice = flowunit_trace_->Slice(TraceSliceType::PROCESS, "");
  slice->SetBatchSize(total_input_count);
  slice->Begin();
  return slice;
}

void FlowUnitGroup::StopTrace(std::shared_ptr<TraceSlice> &slice) {
  if (slice != nullptr) {
    slice->End();
  }
}

void FlowUnitGroup::PreProcess(FUExecContextList &exec_ctx_list) {
  auto exec_ctx_iter = exec_ctx_list.begin();
  while (exec_ctx_iter != exec_ctx_list.end()) {
    auto exec_ctx = *exec_ctx_iter;
    const auto &data_ctx = exec_ctx->GetDataCtx();
    const auto &flowunit = exec_ctx->GetFlowUnit();

    // stream start
    if (data_ctx->IsDataPre()) {
      auto status =
          flowunit->DataPre(std::dynamic_pointer_cast<DataContext>(data_ctx));
      if (status != STATUS_SUCCESS) {
        MBLOG_INFO << "flowunit " << unit_name_
                   << " data pre return: " << status;
        const auto &error_msg = status.Errormsg();
        data_ctx->DealWithDataPreError(unit_name_ + ".DataPreError", error_msg);
      }
    }

    ++exec_ctx_iter;
  }
}

Status FlowUnitGroup::Process(FUExecContextList &exec_ctx_list) {
  FUExecContextList actual_exec_ctx_list;
  // will skip end_buffer create by framework
  for (auto &exec_ctx : exec_ctx_list) {
    const auto &data_ctx = exec_ctx->GetDataCtx();
    if (!data_ctx->IsSkippable()) {
      actual_exec_ctx_list.emplace_back(exec_ctx);
    } else {
      data_ctx->SetStatus(data_ctx->GetLastStatus());
      data_ctx->SetSkippable(false);
    }
  }

  if (actual_exec_ctx_list.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto slice = StartTrace(actual_exec_ctx_list);
  auto status = executor_->Process(actual_exec_ctx_list);
  StopTrace(slice);
  if (!status) {
    MBLOG_WARN << "execute unit " << unit_name_ << " failed: " << status;
    return STATUS_STOP;
  }

  return status;
}

Status FlowUnitGroup::PostProcess(FUExecContextList &exec_ctx_list) {
  auto exec_ctx_iter = exec_ctx_list.begin();
  auto status = STATUS_OK;
  auto ret_status = STATUS_OK;
  while (exec_ctx_iter != exec_ctx_list.end()) {
    auto exec_ctx = *exec_ctx_iter;
    const auto &data_ctx = exec_ctx->GetDataCtx();

    status = data_ctx->PostProcess();
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    const auto &flowunit = exec_ctx->GetFlowUnit();
    if (data_ctx->IsDataPost()) {
      status =
          flowunit->DataPost(std::dynamic_pointer_cast<DataContext>(data_ctx));
      if (!status) {
        MBLOG_INFO << "flowunit " << unit_name_
                   << " data post return: " << status;
      }
    }

    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    // make sure ctx state is right for next process
    data_ctx->UpdateProcessState();

    exec_ctx_iter++;
  }
  return ret_status;
}

void FlowUnitGroup::PostProcessEvent(FUExecContextList &exec_ctx_list) {
  std::vector<std::shared_ptr<FlowUnitInnerEvent>> event_vector;
  for (auto &exec_ctx : exec_ctx_list) {
    const auto &data_ctx = exec_ctx->GetDataCtx();
    auto event = data_ctx->GenerateSendEvent();
    if (event != nullptr) {
      event_vector.push_back(event);
    }
  }
  auto node = node_.lock();
  if (node == nullptr) {
    return;
  }
  node->SendBatchEvent(event_vector);
}

FUExecContextList FlowUnitGroup::CreateExecCtx(
    std::list<std::shared_ptr<FlowUnitDataContext>> &data_ctx_list) {
  FUExecContextList exec_ctx_list;
  for (auto &data_ctx : data_ctx_list) {
    auto exec_ctx = std::make_shared<FlowUnitExecContext>(data_ctx);
    exec_ctx->SetFlowUnit(balancer_->GetFlowUnit(data_ctx));
    exec_ctx_list.push_back(exec_ctx);
  }

  return exec_ctx_list;
}

Status FlowUnitGroup::Run(
    std::list<std::shared_ptr<FlowUnitDataContext>> &data_ctx_list) {
  Status status = STATUS_OK;
  Status ret_status = STATUS_OK;
  auto exec_ctx_list = CreateExecCtx(data_ctx_list);
  try {
    PreProcess(exec_ctx_list);

    status = Process(exec_ctx_list);
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    status = PostProcess(exec_ctx_list);
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    PostProcessEvent(exec_ctx_list);
    // rearrange data ctx order
    std::list<std::shared_ptr<FlowUnitDataContext>> processed_ctx_list;
    for (auto &exec_ctx : exec_ctx_list) {
      processed_ctx_list.push_back(exec_ctx->GetDataCtx());
    }
    data_ctx_list.swap(processed_ctx_list);
  } catch (const std::exception &e) {
    ret_status = {STATUS_FAULT, unit_name_ + " process failed, " + e.what()};
  }

  return ret_status;
}

void FlowUnitGroup::SetNode(const std::shared_ptr<Node> &node) { node_ = node; }

std::shared_ptr<FlowUnit> FlowUnitGroup::GetExecutorUnit() {
  return flowunit_group_[0];
}

Status FlowUnitGroup::CheckInputAndOutput(
    const std::set<std::string> &input_ports_name,
    const std::set<std::string> &output_ports_name) {
  auto flowunit_iter = flowunit_group_.begin();
  while (flowunit_iter != flowunit_group_.end()) {
    auto flowunit_desc = flowunit_iter->get()->GetFlowUnitDesc();
    auto check_failed = false;

    auto input_set = flowunit_desc->GetFlowUnitInput();
    auto input_ports_in_cfg = input_ports_name;
    for (auto &input_item : input_set) {
      auto item = input_ports_in_cfg.find(input_item.GetPortName());
      if (item == input_ports_in_cfg.end()) {
        MBLOG_WARN << "node input port: " << input_item.GetPortName()
                   << " is not connected";
        check_failed = true;
        continue;
      }

      input_ports_in_cfg.erase(item);
    }

    if (!input_ports_in_cfg.empty()) {
      std::string err_msg = "config input port [ ";
      for (const auto &port_name : input_ports_in_cfg) {
        err_msg += port_name + " ";
      }

      err_msg += "] not defined in flowunit";
      MBLOG_WARN << err_msg;
      check_failed = true;
    }

    auto output_set = flowunit_desc->GetFlowUnitOutput();
    auto output_ports_in_cfg = output_ports_name;
    for (auto &output_item : output_set) {
      auto item = output_ports_in_cfg.find(output_item.GetPortName());
      if (item == output_ports_in_cfg.end()) {
        MBLOG_WARN << "node output port: " << output_item.GetPortName()
                   << " is not connected";
        check_failed = true;
        continue;
      }

      output_ports_in_cfg.erase(item);
    }

    if (!output_ports_in_cfg.empty()) {
      std::string err_msg = "config output port [ ";
      for (const auto &port_name : output_ports_in_cfg) {
        err_msg += port_name + " ";
      }

      err_msg += "] not defined in flowunit";
      MBLOG_WARN << err_msg;
      check_failed = true;
    }

    if (check_failed) {
      MBLOG_WARN << "flowunit " << flowunit_desc->GetFlowUnitName()
                 << " port check failed.";
      flowunit_iter = flowunit_group_.erase(flowunit_iter);
    } else {
      flowunit_iter++;
    }
  }

  if (flowunit_group_.size() == 0) {
    return {STATUS_BADCONF, "flowunit '" + unit_name_ +
                                "' config error, port not connect correctly."};
  }
  return STATUS_SUCCESS;
}

FlowUnitGroup::FlowUnitGroup(std::string unit_name, std::string unit_type,
                             std::string unit_device_id,
                             std::shared_ptr<Configuration> config,
                             std::shared_ptr<Profiler> profiler)
    : batch_size_(1),
      unit_name_(std::move(unit_name)),
      unit_type_(std::move(unit_type)),
      unit_device_id_(std::move(unit_device_id)),
      config_(std::move(config)),
      profiler_(std::move(profiler)){};

FlowUnitGroup::~FlowUnitGroup() = default;

Status FlowUnitGroup::Init(const std::set<std::string> &input_ports_name,
                           const std::set<std::string> &output_ports_name,
                           const std::shared_ptr<FlowUnitManager> &flowunit_mgr,
                           bool checkport) {
  if (flowunit_mgr == nullptr) {
    return {STATUS_FAULT, "flowunit manager is null"};
  }

  flowunit_group_ =
      flowunit_mgr->CreateFlowUnit(unit_name_, unit_type_, unit_device_id_);
  if (flowunit_group_.size() == 0) {
    if (StatusError == STATUS_OK) {
      StatusError = STATUS_NOTFOUND;
    }
    return {StatusError,
            std::string("create flowunit '") + unit_name_ + "' failed."};
  }

  if (flowunit_group_.size() == 0) {
    return {STATUS_BADCONF, std::string("flowunit '") + unit_name_ +
                                "' config error, port not connect correctly."};
  }

  if (checkport) {
    auto status = CheckInputAndOutput(input_ports_name, output_ports_name);
    if (status != STATUS_SUCCESS) {
      return status;
    }
  }
  return STATUS_OK;
}

Status FlowUnitGroup::Open(const CreateExternalDataFunc &create_func) {
  auto status = STATUS_OK;
  auto open_func = [&](const std::shared_ptr<FlowUnit> &flowunit) -> modelbox::Status {
    if (!flowunit) {
      MBLOG_WARN << "flow unit is nullptr.";
      return STATUS_INVALID;
    }

    auto flowunit_desc = flowunit->GetFlowUnitDesc();
    flowunit->SetExternalData(create_func);
    try {
      status = flowunit->Open(config_);
    } catch (const std::exception &e) {
      status = {STATUS_FAULT,
                flowunit_desc->GetFlowUnitName() + " open failed, " + e.what()};
    }

    if (!status) {
      MBLOG_WARN << flowunit_desc->GetFlowUnitName() << ":"
                 << flowunit_desc->GetFlowUnitAliasName()
                 << " open failed: " << status;
      status = {status, "open flowunit '" + flowunit_desc->GetFlowUnitName() +
                            "', type '" +
                            flowunit_desc->GetDriverDesc()->GetType() +
                            "' failed."};
      return status;
    }

    MBLOG_DEBUG << flowunit_desc->GetFlowUnitName() << ":"
                << flowunit_desc->GetFlowUnitAliasName() << " opened.";

    return STATUS_OK;
  };

  ThreadPool pool(std::thread::hardware_concurrency());
  pool.SetName(unit_name_ + "-Open");
  std::vector<std::future<Status>> result;

  for (auto &flowunit : flowunit_group_) {
    auto ret = pool.Submit(open_func, flowunit);
    result.push_back(std::move(ret));
  }

  for (auto &fut : result) {
    const auto *msg = "open flowunit failed, please check log.";
    if (!fut.valid()) {
      return {STATUS_FAULT, msg};
    }

    auto ret = fut.get();
    if (!ret) {
      return ret;
    }
  }

  bool need_check_output = false;
  if (config_) {
    uint32_t default_batch_size =
        GetExecutorUnit()->GetFlowUnitDesc()->GetDefaultBatchSize();
    batch_size_ =
        config_->GetProperty<uint32_t>("batch_size", default_batch_size);
    uint32_t max_batch_size =
        GetExecutorUnit()->GetFlowUnitDesc()->GetMaxBatchSize();
    if (max_batch_size != 0 && batch_size_ > max_batch_size) {
      batch_size_ = max_batch_size;
    }
    need_check_output = config_->GetProperty<bool>("need_check_output", false);
  }

  auto node = node_.lock();
  if (node != nullptr) {
    MBLOG_INFO << "node: " << node->GetName() << " get batch size is "
               << batch_size_;
  }

  balancer_ = FlowUnitBalancerFactory::GetInstance().CreateBalancer();
  if (balancer_ == nullptr) {
    return {STATUS_FAULT, "Get flowunit balancer failed"};
  }

  auto ret = balancer_->Init(flowunit_group_);
  if (!ret) {
    return {STATUS_FAULT, "Init balancer failed: " + ret.Errormsg()};
  }

  executor_ = std::make_shared<FlowUnitDataExecutor>(node_, batch_size_);
  executor_->SetNeedCheckOutput(need_check_output);
  return status;
}

Status FlowUnitGroup::Close() {
  auto status = STATUS_OK;
  for (auto &flowunit : flowunit_group_) {
    if (!flowunit) {
      MBLOG_WARN << "flow unit is nullptr.";
      continue;
    }

    auto flowunit_desc = flowunit->GetFlowUnitDesc();
    try {
      status = flowunit->Close();
    } catch (const std::exception &e) {
      status = {STATUS_FAULT, flowunit_desc->GetFlowUnitName() +
                                  " close failed, " + e.what()};
    }

    if (!status) {
      MBLOG_WARN << flowunit_desc->GetFlowUnitName() << ":"
                 << flowunit_desc->GetFlowUnitAliasName()
                 << " close failed: " << status;
      break;
    }

    MBLOG_DEBUG << flowunit_desc->GetFlowUnitName() << ":"
                << flowunit_desc->GetFlowUnitAliasName() << " closed.";
  }

  return status;
}

Status FlowUnitGroup::Destory() { return STATUS_OK; }

}  // namespace modelbox