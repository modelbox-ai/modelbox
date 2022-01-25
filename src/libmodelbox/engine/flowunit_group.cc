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

void FlowUnitGroup::PreProcess(FUExecContextList &exec_ctx_list,
                               FUExecContextList &err_exec_ctx_list) {
  auto exec_ctx_iter = exec_ctx_list.begin();
  while (exec_ctx_iter != exec_ctx_list.end()) {
    auto exec_ctx = *exec_ctx_iter;
    const auto &data_ctx = exec_ctx->GetDataCtx();
    const auto &flowunit = exec_ctx->GetFlowUnit();

    if (!data_ctx->IsDataErrorVisible()) {
      auto error = data_ctx->GetError();
      if (error != nullptr) {
        data_ctx->DealWithDataError(error);
        err_exec_ctx_list.push_back(exec_ctx);
        exec_ctx_iter = exec_ctx_list.erase(exec_ctx_iter);
        continue;
      }
    }

    if (data_ctx->IsDataGroupPre()) {
      auto status = flowunit->DataGroupPre(
          std::dynamic_pointer_cast<DataContext>(data_ctx));
      if (status != STATUS_SUCCESS) {
        auto error = std::make_shared<FlowUnitError>(this->unit_name_,
                                                     "DataGroupPre", status);
        std::dynamic_pointer_cast<StreamCollapseFlowUnitDataContext>(data_ctx)
            ->DealWithDataGroupPreError(error);

        exec_ctx_iter = exec_ctx_list.erase(exec_ctx_iter);
        err_exec_ctx_list.push_back(exec_ctx);
        continue;
      }

      std::dynamic_pointer_cast<StreamCollapseFlowUnitDataContext>(data_ctx)
          ->UpdateDataGroupPostFlag(true);
    }

    if (data_ctx->IsDataPre()) {
      auto status =
          flowunit->DataPre(std::dynamic_pointer_cast<DataContext>(data_ctx));
      data_ctx->UpdateStartFlag();
      if (status != STATUS_SUCCESS) {
        auto error = std::make_shared<FlowUnitError>(this->unit_name_,
                                                     "DataPre", status);
        data_ctx->DealWithDataPreError(error);
        exec_ctx_iter = exec_ctx_list.erase(exec_ctx_iter);
        err_exec_ctx_list.push_back(exec_ctx);
        continue;
      } else {
        data_ctx->UpdateDataPostFlag(true);
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
      data_ctx->FillEmptyOutput();
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

Status FlowUnitGroup::PostProcessData(FUExecContextList &exec_ctx_list,
                                      FUExecContextList &err_exec_ctx_list) {
  auto exec_ctx_iter = exec_ctx_list.begin();
  auto status = STATUS_OK;
  auto ret_status = STATUS_OK;
  while (exec_ctx_iter != exec_ctx_list.end()) {
    auto exec_ctx = *exec_ctx_iter;
    const auto &data_ctx = exec_ctx->GetDataCtx();
    if (data_ctx->IsErrorStatus()) {
      auto error = std::make_shared<FlowUnitError>(this->unit_name_, "Process",
                                                   data_ctx->process_status_);
      data_ctx->DealWithProcessError(error);
    }

    if (data_ctx->IsOutputStreamError()) {
      err_exec_ctx_list.push_back(exec_ctx);
      exec_ctx_iter = exec_ctx_list.erase(exec_ctx_iter);
      continue;
    }

    status = data_ctx->LabelData();
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    const auto &flowunit = exec_ctx->GetFlowUnit();
    if (data_ctx->IsDataPost()) {
      status =
          flowunit->DataPost(std::dynamic_pointer_cast<DataContext>(data_ctx));
    }

    if (data_ctx->IsDataGroupPost()) {
      status = flowunit->DataGroupPost(
          std::dynamic_pointer_cast<DataContext>(data_ctx));
    }

    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }
    exec_ctx_iter++;
  }
  return ret_status;
}

Status FlowUnitGroup::PostProcessError(FUExecContextList &err_exec_ctx_list) {
  auto status = STATUS_OK;
  for (auto &err_exec_ctx : err_exec_ctx_list) {
    const auto &err_data_ctx = err_exec_ctx->GetDataCtx();
    const auto &flowunit = err_exec_ctx->GetFlowUnit();
    status = err_data_ctx->LabelError();
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      return status;
    }
    if (err_data_ctx->IsDataPost()) {
      flowunit->DataPost(std::dynamic_pointer_cast<DataContext>(err_data_ctx));
    }

    if (err_data_ctx->IsDataGroupPost()) {
      flowunit->DataGroupPost(
          std::dynamic_pointer_cast<DataContext>(err_data_ctx));
    }
  }
  return status;
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
  FUExecContextList err_exec_ctx_list;
  Status status = STATUS_OK;
  Status ret_status = STATUS_OK;
  auto exec_ctx_list = CreateExecCtx(data_ctx_list);
  try {
    PreProcess(exec_ctx_list, err_exec_ctx_list);

    if (exec_ctx_list.size() != 0) {
      status = Process(exec_ctx_list);
      if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
        ret_status = status;
        return ret_status;
      }

      status = PostProcessData(exec_ctx_list, err_exec_ctx_list);
      if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
        ret_status = status;
        return ret_status;
      }
    }

    status = PostProcessError(err_exec_ctx_list);
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      ret_status = status;
      return ret_status;
    }

    for (auto &err_ctx : err_exec_ctx_list) {
      exec_ctx_list.push_back(err_ctx);
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

void FlowUnitGroup::SetNode(std::shared_ptr<Node> node) { node_ = node; }

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
      for (auto &port_name : input_ports_in_cfg) {
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
      for (auto &port_name : output_ports_in_cfg) {
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

Status FlowUnitGroup::Init(const std::set<std::string> &input_ports_name,
                           const std::set<std::string> &output_ports_name,
                           std::shared_ptr<FlowUnitManager> flowunit_mgr,
                           bool checkport) {
  if (flowunit_mgr == nullptr) {
    return STATUS_FAULT;
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
  for (auto &flowunit : flowunit_group_) {
    if (!flowunit) {
      MBLOG_WARN << "flow unit is nullptr.";
      continue;
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
      status = {status, "open flowunit " + flowunit_desc->GetFlowUnitName() +
                            ", type " +
                            flowunit_desc->GetDriverDesc()->GetType() +
                            " failed."};
      break;
    }

    MBLOG_DEBUG << flowunit_desc->GetFlowUnitName() << ":"
                << flowunit_desc->GetFlowUnitAliasName() << " opened.";
  }

  bool need_check_output = false;
  if (config_) {
    batch_size_ = config_->GetProperty<uint32_t>("batch_size", 1);
    uint32_t max_batch_size =
        GetExecutorUnit()->GetFlowUnitDesc()->GetMaxBatchSize();
    if (max_batch_size != 0 && batch_size_ > max_batch_size) {
      batch_size_ = max_batch_size;
    }
    need_check_output = config_->GetProperty<bool>("need_check_output", false);
  }

  auto node = node_.lock();
  if (node != nullptr) {
    MBLOG_INFO << "node: " << node->GetName() << " get batch size is " << batch_size_;
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