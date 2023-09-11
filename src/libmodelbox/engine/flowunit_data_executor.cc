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

#include "modelbox/flowunit_data_executor.h"

#include <utility>

#include "modelbox/node.h"

namespace modelbox {

Executor::Executor() {
  thread_pool_ = std::make_shared<ThreadPool>();
  thread_pool_->SetName("Executor");
}

Executor::Executor(int thread_count) {
  thread_pool_ = std::make_shared<ThreadPool>(thread_count);
  thread_pool_->SetName("Executor");
}

Executor::~Executor() { thread_pool_ = nullptr; }

void Executor::SetThreadCount(int thread_count) {
  thread_pool_->SetThreadSize(thread_count);
}

FlowUnitExecContext::FlowUnitExecContext(
    std::shared_ptr<FlowUnitDataContext> data_ctx)
    : data_ctx_(std::move(data_ctx)) {}

void FlowUnitExecContext::SetFlowUnit(std::shared_ptr<FlowUnit> fu) {
  bind_fu_ = std::move(fu);
}

const std::shared_ptr<FlowUnit> &FlowUnitExecContext::GetFlowUnit() {
  return bind_fu_;
}

const std::shared_ptr<FlowUnitDataContext> &FlowUnitExecContext::GetDataCtx() {
  return data_ctx_;
}

FlowUnitExecData::FlowUnitExecData(const std::shared_ptr<FlowUnit> &fu)
    : fu_(fu) {
  // Prepare data container
  const auto &fu_desc = fu->GetFlowUnitDesc();
  const auto &in_list = fu_desc->GetFlowUnitInput();
  auto device = fu->GetBindDevice();
  in_data_ = std::make_shared<BufferListMap>();
  for (const auto &in_item : in_list) {
    auto in_device = in_item.GetDevice();
    in_data_->emplace(
        in_item.GetPortName(),
        std::make_shared<BufferList>(in_device, in_item.GetDeviceMemFlags()));
  }

  const auto &out_list = fu_desc->GetFlowUnitOutput();
  out_data_ = std::make_shared<BufferListMap>();
  for (const auto &out_item : out_list) {
    out_data_->emplace(
        out_item.GetPortName(),
        std::make_shared<BufferList>(device, out_item.GetDeviceMemFlags()));
  }

  ext_data_ = std::make_shared<BufferListMap>();
  if (in_list.empty()) {
    ext_data_->emplace(EXTERNAL_PORT_NAME,
                       std::make_shared<BufferList>(device));
  }
}

FlowUnitExecData::~FlowUnitExecData() = default;

void FlowUnitExecData::ReserveCache(size_t buffer_count, DataType type) {
  auto data = in_data_;
  auto *cache = &in_data_cache_;
  if (type == OUT_DATA) {
    data = out_data_;
    cache = &out_data_cache_;
  }

  for (auto &port_item : *data) {
    const auto &port_name = port_item.first;
    auto &cache_buffer_list = (*cache)[port_name];
    cache_buffer_list.clear();
    cache_buffer_list.reserve(buffer_count);
  }
}

void FlowUnitExecData::AppendToCache(
    const std::shared_ptr<FlowUnitExecData> &src, size_t start_idx,
    size_t count, DataType type) {
  auto src_data = src->in_data_;
  auto *cache = &in_data_cache_;
  if (type == OUT_DATA) {
    cache = &out_data_cache_;
    src_data = src->out_data_;
  }

  for (auto &port_item : *src_data) {
    const auto &port_name = port_item.first;
    auto &port_data = port_item.second;
    auto &cache_buffer_list = (*cache)[port_name];
    auto end_idx = start_idx + count;
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      if (port_data->Size() == 0) {
        // For if_else, only one port has data, need push nullptr to result
        cache_buffer_list.push_back(nullptr);
        continue;
      }

      cache_buffer_list.push_back(port_data->At(idx));
    }
  }
}

void FlowUnitExecData::FlushCache(DataType type) {
  auto data = in_data_;
  auto *cache = &in_data_cache_;
  if (type == OUT_DATA) {
    data = out_data_;
    cache = &out_data_cache_;
  }

  for (auto &port_item : *data) {
    const auto &port_name = port_item.first;
    auto &port_data = port_item.second;
    auto &cache_buffer_list = (*cache)[port_name];
    port_data->Swap(cache_buffer_list);
  }
}

std::shared_ptr<BufferListMap> FlowUnitExecData::GetInData() {
  return in_data_;
}

std::shared_ptr<BufferListMap> FlowUnitExecData::GetInDataForUser() {
  return in_data_for_user_;
}

std::shared_ptr<BufferList> FlowUnitExecData::GetInDataForUser(
    const std::string &name) {
  return HasInData(name) ? in_data_for_user_->at(name) : nullptr;
}

void FlowUnitExecData::SetInData(
    const std::string &name,
    const std::vector<std::shared_ptr<Buffer>> &buffer_list) {
  // if in_data_ is empty it means the input is a external data
  if (in_data_->empty()) {
    SetExternalData(name, buffer_list);
    return;
  }

  (*in_data_)[name]->Assign(buffer_list);
}

std::shared_ptr<BufferListMap> FlowUnitExecData::GetOutData() {
  return out_data_;
}

std::shared_ptr<BufferList> FlowUnitExecData::GetOutData(
    const std::string &name) {
  return HasOutData(name) ? out_data_->at(name) : nullptr;
}

Status FlowUnitExecData::SetExternalData(
    const std::string &name,
    const std::vector<std::shared_ptr<Buffer>> &buffer_list) {
  auto iter = ext_data_->find(name);
  if (iter == ext_data_->end()) {
    return {STATUS_INVALID, "can not find external port"};
  }

  auto &ext_buffer_list = iter->second;
  if (!ext_buffer_list) {
    return {STATUS_INVALID, "external port must not be nullptr"};
  }

  ext_buffer_list->Assign(buffer_list);
  return STATUS_OK;
}

std::shared_ptr<BufferListMap> FlowUnitExecData::GetExternalData() {
  return ext_data_;
}

std::shared_ptr<BufferListMap> FlowUnitExecData::GetExternalDataForUser() {
  return ext_data_for_user_;
}

std::shared_ptr<BufferList> FlowUnitExecData::GetExternalDataForUser(
    const std::string &name) {
  return HasExternData(name) ? ext_data_for_user_->at(name) : nullptr;
}

size_t FlowUnitExecData::GetInBufferNum() {
  // All port data number is same
  if (in_data_->empty() || !(in_data_->begin()->second)) {
    return 0;
  }

  return in_data_->begin()->second->Size();
}

size_t FlowUnitExecData::GetExtBufferNum() {
  if (ext_data_->empty() || !(ext_data_->begin()->second)) {
    return 0;
  }

  return ext_data_->begin()->second->Size();
}

size_t FlowUnitExecData::GetOutBufferNum(bool accumulate_all_port) {
  // All port data number is same
  if (out_data_->empty() || !(out_data_->begin()->second)) {
    return 0;
  }

  if (!accumulate_all_port) {
    return out_data_->begin()->second->Size();
  }

  size_t sum = 0;
  for (auto &port_item : *out_data_) {
    sum += port_item.second->Size();
  }

  return sum;
}

Status FlowUnitExecData::GetStatus() const { return status_; }

void FlowUnitExecData::SetStatus(const Status &status) { status_ = status; }

bool FlowUnitExecData::HasInData(const std::string &name) const {
  return in_data_->find(name) != in_data_->end();
}

bool FlowUnitExecData::HasOutData(const std::string &name) const {
  return out_data_->find(name) != out_data_->end();
}

bool FlowUnitExecData::HasExternData(const std::string &name) const {
  return ext_data_->find(name) != ext_data_->end();
}

void FlowUnitExecData::SetupUserInput() {
  // freeze data and make copy to avoid user modify origin input
  in_data_for_user_ = std::make_shared<BufferListMap>();
  for (auto &in_item : *in_data_) {
    auto in_buffer_list_copy = std::make_shared<BufferList>(*in_item.second);
    in_buffer_list_copy->SetMutable(false);
    (*in_data_for_user_)[in_item.first] = in_buffer_list_copy;
  }

  ext_data_for_user_ = std::make_shared<BufferListMap>();
  for (auto &ext_item : *ext_data_) {
    auto ext_buffer_list_copy = std::make_shared<BufferList>(*ext_item.second);
    ext_buffer_list_copy->SetMutable(false);
    (*ext_data_for_user_)[ext_item.first] = ext_buffer_list_copy;
  }
}

Status FlowUnitExecData::CheckStatus(bool one_to_one, bool data_in_one_port) {
  if (status_ == STATUS_OK || status_ == STATUS_CONTINUE ||
      status_ == STATUS_SHUTDOWN || status_ == STATUS_STOP) {
    return STATUS_OK;
  }
  MBLOG_INFO << "flowunit " << fu_->GetFlowUnitDesc()->GetFlowUnitName()
             << " process return: " << status_;

  auto in_count = GetInBufferNum();
  if (in_count == 0) {
    in_count = GetExtBufferNum();
  }

  size_t out_count = in_count;
  if (!one_to_one || out_count == 0) {
    out_count = 1;
  }

  if (!out_data_->empty()) {
    FillErrorOutput(out_count, data_in_one_port);
    status_ = STATUS_OK;
  }

  return STATUS_OK;
}

void FlowUnitExecData::FillErrorOutput(size_t out_count,
                                       bool data_in_one_port) {
  bool first_port = true;
  for (auto &out_item : *out_data_) {
    auto &port_data_list = out_item.second;
    port_data_list->Reset();

    if (data_in_one_port && !first_port) {
      continue;
    }
    for (size_t i = 0; i < out_count; ++i) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(
          fu_->GetFlowUnitDesc()->GetFlowUnitName() + ".ProcessError",
          status_.Errormsg());
      port_data_list->PushBack(buffer);
    }
    first_port = false;
  }
}

Status FlowUnitExecData::SetupUserOutput(bool one_to_one,
                                         bool data_in_one_port) {
  if (status_ != STATUS_OK && status_ != STATUS_CONTINUE) {
    // process error, no need to save inherit info
    return STATUS_OK;
  }

  // avoid user push same buffer as multi output
  MakeCopyForUserOutput();

  auto in_count = GetInBufferNum();
  auto parent_data = in_data_;
  if (in_count == 0) {
    in_count = GetExtBufferNum();
    parent_data = ext_data_;
  }

  if (in_count == 0) {
    // event driven
    return STATUS_SUCCESS;
  }

  if (one_to_one) {
    return SaveProcessOneToOne(parent_data, in_count, data_in_one_port);
  }

  return SaveProcessNToM(parent_data);
}

void FlowUnitExecData::MakeCopyForUserOutput() {
  auto output = std::make_shared<BufferListMap>();
  for (auto &out_item : *out_data_) {
    auto output_buffer_list_copy =
        std::make_shared<BufferList>(*out_item.second);
    (*output)[out_item.first] = output_buffer_list_copy;
    for (auto &buffer : *output_buffer_list_copy) {
      buffer->ClearDelayedCopyDestinationInfo();
    }
  }

  out_data_ = output;
}

Status FlowUnitExecData::SaveProcessOneToOne(
    const std::shared_ptr<BufferListMap> &parent_data, size_t data_count,
    bool data_in_one_port) {
  // input n, output n, and inherit one to one
  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list;

  auto out_count = GetOutBufferNum(data_in_one_port);
  if (data_count != out_count) {
    return {STATUS_FAULT, "input buffer count " + std::to_string(data_count) +
                              " should equal output buffer count " +
                              std::to_string(out_count)};
  }

  process_info_list.reserve(data_count);
  for (size_t i = 0; i < data_count; ++i) {
    process_info_list.push_back(std::make_shared<BufferProcessInfo>());
  }

  for (auto &in_item : *parent_data) {
    const auto &port_name = in_item.first;
    auto &port_data_list = in_item.second;
    for (size_t i = 0; i < port_data_list->Size(); ++i) {
      auto process_info = process_info_list[i];
      auto buffer = port_data_list->At(i);
      auto buffer_index_info = BufferManageView::GetIndexInfo(buffer);
      process_info->SetParentBuffers(port_name, {buffer_index_info});
    }
  }

  for (auto &out_item : *out_data_) {
    auto &port_data_list = out_item.second;
    for (size_t i = 0; i < port_data_list->Size(); ++i) {
      auto process_info = process_info_list[i];
      auto buffer = port_data_list->At(i);
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      index_info->SetProcessInfo(process_info);
    }
  }

  return STATUS_OK;
}

Status FlowUnitExecData::SaveProcessNToM(
    const std::shared_ptr<BufferListMap> &parent_data) {
  // input n, output m
  auto process_info = std::make_shared<BufferProcessInfo>();
  for (auto &in_item : *parent_data) {
    const auto &port_name = in_item.first;
    auto &port_data_list = in_item.second;
    std::list<std::shared_ptr<BufferIndexInfo>> in_port_buffer_index_info_list;
    for (auto &buffer : *port_data_list) {
      auto buffer_index_info = BufferManageView::GetIndexInfo(buffer);
      in_port_buffer_index_info_list.push_back(buffer_index_info);
    }

    process_info->SetParentBuffers(port_name,
                                   std::move(in_port_buffer_index_info_list));
  }

  for (auto &out_item : *out_data_) {
    auto &port_data_list = out_item.second;
    for (auto &buffer : *port_data_list) {
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      index_info->SetProcessInfo(process_info);
    }
  }

  return STATUS_OK;
}

void FlowUnitExecDataMapper::AddExecCtx(
    const std::shared_ptr<FlowUnitExecContext> &exec_ctx) {
  origin_exec_ctx_list_.push_back(exec_ctx);
}

void FlowUnitExecDataMapper::LoadDataFromExecCtx() {
  auto ctx_count = origin_exec_ctx_list_.size();
  origin_data_list_.reserve(ctx_count);
  origin_shapes_.reserve(ctx_count);
  for (auto &exec_ctx : origin_exec_ctx_list_) {
    auto exec_data =
        std::make_shared<FlowUnitExecData>(exec_ctx->GetFlowUnit());
    const auto &inputs = exec_ctx->GetDataCtx()->GetInputs();
    for (const auto &item : inputs) {
      const auto &port_name = item.first;
      const auto &port_data_list = item.second;
      if (port_data_list.empty()) {
        continue;
      }

      exec_data->SetInData(port_name, port_data_list);
    }

    origin_data_list_.push_back(exec_data);
    origin_shapes_.push_back(exec_data->GetInBufferNum());
  }
}

Status FlowUnitExecDataMapper::MapData(bool need_reshape, size_t batch_size,
                                       bool is_stream) {
  if (!need_reshape || !NeedReshape(batch_size)) {
    map_type_ = DIRECT_MAP;
    return DirectMap();
  }

  if (batch_size == 0) {
    return {STATUS_FAULT, "batch_size should not be zero"};
  }

  if (is_stream) {
    map_type_ = RESHAPE_STREAM;
    return ReshapeStream(batch_size);
  }

  map_type_ = RESHAPE_NORMAL;
  return ReshapeNormal(batch_size);
}

Status FlowUnitExecDataMapper::MoveToTargetDevice(bool need_contiguous) {
  for (auto &batched_data : mapped_data_list_) {
    for (auto &data : batched_data) {
      auto in_data = data->GetInData();
      auto ret = MoveDataToTargetDevice(in_data, need_contiguous);
      if (!ret) {
        MBLOG_ERROR << "Move input data to target dev failed, err: " << ret;
        return ret;
      }

      auto ext_data = data->GetExternalData();
      ret = MoveDataToTargetDevice(ext_data, need_contiguous);
      if (!ret) {
        MBLOG_ERROR << "Move external data to target dev failed, err: " << ret;
        return ret;
      }
    }
  }

  return STATUS_SUCCESS;
}

Status FlowUnitExecDataMapper::MoveDataToTargetDevice(
    std::shared_ptr<BufferListMap> &data, bool need_contiguous) {
  for (auto &item : *data) {
    auto &buffer_list = item.second;
    if (need_contiguous && buffer_list->SupportMemContiguous()) {
      if (!buffer_list->MakeContiguous()) {
        return {STATUS_FAULT, "make contiguous failed, port:" + item.first};
      }
    } else {
      if (!buffer_list->MoveAllBufferToTargetDevice()) {
        return {STATUS_FAULT,
                "move buffer to target dev failed, port:" + item.first};
      }
    }
  }

  return STATUS_SUCCESS;
}

void FlowUnitExecDataMapper::SetupUserInput() {
  for (auto &data_batch : mapped_data_list_) {
    for (auto &data : data_batch) {
      data->SetupUserInput();
    }
  }
}

BatchedFUExecDataCtxList FlowUnitExecDataMapper::GetBatchedExecDataCtxList() {
  return mapped_exec_data_ctx_list_;
}

Status FlowUnitExecDataMapper::CheckOutputDataNumber(bool data_in_one_port) {
  for (auto &batched_data : mapped_data_list_) {
    for (auto &data : batched_data) {
      auto status = data->GetStatus();
      if (status != STATUS_OK && status != STATUS_CONTINUE) {
        // Flowunit process failed, skip this batch
        continue;
      }

      auto outputs = data->GetOutData();
      if (outputs == nullptr) {
        return {STATUS_FAULT, "output data is nullptr"};
      }

      if (outputs->empty()) {
        // Flowunit has no output port
        continue;
      }

      auto ret = CheckAllOutputNumEqual(data, data_in_one_port);
      if (!ret) {
        return ret;
      }

      ret = CheckOutputNumEqualInput(data, data_in_one_port);
      if (!ret) {
        return ret;
      }
    }
  }

  return true;
}

Status FlowUnitExecDataMapper::CheckAllOutputNumEqual(
    const std::shared_ptr<FlowUnitExecData> &data, bool data_in_one_port) {
  auto outputs = data->GetOutData();
  if (outputs->size() == 1) {
    return STATUS_OK;
  }

  size_t none_empty_port_num = 0;
  size_t port_data_num = 0;
  bool first_buffer = true;
  for (auto &port_item : *outputs) {
    auto &port_data_list = port_item.second;
    auto cur_port_data_num = port_data_list->Size();
    // For if else: only one port has data
    if (data_in_one_port) {
      none_empty_port_num += (cur_port_data_num == 0 ? 0 : 1);
      if (none_empty_port_num > 1) {
        return {STATUS_FAULT,
                "For condition flowunit, should only one port has data"};
      }

      continue;
    }

    // For other: all port has same output number
    if (first_buffer) {
      port_data_num = cur_port_data_num;
      first_buffer = false;
      continue;
    }

    if (port_data_num != cur_port_data_num) {
      return {STATUS_FAULT, "Output port " + port_item.first +
                                " data is not same with other port"};
    }
  }

  return STATUS_OK;
}

Status FlowUnitExecDataMapper::CheckOutputNumEqualInput(
    const std::shared_ptr<FlowUnitExecData> &data, bool data_in_one_port) {
  if (map_type_ != RESHAPE_NORMAL) {
    // Only reshape normal needs input == output
    return STATUS_OK;
  }

  auto in_num = data->GetInBufferNum();
  auto accumulate_all_port_data = data_in_one_port;
  auto out_num = data->GetOutBufferNum(accumulate_all_port_data);

  if (in_num != out_num) {
    return {STATUS_FAULT,
            "Output number must equals input number in normal flowunit"};
  }

  return STATUS_OK;
}

Status FlowUnitExecDataMapper::CheckStatus(bool one_to_one,
                                           bool data_in_one_port) {
  for (auto &mapped_ctx_data : mapped_data_list_) {
    for (auto &mapped_batch_data : mapped_ctx_data) {
      auto ret = mapped_batch_data->CheckStatus(one_to_one, data_in_one_port);
      if (!ret) {
        return ret;
      }
    }
  }

  return STATUS_OK;
}

Status FlowUnitExecDataMapper::SetupUserOutput(bool one_to_one,
                                               bool data_in_one_port) {
  for (auto &mapped_ctx_data : mapped_data_list_) {
    for (auto &mapped_batch_data : mapped_ctx_data) {
      auto ret =
          mapped_batch_data->SetupUserOutput(one_to_one, data_in_one_port);
      if (!ret) {
        return ret;
      }
    }
  }

  return STATUS_OK;
}

Status FlowUnitExecDataMapper::SaveDataToExecCtx() {
  /**
   * case DirectMap:
   * {origin_data(8)} ====> origin_data(8)
   * case Reshape Normal:
   * {mapped_data(5)}       origin_data(8)
   * {mapped_data(5)} ====> origin_data(5)
   * {mapped_data(3)}
   * case Reshape Stream: data number not same with origin input
   * {mapped_data, mapped_data} ===> origin_data
   **/
  auto ret = STATUS_OK;
  switch (map_type_) {
    case RESHAPE_STREAM:
      ret = WriteBackStream();
      break;

    case RESHAPE_NORMAL:
      ret = WriteBackNormal();
      break;

    case DIRECT_MAP:
    default:
      break;
  }

  if (!ret) {
    MBLOG_ERROR << "Write back data failed, err " << ret;
    return ret;
  }

  return FillExecCtxOutput();
}

void FlowUnitExecDataMapper::Clear() {
  // release processing data
  for (auto &batched_data_ctx : mapped_exec_data_ctx_list_) {
    for (auto &data_ctx : batched_data_ctx) {
      data_ctx->Clear();
    }
  }
}

Status FlowUnitExecDataMapper::WriteBackStream() {
  auto ctx_count = mapped_data_list_.size();
  for (size_t ctx_idx = 0; ctx_idx < ctx_count; ++ctx_idx) {
    auto &mapped_batch_data = mapped_data_list_[ctx_idx];
    auto &origin_data = origin_data_list_[ctx_idx];
    origin_data->SetStatus(STATUS_OK);
    auto output_size = std::accumulate(
        mapped_batch_data.begin(), mapped_batch_data.end(), size_t(0),
        [](size_t sum, const std::shared_ptr<FlowUnitExecData> &mapped_data) {
          return sum + mapped_data->GetOutBufferNum();
        });
    auto type = FlowUnitExecData::OUT_DATA;
    origin_data->ReserveCache(output_size, type);
    for (auto &mapped_data : mapped_batch_data) {
      if (origin_data->GetStatus() == STATUS_OK ||
          origin_data->GetStatus() == STATUS_CONTINUE) {
        origin_data->SetStatus(mapped_data->GetStatus());
      }

      origin_data->AppendToCache(mapped_data, 0, mapped_data->GetOutBufferNum(),
                                 type);
    }
    origin_data->FlushCache(type);
  }

  return STATUS_SUCCESS;
}

Status FlowUnitExecDataMapper::WriteBackNormal() {
  auto mapped_ctx_count = mapped_data_list_.size();
  size_t mapped_ctx_idx = 0;
  size_t buffer_idx_in_mapped_data = 0;
  size_t origin_ctx_idx = 0;
  size_t buffer_idx_in_origin_data = 0;
  const size_t buffer_count = 1;
  auto type = FlowUnitExecData::OUT_DATA;
  while (mapped_ctx_idx < mapped_ctx_count) {
    // only one data per batch for normal
    auto &mapped_data = mapped_data_list_[mapped_ctx_idx].front();
    auto mapped_shape = mapped_shapes_[mapped_ctx_idx].front();
    auto &origin_data = origin_data_list_[origin_ctx_idx];
    auto origin_shape = origin_shapes_[origin_ctx_idx];

    if (buffer_idx_in_mapped_data >= mapped_shape) {
      buffer_idx_in_mapped_data = 0;
      ++mapped_ctx_idx;
      if (mapped_ctx_idx >= mapped_ctx_count) {
        // The end buffer
        origin_data->FlushCache(type);
      }

      continue;
    }

    if (buffer_idx_in_origin_data == 0) {
      // Start to write a new ctx output
      origin_data->ReserveCache(origin_shape, type);
    }

    if (buffer_idx_in_origin_data >= origin_shape) {
      origin_data->FlushCache(type);
      buffer_idx_in_origin_data = 0;
      ++origin_ctx_idx;
      continue;
    }

    if (!mapped_data->GetStatus()) {
      origin_data->SetStatus(mapped_data->GetStatus());
    } else {
      origin_data->AppendToCache(mapped_data, buffer_idx_in_mapped_data,
                                 buffer_count, type);
    }

    ++buffer_idx_in_mapped_data;
    ++buffer_idx_in_origin_data;
  }

  return STATUS_SUCCESS;
}

Status FlowUnitExecDataMapper::FillExecCtxOutput() {
  size_t ctx_idx = 0;
  for (auto &exec_ctx : origin_exec_ctx_list_) {
    auto &data = origin_data_list_[ctx_idx];
    const auto &data_ctx = exec_ctx->GetDataCtx();
    data_ctx->SetStatus(data->GetStatus());
    data_ctx->SetOutput(*data->GetOutData());
    ++ctx_idx;
  }

  return STATUS_SUCCESS;
}

bool FlowUnitExecDataMapper::NeedReshape(size_t batch_size) {
  /**
   * if any one input is 0, then it cannot be reshaped
   * in case: external data, event data
   **/
  if (std::any_of(origin_data_list_.begin(), origin_data_list_.end(),
                  [](std::shared_ptr<FlowUnitExecData> &data) {
                    return data->GetInBufferNum() == 0;
                  })) {
    return false;
  }

  // Input buffer num might less than batch_size, but we still need use reshape
  // process to check output
  return true;
}

Status FlowUnitExecDataMapper::DirectMap() {
  /** Direct map
   * origin_data(8)          {origin_data(8)}
   * origin_data(7) =======> {origin_data(7)}
   * origin_data(3)          {origin_data(3)}
   **/
  auto data_ctx_count = origin_data_list_.size();
  mapped_data_list_.resize(data_ctx_count);
  mapped_exec_data_ctx_list_.resize(data_ctx_count);
  for (size_t ctx_idx = 0; ctx_idx < data_ctx_count; ++ctx_idx) {
    auto &origin_data = origin_data_list_[ctx_idx];
    auto &origin_ctx = origin_exec_ctx_list_[ctx_idx];
    mapped_data_list_[ctx_idx].push_back(origin_data);
    auto mapped_ctx = std::make_shared<ExecutorDataContext>(
        origin_ctx->GetDataCtx(), origin_data);
    mapped_exec_data_ctx_list_[ctx_idx].push_back(mapped_ctx);
  }

  return STATUS_SUCCESS;
}

Status FlowUnitExecDataMapper::ReshapeNormal(size_t batch_size) {
  /** Will mix diff data ctx
   * origin_data(8)  batch_size = 5   {mapped_data(5)}
   * origin_data(7) ================> {mapped_data(5)}
   * origin_data(3)                   {mapped_data(5)}
   *                                  {mapped_data(3)}
   * total_input_buffer = 18
   **/
  BuildMappedDataNormal(batch_size);
  FillMappedDataNormal(batch_size);
  return STATUS_SUCCESS;
}

void FlowUnitExecDataMapper::BuildMappedDataNormal(size_t batch_size) {
  size_t total_input_buffer = std::accumulate(
      origin_data_list_.begin(), origin_data_list_.end(), (size_t)0,
      [](size_t sum, std::shared_ptr<FlowUnitExecData> &data) {
        return sum + data->GetInBufferNum();
      });
  size_t new_exec_data_count =
      (total_input_buffer + batch_size - 1) / batch_size;
  mapped_data_list_.resize(new_exec_data_count);
  mapped_shapes_.resize(new_exec_data_count);
  mapped_exec_data_ctx_list_.resize(new_exec_data_count);
}

void FlowUnitExecDataMapper::FillMappedDataNormal(size_t batch_size) {
  size_t origin_data_count = origin_data_list_.size();
  size_t origin_index = 0;
  size_t index_in_origin_bufferlist = 0;
  size_t mapped_index = 0;
  size_t index_in_mapped_bufferlist = 0;
  const size_t buffer_count = 1;
  while (origin_index < origin_data_count) {
    auto &origin_data = origin_data_list_[origin_index];
    auto &origin_shape = origin_shapes_[origin_index];
    auto &mapped_batch_data = mapped_data_list_[mapped_index];
    auto &mapped_batch_shape = mapped_shapes_[mapped_index];
    if (mapped_batch_data.empty()) {
      auto &origin_exec_ctx = origin_exec_ctx_list_[origin_index];
      auto &mapped_batch_data_ctx = mapped_exec_data_ctx_list_[mapped_index];
      auto mapped_data =
          std::make_shared<FlowUnitExecData>(origin_exec_ctx->GetFlowUnit());
      auto mapped_data_ctx = std::make_shared<ExecutorDataContext>(
          origin_exec_ctx->GetDataCtx(), mapped_data);
      mapped_data->ReserveCache(batch_size);
      mapped_batch_data.push_back(mapped_data);
      mapped_batch_shape.push_back(0);
      mapped_batch_data_ctx.push_back(mapped_data_ctx);
    }

    auto &mapped_data = mapped_batch_data[0];
    mapped_data->AppendToCache(origin_data, index_in_origin_bufferlist,
                               buffer_count);
    ++index_in_origin_bufferlist;
    ++index_in_mapped_bufferlist;

    if (index_in_origin_bufferlist >= origin_shape) {
      // Read next data
      ++origin_index;
      index_in_origin_bufferlist = 0;
      if (origin_index >= origin_data_count) {
        // The end buffer
        mapped_batch_shape[0] = index_in_mapped_bufferlist;
        mapped_data->FlushCache();
        continue;
      }
    }

    if (index_in_mapped_bufferlist >= batch_size) {
      // Save last mapped data
      mapped_batch_shape[0] = batch_size;
      mapped_data->FlushCache();
      // Fill next mapped data
      ++mapped_index;
      index_in_mapped_bufferlist = 0;
    }
  }
}

Status FlowUnitExecDataMapper::ReshapeStream(size_t batch_size) {
  /** Will not mix diff data ctx
   * origin_data(8)  batch_size = 5   {mapped_data(5), mapped_data(3)}
   * origin_data(7) ================> {mapped_data(5), mapped_data(2)}
   * origin_data(3)                   {mapped_data(3)}
   * total_input_buffer = 18
   **/
  BuildMappedDataStream();
  FillMappedDataStream(batch_size);
  return STATUS_SUCCESS;
}

void FlowUnitExecDataMapper::BuildMappedDataStream() {
  size_t data_ctx_count = origin_data_list_.size();
  mapped_data_list_.resize(data_ctx_count);
  mapped_shapes_.resize(data_ctx_count);
  mapped_exec_data_ctx_list_.resize(data_ctx_count);
}

void FlowUnitExecDataMapper::FillMappedDataStream(size_t batch_size) {
  size_t data_ctx_count = origin_data_list_.size();
  for (size_t data_ctx_idx = 0; data_ctx_idx < data_ctx_count; ++data_ctx_idx) {
    auto &origin_data = origin_data_list_[data_ctx_idx];
    auto &origin_shape = origin_shapes_[data_ctx_idx];
    auto &origin_exec_ctx = origin_exec_ctx_list_[data_ctx_idx];
    auto &mapped_batch_data = mapped_data_list_[data_ctx_idx];
    auto &mapped_batch_shape = mapped_shapes_[data_ctx_idx];
    auto &mapped_batch_data_ctx = mapped_exec_data_ctx_list_[data_ctx_idx];
    auto batch_count = (origin_shape + batch_size - 1) / batch_size;
    mapped_batch_data.reserve(batch_count);
    mapped_batch_data_ctx.reserve(batch_count);
    for (size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
      auto mapped_data =
          std::make_shared<FlowUnitExecData>(origin_exec_ctx->GetFlowUnit());
      auto mapped_ctx = std::make_shared<ExecutorDataContext>(
          origin_exec_ctx->GetDataCtx(), mapped_data);
      mapped_batch_data.push_back(mapped_data);
      mapped_batch_data_ctx.push_back(mapped_ctx);
      auto buffer_idx_in_origin = batch_idx * batch_size;
      auto buffer_count =
          std::min(batch_size, origin_shape - buffer_idx_in_origin);
      mapped_batch_shape.push_back(buffer_count);
      mapped_data->ReserveCache(buffer_count);
      mapped_data->AppendToCache(origin_data, buffer_idx_in_origin,
                                 buffer_count);
      mapped_data->FlushCache();
    }
  }
}

FlowUnitExecDataView::FlowUnitExecDataView(FUExecContextList exec_ctx_list)
    : exec_ctx_list_(std::move(exec_ctx_list)) {}

FlowUnitExecDataView::~FlowUnitExecDataView() = default;

Status FlowUnitExecDataView::LoadInputFromExecCtx(bool need_reshape,
                                                  bool is_stream,
                                                  size_t batch_size,
                                                  bool need_contiguous) {
  auto ret = DevideExecCtxByFlowunit();
  if (!ret) {
    return ret;
  }

  LoadConfig cfg(need_reshape, is_stream, batch_size, need_contiguous);
  std::vector<std::shared_ptr<Executor>> executor_of_flownit;
  std::vector<std::function<Status()>> task_of_flowunit;
  ret = PackLoadTasks(cfg, executor_of_flownit, task_of_flowunit);
  if (!ret) {
    return ret;
  }

  std::vector<std::future<Status>> status_list;
  size_t fu_count = task_of_flowunit.size() - 1;
  status_list.reserve(fu_count);
  for (size_t fu_idx = 0; fu_idx < fu_count; ++fu_idx) {
    auto &fu_executor = executor_of_flownit[fu_idx];
    auto ret = fu_executor->Run(task_of_flowunit[fu_idx], 0);
    status_list.push_back(std::move(ret));
  }

  // Use current thread to process last one
  auto &last_prepare_task = task_of_flowunit.back();
  auto task_ret = last_prepare_task();
  if (!task_ret) {
    return task_ret;
  }

  // Wait async process result
  for (auto &status : status_list) {
    auto ret = status.get();
    if (!ret) {
      return ret;
    }
  }

  return STATUS_SUCCESS;
}

Status FlowUnitExecDataView::DataLoadTask(
    const LoadConfig &cfg, FlowUnit *flowunit,
    const std::shared_ptr<FlowUnitExecDataMapper> &exec_data_mapper) {
  exec_data_mapper->LoadDataFromExecCtx();
  auto ret = exec_data_mapper->MapData(cfg.need_reshape_, cfg.batch_size_,
                                       cfg.is_stream_);
  if (!ret) {
    return ret;
  }

  ret = exec_data_mapper->MoveToTargetDevice(cfg.need_contiguous_);
  if (!ret) {
    return ret;
  }

  exec_data_mapper->SetupUserInput();
  std::lock_guard<std::mutex> lock(data_of_flowunit_lock_);
  data_of_flowunit_[flowunit] = exec_data_mapper->GetBatchedExecDataCtxList();
  return STATUS_OK;
}

Status FlowUnitExecDataView::PackLoadTasks(
    const LoadConfig &cfg, std::vector<std::shared_ptr<Executor>> &executors,
    std::vector<std::function<Status()>> &tasks) {
  executors.reserve(mapper_of_flowunit_.size());
  tasks.reserve(mapper_of_flowunit_.size());
  for (auto &item : mapper_of_flowunit_) {
    const auto &flowunit = item.first;
    auto device = flowunit->GetBindDevice();
    if (device == nullptr) {
      MBLOG_ERROR << "Get bind device failed";
      return STATUS_FAULT;
    }

    auto executor = device->GetDeviceExecutor();
    if (executor == nullptr) {
      MBLOG_ERROR << "Get device executor failed";
      return STATUS_FAULT;
    }

    auto exec_data_mapper = item.second;
    executors.push_back(executor);
    tasks.emplace_back(std::bind(&FlowUnitExecDataView::DataLoadTask, this, cfg,
                                 flowunit, exec_data_mapper));
  }

  return STATUS_OK;
}

const std::vector<FlowUnit *> &FlowUnitExecDataView::GetFlowUnits() {
  return flowunit_list_;
}

const BatchedFUExecDataCtxList &FlowUnitExecDataView::GetFlowUnitProcessData(
    FlowUnit *flowunit) {
  return data_of_flowunit_[flowunit];
}

Status FlowUnitExecDataView::CheckOutputDataNumber(bool data_in_one_port) {
  for (auto &mapper_item : mapper_of_flowunit_) {
    auto &mapper = mapper_item.second;
    auto ret = mapper->CheckOutputDataNumber(data_in_one_port);
    if (!ret) {
      return ret;
    }
  }

  return STATUS_OK;
}

Status FlowUnitExecDataView::CheckStatus(bool one_to_one,
                                         bool data_in_one_port) {
  for (auto &mapper_item : mapper_of_flowunit_) {
    auto &mapper = mapper_item.second;
    auto ret = mapper->CheckStatus(one_to_one, data_in_one_port);
    if (!ret) {
      return ret;
    }
  }
  return STATUS_OK;
}

Status FlowUnitExecDataView::SetupUserOutput(bool one_to_one,
                                             bool data_in_one_port) {
  for (auto &mapper_item : mapper_of_flowunit_) {
    auto &mapper = mapper_item.second;
    auto ret = mapper->SetupUserOutput(one_to_one, data_in_one_port);
    if (!ret) {
      return ret;
    }
  }
  return STATUS_OK;
}

Status FlowUnitExecDataView::SaveOutputToExecCtx() {
  std::vector<std::shared_ptr<Executor>> executor_of_flowunit;
  std::vector<std::function<Status()>> task_of_flowunit;
  auto ret = PackSaveTasks(executor_of_flowunit, task_of_flowunit);
  if (!ret) {
    return ret;
  }

  size_t fu_count = task_of_flowunit.size() - 1;
  std::vector<std::future<Status>> status_list;
  status_list.reserve(fu_count);
  for (size_t fu_idx = 0; fu_idx < fu_count; ++fu_idx) {
    auto &flowunit_executor = executor_of_flowunit[fu_idx];
    auto ret = flowunit_executor->Run(task_of_flowunit[fu_idx], 0);
    status_list.push_back(std::move(ret));
  }

  auto &last_save_task = task_of_flowunit.back();
  auto task_ret = last_save_task();
  if (!task_ret) {
    return ret;
  }

  for (auto &status : status_list) {
    auto ret = status.get();
    if (!ret) {
      return ret;
    }
  }

  return STATUS_SUCCESS;
}

void FlowUnitExecDataView::Clear() {
  // release processing data
  for (auto &mapper : mapper_of_flowunit_) {
    mapper.second->Clear();
  }
}

Status FlowUnitExecDataView::PackSaveTasks(
    std::vector<std::shared_ptr<Executor>> &executors,
    std::vector<std::function<Status()>> &tasks) {
  executors.reserve(mapper_of_flowunit_.size());
  tasks.reserve(mapper_of_flowunit_.size());
  for (auto &mapper_item : mapper_of_flowunit_) {
    auto *flowunit = mapper_item.first;
    auto device = flowunit->GetBindDevice();
    if (device == nullptr) {
      MBLOG_ERROR << "Get bind device failed";
      return STATUS_FAULT;
    }

    auto executor = device->GetDeviceExecutor();
    if (executor == nullptr) {
      MBLOG_ERROR << "Get device executor failed";
      return STATUS_FAULT;
    }

    auto &mapper = mapper_item.second;
    executors.push_back(executor);
    tasks.emplace_back(
        std::bind(&FlowUnitExecDataMapper::SaveDataToExecCtx, mapper.get()));
  }

  return STATUS_OK;
}

Status FlowUnitExecDataView::DevideExecCtxByFlowunit() {
  for (auto &exec_ctx : exec_ctx_list_) {
    const auto &flowunit = exec_ctx->GetFlowUnit();
    auto item = mapper_of_flowunit_.find(flowunit.get());
    std::shared_ptr<FlowUnitExecDataMapper> mapper;
    if (item != mapper_of_flowunit_.end()) {
      mapper = item->second;
    } else {
      mapper = std::make_shared<FlowUnitExecDataMapper>();
      mapper_of_flowunit_[flowunit.get()] = mapper;
      flowunit_list_.push_back(flowunit.get());
    }

    mapper->AddExecCtx(exec_ctx);
  }

  return STATUS_SUCCESS;
}

FlowUnitDataExecutor::FlowUnitDataExecutor(std::weak_ptr<Node> node_ref,
                                           size_t batch_size)
    : node_ref_(std::move(node_ref)), batch_size_(batch_size) {}

FlowUnitDataExecutor::~FlowUnitDataExecutor() = default;

Status FlowUnitDataExecutor::DataCtxExecuteFunc(
    FlowUnit *flowunit, const BatchedFUExecDataCtxList &process_data,
    size_t data_ctx_idx) {
  const auto &batched_fu_data_ctx = process_data[data_ctx_idx];
  for (const auto &data_ctx : batched_fu_data_ctx) {
    Status status = STATUS_FAULT;
    try {
      status = flowunit->Process(data_ctx);
    } catch (const std::exception &e) {
      std::string msg("Exception caught in flowunit process");
      msg += ", detail:";
      msg += e.what();
      status = {STATUS_SHUTDOWN, msg};
    }

    data_ctx->SetStatus(status);
    /** Only STOP and SHUTDOWN will be transparent
     * STOP means to stop scheduling, SHUTDOWN means that a fatal error
     * has occurred and needs to exit
     **/
    if (status == STATUS_STOP || status == STATUS_SHUTDOWN) {
      return status;
    }
  }

  return STATUS_SUCCESS;
}

void FlowUnitDataExecutor::SetNeedCheckOutput(bool need_check) {
  need_check_output_ = need_check;
}

Status FlowUnitDataExecutor::Process(const FUExecContextList &exec_ctx_list) {
  /**
   * for event type data ctx list, all inputs is 0. (videodemuxer event input)
   * for data type data ctx list, inputs is ports, port buffer list size will
   * not be 0
   **/
  FlowUnitExecDataView exec_view(exec_ctx_list);
  auto node = node_ref_.lock();
  if (node == nullptr) {
    return {STATUS_FAULT, "Node has been released"};
  }

  auto node_name = node->GetName();
  auto ret = LoadExecuteInput(node, exec_view);
  if (!ret) {
    MBLOG_ERROR << "node: " << node_name << ", load execute input failed, err "
                << ret;
    return ret;
  }

  ret = Execute(exec_view);
  if (!ret) {
    MBLOG_ERROR << "node: " << node_name << ", execute failed, err " << ret;
    return ret;
  }

  ret = SaveExecuteOutput(node, exec_view);
  if (!ret) {
    MBLOG_ERROR << "node: " << node_name << ", save execute output failed, err "
                << ret;
    return ret;
  }

  return STATUS_OK;
}

Status FlowUnitDataExecutor::LoadExecuteInput(const std::shared_ptr<Node> &node,
                                              FlowUnitExecDataView &exec_view) {
  bool need_reshape = false;

  if (node->GetOutputType() == ORIGIN) {
    need_reshape = true;
  }

  if (node->GetConditionType() == IF_ELSE) {
    if (batch_size_ != 1) {
      MBLOG_WARN
          << "Batch size not available for condition flowunit, auto set to 1";
      batch_size_ = 1;
    }

    need_reshape = true;
  }

  auto is_stream = (node->GetFlowType() == STREAM);
  auto need_contiguous = node->IsInputContiguous();

  auto ret = exec_view.LoadInputFromExecCtx(need_reshape, is_stream,
                                            batch_size_, need_contiguous);
  if (!ret) {
    MBLOG_ERROR << "Prepare exec view by batch size failed, " << ret;
    return ret;
  }

  return STATUS_OK;
}

Status FlowUnitDataExecutor::Execute(FlowUnitExecDataView &exec_view) {
  const int32_t priority = 0;
  std::list<std::future<Status>> status_list;
  // each flowunit has a device executor which manages thread pool
  const auto &flowunits = exec_view.GetFlowUnits();
  for (const auto &flowunit : flowunits) {
    const auto &process_data = exec_view.GetFlowUnitProcessData(flowunit);
    auto data_ctx_count = process_data.size();
    auto exec_func = std::bind(&FlowUnitDataExecutor::DataCtxExecuteFunc, this,
                               flowunit, process_data, std::placeholders::_1);
    auto resource_nice = flowunit->GetFlowUnitDesc()->IsResourceNice();
    auto exec_device = flowunit->GetBindDevice();
    auto future_status_list = exec_device->DeviceExecuteAsync(
        exec_func, priority, data_ctx_count, resource_nice);
    status_list.splice(status_list.begin(), future_status_list);
  }

  auto status_count = status_list.size();
  std::vector<Status> exec_results(status_count, STATUS_OK);
  size_t result_idx = 0;
  for (auto &fu_status : status_list) {
    exec_results[result_idx] = fu_status.get();
    ++result_idx;
  }

  for (const auto &result : exec_results) {
    if (!result) {
      return result;
    }
  }

  return STATUS_OK;
}

Status FlowUnitDataExecutor::SaveExecuteOutput(
    const std::shared_ptr<Node> &node, FlowUnitExecDataView &exec_view) {
  /**
   * input num must equals output num in normal flowunit
   * condition flowunit only one port has data
   * all port data num should be same
   *
   * as usual, only need check output in develop mode
   * user could close check at running mode
   **/
  auto data_in_one_port = (node->GetConditionType() != ConditionType::NONE ||
                           node->GetLoopType() == LOOP);
  auto node_has_output = node->GetOutputNum() != 0;

  if (need_check_output_ && node_has_output) {
    auto ret = exec_view.CheckOutputDataNumber(data_in_one_port);
    if (!ret) {
      MBLOG_ERROR << "check output failed, err " << ret;
      return ret;
    }
  }

  bool one_to_one =
      node->GetFlowType() == NORMAL && node->GetOutputType() == ORIGIN;
  auto ret = exec_view.CheckStatus(one_to_one, data_in_one_port);
  if (!ret) {
    MBLOG_ERROR << "check data context status failed, err " << ret;
    return STATUS_FAULT;
  }

  if (node_has_output) {
    ret = exec_view.SetupUserOutput(one_to_one, data_in_one_port);
    if (!ret) {
      MBLOG_ERROR << "save buffer inherit info failed, err " << ret;
      return STATUS_FAULT;
    }
  }

  ret = exec_view.SaveOutputToExecCtx();
  if (!ret) {
    MBLOG_ERROR << "setup failed, err " << ret;
    return ret;
  }

  // release processing data
  exec_view.Clear();

  return STATUS_OK;
}

}  // namespace modelbox