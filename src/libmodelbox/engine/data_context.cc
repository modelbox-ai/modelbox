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

#include <modelbox/data_context.h>
#include <modelbox/match_stream.h>
#include <modelbox/node.h>
#include <modelbox/port.h>
#include <modelbox/session.h>
#include <modelbox/session_context.h>

namespace modelbox {

ExternalDataImpl::ExternalDataImpl(std::shared_ptr<InPort> port,
                                   std::shared_ptr<Device> device,
                                   std::shared_ptr<Stream> init_stream)
    : root_buffer_(std::make_shared<BufferIndexInfo>()),
      ext_port_(port),
      device_(device),
      input_stream_(init_stream),
      session_(init_stream->GetSession()),
      session_ctx_(init_stream->GetSession()->GetSessionCtx()) {}

ExternalDataImpl::~ExternalDataImpl() { Close(); }

std::shared_ptr<BufferList> ExternalDataImpl::CreateBufferList() {
  if (!device_) {
    MBLOG_ERROR << "device is null";
    return nullptr;
  }

  return std::make_shared<BufferList>(device_);
}

Status ExternalDataImpl::Send(std::shared_ptr<BufferList> buffer_list) {
  if (!buffer_list) {
    return {STATUS_INVALID, "input buffer list is null"};
  }

  if (!ext_port_) {
    return {STATUS_INVALID, "external port is null"};
  }

  for (auto &buffer : *buffer_list) {
    auto index_info = BufferManageView::GetIndexInfo(buffer);
    index_info->SetStream(input_stream_);
    index_info->SetIndex(input_stream_->GetBufferCount());
    input_stream_->IncreaseBufferCount();
    auto inherit_info = std::make_shared<BufferInheritInfo>();
    inherit_info->SetType(BufferProcessType::EXPAND);
    inherit_info->SetInheritFrom(root_buffer_);
    index_info->SetInheritInfo(inherit_info);
    ext_port_->Send(buffer);
  }

  ext_port_->NotifyPushEvent();
  return STATUS_OK;
}

std::shared_ptr<SessionContext> ExternalDataImpl::GetSessionContext() {
  return session_ctx_.lock();
}

Status ExternalDataImpl::SetOutputMeta(std::shared_ptr<DataMeta> meta) {
  input_stream_->SetStreamMeta(meta);
  return STATUS_OK;
}

/**
 * @brief close input stream, wait process
 **/
Status ExternalDataImpl::Close() {
  if (input_stream_ == nullptr) {
    return STATUS_OK;
  }

  auto buffer = std::make_shared<Buffer>();
  auto index_info = BufferManageView::GetIndexInfo(buffer);
  index_info->SetStream(input_stream_);
  index_info->SetIndex(input_stream_->GetBufferCount());
  index_info->MarkAsEndFlag();
  input_stream_->IncreaseBufferCount();
  auto inherit_info = std::make_shared<BufferInheritInfo>();
  inherit_info->SetType(BufferProcessType::EXPAND);
  inherit_info->SetInheritFrom(root_buffer_);
  index_info->SetInheritInfo(inherit_info);
  ext_port_->Send(buffer);
  ext_port_->NotifyPushEvent();
  input_stream_ = nullptr;
  root_buffer_ = nullptr;
  return STATUS_OK;
}

/**
 * @brief stop task immediately
 **/
Status ExternalDataImpl::Shutdown() {
  auto session = session_.lock();
  if (session == nullptr) {
    return STATUS_OK;
  }

  session->Close();
  Close();  // make sure data end has been sent
  return STATUS_OK;
}

std::shared_ptr<Configuration> ExternalDataImpl::GetSessionConfig() {
  auto ctx = session_ctx_.lock();
  if (ctx == nullptr) {
    return nullptr;
  }

  return ctx->GetConfig();
}

bool FlowUnitDataContext::HasError() {
  if (error_ == nullptr) {
    return false;
  }

  return true;
}

FlowUnitDataContext::~FlowUnitDataContext() {
  for (auto &callback : destroy_callback_list_) {
    callback();
  }
}

FlowUnitDataContext::FlowUnitDataContext(Node *node,
                                         MatchKey *data_ctx_match_key,
                                         std::shared_ptr<Session> session) {
  user_event_ = nullptr;
  node_ = node;
  data_ctx_match_key_ = data_ctx_match_key;
  session_ = session;
  if (session != nullptr) {
    session_context_ = session->GetSessionCtx();
  }
  is_exception_visible_ = node->IsExceptionVisible();
  InitStatistic();
}

void FlowUnitDataContext::WriteInputData(
    std::shared_ptr<PortDataMap> stream_data_map) {
  SetCurrentInputData(stream_data_map);
}

void FlowUnitDataContext::SetCurrentInputData(
    std::shared_ptr<PortDataMap> stream_data_map) {
  cur_input_ = stream_data_map;
  for (auto &port_data_item : *cur_input_) {
    auto &port_name = port_data_item.first;
    auto &data_list = port_data_item.second;
    for (auto &buffer : data_list) {
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      if (index_info->IsPlaceholder()) {
        cur_input_placeholder_[port_name].push_back(buffer);
        continue;
      }

      if (index_info->IsEndFlag()) {
        end_flag_received_ = true;
        if (index_info->GetIndex() == 0) {  // empty stream
          is_empty_stream = true;
        }
        input_stream_max_buffer_count_ = index_info->GetIndex() + 1;
        cur_input_end_flag_[port_name].push_back(buffer);
        continue;
      }

      auto error = buffer->GetError();
      if (buffer->GetError() != nullptr) {
        error_ = error;  // error will record during this process
      }

      cur_input_valid_data_[port_name].push_back(buffer);
    }
  }

  bool has_no_data = cur_input_valid_data_.empty() ||
                     cur_input_valid_data_.begin()->second.empty();
  SetSkippable(has_no_data);
  UpdateInputInfo();
}

std::shared_ptr<BufferList> FlowUnitDataContext::Input(
    const std::string &port) const {
  auto port_iter = cur_input_valid_data_.find(port);
  if (port_iter == cur_input_valid_data_.end()) {
    return nullptr;
  }
  auto buffer_list = std::make_shared<BufferList>(port_iter->second);
  return buffer_list;
}

std::shared_ptr<BufferList> FlowUnitDataContext::Output(
    const std::string &port) {
  auto item = cur_output_valid_data_.find(port);
  if (item == cur_output_valid_data_.end()) {
    return nullptr;
  }
  return item->second;
}

std::shared_ptr<BufferListMap> FlowUnitDataContext::Input() const {
  return nullptr;
}

std::shared_ptr<BufferListMap> FlowUnitDataContext::Output() {
  return std::shared_ptr<BufferListMap>(&(cur_output_valid_data_),
                                        [](BufferListMap *buffer_list_map) {});
};

std::shared_ptr<BufferList> FlowUnitDataContext::External() { return nullptr; }

void FlowUnitDataContext::SetEvent(std::shared_ptr<FlowUnitEvent> event) {
  wait_user_events_.erase(event);
  user_event_ = event;
}

std::shared_ptr<FlowUnitEvent> FlowUnitDataContext::Event() {
  return user_event_;
}

std::shared_ptr<FlowUnitError> FlowUnitDataContext::GetError() {
  return error_;
}

void FlowUnitDataContext::SetPrivate(const std::string &key,
                                     std::shared_ptr<void> private_content) {
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    private_map_.emplace(key, private_content);
  } else {
    private_map_[key] = private_content;
  }
}

std::shared_ptr<void> FlowUnitDataContext::GetPrivate(const std::string &key) {
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    return nullptr;
  }
  return private_map_[key];
}

void FlowUnitDataContext::SendEvent(std::shared_ptr<FlowUnitEvent> event) {
  if (session_ != nullptr && session_->IsClosed()) {
    // stop event driven
    return;
  }

  wait_user_events_.insert(event);
  auto inner_event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  inner_event->SetUserEvent(event);
  inner_event->SetDataCtxMatchKey(data_ctx_match_key_);
  if (node_ == nullptr) {
    return;
  }
  node_->SendEvent(inner_event);

  // save last unfinished input
  if (!cur_input_valid_data_.empty()) {
    // driven by data
    cur_event_input_data_.clear();
  }  // else driven by event, use last valid input
  for (auto &in_port_data : cur_input_valid_data_) {
    auto &port_name = in_port_data.first;
    auto &port_data_list = in_port_data.second;
    cur_event_input_data_[port_name].push_back(port_data_list.back());
  }
}

const std::shared_ptr<DataMeta> FlowUnitDataContext::GetInputMeta(
    const std::string &port) {
  if (input_port_meta_.find(port) == input_port_meta_.end()) {
    return nullptr;
  }
  return input_port_meta_.find(port)->second;
}

const std::shared_ptr<DataMeta> FlowUnitDataContext::GetInputGroupMeta(
    const std::string &port) {
  return GetInputMeta(port);
}

void FlowUnitDataContext::SetOutputMeta(const std::string &port,
                                        std::shared_ptr<DataMeta> data_meta) {
  if (output_port_meta_.find(port) != output_port_meta_.end()) {
    return;
  }
  output_port_meta_.emplace(port, data_meta);
}

std::shared_ptr<SessionContext> FlowUnitDataContext::GetSessionContext() {
  auto session_context = session_context_.lock();
  return session_context;
}

std::shared_ptr<Configuration> FlowUnitDataContext::GetSessionConfig() {
  auto session_context = session_context_.lock();
  if (session_context == nullptr) {
    ConfigurationBuilder config_builder;
    auto config = config_builder.Build();
    return config;
  }

  auto config = session_context->GetConfig();
  auto node_prop_name = CONFIG_NODE + node_->GetName();
  auto flowunit_prop_name =
      CONFIG_FLOWUNIT + node_->GetFlowUnitDesc()->GetFlowUnitName();
  auto all_node_prop_name = CONFIG_NODES;

  auto all_node_config = config->GetSubConfig(all_node_prop_name);
  auto flowunit_config = config->GetSubConfig(flowunit_prop_name);
  auto node_config = config->GetSubConfig(node_prop_name);

  flowunit_config->Add(*(node_config.get()));
  all_node_config->Add(*(flowunit_config.get()));

  return all_node_config;
}

std::shared_ptr<StatisticsItem> FlowUnitDataContext::GetStatistics(
    DataContextStatsType type) {
  switch (type) {
    case DataContextStatsType::NODE:
      return node_stats_;

    case DataContextStatsType::SESSION:
      return session_stats_;

    case DataContextStatsType::GRAPH:
      return graph_stats_;

    default:
      return nullptr;
  }
}

const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
    &FlowUnitDataContext::GetInputs() const {
  return cur_input_valid_data_;
}

const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
    &FlowUnitDataContext::GetExternals() const {
  return cur_input_valid_data_;
}

const std::unordered_map<std::string, std::shared_ptr<BufferList>>
    &FlowUnitDataContext::GetOutputs() const {
  return cur_output_valid_data_;
}

void FlowUnitDataContext::SetOutput(
    const std::unordered_map<std::string, std::shared_ptr<BufferList>>
        &data_list) {
  cur_output_valid_data_ = data_list;
}

void FlowUnitDataContext::SetStatus(Status status) {
  process_status_ = status;
  last_process_status_ = status;
}

Status FlowUnitDataContext::GetStatus() { return process_status_; }

Status FlowUnitDataContext::GetLastStatus() { return last_process_status_; }

bool FlowUnitDataContext::IsErrorStatus() {
  auto code = process_status_.Code();
  if (code != STATUS_SUCCESS && code != STATUS_CONTINUE &&
      code != STATUS_STOP && code != STATUS_SHUTDOWN) {
    return true;
  }
  return false;
}

void FlowUnitDataContext::UpdateProcessState() {}

void FlowUnitDataContext::ClearData() {
  cur_input_ = nullptr;
  cur_input_valid_data_.clear();
  cur_input_placeholder_.clear();
  cur_input_end_flag_.clear();

  cur_output_valid_data_.clear();
  cur_output_placeholder_.clear();
  cur_output_.clear();

  user_event_ = nullptr;
  error_ = nullptr;

  input_has_stream_start_ = false;
  input_has_stream_end_ = false;

  is_empty_stream = false;
  is_skippable_ = false;
}

bool FlowUnitDataContext::IsSkippable() { return is_skippable_; }
void FlowUnitDataContext::SetSkippable(bool skippable) {
  is_skippable_ = skippable;
}

// after flowunit process
Status FlowUnitDataContext::PostProcess() {
  auto ret = GenerateOutputPlaceholder();
  if (!ret) {
    return ret;
  }

  ret = GenerateOutput();
  if (!ret) {
    return ret;
  }

  ret = AppendEndFlag();
  if (!ret) {
    return ret;
  }

  ret = UpdateOutputIndexInfo();
  if (!ret) {
    return ret;
  }

  ret = CheckOutputData();
  if (!ret) {
    return STATUS_STOP;  // fatal error
  }

  return STATUS_OK;
}

Status FlowUnitDataContext::GenerateOutputPlaceholder() {
  FillPlaceholderOutput();
  return STATUS_OK;
}

Status FlowUnitDataContext::AppendEndFlag() {
  if (!NeedStreamEndFlag()) {
    return STATUS_OK;
  }

  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list;
  auto end_flag_parent = &cur_input_end_flag_;
  if (end_flag_parent->empty()) {
    // need append a new end flag. inherit input directly. in case expand
    end_flag_parent = &cur_input_valid_data_;
  }
  if (end_flag_parent->empty()) {
    // event driven expand
    end_flag_parent = &cur_event_input_data_;
  }
  if (end_flag_parent->empty()) {
    // expand empty buffer
    end_flag_parent = &cur_input_placeholder_;
  }
  if (end_flag_parent->empty()) {
    // no available input buffer
    return STATUS_OK;
  }
  BufferManageView::GenProcessInfo<std::vector<std::shared_ptr<Buffer>>>(
      *end_flag_parent, 1,
      [](const std::vector<std::shared_ptr<Buffer>> &container, size_t idx) {
        return container[idx];
      },
      process_info_list);
  for (auto &port_name : node_->GetOutputNames()) {
    auto &port_data_list = cur_output_[port_name];
    auto buffer = std::make_shared<Buffer>();
    auto index_info = BufferManageView::GetIndexInfo(buffer);
    index_info->MarkAsEndFlag();
    index_info->SetProcessInfo(process_info_list.front());
    port_data_list.push_back(buffer);
  }

  return STATUS_OK;
}

void FlowUnitDataContext::FillPlaceholderOutput(bool from_valid_input,
                                                bool same_with_input_num) {
  PortDataMap *cur_parent = &cur_input_placeholder_;
  if (from_valid_input) {
    // in case user has no output, we generate new empty
    cur_parent = &cur_input_valid_data_;
  }

  if (cur_parent->empty()) {
    return;
  }

  auto first_input_port = cur_parent->begin()->second;
  if (first_input_port.empty()) {
    return;
  }
  auto input_num = first_input_port.size();

  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list;
  BufferManageView::GenProcessInfo<std::vector<std::shared_ptr<Buffer>>>(
      *cur_parent, input_num,
      [](const std::vector<std::shared_ptr<Buffer>> &container, size_t idx) {
        return container[idx];
      },
      process_info_list, !same_with_input_num);
  // generate empty output
  bool is_condition = node_->GetConditionType() != ConditionType::NONE;
  bool first_port = true;
  cur_output_placeholder_.clear();
  size_t output_num = 1;
  if (same_with_input_num) {
    output_num = input_num;
  }

  for (auto &port_name : node_->GetOutputNames()) {
    auto &port_data_list = cur_output_placeholder_[port_name];
    if (is_condition && !first_port) {
      port_data_list.resize(output_num, nullptr);
      continue;
    }

    port_data_list.reserve(output_num);
    for (size_t i = 0; i < output_num; ++i) {
      auto buffer = std::make_shared<Buffer>();
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      index_info->SetProcessInfo(process_info_list[i]);
      index_info->MarkAsPlaceholder();
      port_data_list.push_back(buffer);
    }

    first_port = false;
  }
}

void FlowUnitDataContext::FillErrorOutput(std::shared_ptr<FlowUnitError> error,
                                          bool same_with_input_num) {
  if (cur_input_valid_data_.empty()) {
    // input for this process is empty, error might be last process
    return;
  }

  bool is_condition = node_->GetConditionType() != ConditionType::NONE;
  bool first_port = true;
  cur_output_valid_data_.clear();
  auto input_num = cur_input_valid_data_.begin()->second.size();
  size_t output_num = 1;
  if (same_with_input_num) {
    output_num = input_num;
  }

  std::vector<std::shared_ptr<BufferProcessInfo>> process_info_list;
  BufferManageView::GenProcessInfo<std::vector<std::shared_ptr<Buffer>>>(
      cur_input_valid_data_, input_num,
      [](const std::vector<std::shared_ptr<Buffer>> &container, size_t idx) {
        return container[idx];
      },
      process_info_list, !same_with_input_num);

  for (auto &port_name : node_->GetOutputNames()) {
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    if (is_condition & !first_port) {
      buffer_vector.resize(output_num, nullptr);
      auto buffer_list = std::make_shared<BufferList>(buffer_vector);
      cur_output_valid_data_[port_name] = buffer_list;
      continue;
    }

    buffer_vector.reserve(output_num);
    for (size_t i = 0; i < output_num; ++i) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto index_info = BufferManageView::GetIndexInfo(buffer);
      index_info->SetProcessInfo(process_info_list[i]);
      buffer_vector.push_back(buffer);
    }
    first_port = false;
    auto buffer_list = std::make_shared<BufferList>(buffer_vector);
    cur_output_valid_data_[port_name] = buffer_list;
  }
}

bool FlowUnitDataContext::HasValidOutput() {
  if (cur_output_valid_data_.empty()) {
    return false;
  }

  auto &first_output_port = cur_output_valid_data_.begin()->second;
  return first_output_port->Size() != 0;
}

bool FlowUnitDataContext::IsContinueProcess() {
  return process_status_ == STATUS_CONTINUE && !wait_user_events_.empty();
}

size_t FlowUnitDataContext::GetOutputBufferNum() {
  if (cur_output_.empty()) {
    return 0;
  }

  auto &first_port = cur_output_.begin()->second;
  return first_port.size();
}

Status FlowUnitDataContext::GenerateOutput() {
  for (auto &port_name : node_->GetOutputNames()) {
    auto &valid_data_list = cur_output_valid_data_[port_name];
    if (valid_data_list == nullptr) {
      // no output
      valid_data_list = std::make_shared<BufferList>();
    }
    auto &placeholder_data_list = cur_output_placeholder_[port_name];
    auto &port_data_list = cur_output_[port_name];
    port_data_list.resize(valid_data_list->Size() +
                          placeholder_data_list.size());
    // merge buffer by input index
    std::merge(
        valid_data_list->begin(), valid_data_list->end(),
        placeholder_data_list.begin(), placeholder_data_list.end(),
        port_data_list.begin(),
        [](std::shared_ptr<Buffer> b1, std::shared_ptr<Buffer> b2) {
          if (b1 == nullptr) {
            // condition output, will be removed in node stream manage
            return true;
          }
          auto parent_index_info1 = BufferManageView::GetFirstParentBuffer(b1);
          auto parent_index_info2 = BufferManageView::GetFirstParentBuffer(b2);
          return parent_index_info1->GetIndex() <
                 parent_index_info2->GetIndex();
        });
  }
  return STATUS_OK;
}

Status FlowUnitDataContext::UpdateOutputIndexInfo() {
  for (auto &output_item : cur_output_) {
    auto &output_data_list = output_item.second;
    for (auto &buffer : output_data_list) {
      if (buffer == nullptr) {
        // condition output, will be removed in node stream manage
        continue;
      }
      auto cur_buffer_index_info = BufferManageView::GetIndexInfo(buffer);
      auto cur_node_process_info = GetCurNodeProcessInfo(cur_buffer_index_info);
      auto first_input_port = cur_node_process_info->GetParentBuffers().begin();
      auto first_buffer_info_in_port = first_input_port->second.front();
      UpdateBufferIndexInfo(cur_buffer_index_info, first_buffer_info_in_port);
    }
  }
  return STATUS_OK;
}

std::shared_ptr<BufferProcessInfo> FlowUnitDataContext::GetCurNodeProcessInfo(
    std::shared_ptr<BufferIndexInfo> index_info) {
  auto cur_node_process_info = index_info->GetProcessInfo();
  if (cur_node_process_info != nullptr) {
    return cur_node_process_info;
  }

  // event driven
  cur_node_process_info = std::make_shared<BufferProcessInfo>();
  for (auto &in_port_data_item : cur_event_input_data_) {
    auto &in_port_name = in_port_data_item.first;
    auto &in_port_data_list = in_port_data_item.second;
    std::list<std::shared_ptr<BufferIndexInfo>> index_info_list;
    for (auto &in_buffer : in_port_data_list) {
      auto in_buffer_index_info = BufferManageView::GetIndexInfo(in_buffer);
      index_info_list.push_back(in_buffer_index_info);
    }
    cur_node_process_info->SetParentBuffers(in_port_name,
                                            std::move(index_info_list));
  }

  index_info->SetProcessInfo(cur_node_process_info);
  return cur_node_process_info;
}

std::shared_ptr<Session> FlowUnitDataContext::GetSession() { return session_; }

void FlowUnitDataContext::DealWithDataError() {
  DealWithProcessError(GetError());
}

void FlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void FlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  FillErrorOutput(error);
  process_status_ = STATUS_SUCCESS;
}

bool FlowUnitDataContext::IsFinished() { return is_finished_; }

void FlowUnitDataContext::InitStatistic() {
  if (node_ == nullptr) {
    MBLOG_DEBUG << "Node is null, init statistics ctx failed";
    return;
  }
  auto session_context = session_context_.lock();
  if (session_context == nullptr) {
    MBLOG_DEBUG
        << "session_context is null, init statistics ctx failed, node : "
        << node_->GetName();
    return;
  }

  graph_stats_ = session_context->GetStatistics(SessionContexStatsType::GRAPH);
  session_stats_ = session_context->GetStatistics();
  if (session_stats_ == nullptr) {
    MBLOG_DEBUG << "Get session statistics ctx failed, node : "
                << node_->GetName();
    return;
  }

  node_stats_ = session_stats_->AddItem(node_->GetName());
  if (node_stats_ == nullptr) {
    MBLOG_WARN << "Get statistics ctx failed for " << node_->GetName();
  }
}

void FlowUnitDataContext::AddDestroyCallback(
    const std::function<void()> &func) {
  destroy_callback_list_.push_back(func);
}

bool FlowUnitDataContext::IsDataErrorVisible() { return is_exception_visible_; }

Status FlowUnitDataContext::PopOutputData(PortDataMap &stream_data_map) {
  cur_output_.swap(stream_data_map);
  return STATUS_OK;
}

std::unordered_map<std::string, std::shared_ptr<DataMeta>>
FlowUnitDataContext::GetOutputPortStreamMeta() {
  return output_port_meta_;
}

void FlowUnitDataContext::UpdateInputInfo() {
  for (auto &input_item : *cur_input_) {
    auto &input_port_name = input_item.first;
    auto &input_port_data_list = input_item.second;
    if (input_port_data_list.empty()) {
      continue;
    }

    auto input_stream =
        BufferManageView::GetIndexInfo(input_port_data_list.front())
            ->GetStream();
    if (input_stream == nullptr) {
      continue;
    }

    input_port_meta_[input_port_name] = input_stream->GetStreamMeta();
  }

  auto first_port_data_list = cur_input_->begin()->second;
  if (first_port_data_list.empty()) {
    return;
  }

  input_stream_cur_buffer_count_ += first_port_data_list.size();

  auto first_buffer = first_port_data_list.front();
  auto buffer_index_info = BufferManageView::GetIndexInfo(first_buffer);
  input_has_stream_start_ = buffer_index_info->IsFirstBufferInStream();

  auto last_buffer = first_port_data_list.back();
  buffer_index_info = BufferManageView::GetIndexInfo(last_buffer);
  input_has_stream_end_ = buffer_index_info->IsEndFlag();
}

ExecutorDataContext::ExecutorDataContext(
    std::shared_ptr<FlowUnitDataContext> origin_ctx,
    std::shared_ptr<FlowUnitExecData> data)
    : origin_ctx_(origin_ctx), data_(data){};

std::shared_ptr<BufferList> ExecutorDataContext::Input(
    const std::string &port) const {
  return data_->GetInData(port);
}

std::shared_ptr<BufferList> ExecutorDataContext::Output(
    const std::string &port) {
  return data_->GetOutData(port);
}

std::shared_ptr<BufferListMap> ExecutorDataContext::Input() const {
  return data_->GetInData();
}

std::shared_ptr<BufferListMap> ExecutorDataContext::Output() {
  return data_->GetOutData();
}

std::shared_ptr<BufferList> ExecutorDataContext::External() {
  return data_->GetExternalData(EXTERNAL_PORT_NAME);
}

bool ExecutorDataContext::HasError() { return origin_ctx_->HasError(); };

std::shared_ptr<FlowUnitError> ExecutorDataContext::GetError() {
  return origin_ctx_->GetError();
};

std::shared_ptr<FlowUnitEvent> ExecutorDataContext::Event() {
  return origin_ctx_->Event();
}

void ExecutorDataContext::SendEvent(std::shared_ptr<FlowUnitEvent> event) {
  origin_ctx_->SendEvent(event);
}

void ExecutorDataContext::SetPrivate(const std::string &key,
                                     std::shared_ptr<void> private_content) {
  origin_ctx_->SetPrivate(key, private_content);
};

std::shared_ptr<void> ExecutorDataContext::GetPrivate(const std::string &key) {
  return origin_ctx_->GetPrivate(key);
};

const std::shared_ptr<DataMeta> ExecutorDataContext::GetInputMeta(
    const std::string &port) {
  return origin_ctx_->GetInputMeta(port);
};

const std::shared_ptr<DataMeta> ExecutorDataContext::GetInputGroupMeta(
    const std::string &port) {
  return origin_ctx_->GetInputGroupMeta(port);
};

void ExecutorDataContext::SetOutputMeta(const std::string &port,
                                        std::shared_ptr<DataMeta> data_meta) {
  origin_ctx_->SetOutputMeta(port, data_meta);
};

std::shared_ptr<SessionContext> ExecutorDataContext::GetSessionContext() {
  return origin_ctx_->GetSessionContext();
};

void ExecutorDataContext::SetStatus(Status status) { data_->SetStatus(status); }

std::shared_ptr<Configuration> ExecutorDataContext::GetSessionConfig() {
  return origin_ctx_->GetSessionConfig();
}

std::shared_ptr<StatisticsItem> ExecutorDataContext::GetStatistics(
    DataContextStatsType type) {
  return origin_ctx_->GetStatistics(type);
}

NormalFlowUnitDataContext::NormalFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

void NormalFlowUnitDataContext::UpdateProcessState() {
  // input process over, normal data ctx is over
  if (input_stream_max_buffer_count_ == 0) {
    is_finished_ = false;
    return;
  }

  is_finished_ =
      input_stream_cur_buffer_count_ == input_stream_max_buffer_count_;
}

bool NormalFlowUnitDataContext::NeedStreamEndFlag() {
  return input_has_stream_end_;
}

void NormalFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  if (node_->GetConditionType() == ConditionType::IF_ELSE) {
    cur_buffer->GetProcessInfo()->SetType(BufferProcessType::CONDITION_START);
    auto inherit_info = std::make_shared<BufferInheritInfo>();
    inherit_info->SetType(BufferProcessType::CONDITION_START);
    inherit_info->SetInheritFrom(parent_buffer);
    cur_buffer->SetInheritInfo(inherit_info);
    return;
  }

  cur_buffer->SetIndex(parent_buffer->GetIndex());
  cur_buffer->SetInheritInfo(parent_buffer->GetInheritInfo());
}

LoopNormalFlowUnitDataContext::LoopNormalFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : NormalFlowUnitDataContext(node, data_ctx_match_key, session) {}

Status LoopNormalFlowUnitDataContext::GenerateOutput() {
  // need know output port for this loop
  if (HasValidOutput()) {
    // has user output
    for (auto &port_data_item : cur_output_valid_data_) {
      auto &port_name = port_data_item.first;
      auto &port_data_list = port_data_item.second;
      if (port_data_list->Front() != nullptr) {
        output_port_for_this_loop_ = port_name;
        break;
      }
    }
  }

  for (auto &port_data_item : cur_output_placeholder_) {
    auto &port_name = port_data_item.first;
    auto &port_data_list = port_data_item.second;
    auto &cached_port_data_list = cached_output_placeholder_[port_name];
    cached_port_data_list.insert(cached_port_data_list.end(),
                                 port_data_list.begin(), port_data_list.end());
  }
  if (!cur_input_end_flag_.empty()) {
    cached_input_end_flag_.swap(cur_input_end_flag_);
  }

  if (output_port_for_this_loop_.empty() &&
      input_stream_cur_buffer_count_ < input_stream_max_buffer_count_) {
    // not decide this loop yet and input stream not over,
    // then not process cache
    return STATUS_OK;
  }

  if (output_port_for_this_loop_.empty() &&
      input_stream_cur_buffer_count_ == input_stream_max_buffer_count_) {
    // input stream end, but no user output, then just go through this loop
    output_port_for_this_loop_ = node_->GetLoopOutPortName();
  }

  cur_output_placeholder_.swap(cur_output_placeholder_);
  NormalFlowUnitDataContext::GenerateOutput();
  return STATUS_OK;
}

Status LoopNormalFlowUnitDataContext::AppendEndFlag() {
  if (output_port_for_this_loop_.empty()) {
    // not decide this loop
    return STATUS_OK;
  }

  if (cached_input_end_flag_.empty()) {
    // not end
    return STATUS_OK;
  }

  cur_input_end_flag_.swap(cached_input_end_flag_);
  input_has_stream_end_ = true;
  NormalFlowUnitDataContext::AppendEndFlag();
  return STATUS_OK;
}

Status LoopNormalFlowUnitDataContext::CheckOutputData() {
  if (output_port_for_this_loop_.empty()) {
    return STATUS_OK;
  }

  for (auto iter = cur_output_.begin(); iter != cur_output_.end();) {
    auto &port_name = iter->first;
    if (port_name != output_port_for_this_loop_) {
      iter = cur_output_.erase(iter);
    } else {
      ++iter;
    }
  }

  return STATUS_OK;
}

StreamFlowUnitDataContext::StreamFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

bool StreamFlowUnitDataContext::IsDataPre() {
  return input_has_stream_start_ && !is_empty_stream;
}

bool StreamFlowUnitDataContext::IsDataPost() {
  return input_has_stream_end_ && !is_empty_stream;
}

void StreamFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void StreamFlowUnitDataContext::UpdateProcessState() {
  is_finished_ = end_flag_received_ && !IsContinueProcess();
}

bool StreamFlowUnitDataContext::NeedStreamEndFlag() {
  return end_flag_received_ && !IsContinueProcess();
}

void StreamFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  cur_buffer->SetInheritInfo(parent_buffer->GetInheritInfo());
}

NormalExpandFlowUnitDataContext::NormalExpandFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

void NormalExpandFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void NormalExpandFlowUnitDataContext::UpdateProcessState() {
  // each buffer in stream has one data ctx, finish after buffer expand end
  is_finished_ = !IsContinueProcess();
}

bool NormalExpandFlowUnitDataContext::NeedStreamEndFlag() {
  return !IsContinueProcess();
}

void NormalExpandFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  cur_buffer->GetProcessInfo()->SetType(BufferProcessType::EXPAND);

  auto inherit_info = std::make_shared<BufferInheritInfo>();
  inherit_info->SetType(BufferProcessType::EXPAND);
  inherit_info->SetInheritFrom(parent_buffer);
  cur_buffer->SetInheritInfo(inherit_info);
}

StreamExpandFlowUnitDataContext::StreamExpandFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

void StreamExpandFlowUnitDataContext::WriteInputData(
    std::shared_ptr<PortDataMap> stream_data_map) {
  stream_data_cache_.push_back(stream_data_map);
  ExpandNextBuffer();
}

void StreamExpandFlowUnitDataContext::ExpandNextBuffer() {
  if (stream_data_cache_.empty()) {
    // no data to process
    SetSkippable(true);
    return;
  }

  auto front_cache = stream_data_cache_.front();
  auto &first_port = front_cache->begin()->second;
  if (first_port.size() <= cur_data_pose_in_first_cache_) {
    // remove processed data
    stream_data_cache_.pop_front();
    cur_data_pose_in_first_cache_ = 0;
    if (stream_data_cache_.empty()) {
      // No data to process
      SetSkippable(true);
      return;
    }

    front_cache = stream_data_cache_.front();
  }

  auto cur_input_data = std::make_shared<PortDataMap>();
  for (auto &port_data : *front_cache) {
    auto &port_name = port_data.first;
    auto &data_list = port_data.second;
    (*cur_input_data)[port_name].push_back(
        data_list[cur_data_pose_in_first_cache_]);
  }

  // test cur input is next buffer to process
  auto &first_input = cur_input_data->begin()->second.front();
  auto first_input_index = BufferManageView::GetIndexInfo(first_input);
  if (first_input_index->GetIndex() != cur_expand_buffer_index_) {
    // not the index to process next
    SetSkippable(true);
    return;
  }

  cur_expand_buffer_index_received_ = true;
  ++cur_data_pose_in_first_cache_;
  SetCurrentInputData(cur_input_data);
}

bool StreamExpandFlowUnitDataContext::IsDataPre() {
  return input_has_stream_start_ && !is_empty_stream;
}

bool StreamExpandFlowUnitDataContext::IsDataPost() {
  return input_has_stream_end_ && !is_empty_stream;
}

std::shared_ptr<FlowUnitInnerEvent>
StreamExpandFlowUnitDataContext::GenerateSendEvent() {
  if (IsContinueProcess()) {
    // user event driven
    return nullptr;
  }

  if (end_flag_received_) {
    // all data processed
    return nullptr;
  }

  if (stream_data_cache_.empty()) {
    // no cache data
    return nullptr;
  }

  auto expand_event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_NEXT_STREAM);
  expand_event->SetDataCtxMatchKey(data_ctx_match_key_);
  return expand_event;
}

void StreamExpandFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void StreamExpandFlowUnitDataContext::UpdateProcessState() {
  is_finished_ = end_flag_received_ && !IsContinueProcess();
  if (!IsContinueProcess() && cur_expand_buffer_index_received_) {
    ++cur_expand_buffer_index_;
    cur_expand_buffer_index_received_ = false;
  }
}

bool StreamExpandFlowUnitDataContext::NeedStreamEndFlag() {
  return !IsContinueProcess();
}

void StreamExpandFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  cur_buffer->GetProcessInfo()->SetType(BufferProcessType::EXPAND);

  auto inherit_info = std::make_shared<BufferInheritInfo>();
  inherit_info->SetType(BufferProcessType::EXPAND);
  inherit_info->SetInheritFrom(parent_buffer);
  cur_buffer->SetInheritInfo(inherit_info);
}

NormalCollapseFlowUnitDataContext::NormalCollapseFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

bool NormalCollapseFlowUnitDataContext::IsDataPre() {
  return input_has_stream_start_ && !is_empty_stream;
}

bool NormalCollapseFlowUnitDataContext::IsDataPost() {
  return input_has_stream_end_ && !is_empty_stream;
}

void NormalCollapseFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  if (output_buffer_for_current_stream_ == 0 && input_has_stream_end_) {
    // no valid data generated, will generate an error buffer
    FillErrorOutput(error, false);
  }
  process_status_ = STATUS_SUCCESS;
};

void NormalCollapseFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error){};

void NormalCollapseFlowUnitDataContext::UpdateProcessState() {
  is_finished_ = end_flag_received_;
};

bool NormalCollapseFlowUnitDataContext::NeedStreamEndFlag() {
  if (!input_has_stream_end_) {
    return false;
  }

  // check stream end flag before expand
  auto &first_port = cur_input_end_flag_.begin()->second;
  auto &input_end_buffer = first_port.front();
  auto input_end_buffer_index =
      BufferManageView::GetIndexInfo(input_end_buffer);
  auto expand_from = input_end_buffer_index->GetInheritInfo()->GetInheritFrom();
  return expand_from->IsEndFlag();
}

Status NormalCollapseFlowUnitDataContext::CheckOutputData() {
  output_buffer_for_current_stream_ += GetOutputBufferNum();
  if (output_buffer_for_current_stream_ >
      1) {  // collapse for one stream should only generate one buffer
    MBLOG_ERROR << "node " << node_->GetName() << " output buffer is "
                << output_buffer_for_current_stream_
                << ", should generate one buffer for one stream collapse";
    return STATUS_INVALID;
  }

  if (output_buffer_for_current_stream_ == 0 && input_has_stream_end_) {
    // collapse over, but has no data, we need generate empty from valid input
    FillPlaceholderOutput(true, false);
  }

  return STATUS_OK;
}

Status NormalCollapseFlowUnitDataContext::GenerateOutputPlaceholder() {
  if (!(end_flag_received_ && !HasValidOutput())) {
    return STATUS_OK;
  }
  // receive end and no buffer generated, try generate empty from input
  // placehold
  FillPlaceholderOutput(false, false);
  return STATUS_OK;
}

void NormalCollapseFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  cur_buffer->GetProcessInfo()->SetType(BufferProcessType::COLLAPSE);

  auto expand_from_buffer = parent_buffer->GetInheritInfo()->GetInheritFrom();
  cur_buffer->SetInheritInfo(expand_from_buffer->GetInheritInfo());
  cur_buffer->SetIndex(expand_from_buffer->GetIndex());
}

StreamCollapseFlowUnitDataContext::StreamCollapseFlowUnitDataContext(
    Node *node, MatchKey *data_ctx_match_key, std::shared_ptr<Session> session)
    : FlowUnitDataContext(node, data_ctx_match_key, session) {}

void StreamCollapseFlowUnitDataContext::WriteInputData(
    std::shared_ptr<PortDataMap> stream_data_map) {
  AppendToCache(stream_data_map);
  CollapseNextStream();
}

void StreamCollapseFlowUnitDataContext::AppendToCache(
    std::shared_ptr<PortDataMap> stream_data_map) {
  auto first_buffer = stream_data_map->begin()->second.front();
  auto buffer_index = BufferManageView::GetIndexInfo(first_buffer);
  auto expand_from = buffer_index->GetInheritInfo()->GetInheritFrom();
  auto index_before_expand = expand_from->GetIndex();
  // find cache
  auto cache_item = stream_data_cache_.find(index_before_expand);
  if (cache_item == stream_data_cache_.end()) {
    stream_data_cache_[index_before_expand] = stream_data_map;
    return;
  }

  auto &cache_stream_data = cache_item->second;
  for (auto &port_item : *stream_data_map) {
    auto &port_name = port_item.first;
    auto &port_new_data = port_item.second;
    auto &old_data = (*cache_stream_data)[port_name];
    old_data.insert(old_data.end(), port_new_data.begin(), port_new_data.end());
  }
}

void StreamCollapseFlowUnitDataContext::UpdateInputInfo() {
  FlowUnitDataContext::UpdateInputInfo();
  auto &first_input_port = cur_input_->begin()->second;
  auto &first_buffer_in_port = first_input_port.front();
  auto index_info_first_buffer_in_port =
      BufferManageView::GetIndexInfo(first_buffer_in_port);
  auto expand_buffer_index_info =
      index_info_first_buffer_in_port->GetInheritInfo()->GetInheritFrom();
  input_is_expand_from_end_buffer_ = expand_buffer_index_info->IsEndFlag();
}

void StreamCollapseFlowUnitDataContext::CollapseNextStream() {
  auto next_stream_item = stream_data_cache_.find(current_collapse_order_);
  if (next_stream_item == stream_data_cache_.end()) {
    // in single node run, multi stream data expand from same one will cache at
    // same data context
    // this stream data not the next, but cur_input_ might be ready
    SetSkippable(cur_input_ == nullptr);
    return;
  }

  SetCurrentInputData(next_stream_item->second);
  stream_data_cache_.erase(current_collapse_order_);
}

bool StreamCollapseFlowUnitDataContext::IsDataPre() {
  return input_has_stream_start_ && !is_empty_stream;
}

bool StreamCollapseFlowUnitDataContext::IsDataPost() {
  return input_has_stream_end_ && !is_empty_stream;
}

std::shared_ptr<FlowUnitInnerEvent>
StreamCollapseFlowUnitDataContext::GenerateSendEvent() {
  if (!input_is_expand_from_end_buffer_ && input_has_stream_end_) {
    auto event = std::make_shared<FlowUnitInnerEvent>(
        FlowUnitInnerEvent::COLLAPSE_NEXT_STREAM);
    event->SetDataCtxMatchKey(data_ctx_match_key_);
    return event;
  }

  return nullptr;
}

void StreamCollapseFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void StreamCollapseFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  if (output_buffer_for_current_stream_ == 0 && input_has_stream_end_) {
    // no valid data generated before error happen, and stream will end at this
    // process, we need generate an error buffer
    FillErrorOutput(error, false);
  }
  process_status_ = STATUS_SUCCESS;
}

void StreamCollapseFlowUnitDataContext::UpdateProcessState() {
  if (!input_has_stream_end_) {
    return;
  }

  // last stream collapse over, process next stream, reset stream state
  ++current_collapse_order_;
  is_empty_stream = false;
  end_flag_received_ = false;
  input_stream_cur_buffer_count_ = 0;
  input_stream_max_buffer_count_ = 0;
  output_buffer_for_current_stream_ = 0;

  // test ctx finish
  if (input_is_expand_from_end_buffer_) {
    is_finished_ = true;  // this is last packet to collapse
  }
}

bool StreamCollapseFlowUnitDataContext::NeedStreamEndFlag() {
  if (!end_flag_received_) {
    return false;
  }

  // check stream end flag before expand
  auto &first_port = cur_input_end_flag_.begin()->second;
  auto &input_end_buffer = first_port.front();
  auto input_end_buffer_index =
      BufferManageView::GetIndexInfo(input_end_buffer);
  auto expand_from = input_end_buffer_index->GetInheritInfo()->GetInheritFrom();
  return expand_from->IsEndFlag();
}

Status StreamCollapseFlowUnitDataContext::CheckOutputData() {
  output_buffer_for_current_stream_ += GetOutputBufferNum();
  if (output_buffer_for_current_stream_ >
      1) {  // collapse for one stream should only generate one buffer
    MBLOG_ERROR << "node " << node_->GetName() << " output buffer is "
                << output_buffer_for_current_stream_
                << ", should generate one buffer for one stream collapse";
    return STATUS_INVALID;
  }

  if (output_buffer_for_current_stream_ == 0 && input_has_stream_end_) {
    // collapse over, but has no data, we need generate empty from valid input
    FillPlaceholderOutput(true, false);
  }

  return STATUS_OK;
}

Status StreamCollapseFlowUnitDataContext::GenerateOutputPlaceholder() {
  if (!(end_flag_received_ && !HasValidOutput())) {
    return STATUS_OK;
  }
  // receive end and no buffer generated, try generate empty from input
  // placehold
  FillPlaceholderOutput(false, false);
  return STATUS_OK;
}

void StreamCollapseFlowUnitDataContext::UpdateBufferIndexInfo(
    std::shared_ptr<BufferIndexInfo> cur_buffer,
    std::shared_ptr<BufferIndexInfo> parent_buffer) {
  cur_buffer->GetProcessInfo()->SetType(BufferProcessType::COLLAPSE);

  auto expand_from_buffer = parent_buffer->GetInheritInfo()->GetInheritFrom();
  cur_buffer->SetInheritInfo(expand_from_buffer->GetInheritInfo());
  cur_buffer->SetIndex(expand_from_buffer->GetIndex());
}

}  // namespace modelbox
