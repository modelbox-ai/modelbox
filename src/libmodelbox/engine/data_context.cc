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
#include <modelbox/node.h>
#include <modelbox/session_context.h>

namespace modelbox {

VirtualStream::VirtualStream(std::shared_ptr<BufferGroup> bg, int priority)
    : priority_(priority) {
  if (bg == nullptr) {
    auto root = std::make_shared<BufferGroup>();
    stream_bg_ = root->AddSubGroup(true, true);
    port_id_ = 0;
    return;
  }
  port_id_ = bg->GetPortId() + 1;
  stream_bg_ = bg->GetGroup();
}

VirtualStream::~VirtualStream() { Close(); }

void VirtualStream::LabelIndexBuffer(
    std::shared_ptr<IndexBufferList> index_buffer_list) {
  for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
    auto new_group_ptr = stream_bg_->AddNewSubGroup(port_id_);
    auto new_node_ptr = new_group_ptr->GenerateSameLevelGroup();
    auto other_buffer = index_buffer_list->GetBuffer(i);
    other_buffer->SetBufferGroup(new_node_ptr);
    other_buffer->SetPriority(priority_);
  }
}

bool VirtualStream::IsClosed() { return stream_bg_->IsFullGroup(port_id_); }

std::shared_ptr<BufferGroup> VirtualStream::GetBufferGroup() {
  return stream_bg_;
}

void VirtualStream::SetSessionContext(
    std::shared_ptr<SessionContext> session_cxt) {
  if (stream_bg_ == nullptr) {
    return;
  }
  stream_bg_->SetSessionContex(session_cxt);
}

std::shared_ptr<SessionContext> VirtualStream::GetSessionContext() {
  if (stream_bg_ == nullptr) {
    return nullptr;
  }
  return stream_bg_->GetSessionContext();
}

void VirtualStream::Close() { stream_bg_->FinishSubGroup(port_id_); }

std::shared_ptr<BufferGroup> VirtualStream::GetLastBufferGroup() {
  return stream_bg_->AddOneMoreSubGroup(port_id_);
}

bool FlowUnitDataContext::HasError() {
  if (error_ == nullptr) {
    return false;
  }

  return true;
}

ExternalDataImpl::ExternalDataImpl(std::shared_ptr<InPort> port,
                                   std::shared_ptr<Device> device,
                                   std::shared_ptr<StatisticsItem> graph_stats)
    : device_(device), ext_port_(port) {
  virtual_stream_ = std::make_shared<VirtualStream>(nullptr, 0);
  auto session_context = std::make_shared<SessionContext>(graph_stats);
  session_context_ = session_context;
  virtual_stream_->SetSessionContext(session_context);
}

std::shared_ptr<BufferList> ExternalDataImpl::CreateBufferList() {
  if (!device_) {
    MBLOG_WARN << "device_ must not be nullptr";
    return nullptr;
  }

  return std::make_shared<BufferList>(device_);
}

Status ExternalDataImpl::SetOutputMeta(std::shared_ptr<DataMeta> meta) {
  if (input_meta_ != nullptr) {
    return STATUS_NOTFOUND;
  }
  input_meta_ = meta;
  return STATUS_OK;
}

Status ExternalDataImpl::Send(std::shared_ptr<BufferList> buffer_list) {
  if (!buffer_list) {
    return {STATUS_INVALID, "buffer_list must not be nullptr."};
  }

  if (!ext_port_) {
    return {STATUS_INVALID, "external port must not be nullptr."};
  }

  Status status;
  std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;
  for (auto &buffer : *buffer_list) {
    auto index_buffer = std::make_shared<IndexBuffer>(buffer);
    buffer_vector.push_back(index_buffer);
  }

  auto index_buffer_list = std::make_shared<IndexBufferList>(buffer_vector);
  virtual_stream_->LabelIndexBuffer(index_buffer_list);
  if (index_buffer_list != nullptr && input_meta_ != nullptr) {
    index_buffer_list->SetDataMeta(input_meta_);
  }

  for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
    ext_port_->Send(index_buffer_list->GetBuffer(i));
  }
  ext_port_->NotifyPushEvent();
  return STATUS_OK;
};

std::shared_ptr<SessionContext> ExternalDataImpl::GetSessionContext() {
  return session_context_.lock();
};

Status ExternalDataImpl::Close() {
  virtual_stream_->Close();
  return STATUS_OK;
};

Status ExternalDataImpl::Shutdown() {
  auto last_bg = virtual_stream_->GetLastBufferGroup();
  auto buffer = std::make_shared<Buffer>();
  auto error = std::make_shared<FlowUnitError>("EOF");
  buffer->SetError(error);
  auto index_buffer = std::make_shared<IndexBuffer>(buffer);
  index_buffer->SetBufferGroup(last_bg);
  ext_port_->Send(index_buffer);
  return STATUS_OK;
};

std::shared_ptr<Configuration> ExternalDataImpl::GetSessionConfig() {
  auto session_context = session_context_.lock();
  if (session_context == nullptr) {
    return nullptr;
  }
  return session_context->GetConfig();
}

FlowUnitDataContext::~FlowUnitDataContext() {
  for (auto &callback : destroy_callback_list_) {
    callback();
  }
}

FlowUnitDataContext::FlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg,
                                         Node *node) {
  user_event_ = nullptr;
  priority_ = node->GetPriority();
  node_ = node;
  stream_index_ = stream_bg;
  session_context_ = stream_index_->GetSessionContext();
  is_exception_visible_ = node->IsExceptionVisible();
  is_finished_ = false;
  output_stream_error_ = nullptr;
  start_flag_ = false;
  end_flag_ = false;
  closed_ = false;
  data_post_flag_ = false;

  is_skippable_ = false;

  is_input_meta_available_ = false;

  is_input_group_meta_available_ = false;

  is_output_meta_available_ = false;

  InitStatistic();
}

void FlowUnitDataContext::UpdateInputInfo(
    std::shared_ptr<IndexBufferList> buffer_list) {
  if ((buffer_list == nullptr) || (buffer_list->GetBufferNum() == 0)) {
    return;
  }

  if (buffer_list->GetBuffer(0)->GetSameLevelGroup() == nullptr) {
    return;
  }

  for (uint32_t i = 0; i < buffer_list->GetBufferNum(); i++) {
    auto buffer = buffer_list->GetBuffer(i);
    if (buffer->GetSameLevelGroup()->IsStartGroup()) {
      start_flag_ = true;
    }
    if (buffer->GetSameLevelGroup()->IsEndGroup()) {
      end_flag_ = true;
    }
  }
}

void FlowUnitDataContext::UpdateInputDataMeta(
    std::string key, std::shared_ptr<IndexBufferList> buffer_list) {
  if (buffer_list == nullptr) {
    return;
  }

  if (input_port_meta_.find(key) == input_port_meta_.end()) {
    input_port_meta_.emplace(key, buffer_list->GetDataMeta());
  } else {
    input_port_meta_[key] = buffer_list->GetDataMeta();
  }
}

std::shared_ptr<FlowUnitError> FlowUnitDataContext::GetInputError() {
  for (auto &error_index : input_error_index_) {
    auto error = stream_index_->GetDataError(error_index.second);
    if (error != nullptr) {
      return error;
    }
  }
  return nullptr;
}

void FlowUnitDataContext::UpdateErrorIndex(
    std::shared_ptr<InputData> input_data) {
  if (input_data == nullptr) {
    return;
  }
  input_error_index_.clear();

  auto input_iter = input_data->begin();
  while (input_iter != input_data->end()) {
    auto key = input_iter->first;
    auto index_buffer_list = input_iter->second;
    auto error_index = index_buffer_list->GetDataErrorIndex();
    input_error_index_.emplace(key, error_index);
    input_iter++;
  }
}

void FlowUnitDataContext::UpdateOutputDataMeta() {
  if (output_data_ == nullptr) {
    return;
  }

  auto output_iter = output_port_meta_.begin();
  while (output_iter != output_port_meta_.end()) {
    auto key = output_iter->first;
    auto data_meta = output_iter->second;
    auto index_buffer_list = output_data_->GetBufferList(key);
    if (index_buffer_list != nullptr) {
      index_buffer_list->SetDataMeta(data_meta);
    }
    output_iter++;
  }
}

bool FlowUnitDataContext::IsErrorStatus() {
  auto code = process_status_.Code();
  if (code != STATUS_SUCCESS && code != STATUS_CONTINUE &&
      code != STATUS_STOP && code != STATUS_SHUTDOWN) {
    return true;
  }
  return false;
}

void FlowUnitDataContext::SetError(const std::shared_ptr<FlowUnitError> error) {
  error_ = error;
}

std::shared_ptr<BufferList> FlowUnitDataContext::Input(
    const std::string &port) const {
  auto port_iter = input_.find(port);
  if (port_iter == input_.end()) {
    return nullptr;
  }
  auto buffer_list = std::make_shared<BufferList>(port_iter->second);
  return buffer_list;
}

std::shared_ptr<BufferList> FlowUnitDataContext::Output(
    const std::string &port) {
  if (output_.find(port) == output_.end()) {
    return nullptr;
  }
  return output_.find(port)->second;
}

std::shared_ptr<BufferListMap> FlowUnitDataContext::Input() const {
  return nullptr;
}

std::shared_ptr<BufferListMap> FlowUnitDataContext::Output() {
  return std::shared_ptr<BufferListMap>(&(output_),
                                        [](BufferListMap *buffer_list_map) {});
};

std::shared_ptr<BufferList> FlowUnitDataContext::External() { return nullptr; }

void FlowUnitDataContext::SetEvent(std::shared_ptr<FlowUnitEvent> event) {
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
  auto inner_event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  inner_event->SetUserEvent(event);
  inner_event->SetBufferGroup(stream_index_);
  if (node_ == nullptr) {
    return;
  }
  node_->SendEvent(inner_event);
}

const std::shared_ptr<DataMeta> FlowUnitDataContext::GetInputMeta(
    const std::string &port) {
  if (is_input_meta_available_ == false) {
    return nullptr;
  }

  if (input_port_meta_.find(port) == input_port_meta_.end()) {
    return nullptr;
  }
  return input_port_meta_.find(port)->second;
}

const std::shared_ptr<DataMeta> FlowUnitDataContext::GetInputGroupMeta(
    const std::string &port) {
  return nullptr;
}

void FlowUnitDataContext::SetOutputMeta(const std::string &port,
                                        std::shared_ptr<DataMeta> data_meta) {
  if (is_output_meta_available_ == false) {
    return;
  }

  if (output_port_meta_.find(port) != output_port_meta_.end()) {
    return;
  }
  output_port_meta_.emplace(port, data_meta);
}

std::shared_ptr<SessionContext> FlowUnitDataContext::GetSessionContext() {
  auto session_context = session_context_.lock();
  return session_context;
}

const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
    &FlowUnitDataContext::GetInputs() const {
  return input_;
}

const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
    &FlowUnitDataContext::GetExternals() const {
  return input_;
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

const std::unordered_map<std::string, std::shared_ptr<BufferList>>
    &FlowUnitDataContext::GetOutputs() const {
  return output_;
}

void FlowUnitDataContext::SetOutput(
    const std::unordered_map<std::string, std::shared_ptr<BufferList>>
        &data_list) {
  output_ = data_list;
}

void FlowUnitDataContext::ClearData() {
  input_.clear();
  output_.clear();
  output_data_ = nullptr;
  user_event_ = nullptr;
  output_stream_error_ = nullptr;
  error_ = nullptr;
}

void FlowUnitDataContext::UpdateDataPostFlag(bool flag) {
  data_post_flag_ = flag;
}

bool FlowUnitDataContext::IsSkippable() { return is_skippable_; }
void FlowUnitDataContext::SetSkippable(bool skippable) {
  is_skippable_ = skippable;
}

void FlowUnitDataContext::FillEmptyOutput() {
  for (auto &output : node_->GetOutputNames()) {
    auto buffer_list = std::make_shared<BufferList>();
    output_.emplace(output, buffer_list);
  }
}

bool FlowUnitDataContext::IsOutputStreamError() {
  if (output_stream_error_ != nullptr) {
    return true;
  }
  return false;
}

void FlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {
  FlowUnitDataContext::DealWithProcessError(error);
}

void FlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
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

void FlowUnitDataContext::AddDestroyCallback(
    const std::function<void()> &func) {
  destroy_callback_list_.push_back(func);
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

void FlowUnitDataContext::UpdateStartFlag() {
  if (start_flag_) {
    start_flag_ = false;
  }
}

Status FlowUnitDataContext::AppendOutputMap(OutputIndexBuffer *map) {
  if (output_data_ == nullptr) {
    return {STATUS_FAULT, "output data is invalid."};
  }
  if (output_data_->IsEmpty()) {
    return {STATUS_FAULT, "output data is empty."};
  }
  return output_data_->AppendOutputMap(map);
}

void FlowUnitDataContext::SetStatus(Status status) {
  process_status_ = status;
  last_process_status_ = status;
}

Status FlowUnitDataContext::GetStatus() { return process_status_; }

Status FlowUnitDataContext::GetLastStatus() { return last_process_status_; }

NormalExpandFlowUnitDataContext::NormalExpandFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {
  expand_level_stream_ = nullptr;
  start_flag_ = true;
  end_flag_ = true;
}

void NormalExpandFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  auto buffer_list = input_data->begin()->second;

  auto first_buffer = buffer_list->GetBuffer(0);
  // It needs to be modified after refactoring
  auto buffer_ptr = first_buffer->GetBufferGroup()
                        ->GetOneLevelGroup()
                        ->GenerateSameLevelGroup();
  auto priority = first_buffer->GetPriority();
  expand_level_stream_ = std::make_shared<VirtualStream>(buffer_ptr, priority);

  auto first_buffer_list = input_data->begin()->second;
  if (first_buffer_list->GetBufferPtrList().size() == 0) {
    SetSkippable(true);
    backfill_set_.insert(0);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);
    input_.emplace(key, index_buffer_list->GetBufferPtrList());
    input_data_iter++;
  }

  UpdateErrorIndex(input_data);
}

bool NormalExpandFlowUnitDataContext::IsDataGroupPre() { return false; }
bool NormalExpandFlowUnitDataContext::IsDataGroupPost() { return false; }
bool NormalExpandFlowUnitDataContext::IsDataPre() { return false; }
bool NormalExpandFlowUnitDataContext::IsDataPost() { return false; }
bool NormalExpandFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }

  if (is_exception_visible_) {
    return true;
  }

  return false;
}

void NormalExpandFlowUnitDataContext::SendEvent(
    std::shared_ptr<FlowUnitEvent> event) {
  auto inner_event = std::make_shared<FlowUnitInnerEvent>(
      FlowUnitInnerEvent::EXPAND_UNFINISH_DATA);
  inner_event->SetUserEvent(event);
  inner_event->SetBufferGroup(
      expand_level_stream_->GetBufferGroup()->GetGroup());
  std::vector<std::shared_ptr<FlowUnitInnerEvent>> event_vector;
  event_vector.push_back(inner_event);
  if (node_ == nullptr) {
    return;
  }
  node_->SendBatchEvent(event_vector);
}

std::shared_ptr<FlowUnitInnerEvent>
NormalExpandFlowUnitDataContext::GenerateSendEvent() {
  if (closed_) {
    process_status_ = STATUS_SUCCESS;
  }

  return nullptr;
}

void NormalExpandFlowUnitDataContext::CloseExpandStream() {
  if (expand_level_stream_ != nullptr) {
    expand_level_stream_->Close();
  }
}

Status NormalExpandFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  if (closed_) {
    CloseExpandStream();
    process_status_ = STATUS_SUCCESS;
    return STATUS_SUCCESS;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto &input_error : input_error_index_) {
      stream_index_->SetDataError(input_error.second, output_error);
    }
    output_data_->SetAllPortError(output_error);
    if (!closed_) {
      expand_level_stream_->Close();
    }
    closed_ = true;
    process_status_ = STATUS_SUCCESS;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

Status NormalExpandFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  if (closed_) {
    output_data_->SetAllPortError(output_stream_error_);
    CloseExpandStream();
    process_status_ = STATUS_SUCCESS;
    return STATUS_SUCCESS;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    output_stream_error_ = nullptr;
  } else if (output_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_stream_error_);
  }

  if (!closed_) {
    CloseExpandStream();
  }
  closed_ = true;
  return STATUS_SUCCESS;
}

void NormalExpandFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  if ((process_status_ == STATUS_SUCCESS) && (end_flag_)) {
    is_finished_ = true;
  }

  process_status_ = STATUS_SUCCESS;
}

void NormalExpandFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  output_stream_error_ = error;
  process_status_ = STATUS_SUCCESS;
}

void NormalExpandFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void NormalExpandFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

void NormalExpandFlowUnitDataContext::CloseStreamIfNecessary() {
  if (process_status_ == STATUS_SUCCESS) {
    if (output_data_ != nullptr) {
      auto index_buffer_list = output_data_->GetOneBufferList();
      if (index_buffer_list->GetBufferNum() == 0) {
        InsertPlaceholderToTheOutput();
      }
    }
    CloseExpandStream();
  }
}

Status NormalExpandFlowUnitDataContext::GenerateOutData() {
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  if (expand_level_stream_ == nullptr) {
    output_data_->Clear();
    auto msg = "expand stream can not find in " + node_->GetName();
    MBLOG_ERROR << msg;
    return {STATUS_NOTFOUND, msg};
  }
  output_data_->BackfillOutput(backfill_set_);

  auto index_buffer_list = output_data_->GetOneBufferList();
  expand_level_stream_->LabelIndexBuffer(index_buffer_list);
  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
}

void NormalExpandFlowUnitDataContext::InsertPlaceholderToTheOutput() {
  output_data_->BackfillOutput(std::set<uint32_t>{0});
  auto index_buffer_list = output_data_->GetOneBufferList();
  expand_level_stream_->LabelIndexBuffer(index_buffer_list);
  output_data_->BroadcastMetaToAll();
}

StreamExpandFlowUnitDataContext::StreamExpandFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {
  current_expand_order_ = 0;
  expand_cache_num_ = 0;
  current_expand_group_ = nullptr;
  input_spilt_cache_ = nullptr;
  expand_level_stream_ = nullptr;
  is_input_group_meta_available_ = false;
  is_input_meta_available_ = true;
  is_output_meta_available_ = true;
  input_spilt_cache_ = std::make_shared<InputData>();
}

bool StreamExpandFlowUnitDataContext::IsDataGroupPre() { return false; }

bool StreamExpandFlowUnitDataContext::IsDataGroupPost() { return false; }

bool StreamExpandFlowUnitDataContext::IsDataPre() {
  if (start_flag_) {
    return true;
  }
  return false;
}

bool StreamExpandFlowUnitDataContext::IsDataPost() {
  if (end_flag_ && data_post_flag_ && (process_status_ == STATUS_SUCCESS)) {
    UpdateDataPostFlag(false);
    return true;
  }
  return false;
}

bool StreamExpandFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }

  if (is_exception_visible_) {
    return true;
  }

  return false;
}

void StreamExpandFlowUnitDataContext::UpdateCurrentOrder() {
  current_expand_order_++;
}

std::shared_ptr<FlowUnitInnerEvent>
StreamExpandFlowUnitDataContext::GenerateSendEvent() {
  if (closed_) {
    process_status_ = STATUS_SUCCESS;
  }

  if (process_status_ == STATUS_SUCCESS) {
    if (!end_flag_) {
      auto expand_event = std::make_shared<FlowUnitInnerEvent>(
          FlowUnitInnerEvent::EXPAND_NEXT_STREAM);
      expand_event->SetBufferGroup(stream_index_);
      return expand_event;
    }
  }
  return nullptr;
}

void StreamExpandFlowUnitDataContext::CloseStreamIfNecessary() {
  if (process_status_ == STATUS_SUCCESS) {
    if (output_data_ != nullptr) {
      auto index_buffer_list = output_data_->GetOneBufferList();
      if (index_buffer_list->GetBufferNum() == 0) {
        InsertPlaceholderToTheOutput();
      }
    }
    CloseExpandStream();
  }
}

Status StreamExpandFlowUnitDataContext::GenerateOutData() {
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  if (expand_level_stream_ == nullptr) {
    output_data_->Clear();
    auto msg = "expand stream can not find in " + node_->GetName();
    MBLOG_ERROR << msg;
    return {STATUS_NOTFOUND, msg};
  }

  auto index_buffer_list = output_data_->GetOneBufferList();
  output_data_->BackfillOutput(backfill_set_);
  expand_level_stream_->LabelIndexBuffer(index_buffer_list);
  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
}

Status StreamExpandFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  if (closed_) {
    output_data_->SetAllPortError(output_stream_error_);
    CloseExpandStream();
    process_status_ = STATUS_SUCCESS;
    return STATUS_SUCCESS;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    output_stream_error_ = nullptr;
  } else if (output_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_stream_error_);
  }

  if (!closed_) {
    CloseExpandStream();
  }
  closed_ = true;
  return STATUS_SUCCESS;
}

Status StreamExpandFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  if (closed_) {
    CloseExpandStream();
    process_status_ = STATUS_SUCCESS;
    return STATUS_SUCCESS;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto &input_error : input_error_index_) {
      stream_index_->SetDataError(input_error.second, output_error);
    }
    output_data_->SetAllPortError(output_error);
    if (!closed_) {
      expand_level_stream_->Close();
    }
    closed_ = true;
    process_status_ = STATUS_SUCCESS;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

void StreamExpandFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

void StreamExpandFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {
  DealWithDataError(error);
}

void StreamExpandFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  output_stream_error_ = error;
  process_status_ = STATUS_SUCCESS;
}

void StreamExpandFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  if ((process_status_ == STATUS_SUCCESS) && (end_flag_)) {
    is_finished_ = true;
  }

  process_status_ = STATUS_SUCCESS;
}

void StreamExpandFlowUnitDataContext::ExpandNewStream(
    std::shared_ptr<InputData> &input_data) {
  auto buffer_list = input_data->begin()->second;
  if (expand_cache_num_ == current_expand_order_) {
    return;
  }

  auto first_buffer_vector = std::vector<std::shared_ptr<IndexBuffer>>();
  first_buffer_vector.push_back(buffer_list->GetBuffer(current_expand_order_));
  auto first_buffer_list =
      std::make_shared<IndexBufferList>(first_buffer_vector);
  UpdateInputInfo(first_buffer_list);

  auto first_buffer = first_buffer_list->GetBuffer(0);

  auto buffer_ptr = first_buffer->GetBufferGroup();
  auto priority = first_buffer->GetPriority();
  expand_level_stream_ = std::make_shared<VirtualStream>(buffer_ptr, priority);

  current_expand_group_ = first_buffer->GetBufferGroup();

  auto cur_input_buffer =
      input_data->begin()->second->GetBuffer(current_expand_order_);
  if (!cur_input_buffer->IsPlaceholder()) {
    backfill_set_.clear();
    SetSkippable(false);
  } else {
    backfill_set_.insert(0);
    SetSkippable(true);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);
    auto cur_buffer_list = std::vector<std::shared_ptr<Buffer>>();
    auto cur_buffer = index_buffer_list->GetBuffer(current_expand_order_);
    if (!cur_buffer->IsPlaceholder()) {
      cur_buffer_list.push_back(cur_buffer->GetBufferPtr());
    }

    input_.emplace(key, cur_buffer_list);
    input_data_iter++;
  }
  UpdateErrorIndex(input_data);
}

void StreamExpandFlowUnitDataContext::ExpandNextStream() {
  ExpandNewStream(input_spilt_cache_);
}

void StreamExpandFlowUnitDataContext::CloseExpandStream() {
  if (expand_level_stream_ != nullptr) {
    expand_level_stream_->Close();
  }
}

void StreamExpandFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  if (input_spilt_cache_ == nullptr) {
    input_spilt_cache_ = std::make_shared<InputData>();
  }
  auto origin_data_iter = input_data->begin();
  int cache_buffer_number = 0;
  while (origin_data_iter != input_data->end()) {
    auto key = origin_data_iter->first;
    auto index_buffer_list = origin_data_iter->second;
    if (input_spilt_cache_->find(key) == input_spilt_cache_->end()) {
      input_spilt_cache_->emplace(key, index_buffer_list);
    } else {
      input_spilt_cache_->at(key)->ExtendBufferList(index_buffer_list);
    }

    cache_buffer_number = index_buffer_list->GetBufferNum();
    origin_data_iter++;
  }
  expand_cache_num_ += cache_buffer_number;

  if (expand_cache_num_ == current_expand_order_ + cache_buffer_number) {
    ExpandNewStream(input_spilt_cache_);
  }
}

void StreamExpandFlowUnitDataContext::InsertPlaceholderToTheOutput() {
  output_data_->BackfillOutput(std::set<uint32_t>{0});
  auto index_buffer_list = output_data_->GetOneBufferList();
  expand_level_stream_->LabelIndexBuffer(index_buffer_list);
  output_data_->BroadcastMetaToAll();
}

NormalCollapseFlowUnitDataContext::NormalCollapseFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {}

void NormalCollapseFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  auto buffer_list = input_data->begin()->second;
  stream_index_ = buffer_list->GetStreamBufferGroup();
  UpdateInputInfo(buffer_list);

  auto first_bufferlist = input_data->begin()->second;
  if (first_bufferlist->IsExpandBackfillBufferlist()) {
    backfill_set_.insert(0);
  }

  if (first_bufferlist->GetBufferPtrList().size() == 0) {
    SetSkippable(true);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);
    if (index_buffer_list->IsExpandBackfillBufferlist()) {
      backfill_set_.insert(0);
    }
    input_.emplace(key, index_buffer_list->GetBufferPtrList());
    input_data_iter++;
  }
  UpdateErrorIndex(input_data);
}

void NormalCollapseFlowUnitDataContext::UpdateErrorIndex(
    std::shared_ptr<InputData> input_data) {
  auto input_iter = input_data->begin();
  while (input_iter != input_data->end()) {
    auto key = input_iter->first;
    auto index_buffer_list = input_iter->second;
    auto error_index = index_buffer_list->GetGroupDataErrorIndex();
    input_group_error_index_.emplace(key, error_index);

    input_iter++;
  }
  FlowUnitDataContext::UpdateErrorIndex(input_data);
}

bool NormalCollapseFlowUnitDataContext::IsDataGroupPre() { return false; }

bool NormalCollapseFlowUnitDataContext::IsDataGroupPost() { return false; }

bool NormalCollapseFlowUnitDataContext::IsDataPre() {
  if (start_flag_ && backfill_set_.empty()) {
    return true;
  }
  return false;
}

bool NormalCollapseFlowUnitDataContext::IsDataPost() {
  if (end_flag_ && data_post_flag_ && (process_status_ == STATUS_SUCCESS)) {
    UpdateDataPostFlag(false);
    if (backfill_set_.empty()) {
      return true;
    }
  }
  return false;
}

bool NormalCollapseFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }

  if (is_exception_visible_) {
    return true;
  }
  return false;
}

std::shared_ptr<FlowUnitInnerEvent>
NormalCollapseFlowUnitDataContext::GenerateSendEvent() {
  return nullptr;
};

Status NormalCollapseFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto &input_group_error : input_group_error_index_) {
      auto key = input_group_error.second;
      stream_index_->GetOneLevelGroup()->SetDataError(key, output_error);
    }

    for (auto &input_error : input_error_index_) {
      auto key = input_error.second;
      stream_index_->SetDataError(key, output_error);
    }
    output_data_->SetAllPortError(output_error);

    closed_ = true;
    output_group_stream_error_ = nullptr;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

Status NormalCollapseFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    closed_ = true;
    output_group_stream_error_ = nullptr;
    output_stream_error_ = nullptr;
  } else if (output_group_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_group_stream_error_);
  } else if (output_stream_error_ != nullptr) {
    output_stream_error_ = nullptr;
  }
  return STATUS_SUCCESS;
}

void NormalCollapseFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  input_group_error_index_.clear();
  process_status_ = STATUS_SUCCESS;
};

Status NormalCollapseFlowUnitDataContext::GenerateOutData() {
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  output_data_->BackfillOutput(backfill_set_);
  auto index_buffer_list = output_data_->GetOneBufferList();

  if (index_buffer_list->GetBufferNum() == 0) {
    return STATUS_SUCCESS;
  }

  if (index_buffer_list->GetBufferNum() != 1) {
    auto msg = node_->GetName() + " collapse generate more than one buffer";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  if (stream_index_ == nullptr) {
    output_data_->Clear();
    auto msg = "collapse stream can not find in " + node_->GetName();
    return {STATUS_NOTFOUND, msg};
  }

  auto bg = stream_index_->GetGroup()->GenerateSameLevelGroup();
  auto stream_buffer = index_buffer_list->GetBuffer(0);
  stream_buffer->SetBufferGroup(bg);
  stream_buffer->SetPriority(priority_);

  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
};

void NormalCollapseFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  DealWithDataPreError(error);
};

void NormalCollapseFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  if (end_flag_) {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
};

void NormalCollapseFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_group_error : input_group_error_index_) {
    auto key = input_group_error.second;
    stream_index_->GetOneLevelGroup()->SetDataError(key, error);
  }

  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  if (end_flag_) {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }

  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
};

void NormalCollapseFlowUnitDataContext::CloseStreamIfNecessary() {
  if (end_flag_) {
    is_finished_ = true;
  }
};

StreamCollapseFlowUnitDataContext::StreamCollapseFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {
  is_group_stream_same_count_ = true;
  group_start_flag_ = false;
  group_end_flag_ = false;
  current_collapse_order_ = 0;

  collapse_level_stream_ = nullptr;
  data_group_post_flag_ = false;
  is_input_group_meta_available_ = true;
  is_output_meta_available_ = !is_group_stream_same_count_;
}

void StreamCollapseFlowUnitDataContext::UpdateInputInfo(
    std::shared_ptr<IndexBufferList> buffer_list) {
  group_start_flag_ = buffer_list->IsStartStream();
  group_end_flag_ = buffer_list->IsEndStream();
  FlowUnitDataContext::UpdateInputInfo(buffer_list);
}

void StreamCollapseFlowUnitDataContext::UpdateInputDataMeta(
    std::string key, std::shared_ptr<IndexBufferList> buffer_list) {
  if (input_group_port_meta_.find(key) == input_group_port_meta_.end()) {
    input_group_port_meta_.emplace(key, buffer_list->GetGroupDataMeta());
  }
  FlowUnitDataContext::UpdateInputDataMeta(key, buffer_list);
}

void StreamCollapseFlowUnitDataContext::UpdateErrorIndex(
    std::shared_ptr<InputData> input_data) {
  auto input_iter = input_data->begin();
  while (input_iter != input_data->end()) {
    auto key = input_iter->first;
    auto index_buffer_list = input_iter->second;
    auto error_index = index_buffer_list->GetGroupDataErrorIndex();
    input_group_error_index_.emplace(key, error_index);

    input_iter++;
  }
  FlowUnitDataContext::UpdateErrorIndex(input_data);
}

void StreamCollapseFlowUnitDataContext::CollapseNewStream(
    std::shared_ptr<InputData> &input_data) {
  auto buffer_list = input_data->begin()->second;
  stream_index_ = buffer_list->GetStreamBufferGroup();
  UpdateInputInfo(buffer_list);
  if ((group_start_flag_) && (!is_group_stream_same_count_)) {
    collapse_level_stream_ = std::make_shared<VirtualStream>(
        buffer_list->GetStreamBufferGroup(), buffer_list->GetPriority());
  }

  auto first_bufferlist = input_data->begin()->second;

  if (first_bufferlist->IsExpandBackfillBufferlist()) {
    backfill_set_.insert(0);
  } else {
    backfill_set_.clear();
  }

  if (first_bufferlist->GetBufferPtrList().size() == 0) {
    SetSkippable(true);
  } else {
    SetSkippable(false);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);

    input_.emplace(key, index_buffer_list->GetBufferPtrList());
    input_data_iter++;
  }
  UpdateErrorIndex(input_data);
}

void StreamCollapseFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  auto buffer_list = input_data->begin()->second;
  auto group_order = buffer_list->GetOrder();

  if (group_order != current_collapse_order_) {
    if (input_cache_.find(group_order) != input_cache_.end()) {
      auto origin_data_iter = input_cache_[group_order]->begin();
      while (origin_data_iter != input_cache_[group_order]->end()) {
        auto key = origin_data_iter->first;
        input_cache_[group_order]->at(key)->ExtendBufferList(
            input_data->at(key));
        origin_data_iter++;
      }
    } else {
      input_cache_.emplace(group_order, input_data);
    }
  } else {
    CollapseNewStream(input_data);
  }
}

const std::shared_ptr<DataMeta>
StreamCollapseFlowUnitDataContext::GetInputGroupMeta(const std::string &port) {
  if (is_input_group_meta_available_ == false) {
    return nullptr;
  }
  if (input_group_port_meta_.find(port) == input_group_port_meta_.end()) {
    return nullptr;
  }
  return input_group_port_meta_.find(port)->second;
}

void StreamCollapseFlowUnitDataContext::UpdateCurrentOrder() {
  current_collapse_order_++;
}

void StreamCollapseFlowUnitDataContext::UpdateDataGroupPostFlag(bool flag) {
  data_group_post_flag_ = flag;
  is_input_group_meta_available_ = !flag;
}

void StreamCollapseFlowUnitDataContext::CollapseNextStream() {
  if (input_cache_.find(current_collapse_order_) == input_cache_.end()) {
    start_flag_ = true;
    end_flag_ = false;
    return;
  }
  auto input_data = input_cache_[current_collapse_order_];
  CollapseNewStream(input_data);
}

std::shared_ptr<FlowUnitInnerEvent>
StreamCollapseFlowUnitDataContext::GenerateSendEvent() {
  if ((process_status_ == STATUS_SUCCESS) ||
      (process_status_ == STATUS_CONTINUE)) {
    if ((end_flag_) && (!group_end_flag_)) {
      auto event = std::make_shared<FlowUnitInnerEvent>(
          FlowUnitInnerEvent::COLLAPSE_NEXT_STREAM);
      event->SetBufferGroup(stream_index_->GetOneLevelGroup());
      return event;
    }
  }
  return nullptr;
}

Status StreamCollapseFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }
  if (closed_ && (!is_group_stream_same_count_)) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    if ((!is_group_stream_same_count_) && (!closed_)) {
      collapse_level_stream_->Close();
    }
    closed_ = true;
    output_group_stream_error_ = nullptr;
    output_stream_error_ = nullptr;
  } else if (output_group_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_group_stream_error_);
  } else if (output_stream_error_ != nullptr) {
    output_stream_error_ = nullptr;
  }
  return STATUS_SUCCESS;
}

void StreamCollapseFlowUnitDataContext::CloseStreamIfNecessary() {
  if ((group_end_flag_) && (end_flag_)) {
    CloseCollapseStream();
  }
}

Status StreamCollapseFlowUnitDataContext::GenerateOutData() {
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }
  output_data_->BackfillOutput(backfill_set_);

  auto index_buffer_list = output_data_->GetOneBufferList();

  if (is_group_stream_same_count_) {
    if (index_buffer_list->GetBufferNum() == 0) {
      return STATUS_SUCCESS;
    }

    if (index_buffer_list->GetBufferNum() != 1) {
      auto msg = node_->GetName() + " collapse generate more than one buffer";
      MBLOG_ERROR << msg;
      return {STATUS_INVALID, msg};
    }

    if (stream_index_ == nullptr) {
      output_data_->Clear();
      auto msg = "collapse stream can not find in " + node_->GetName();
      return {STATUS_NOTFOUND, msg};
    }

    auto bg = stream_index_->GetGroup()->GenerateSameLevelGroup();
    auto stream_buffer = index_buffer_list->GetBuffer(0);
    stream_buffer->SetBufferGroup(bg);
    stream_buffer->SetPriority(priority_);
  } else {
    collapse_level_stream_->LabelIndexBuffer(index_buffer_list);
  }
  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
}

Status StreamCollapseFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  if (closed_ && (!is_group_stream_same_count_)) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto &input_group_error : input_group_error_index_) {
      auto key = input_group_error.second;
      stream_index_->GetOneLevelGroup()->SetDataError(key, output_error);
    }

    for (auto &input_error : input_error_index_) {
      auto key = input_error.second;
      stream_index_->SetDataError(key, output_error);
    }
    output_data_->SetAllPortError(output_error);
    if ((!is_group_stream_same_count_) && (!closed_)) {
      collapse_level_stream_->Close();
    }

    closed_ = true;
    output_group_stream_error_ = nullptr;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

bool StreamCollapseFlowUnitDataContext::IsDataGroupPre() {
  if (group_start_flag_ && start_flag_) {
    return true;
  }
  return false;
}

bool StreamCollapseFlowUnitDataContext::IsDataGroupPost() {
  if (group_end_flag_ && end_flag_ && data_group_post_flag_) {
    return true;
  }
  return false;
}
bool StreamCollapseFlowUnitDataContext::IsDataPre() {
  if (start_flag_ && backfill_set_.empty()) {
    return true;
  }
  return false;
}

bool StreamCollapseFlowUnitDataContext::IsDataPost() {
  if (end_flag_ && data_post_flag_ && (process_status_ == STATUS_SUCCESS)) {
    UpdateDataPostFlag(false);
    if (backfill_set_.empty()) {
      return true;
    }
  }
  return false;
}

bool StreamCollapseFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }

  if (is_exception_visible_) {
    return true;
  }
  return false;
}

void StreamCollapseFlowUnitDataContext::CloseCollapseStream() {
  is_finished_ = true;
  if (!is_group_stream_same_count_) {
    if (collapse_level_stream_ != nullptr) {
      collapse_level_stream_->Close();
    }
  }
}

void StreamCollapseFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  input_group_error_index_.clear();
  process_status_ = STATUS_SUCCESS;
}

void StreamCollapseFlowUnitDataContext::DealWithDataGroupPreError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_group_error : input_group_error_index_) {
    auto key = input_group_error.second;
    stream_index_->GetOneLevelGroup()->SetDataError(key, error);
  }

  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  if ((group_end_flag_) && (end_flag_)) {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }

  output_group_stream_error_ = error;
  closed_ = true;
}

void StreamCollapseFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  if (end_flag_) {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

void StreamCollapseFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  DealWithDataPreError(error);
}

void StreamCollapseFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_group_error : input_group_error_index_) {
    auto key = input_group_error.second;
    stream_index_->GetOneLevelGroup()->SetDataError(key, error);
  }

  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  if ((group_end_flag_ && !is_group_stream_same_count_) ||
      (end_flag_ && is_group_stream_same_count_)) {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }

  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

StreamFlowUnitDataContext::StreamFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {
  is_stream_same_count_ = node->IsStreamSameCount();
  same_level_stream_ = nullptr;
  is_input_meta_available_ = true;
  is_input_group_meta_available_ = false;
  is_output_meta_available_ = !is_stream_same_count_;
}

void StreamFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  auto buffer_list = input_data->begin()->second;
  UpdateInputInfo(buffer_list);

  if (is_stream_same_count_) {
    for (uint32_t i = 0; i < buffer_list->GetBufferNum(); i++) {
      auto buffer = buffer_list->GetBuffer(i);
      buffer_index_.push_back(buffer->GetSameLevelGroup());
    }
  } else {
    if (same_level_stream_ == nullptr) {
      same_level_stream_ = std::make_shared<VirtualStream>(
          buffer_list->GetBuffer(0)->GetSameLevelGroup(),
          buffer_list->GetPriority());
    }
  }

  auto first_index_buffer = input_data->begin()->second;
  backfill_set_ = first_index_buffer->GetPlaceholderPos();
  if (first_index_buffer->GetBufferPtrList().size() == 0) {
    SetSkippable(true);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);
    input_.emplace(key, index_buffer_list->GetBufferPtrList());
    input_data_iter++;
  }
  UpdateErrorIndex(input_data);
}

void StreamFlowUnitDataContext::CloseStream() {
  is_finished_ = true;
  if (!is_stream_same_count_) {
    if (same_level_stream_ != nullptr) {
      same_level_stream_->Close();
    }
  }
}

bool StreamFlowUnitDataContext::IsDataGroupPre() { return false; }
bool StreamFlowUnitDataContext::IsDataGroupPost() { return false; }
bool StreamFlowUnitDataContext::IsDataPre() {
  if (start_flag_) {
    return true;
  }
  return false;
}
bool StreamFlowUnitDataContext::IsDataPost() {
  if (end_flag_ && data_post_flag_ && (process_status_ == STATUS_SUCCESS)) {
    UpdateDataPostFlag(false);
    return true;
  }
  return false;
}

bool StreamFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }
  if (is_exception_visible_) {
    return true;
  }
  return false;
}

std::shared_ptr<FlowUnitInnerEvent>
StreamFlowUnitDataContext::GenerateSendEvent() {
  return nullptr;
}

void StreamFlowUnitDataContext::CloseStreamIfNecessary() {
  if (end_flag_ && process_status_ == STATUS_SUCCESS) {
    if (output_data_ != nullptr) {
      auto index_buffer_list = output_data_->GetOneBufferList();
      if (index_buffer_list->GetBufferNum() == 0) {
        output_data_->BackfillOutput(std::set<uint32_t>{0});
        auto index_buffer_list = output_data_->GetOneBufferList();
        same_level_stream_->LabelIndexBuffer(index_buffer_list);
        output_data_->BroadcastMetaToAll();
      }
    }

    CloseStream();
  }
}

Status StreamFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }
  if ((closed_) && (!is_stream_same_count_)) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    if ((!is_stream_same_count_) && (!closed_)) {
      same_level_stream_->Close();
    }
    output_stream_error_ = nullptr;
  } else if (output_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_stream_error_);
    if ((!is_stream_same_count_) && (!closed_)) {
      same_level_stream_->Close();
    }
  }
  closed_ = true;
  return STATUS_SUCCESS;
}

Status StreamFlowUnitDataContext::GenerateOutData() {
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  auto index_buffer_list = output_data_->GetOneBufferList();
  if (is_stream_same_count_) {
    output_data_->BackfillOutput(backfill_set_);
    if (index_buffer_list->GetBufferNum() != buffer_index_.size()) {
      auto msg = node_->GetName() +
                 " can't  generate the same count output buffer as input";
      MBLOG_ERROR << msg;
      return {STATUS_INVALID, msg};
    }

    if (buffer_index_.size() == 0) {
      output_data_->Clear();
      auto msg = node_->GetName() + " the input buffer num is zero";
      MBLOG_ERROR << msg;
      return {STATUS_NOTFOUND, msg};
    }

    for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
      auto stream_buffer = index_buffer_list->GetBuffer(i);
      auto bg = buffer_index_.at(i)->GenerateSameLevelGroup();
      stream_buffer->SetBufferGroup(bg);
      stream_buffer->SetPriority(priority_);
    }

  } else {
    if (same_level_stream_ == nullptr) {
      output_data_->Clear();
      auto msg = "stream can not find in " + node_->GetName();
      MBLOG_ERROR << msg;
      return {STATUS_NOTFOUND, msg};
    }
    same_level_stream_->LabelIndexBuffer(index_buffer_list);
  }
  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
}

Status StreamFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  if ((!is_stream_same_count_) && (closed_)) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto input_error : input_error_index_) {
      stream_index_->SetDataError(input_error.second, output_error);
    }
    output_data_->SetAllPortError(output_error);
    if ((!is_stream_same_count_) && (!closed_)) {
      same_level_stream_->Close();
    }
    closed_ = true;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

void StreamFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  if (is_stream_same_count_) {
    buffer_index_.clear();
  }
  process_status_ = STATUS_SUCCESS;
}

void StreamFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  if (is_stream_same_count_) {
    auto input_num = input_.begin()->second.size();
    for (auto &output : node_->GetOutputNames()) {
      std::vector<std::shared_ptr<Buffer>> buffer_vector;
      for (uint32_t i = 0; i < input_num; i++) {
        auto buffer = std::make_shared<Buffer>();
        buffer->SetError(error);
        buffer_vector.push_back(buffer);
      }
      auto buffer_list = std::make_shared<BufferList>(buffer_vector);
      output_.emplace(output, buffer_list);
    }
  } else {
    for (auto &output : node_->GetOutputNames()) {
      auto buffer = std::make_shared<Buffer>();
      buffer->SetError(error);
      auto buffer_list = std::make_shared<BufferList>(buffer);
      output_.emplace(output, buffer_list);
    }
  }

  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

void StreamFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }

  output_.clear();
  for (auto &output : node_->GetOutputNames()) {
    auto buffer = std::make_shared<Buffer>();
    buffer->SetError(error);
    auto buffer_list = std::make_shared<BufferList>(buffer);
    output_.emplace(output, buffer_list);
  }
  output_stream_error_ = error;
}

void StreamFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  FlowUnitDataContext::DealWithProcessError(error);
}

void StreamFlowUnitDataContext::UpdateStartFlag() {
  is_output_meta_available_ = false;
  FlowUnitDataContext::UpdateStartFlag();
}

NormalFlowUnitDataContext::NormalFlowUnitDataContext(
    std::shared_ptr<BufferGroup> stream_bg, Node *node)
    : FlowUnitDataContext(stream_bg, node) {
  is_input_meta_available_ = false;
  is_input_group_meta_available_ = false;
  is_output_meta_available_ = false;
}

void NormalFlowUnitDataContext::SetInputData(
    std::shared_ptr<InputData> input_data) {
  auto buffer_list = input_data->begin()->second;
  UpdateInputInfo(buffer_list);

  for (uint32_t i = 0; i < buffer_list->GetBufferNum(); i++) {
    auto buffer = buffer_list->GetBuffer(i);
    buffer_index_.push_back(buffer->GetSameLevelGroup());
  }

  auto first_index_buffer = input_data->begin()->second;
  backfill_set_ = first_index_buffer->GetPlaceholderPos();
  if (first_index_buffer->GetBufferPtrList().size() == 0) {
    SetSkippable(true);
  }

  auto input_data_iter = input_data->begin();
  while (input_data_iter != input_data->end()) {
    auto key = input_data_iter->first;
    auto index_buffer_list = input_data_iter->second;
    UpdateInputDataMeta(key, index_buffer_list);
    input_.emplace(key, index_buffer_list->GetBufferPtrList());
    input_data_iter++;
  }

  UpdateErrorIndex(input_data);
}

void NormalFlowUnitDataContext::DealWithDataPreError(
    std::shared_ptr<FlowUnitError> error) {}

void NormalFlowUnitDataContext::DealWithProcessError(
    std::shared_ptr<FlowUnitError> error) {
  output_.clear();
  auto input_num = input_.begin()->second.size();
  bool input_flag = false;
  for (auto &output : node_->GetOutputNames()) {
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    if (input_flag == false) {
      for (uint32_t i = 0; i < input_num; i++) {
        auto buffer = std::make_shared<Buffer>();
        buffer->SetError(error);
        buffer_vector.push_back(buffer);
      }
    } else {
      for (uint32_t i = 0; i < input_num; i++) {
        buffer_vector.push_back(nullptr);
      }
    }
    auto buffer_list = std::make_shared<BufferList>(buffer_vector);
    output_.emplace(output, buffer_list);

    if (input_flag == false && node_->GetConditionType() == IF_ELSE) {
      input_flag = true;
    }
  }
  process_status_ = STATUS_SUCCESS;
  output_stream_error_ = error;
}

void NormalFlowUnitDataContext::DealWithDataError(
    std::shared_ptr<FlowUnitError> error) {
  for (auto &input_error : input_error_index_) {
    auto key = input_error.second;
    stream_index_->SetDataError(key, error);
  }
  auto input_num = input_.begin()->second.size();

  bool input_flag = false;
  for (auto &output : node_->GetOutputNames()) {
    std::vector<std::shared_ptr<Buffer>> buffer_vector;
    if (input_flag == false) {
      for (uint32_t i = 0; i < input_num; i++) {
        auto buffer = std::make_shared<Buffer>();
        buffer->SetError(error);
        buffer_vector.push_back(buffer);
      }
    } else {
      for (uint32_t i = 0; i < input_num; i++) {
        buffer_vector.push_back(nullptr);
      }
    }
    auto buffer_list = std::make_shared<BufferList>(buffer_vector);
    output_.emplace(output, buffer_list);

    if (input_flag == false && node_->GetConditionType() == IF_ELSE) {
      input_flag = true;
    }
  }
  output_stream_error_ = error;
}

bool NormalFlowUnitDataContext::IsDataGroupPre() { return false; }
bool NormalFlowUnitDataContext::IsDataGroupPost() { return false; }
bool NormalFlowUnitDataContext::IsDataPre() { return false; }
bool NormalFlowUnitDataContext::IsDataPost() { return false; }

bool NormalFlowUnitDataContext::IsDataErrorVisible() {
  if (closed_) {
    return false;
  }
  if (is_exception_visible_) {
    return true;
  }
  return false;
}

std::shared_ptr<FlowUnitInnerEvent>
NormalFlowUnitDataContext::GenerateSendEvent() {
  return nullptr;
}

Status NormalFlowUnitDataContext::LabelError() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }

  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    output_stream_error_ = nullptr;
  } else if (output_stream_error_ != nullptr) {
    output_data_->SetAllPortError(output_stream_error_);
  }
  closed_ = true;
  return STATUS_SUCCESS;
}

Status NormalFlowUnitDataContext::GenerateOutData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }
  output_data_ = std::make_shared<OutputRings>(output_);
  auto status = output_data_->IsValid();
  if (status != STATUS_SUCCESS) {
    auto msg = node_->GetName() +
               " output data is invalid,please check the flow output";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  if (node_->GetConditionType() == IF_ELSE) {
    output_data_->BackfillOneOutput(backfill_set_);
  } else {
    output_data_->BackfillOutput(backfill_set_);
  }

  auto index_buffer_list = output_data_->GetOneBufferList();

  if (index_buffer_list->GetBufferNum() != buffer_index_.size()) {
    auto msg = node_->GetName() +
               " don't  generate the same count output buffer as input";
    MBLOG_ERROR << msg;
    return {STATUS_INVALID, msg};
  }

  if (buffer_index_[0] == nullptr) {
    return STATUS_SUCCESS;
  }

  if (buffer_index_.size() == 0) {
    output_data_->Clear();
    auto msg = node_->GetName() + " the input buffer num is zero";
    MBLOG_ERROR << msg;
    return {STATUS_NOTFOUND, msg};
  }

  for (uint32_t i = 0; i < index_buffer_list->GetBufferNum(); i++) {
    auto stream_buffer = index_buffer_list->GetBuffer(i);
    auto bg = buffer_index_.at(i)->GenerateSameLevelGroup();
    stream_buffer->SetBufferGroup(bg);
    stream_buffer->SetPriority(priority_);
  }

  output_data_->BroadcastMetaToAll();
  return STATUS_SUCCESS;
}

Status NormalFlowUnitDataContext::LabelData() {
  if (output_.size() == 0) {
    return STATUS_SUCCESS;
  }
  auto status = GenerateOutData();
  if (status != STATUS_SUCCESS) {
    return STATUS_STOP;
  }

  auto output_error = output_data_->GetError();
  if (output_error != nullptr) {
    for (auto input_error : input_error_index_) {
      stream_index_->SetDataError(input_error.second, output_error);
    }
    output_data_->SetAllPortError(output_error);
    closed_ = true;
    output_stream_error_ = nullptr;
  } else {
    UpdateOutputDataMeta();
  }
  return STATUS_SUCCESS;
}

void NormalFlowUnitDataContext::CloseStreamIfNecessary() {}

void NormalFlowUnitDataContext::ClearData() {
  FlowUnitDataContext::ClearData();
  buffer_index_.clear();
  is_finished_ = true;
  process_status_ = STATUS_SUCCESS;
}

}  // namespace modelbox
