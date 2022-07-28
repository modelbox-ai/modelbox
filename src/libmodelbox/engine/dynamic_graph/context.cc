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

#include "modelbox/context.h"
#include "modelbox/data_handler.h"
#include "modelbox/modelbox_engine.h"

#define MAX_INPUT_QUEUE_SIZE 10
namespace modelbox {
HandlerContext::HandlerContext(std::weak_ptr<ModelBoxEngine> &env) {
  env_ = env;
}

void HandlerContext::SetMeta(const std::string &key, const std::string &value) {
  meta_.emplace(key, value);
}

std::string HandlerContext::GetMeta(const std::string &key) {
  if (meta_.find(key) == meta_.end()) {
    return "";
  }
  return meta_[key];
}

std::shared_ptr<GraphState> HandlerContext::GetGraphState() {
  return graph_state_;
}

std::shared_ptr<FlowUnitDesc> HandlerContext::GetFlowUnitDesc() {
  return desc_;
}

void HandlerContext::SetFlowUnitDesc(
    const std::shared_ptr<FlowUnitDesc> &desc) {
  desc_ = desc;
}

void HandlerContext::SetGraphState(const std::shared_ptr<GraphState> &state) {
  graph_state_ = state;
}

InputContext::InputContext(std::weak_ptr<ModelBoxEngine> env)
    : HandlerContext(env) {}

InputContext::~InputContext() {
  if (extern_buffer_list_) {
    extern_buffer_list_->Reset();
  }
  extern_buffer_list_ = nullptr;
  extern_data_map_ = nullptr;
}

Status InputContext::RunGraph(const std::shared_ptr<DataHandler> &handler) {
  MBLOG_ERROR << "input handler not support next function.";
  return STATUS_FAULT;
}

void InputContext::SetExternPtr(
    std::shared_ptr<void> extern_data_map,
    std::shared_ptr<BufferList> extern_buffer_list) {
  extern_data_map_ = extern_data_map;
  extern_buffer_list_ = extern_buffer_list;
}

void InputContext::Close() {
  if (extern_data_map_) {
    auto externdata =
        std::static_pointer_cast<ExternalDataMap>(extern_data_map_);
    externdata->Shutdown();
  }
}

std::shared_ptr<BufferList> InputContext::GetBufferList(
    const std::string &key) {
  if (data_map_.size() <= 0) {
    return nullptr;
  }
  return data_map_[key];
}
Status InputContext::PushData(const std::string &key,
                              const std::shared_ptr<BufferList> &bufferlist) {
  auto save_buffer = [&]() {
    if (data_map_.find(key) != data_map_.end()) {
      for (auto &buffer : *bufferlist) {
        data_map_[key]->PushBack(buffer);
      }
      bufferlist->Reset();
    }
  };

  if (extern_data_map_ == nullptr || extern_buffer_list_ == nullptr) {
    if (data_map_.find(key) == data_map_.end()) {
      data_map_[key] = std::make_shared<BufferList>();
    }

    if (data_map_[key]->Size() > MAX_INPUT_QUEUE_SIZE) {
      const auto *msg =
          "temp bufferlist store too many buffers,please use it firstly.";
      MBLOG_ERROR << msg;
      return {STATUS_INVALID, msg};
    }
    save_buffer();
  }

  if (extern_data_map_ != nullptr && extern_buffer_list_ != nullptr) {
    auto externdata =
        std::static_pointer_cast<ExternalDataMap>(extern_data_map_);
    auto flowunit_error = externdata->GetSessionContext()->GetError();
    if (flowunit_error) {
      auto error_msg = flowunit_error->GetDesc();
      return {STATUS_FAULT, error_msg};
    }
    if (data_map_.find(key) != data_map_.end() && data_map_[key]->Size() > 0) {
      save_buffer();
      auto status = externdata->Send(key, data_map_[key]);
      data_map_[key]->Reset();
    } else {
      auto status = externdata->Send(key, bufferlist);
      bufferlist->Reset();
    }
  }
  return STATUS_OK;
}

BufferListContext::BufferListContext(std::weak_ptr<ModelBoxEngine> env)
    : HandlerContext(env) {}

BufferListContext::~BufferListContext() {}

Status BufferListContext::RunGraph(
    const std::shared_ptr<DataHandler> &handler) {
  MBLOG_ERROR << "bufferlist handler not support next function.";
  return STATUS_FAULT;
}
Status BufferListContext::PushData(
    const std::string &key, const std::shared_ptr<BufferList> &bufferlist) {
  if (data_map_.find(key) == data_map_.end()) {
    data_map_[key] = std::make_shared<BufferList>();
  }

  for (auto &buffer : *bufferlist) {
    data_map_[key]->PushBack(buffer);
  }
  bufferlist->Reset();

  return STATUS_OK;
}

std::shared_ptr<BufferList> BufferListContext::GetBufferList(
    const std::string &key) {
  if (data_map_.find(key) == data_map_.end()) {
    return nullptr;
  }
  if (data_map_[key]->Size() <= 0) {
    return nullptr;
  }

  return data_map_[key];
}

StreamContext::StreamContext(std::weak_ptr<ModelBoxEngine> env)
    : HandlerContext(env) {
  end_flag_ = false;
}

StreamContext::~StreamContext() {}

std::shared_ptr<BufferList> StreamContext::GetBufferList(
    const std::string &key) {
  MBLOG_ERROR << "stream context not support get bufferlist.";
  return nullptr;
}

Status StreamContext::PushData(const std::string &key,
                               const std::shared_ptr<BufferList> &bufferlist) {
  return STATUS_FAULT;
}

Status StreamContext::RunGraph(const std::shared_ptr<DataHandler> &handler) {
  if (env_.lock() == nullptr) {
    MBLOG_ERROR << "env is nullptr, please check input is right";
    return STATUS_FAULT;
  }
  auto graph_state = GetGraphState();
  if (nullptr == graph_state) {
    MBLOG_ERROR << "graph state is nullptr, please set input as the first node";
    return STATUS_FAULT;
  }
  if (graph_state->graph_ == nullptr) {
    auto env = env_.lock();

    auto dynamic_graph = env->CreateDynamicGraph(graph_state->gcgraph_);

    if (STATUS_OK != env->FeedData(dynamic_graph, graph_state->gcgraph_)) {
      MBLOG_ERROR << "failed feed data into input";
      return STATUS_FAULT;
    }
    graph_state->graph_ = dynamic_graph;
  }
  return STATUS_OK;
}

}  // namespace modelbox