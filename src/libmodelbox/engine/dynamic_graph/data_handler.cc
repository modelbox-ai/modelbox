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

#include "modelbox/data_handler.h"
#include "modelbox/modelbox_engine.h"

namespace modelbox {

DataHandler::DataHandler(BindNodeType type,
                         std::shared_ptr<ModelBoxEngine> env) {
  env_ = env;
  data_type_ = type;
  switch (data_type_) {
    case STREAM_NODE:
      context_ = std::make_shared<StreamContext>(env);
      break;
    case BUFFERLIST_NODE:
      context_ = std::make_shared<BufferListContext>(env);
      break;
    case VIRTUAL_NODE:
      context_ = std::make_shared<InputContext>(env);
      break;
    default:
      MBLOG_ERROR << "failed find right type";
  }
}

DataHandler::~DataHandler() { context_ = nullptr; }

void DataHandler::Close() {
  closed_ = true;
  if (context_) {
    context_->Close();
  }
}

bool DataHandler::IsClosed() { return closed_; }

void DataHandler::SetEnv(const std::shared_ptr<ModelBoxEngine> &env) {
  env_ = env;
}

std::shared_ptr<ModelBoxEngine> DataHandler::GetEnv() { return env_.lock(); }

DataHandlerType DataHandler::GetDataHandlerType() { return data_handler_type_; }

void DataHandler::SetDataHandlerType(const DataHandlerType &type) {
  data_handler_type_ = type;
}

std::shared_ptr<GraphState> DataHandler::GetBindGraph() {
  if (context_ == nullptr) {
    return nullptr;
  }
  return context_->GetGraphState();
}

Status DataHandler::SetBindGraph(const std::shared_ptr<GraphState> &gcgraph) {
  if (context_ == nullptr) {
    MBLOG_ERROR << "context_ is null, SetBindGraph failed";
    return STATUS_FAULT;
  }
  context_->SetGraphState(gcgraph);
  return STATUS_SUCCESS;
}

void DataHandler::SetNodeName(const std::string &name) { node_name_ = name; }

std::string DataHandler::GetNodeName() { return node_name_; }

std::set<std::string> DataHandler::GetPortNames() { return port_names_; }

BindNodeType DataHandler::GetBindNodeType() { return data_type_; }

void DataHandler::SetBindNodeType(BindNodeType type) { data_type_ = type; }

void DataHandler::SetExternData(std::shared_ptr<void> extern_map,
                                std::shared_ptr<BufferList> &bufferlist) {
  if (context_ == nullptr) {
    MBLOG_ERROR << "context_ is null, SetExternData failed";
    return;
  }
  if (data_type_ == VIRTUAL_NODE) {
    auto context = std::static_pointer_cast<InputContext>(context_);
    context->SetExternPtr(extern_map, bufferlist);
  }
}

Status DataHandler::PushData(std::shared_ptr<DataHandler> &data,
                             const std::string key) {
  if (GetDataHandlerType() != INPUT) {
    MBLOG_ERROR << "only input data can receive other data";
    return STATUS_FAULT;
  }

  if (data->GetPortNames().size() > 0) {
    auto port_name1 = *(data->GetPortNames().begin());
  } else {
    MBLOG_ERROR << "port name is nullptr";
    return STATUS_FAULT;
  }
  auto port_name = *(data->GetPortNames().begin());
  auto bufferlist = data->GetBufferList(port_name);
  return context_->PushData(key, bufferlist);
}

Status DataHandler::PushData(std::shared_ptr<Buffer> &data,
                             const std::string key) {
  if (context_ == nullptr) {
    MBLOG_ERROR << "context is nullptr, datahanler init failed";
    return STATUS_FAULT;
  }
  port_names_.emplace(key);
  auto bufferlist = std::make_shared<BufferList>(data);
  return context_->PushData(key, bufferlist);
}

Status DataHandler::PushData(std::shared_ptr<BufferList> &data,
                             const std::string key) {
  if (context_ == nullptr) {
    MBLOG_ERROR << "context is nullptr, datahanler init failed";
    return STATUS_FAULT;
  }
  port_names_.emplace(key);
  return context_->PushData(key, data);
}

Status DataHandler::SetMeta(const std::string &key, const std::string &data) {
  if (context_ == nullptr) {
    return STATUS_FAULT;
  }
  if (key == "" || data == "") {
    MBLOG_ERROR << "input key or value is invalid";
    return STATUS_FAULT;
  }
  context_->SetMeta(key, data);

  return STATUS_OK;
}

std::shared_ptr<DataHandler> DataHandler::GetDataHandler(
    const std::string &key) {
  if (data_type_ == VIRTUAL_NODE) {
    MBLOG_ERROR << "input node not support GetDataHandler function";
    return nullptr;
  }
  if (port_names_.find(key) == port_names_.end()) {
    MBLOG_ERROR << "faild find port name: " << key
                << " in node: " << node_name_;
    return nullptr;
  }
  auto data = std::make_shared<DataHandler>(data_type_);

  data->SetNodeName(node_name_);
  auto graph_state = GetBindGraph();
  data->SetBindGraph(graph_state);
  std::set<std::string> ports = {key};
  data->SetPortNames(ports);
  if (data_type_ == BUFFERLIST_NODE) {
    auto bufferlist = data->GetBufferList(key);
    data->context_->PushData(key, bufferlist);
  }
  return data;
}

Status DataHandler::SetDataHandler(
    const std::map<std::string, std::shared_ptr<DataHandler>> &data_map) {
  if (GetBindNodeType() == BUFFERLIST_NODE) {
    MBLOG_ERROR
        << "function SetDataHandler not support node type: bufferlistnode";
    return STATUS_FAULT;
  }

  for (auto &iter : data_map) {
    if (iter.second->GetBindNodeType() == BUFFERLIST_NODE) {
      MBLOG_ERROR
          << "function SetDataHandler not support node type: bufferlistnode";
      return STATUS_FAULT;
    }

    auto ports = iter.second->GetPortNames();
    if (ports.size() != 1) {
      std::string err_msg = "input data handler has one more ports";
      return {STATUS_FAULT, err_msg};
    }

    auto in_port_name = iter.first;
    auto temp_data = iter.second;
    auto node_name = temp_data->GetNodeName();
    auto out_port_name = *(ports.begin());
    context_->data_map_[in_port_name] = temp_data->GetBufferList(out_port_name);
    port_to_port_[in_port_name] = out_port_name;

    port_to_node_[in_port_name] = node_name;
    node_type_map_[node_name] = temp_data->GetBindNodeType();
    if (GetBindGraph() == nullptr) {
      SetBindGraph(iter.second->GetBindGraph());
    }
    if (GetBindGraph() != iter.second->GetBindGraph()) {
      std::string msg = "sub datahandler bind different graph";
      MBLOG_ERROR << msg;
      return {STATUS_FAULT, msg};
    }
  }
  return STATUS_SUCCESS;
}

Status DataHandler::CheckInputType(BindNodeType &node_type) {
  if (node_type_map_.size() == 0) {
    return STATUS_OK;
  }

  auto type = node_type_map_.begin()->second;
  for (auto &iter : node_type_map_) {
    if (type != iter.second) {
      return STATUS_FAULT;
    }
  }
  node_type = type;
  return STATUS_OK;
}

std::shared_ptr<BufferList> DataHandler::GetBufferList(const std::string &key) {
  if (context_ == nullptr || data_type_ != BUFFERLIST_NODE) {
    return nullptr;
  }

  return context_->GetBufferList(key);
}

std::shared_ptr<DataHandler> DataHandler::operator[](
    const std::string &port_name) {
  return GetDataHandler(port_name);
}

Status DataHandler::InsertOutputNode(std::shared_ptr<HandlerContext> &context) {
  std::shared_ptr<FlowUnitDesc> desc = context->GetFlowUnitDesc();
  if (desc == nullptr) {
    return STATUS_FAULT;
  }

  auto gcgraph = context->GetGraphState()->gcgraph_;
  auto node = context->GetGraphState()->gcgraph_->GetNode(node_name_);
  auto outputs = desc->GetFlowUnitOutput();
  if (outputs.size() > 0) {
    for (auto &iter : outputs) {
      auto outport_name = iter.GetPortName();  //
      auto outnode = std::make_shared<GCNode>();
      if (context->GetGraphState()->gcgraph_) {
        outnode->Init(outport_name, gcgraph);
        outnode->SetConfiguration("type", "output");
        context->GetGraphState()->gcgraph_->AddNode(outnode);

        env_.lock()->InsertGrahEdge(gcgraph, node, outport_name, outnode,
                                    outport_name);
      }
    }
  }
  return STATUS_OK;
}

std::shared_ptr<DataHandler> DataHandler::GetData() {
  if (context_ == nullptr || GetBindNodeType() != STREAM_NODE) {
    return nullptr;
  }

  if (context_->GetGraphState()->graph_ == nullptr) {
    if (InsertOutputNode(context_) != STATUS_OK) {
      return nullptr;
    }

    auto status = context_->RunGraph(shared_from_this());
    if (status != STATUS_OK) {
      return nullptr;
    }
  }

  OutputBufferList map_buffer_list;
  auto external_data = std::static_pointer_cast<ExternalDataMap>(
      context_->GetGraphState()->external_data_);
  auto status = external_data->Recv(map_buffer_list);
  if (status != STATUS_SUCCESS) {
    return nullptr;
  }

  auto buffer = std::make_shared<DataHandler>(BUFFERLIST_NODE);
  for (auto iter = map_buffer_list.begin(); iter != map_buffer_list.end();
       iter++) {
    auto temp_buffer = iter->second;
    buffer->PushData(temp_buffer, iter->first);
  }
  return buffer;
}
Status DataHandler::SetPortNames(std::set<std::string> &port_names) {
  port_names_ = port_names;
  return STATUS_OK;
}

std::string DataHandler::GetMeta(std::string &key) {
  if (context_ == nullptr) {
    return "";
  }
  return context_->GetMeta(key);
}

}  // namespace modelbox
