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

#include <utility>

#include "modelbox/flow_stream_io.h"

namespace modelbox {
FlowStreamIO::FlowStreamIO(std::shared_ptr<ExternalDataMap> data_map)
    : data_map_(std::move(data_map)) {}

FlowStreamIO::~FlowStreamIO() { data_map_->Shutdown(); }

std::shared_ptr<Buffer> FlowStreamIO::CreateBuffer() {
  auto buffer_list = data_map_->CreateBufferList();
  return std::make_shared<Buffer>(buffer_list->GetDevice());
}

Status FlowStreamIO::Send(const std::string &input_name,
                          const std::shared_ptr<Buffer> &buffer) {
  auto buffer_list = data_map_->CreateBufferList();
  buffer_list->PushBack(buffer);
  return data_map_->Send(input_name, buffer_list);
}

Status FlowStreamIO::Recv(const std::string &output_name,
                          std::shared_ptr<Buffer> &buffer, size_t timeout) {
  auto port_data_cache_item = port_data_cache_map_.find(output_name);
  if (port_data_cache_item == port_data_cache_map_.end() ||
      port_data_cache_item->second.empty()) {
    OutputBufferList map_buffer_list;
    auto status = data_map_->Recv(map_buffer_list, timeout);

    if (!status) {
      MBLOG_ERROR << "Recv data failed, ret " << status;
      return status;
    }

    for (auto &port_item : map_buffer_list) {
      const auto &port_name = port_item.first;
      auto &port_buffer_list = port_item.second;
      auto &data_cache = port_data_cache_map_[port_name];
      data_cache.insert(data_cache.end(), port_buffer_list->begin(),
                        port_buffer_list->end());
    }
  }

  buffer = port_data_cache_map_[output_name].front();
  port_data_cache_map_[output_name].pop_front();
  return STATUS_OK;
}

void FlowStreamIO::CloseInput() { data_map_->Close(); }

}  // namespace modelbox