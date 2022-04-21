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

#include <modelbox/external_data_simple.h>
#include <securec.h>

namespace modelbox {
ExternalDataSimple::ExternalDataSimple(
    std::shared_ptr<ExternalDataMap>& data_map)
    : data_map_(data_map) {}

ExternalDataSimple::~ExternalDataSimple() {
  if (data_map_) {
    data_map_->Close();
  }
}

std::shared_ptr<BufferList> ExternalDataSimple::CreateBufferList() {
  if (data_map_) {
    return data_map_->CreateBufferList();
  }
  return nullptr;
}

Status ExternalDataSimple::PushData(const std::string& port_name,
                                    std::shared_ptr<BufferList>& bufferlist) {
  if (data_map_ == nullptr) {
    return STATUS_INVALID;
  }

  auto temp = data_map_->CreateBufferList();
  if (temp->GetDevice() != bufferlist->GetDevice()) {
    MBLOG_ERROR << "pushed buffer is on different device";
    return STATUS_INVALID;
  }

  auto status = data_map_->Send(port_name, bufferlist);
  if (!status) {
    MBLOG_ERROR << "failed send data to graph: " << status.Errormsg();
    return status;
  }

  return data_map_->Shutdown();
}

Status ExternalDataSimple::PushData(
    const std::string& port_name, const void* data, const size_t& data_len,
    const std::map<std::string, std::string>& meta) {
  if ((data_map_ == nullptr) || ((data == nullptr) && (data_len != 0))) {
    MBLOG_ERROR
        << "push data failed,because data map is null or data is nullptr";
    return STATUS_FAULT;
  }

  Status status = STATUS_OK;
  auto input_buffer = data_map_->CreateBufferList();

  if (data_len == 0) {
    status = input_buffer->Build({1});
  } else {
    status = input_buffer->Build({data_len});
  }

  if (status != STATUS_OK) {
    MBLOG_ERROR << "failed build buffer for len :" << data_len;
    return STATUS_FAULT;
  }

  auto buffer = input_buffer->At(0);
  auto buffer_data = buffer->MutableData();
  if (data_len > 0) {
    auto ret = memcpy_s(buffer_data, data_len, data, data_len);
    if (ret < 0) {
      MBLOG_ERROR << "copy data to external buffer failed.";
      return STATUS_FAULT;
    }
  }

  for (auto& iter : meta) {
    buffer->Set(iter.first, iter.second);
  }

  status = data_map_->Send(port_name, input_buffer);
  if (!status) {
    MBLOG_ERROR << "failed send data to graph: " << status.Errormsg();
    return status;
  }
  
  return data_map_->Shutdown();
}

Status ExternalDataSimple::GetResult(const std::string& port_name,
                                     std::shared_ptr<Buffer>& buffer,
                                     const int& timeout) {
  if (buffer_list_map_[port_name].size() == 0) {
    if (status_ != STATUS_OK) {
      return status_;
    }

    OutputBufferList map_buffer_list;
    status_ = data_map_->Recv(map_buffer_list, timeout);
    Defer {
      if (status_ != STATUS_SUCCESS) {
        data_map_->Close();
      }
    };

    if (status_ != STATUS_SUCCESS) {
      MBLOG_ERROR << "recv failed, error is " << data_map_->GetLastError();
      return status_;
    }

    for (auto& iter : map_buffer_list) {
      auto buffers = std::vector<std::shared_ptr<Buffer>>(iter.second->begin(),
                                                          iter.second->end());
      auto temp_buffer = data_map_->CreateBufferList();
      temp_buffer->Assign(buffers);
      temp_buffer->MoveAllBufferToTargetDevice();
      for (auto buffer_iter = temp_buffer->begin();
           buffer_iter != temp_buffer->end(); buffer_iter++) {
        buffer_list_map_[iter.first].push(*buffer_iter);
      }
    }
  }

  buffer = buffer_list_map_[port_name].front();
  buffer_list_map_[port_name].pop();
  return STATUS_OK;
}

Status ExternalDataSimple::GetResult(const std::string& port_name,
                                     std::shared_ptr<void>& data, size_t& len,
                                     const int& timeout) {
  std::shared_ptr<Buffer> buffer;
  auto status = GetResult(port_name, buffer, timeout);
  if (status != STATUS_OK) {
    return status;
  }

  len = buffer->GetBytes();
  data.reset(buffer->MutableData(), [buffer](void* data) {});

  return STATUS_OK;
}

void ExternalDataSimple::Close() {
  if (data_map_) {
    data_map_->Close();
    data_map_ = nullptr;
  }
}

}  // namespace modelbox
