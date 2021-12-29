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

#include "modelbox/buffer_list.h"

namespace modelbox {

BufferList::BufferList() : is_contiguous_(false), dev_mem_(nullptr) {}

BufferList::BufferList(const std::shared_ptr<Device>& device,
                       uint32_t device_mem_flags)
    : is_contiguous_(false), dev_mem_flags_(device_mem_flags) {
  if (device) {
    dev_mem_ = device->MemAlloc(0, device_mem_flags);
  }
}

BufferList::BufferList(const std::shared_ptr<Buffer>& buffer) : BufferList() {
  buffer_list_.push_back(buffer);
}

BufferList::BufferList(
    const std::vector<std::shared_ptr<Buffer>>& buffer_vector)
    : BufferList() {
  buffer_list_.assign(buffer_vector.begin(), buffer_vector.end());
}

void BufferList::Copy(
    const std::vector<std::shared_ptr<Buffer>>& buffer_vector) {
  buffer_list_.assign(buffer_vector.begin(), buffer_vector.end());
}

Status BufferList::Reset() {
  buffer_list_.clear();
  is_contiguous_ = false;
  if (dev_mem_) {
    auto device = dev_mem_->GetDevice();
    dev_mem_ = device->MemAlloc(0, dev_mem_flags_);
    return dev_mem_ ? STATUS_SUCCESS : STATUS_FAULT;
  }

  return STATUS_OK;
}

size_t BufferList::Size() const { return buffer_list_.size(); }

size_t BufferList::GetBytes() const {
  size_t byte_size = 0;
  for (auto& buffer : buffer_list_) {
    byte_size += buffer->GetBytes();
  }

  return byte_size;
}

std::vector<std::shared_ptr<Buffer>>::iterator BufferList::begin() {
  return buffer_list_.begin();
}

std::vector<std::shared_ptr<Buffer>>::const_iterator BufferList::begin() const {
  return buffer_list_.begin();
}

std::vector<std::shared_ptr<Buffer>>::iterator BufferList::end() {
  return buffer_list_.end();
}

std::vector<std::shared_ptr<Buffer>>::const_iterator BufferList::end() const {
  return buffer_list_.end();
}

std::shared_ptr<Buffer>& BufferList::operator[](size_t pos) {
  return buffer_list_[pos];
}

const std::shared_ptr<Buffer>& BufferList::operator[](size_t pos) const {
  return buffer_list_[pos];
}

void BufferList::PushBack(const std::shared_ptr<Buffer>& buf) {
  buffer_list_.push_back(buf);
  SetNoContiguous();
}

void BufferList::Assign(
    const std::vector<std::shared_ptr<Buffer>>& buffer_list) {
  buffer_list_ = buffer_list;
  SetNoContiguous();
}

void BufferList::Swap(std::vector<std::shared_ptr<Buffer>>& buffer_list) {
  buffer_list_.swap(buffer_list);
  SetNoContiguous();
}

void BufferList::SetNoContiguous() {
  is_contiguous_ = false;
  if (dev_mem_) {
    auto device = dev_mem_->GetDevice();
    dev_mem_ = device->MemAlloc(0, dev_mem_flags_);
  }
}

bool BufferList::IsContiguous() const { return is_contiguous_; }

Status BufferList::SetMutable(bool is_mutable) {
  for (auto& buff : buffer_list_) {
    auto status = buff->SetBufferMutable(is_mutable);
    if (!status) {
      MBLOG_WARN << "SetBufferMutable failed:" << status;
      return status;
    }
  }

  if (dev_mem_) {
    return dev_mem_->SetContentMutable(is_mutable);
  }

  return STATUS_OK;
}

void* BufferList::MutableBufferData(size_t idx) {
  if (idx >= buffer_list_.size()) {
    MBLOG_WARN << "invalid idx: " << idx
               << " buff_vec_view_.size(): " << buffer_list_.size();
    return nullptr;
  }

  return buffer_list_[idx]->MutableData();
}

const void* BufferList::ConstBufferData(size_t idx) const {
  if (idx >= buffer_list_.size()) {
    MBLOG_WARN << "invalid idx: " << idx
               << " buff_vec_view_.size(): " << buffer_list_.size();
    return nullptr;
  }

  return buffer_list_[idx]->ConstData();
}

void* BufferList::MutableData() {
  auto size = Size();
  if (size == 1) {
    return MutableBufferData(0);
  }

  if (size > 1 && IsContiguous() && dev_mem_) {
    auto&& ptr = dev_mem_->GetPtr<void>();
    return !ptr ? nullptr : ptr.get();
  }

  return nullptr;
}

const void* BufferList::ConstData() const {
  auto size = Size();
  if (size == 1) {
    return ConstBufferData(0);
  }

  if (size > 1 && IsContiguous() && dev_mem_) {
    auto&& ptr = dev_mem_->GetConstPtr<void>();
    return !ptr ? nullptr : ptr.get();
  }

  return nullptr;
}

std::shared_ptr<Buffer> BufferList::At(size_t idx) {
  if (idx >= buffer_list_.size()) {
    return nullptr;
  }

  return buffer_list_.at(idx);
}

const std::shared_ptr<Buffer> BufferList::At(size_t idx) const {
  if (idx >= buffer_list_.size()) {
    return nullptr;
  }

  return buffer_list_.at(idx);
}

Status BufferList::CopyToNewBufferList(std::shared_ptr<DeviceMemory>& dev_mem) {
  size_t offset = 0, buff_size = 0;
  std::vector<std::shared_ptr<Buffer>> new_buffer_list;
  new_buffer_list.reserve(buffer_list_.size());
  for (auto& buffer : buffer_list_) {
    if (!buffer) {
      new_buffer_list.push_back(buffer);
      continue;
    }

    auto new_buffer = buffer->Copy();
    if (!new_buffer) {
      MBLOG_ERROR << "Buffer copy failed.";
      return STATUS_FAULT;
    }

    new_buffer_list.push_back(new_buffer);
    buff_size = new_buffer->GetBytes();
    if (0 == buff_size) {
      continue;
    }

    new_buffer->dev_mem_ = dev_mem->Cut(offset, buff_size);
    if (!new_buffer->dev_mem_) {
      MBLOG_ERROR << "device memory cut failed.";
      return STATUS_NOMEM;
    }

    offset += buff_size;
  }

  buffer_list_.swap(new_buffer_list);
  return STATUS_OK;
}

Status BufferList::GenerateDeviceMemory(
    const std::vector<std::shared_ptr<DeviceMemory>>& buffer_dev_mems) {
  bool is_contiguous = false;
  if (dev_mem_ && dev_mem_->IsSameDevice(buffer_dev_mems[0])) {
    is_contiguous = DeviceMemory::IsContiguous(buffer_dev_mems);
  }

  if (is_contiguous) {
    is_contiguous_ = true;
    auto device =
        dev_mem_ ? dev_mem_->GetDevice() : buffer_dev_mems[0]->GetDevice();
    auto dev_mem =
        DeviceMemory::Combine(buffer_dev_mems, device, dev_mem_flags_);
    if (!dev_mem) {
      MBLOG_ERROR << "DeviceMemory Combine failed.";
      return STATUS_NOMEM;
    }

    dev_mem_ = dev_mem;
    return STATUS_OK;
  }

  auto device =
      dev_mem_ ? dev_mem_->GetDevice() : buffer_dev_mems[0]->GetDevice();
  auto dev_mem = DeviceMemory::Combine(buffer_dev_mems, device, dev_mem_flags_);
  if (!dev_mem) {
    MBLOG_ERROR << "DeviceMemory Combine failed.";
    return STATUS_NOMEM;
  }

  auto ret = CopyToNewBufferList(dev_mem);
  if (ret != STATUS_OK) {
    return ret;
  }

  dev_mem_ = dev_mem;
  is_contiguous_ = true;
  return STATUS_OK;
}

Status BufferList::MakeContiguous() {
  std::vector<std::shared_ptr<DeviceMemory>> buffer_dev_mems;
  for (auto& buffer : buffer_list_) {
    if (buffer->HasError() || nullptr == buffer->dev_mem_) {
      continue;
    }

    buffer_dev_mems.push_back(buffer->dev_mem_);
  }

  if (0 == buffer_dev_mems.size()) {
    is_contiguous_ = true;
    return STATUS_OK;
  }

  auto ret = GenerateDeviceMemory(buffer_dev_mems);
  if (ret != STATUS_OK) {
    return ret;
  }

  return STATUS_OK;
}

std::shared_ptr<Device> BufferList::GetDevice() {
  return dev_mem_ ? dev_mem_->GetDevice() : nullptr;
}

std::shared_ptr<DeviceMemory> BufferList::GetDeviceMemory() {
  return dev_mem_;
};

Status BufferList::CopyMeta(const std::shared_ptr<BufferList>& bl,
                            bool is_override) {
  if (!bl || Size() != bl->Size()) {
    return STATUS_FAULT;
  }

  auto status = STATUS_OK;
  for (size_t i = 0; i < buffer_list_.size(); ++i) {
    status = buffer_list_[i]->CopyMeta(bl->At(i), is_override);
    if (!status) {
      MBLOG_WARN << "buffer list copt meta failed:" << status;
      return status;
    }
  }

  return STATUS_OK;
}

Status BufferList::BuildContiguous(std::shared_ptr<modelbox::Device> device,
                                   const std::vector<size_t>& data_size_list) {
  size_t size = std::accumulate(data_size_list.begin(), data_size_list.end(),
                                (size_t)0, std::plus<size_t>());
  auto mem = device->MemAlloc(size, dev_mem_flags_);
  if (!mem) {
    MBLOG_WARN << " MemAlloc " << size << " byte data failed";
    return STATUS_NOMEM;
  }

  dev_mem_ = mem;
  buffer_list_.resize(data_size_list.size(), nullptr);

  size_t offset = 0;
  for (size_t i = 0; i < buffer_list_.size(); i++) {
    auto&& mem = dev_mem_->Cut(offset, data_size_list[i]);
    buffer_list_[i] = std::make_shared<Buffer>(mem);
    offset += data_size_list[i];
  }

  is_contiguous_ = true;
  return STATUS_OK;
}

Status BufferList::BuildSeparate(std::shared_ptr<modelbox::Device> device,
                                 const std::vector<size_t>& data_size_list) {
  if (dev_mem_->GetSize() != 0) {
    dev_mem_ = device->MemAlloc(0, dev_mem_flags_);
  }

  buffer_list_.resize(data_size_list.size(), nullptr);
  for (size_t i = 0; i < buffer_list_.size(); ++i) {
    auto& size = data_size_list[i];
    buffer_list_[i] =
        std::make_shared<Buffer>(device->MemAlloc(size, dev_mem_flags_));
  }

  is_contiguous_ = false;
  return STATUS_OK;
}

Status BufferList::Build(const std::vector<size_t>& data_size_list,
                         bool contiguous) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  auto device = dev_mem_->GetDevice();
  if (device == nullptr) {
    return {STATUS_INVALID, "device is invalid"};
  }

  buffer_list_.clear();
  return contiguous ? BuildContiguous(device, data_size_list)
                    : BuildSeparate(device, data_size_list);
}

Status BufferList::BuildFromHost(const std::vector<size_t>& data_size_list,
                                 void* data, size_t data_size,
                                 DeleteFunction func) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  size_t size = std::accumulate(data_size_list.begin(), data_size_list.end(),
                                (size_t)0, std::plus<size_t>());
  if (data_size < size) {
    MBLOG_WARN << "invalid data size. size: " << size
               << " data_size: " << data_size;
    return STATUS_RANGE;
  }

  auto device = dev_mem_->GetDevice();
  if (dev_mem_->IsHost() && func) {
    std::shared_ptr<void> data_ptr(data, func);
    dev_mem_ = device->MemAcquire(data, data_size, func);
  } else {
    dev_mem_ = device->MemWrite(data, data_size);
    if (!dev_mem_) {
      MBLOG_WARN << " device MemWrite failed.";
      return STATUS_NOMEM;
    }
  }

  buffer_list_.resize(data_size_list.size(), nullptr);

  size_t offset = 0;
  for (size_t i = 0; i < buffer_list_.size(); i++) {
    auto&& mem = dev_mem_->Cut(offset, data_size_list[i]);
    buffer_list_[i] = std::make_shared<Buffer>(mem);
    offset += data_size_list[i];
  }

  is_contiguous_ = true;

  return STATUS_OK;
}

std::vector<std::shared_ptr<DeviceMemory>>
BufferList::GetAllBufferDeviceMemory() {
  std::vector<std::shared_ptr<DeviceMemory>> buffer_dev_mems;
  for (auto& buffer : buffer_list_) {
    if (buffer->HasError() || nullptr == buffer->dev_mem_) {
      continue;
    }

    buffer_dev_mems.push_back(buffer->dev_mem_);
  }
  return buffer_dev_mems;
}

Status BufferList::MoveAllBufferToTargetDevice() {
  if (buffer_list_.empty()) {
    return modelbox::STATUS_OK;
  }

  if (dev_mem_ == nullptr) {
    MBLOG_ERROR << "dev_mem of buffer_list should not be null";
    return modelbox::STATUS_FAULT;
  }

  std::vector<std::shared_ptr<Buffer>> new_buffer_list;
  auto buffer_count = buffer_list_.size();
  new_buffer_list.reserve(buffer_count);
  auto target_device = dev_mem_->GetDevice();
  if (target_device == nullptr) {
    MBLOG_ERROR << "target device is nullptr";
    return STATUS_FAULT;
  }

  for (auto& buffer : buffer_list_) {
    if (buffer == nullptr) {
      MBLOG_ERROR << "buffer in buffer list is nullptr";
      return modelbox::STATUS_FAULT;
    }

    if (dev_mem_->IsSameDevice(buffer->dev_mem_)) {
      // no need to copy real data.
      // Clone dev_mem due to old dev_mem might be used in multi node
      auto new_mem = buffer->dev_mem_->Clone();
      auto new_buffer = std::make_shared<Buffer>(new_mem);
      new_buffer->CopyMeta(buffer);
      new_buffer_list.push_back(new_buffer);
      continue;
    }

    auto data_size = buffer->GetBytes();
    auto new_buffer = std::make_shared<Buffer>(
        target_device->MemAlloc(data_size, dev_mem_flags_));
    new_buffer_list.push_back(new_buffer);
    new_buffer->CopyMeta(buffer);
    if (data_size == 0) {
      continue;
    }

    new_buffer->dev_mem_->ReadFrom(buffer->dev_mem_, 0, data_size);
  }

  buffer_list_.swap(new_buffer_list);
  return modelbox::STATUS_OK;
}

Status BufferList::EmplaceBack(void* device_data, size_t data_size,
                               DeleteFunction func) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  auto device = dev_mem_->GetDevice();
  auto buffer = std::make_shared<Buffer>(device, dev_mem_flags_);
  auto ret = buffer->Build(device_data, data_size, func);
  if (!ret) {
    return ret;
  }

  PushBack(buffer);
  return STATUS_OK;
}

Status BufferList::EmplaceBack(std::shared_ptr<void> device_data,
                               size_t data_size) {
  auto delete_func = [device_data](void*) { /* hold device data */ };
  return EmplaceBack(device_data.get(), data_size, delete_func);
}

Status BufferList::EmplaceBackFromHost(void* host_data, size_t data_size) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  auto device = dev_mem_->GetDevice();
  auto buffer = std::make_shared<Buffer>(device, dev_mem_flags_);
  auto ret = buffer->BuildFromHost(host_data, data_size);
  if (ret != STATUS_OK) {
    return ret;
  }

  PushBack(buffer);
  return STATUS_OK;
}

std::shared_ptr<Buffer> BufferList::Front() {
  if (buffer_list_.empty()) {
    return nullptr;
  }

  return buffer_list_.front();
}

std::shared_ptr<Buffer> BufferList::Back() {
  if (buffer_list_.empty()) {
    return nullptr;
  }

  return buffer_list_.back();
}

}  // namespace modelbox