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

#include "modelbox/buffer.h"

namespace modelbox {

BufferMeta::BufferMeta() : error_(nullptr) {}

BufferMeta::~BufferMeta(){};

BufferMeta& BufferMeta::SetError(const std::shared_ptr<FlowUnitError>& e) {
  error_ = e;
  return *this;
}

const std::shared_ptr<FlowUnitError>& BufferMeta::GetError() const {
  return error_;
}

Status BufferMeta::CopyMeta(const std::shared_ptr<BufferMeta> buf_meta,
                            bool is_override) {
  if (!buf_meta) {
    return STATUS_FAULT;
  }

  custom_meta_.Merge(buf_meta->custom_meta_, is_override);

  return STATUS_OK;
}

BufferMeta& BufferMeta::operator=(const BufferMeta& other) {
  custom_meta_ = other.custom_meta_;
  error_ = other.error_;
  return *this;
}

BufferMeta& BufferMeta::DeepCopy(const BufferMeta& other) {
  custom_meta_ = other.custom_meta_;
  error_ = nullptr;
  if (other.error_) {
    error_ = std::make_shared<FlowUnitError>(*(other.error_));
  }

  return *this;
}

Buffer::Buffer() : dev_mem_(nullptr) { meta_ = std::make_shared<BufferMeta>(); }

Buffer::Buffer(const std::shared_ptr<Device>& device, uint32_t dev_mem_flags)
    : Buffer() {
  dev_mem_flags_ = dev_mem_flags;
  if (device) {
    dev_mem_ = device->MemAlloc(0);
  }
}

Buffer::Buffer(const std::shared_ptr<DeviceMemory>& dev_mem) : Buffer() {
  dev_mem_ = dev_mem;
}

Buffer::Buffer(const Buffer& other) : Buffer() {
  meta_->CopyMeta(other.meta_);
  dev_mem_ = other.dev_mem_;
  type_ = other.type_;
  dev_mem_flags_ = other.dev_mem_flags_;
}

Status Buffer::Build(size_t size) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "Can't get device!"};
  }

  auto&& device = dev_mem_->GetDevice();
  dev_mem_ = device->MemAlloc(size, dev_mem_flags_);

  if (nullptr == dev_mem_) {
    MBLOG_WARN << device << " MemAlloc " << size << " byte data failed!";
    return STATUS_NOMEM;
  }

  return STATUS_OK;
}

Status Buffer::Build(void* data, size_t data_size, DeleteFunction func) {
  if (!dev_mem_) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  if (!func) {
    return {STATUS_INVALID, "DeleteFunction must not be nullptr."};
  }

  auto device = dev_mem_->GetDevice();
  auto dev_mem = device->MemAcquire(data, data_size, func, dev_mem_flags_);
  if (!dev_mem) {
    return {STATUS_NOMEM, "device MemAcquire failed."};
  }

  dev_mem_ = dev_mem;
  return STATUS_OK;
}

Status Buffer::BuildFromHost(void* data, size_t data_size,
                             DeleteFunction func) {
  if (!dev_mem_ && !func) {
    return {STATUS_INVALID, "device memory must not be nullptr."};
  }

  auto device = dev_mem_->GetDevice();
  std::shared_ptr<DeviceMemory> dev_mem = nullptr;
  if (dev_mem_->IsHost() && func) {
    dev_mem = device->MemAcquire(data, data_size, func);
  } else {
    dev_mem = device->MemWrite(data, data_size);
    if (!dev_mem) {
      MBLOG_WARN << " device MemWrite failed.";
      return STATUS_NOMEM;
    }
  }

  dev_mem_ = dev_mem;

  return STATUS_OK;
}

void* Buffer::MutableData() {
  if (!dev_mem_) {
    MBLOG_WARN << "dev_mem_ is nullptr, may be exception buffer.";
    return nullptr;
  }

  auto&& data = dev_mem_->GetPtr<void>();
  if (!data) {
    return nullptr;
  }

  return data.get();
}

const void* Buffer::ConstData() const {
  if (!dev_mem_) {
    MBLOG_WARN << "dev_mem_ is nullptr, may be exception buffer.";
    return nullptr;
  }

  auto&& data = dev_mem_->GetConstPtr<void>();
  if (!data) {
    return nullptr;
  }

  return data.get();
}

Status Buffer::SetBufferMutable(bool is_mutable) {
  if (!dev_mem_) {
    return STATUS_OK;
  }

  return dev_mem_->SetContentMutable(is_mutable);
}

Buffer& Buffer::SetError(const std::shared_ptr<FlowUnitError>& error) {
  meta_->SetError(error);
  dev_mem_ = nullptr;
  return *this;
}

bool Buffer::HasError() const { return nullptr != meta_->GetError(); }
const std::shared_ptr<FlowUnitError>& Buffer::GetError() const {
  return meta_->GetError();
}

size_t Buffer::GetBytes() const { return dev_mem_ ? dev_mem_->GetSize() : 0; }

std::shared_ptr<Device> Buffer::GetDevice() const {
  return dev_mem_ ? dev_mem_->GetDevice() : nullptr;
}

Status Buffer::CopyMeta(const std::shared_ptr<Buffer> buf, bool is_override) {
  if (!buf) {
    return {STATUS_INVALID, "buffer must not be nullptr."};
  }

  auto status = meta_->CopyMeta(buf->meta_, is_override);
  if (!status) {
    MBLOG_WARN << "buffer meta set meta failed." << status;
  }

  return status;
}

std::shared_ptr<Buffer> Buffer::Copy() const {
  return std::make_shared<Buffer>(*this);
}

std::shared_ptr<Buffer> Buffer::DeepCopy() const {
  auto buffer = std::make_shared<Buffer>();
  auto status = buffer->DeepCopy(*this);
  if (!status) {
    MBLOG_ERROR << "Buffer DeepCopy failed: " << status;
    return nullptr;
  }

  return buffer;
}

std::shared_ptr<Buffer> Buffer::CopyTo(
    const std::shared_ptr<Device>& dest_device) const {
  if (dest_device == nullptr) {
    return nullptr;
  }

  auto new_buffer = std::make_shared<Buffer>(dest_device);
  auto status = new_buffer->DeepCopy(*this);
  if (!status) {
    MBLOG_ERROR << "Buffer DeepCopy failed: " << status;
    return nullptr;
  }

  return new_buffer;
}

BufferEnumType Buffer::GetBufferType() const { return type_; }

void Buffer::SetGetBufferType(BufferEnumType type) { type_ = type; }

std::shared_ptr<DeviceMemory> Buffer::GetDeviceMemory() const {
  return dev_mem_;
};

Status Buffer::DeepCopy(const Buffer& other) {
  if (other.meta_) {
    meta_ = std::make_shared<BufferMeta>();
    meta_->DeepCopy(*(other.meta_));
  } else {
    meta_ = nullptr;
  }

  if (!other.dev_mem_) {
    dev_mem_ = nullptr;
    return STATUS_OK;
  }

  if (dev_mem_ == nullptr) {
    dev_mem_ = other.dev_mem_->Clone(true);
  } else {
    auto&& device = dev_mem_->GetDevice();
    dev_mem_ = device->MemAlloc(other.dev_mem_->GetSize());
    dev_mem_->ReadFrom(other.dev_mem_, 0, other.dev_mem_->GetSize());
  }

  if (!dev_mem_) {
    return {STATUS_NOMEM, "device memory copy failed."};
  }

  return STATUS_OK;
}

}  // namespace modelbox