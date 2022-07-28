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


#include <modelbox/tensor.h>

namespace modelbox {

TensorBuffer::TensorBuffer() {}

TensorBuffer::TensorBuffer(const std::shared_ptr<Device>& device)
    : Buffer(device) {}
TensorBuffer::TensorBuffer(const std::shared_ptr<DeviceMemory>& dev_mem)
    : Buffer(dev_mem) {}

TensorBuffer::TensorBuffer(const TensorBuffer& other)
    : Buffer(other), shape_(other.shape_), type_(other.type_) {}

TensorBuffer::~TensorBuffer() {}

const std::vector<size_t>& TensorBuffer::Shape() const { return shape_; }

std::shared_ptr<Buffer> TensorBuffer::Copy() const {
  return std::make_shared<TensorBuffer>(*this);
}

std::shared_ptr<Buffer> TensorBuffer::DeepCopy() const {
  auto tensor = std::make_shared<TensorBuffer>();
  auto status = tensor->DeepCopy(*this);
  if (!status) {
    MBLOG_WARN << "TensorBuffer DeepCopy failed: " << status;
    return nullptr;
  }

  tensor->shape_ = shape_;
  tensor->type_ = type_;
  return tensor;
}

Status TensorBuffer::DeepCopy(const TensorBuffer& other) {
  const auto* other_buffer = dynamic_cast<const Buffer*>(&other);
  if (other_buffer == nullptr) {
    return {STATUS_INVALID, "tensor buffer is invalid."};
  }

  auto status = Buffer::DeepCopy(*(other_buffer));
  if (!status) {
    return status;
  }

  shape_ = other.shape_;
  type_ = other.type_;

  return STATUS_OK;
}

}  // namespace modelbox