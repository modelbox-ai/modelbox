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


#ifndef MODELBOX_TENSOR_LIST_H_
#define MODELBOX_TENSOR_LIST_H_

#include <modelbox/base/device.h>
#include <modelbox/base/log.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer_list.h>
#include <modelbox/buffer_type.h>
#include <modelbox/tensor.h>
#include <modelbox/type.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <utility>

namespace modelbox {

/**
 * @brief Tensor list API
 */
class TensorList {
 public:
  /**
   * @brief Create tensor list from buffer list.
   * @param buffer_list buffer list
   */
  TensorList(std::shared_ptr<BufferList> buffer_list)
      : bl_(std::move(buffer_list)) {
    if (!bl_) {
      throw std::invalid_argument("buffer list must not be nullptr.");
    }

    for (auto& buffer : *bl_) {
      auto tensor = std::dynamic_pointer_cast<TensorBuffer>(buffer);
      if (!tensor) {
        throw std::invalid_argument(
            "the elements of bufferlist must be Tensorbuffer.");
      }
    }
  }

  virtual ~TensorList() = default;

  /**
   * @brief Set tensor list shape
   * @param shape shape list
   * @return set result
   */
  template <typename T>
  Status SetShape(const std::vector<std::vector<size_t>>& shape) {
    if (shape.size() != bl_->buffer_list_.size()) {
      return {STATUS_RANGE, "invalid shape size"};
    }

    Status status{STATUS_SUCCESS};
    auto iter = shape.begin();
    for (auto& buffer : bl_->buffer_list_) {
      auto tensor = std::dynamic_pointer_cast<TensorBuffer>(buffer);
      if (!tensor) {
        return STATUS_FAULT;
      }

      status = tensor->SetShape<T>(*(iter++));
      if (!status) {
        MBLOG_WARN << "tensor set shape failed: " << status;
        return STATUS_FAULT;
      }
    }

    return STATUS_OK;
  }

  /**
   * @brief Set tensor list data type
   * @param type data type
   */
  void SetType(ModelBoxDataType type);

  /**
   * @brief Get tensor list data type
   * @return data type
   */
  std::vector<std::vector<size_t>> GetShape() const;

  template <typename T>
  Status Build(const std::vector<std::vector<size_t>>& shape_list) {
    std::vector<size_t> data_size_list(shape_list.size(), 0);
    std::transform(shape_list.begin(), shape_list.end(), data_size_list.begin(),
                   [](const std::vector<size_t>& shape) {
                     return Volume(shape) * sizeof(T);
                   });

    size_t size = std::accumulate(data_size_list.begin(), data_size_list.end(),
                                  (size_t)0, std::plus<size_t>());
    if (!bl_->dev_mem_) {
      return {STATUS_INVALID, "device memory must not be nullptr."};
    }

    auto device = bl_->dev_mem_->GetDevice();
    bl_->dev_mem_ = device->MemAlloc(size);
    if (!bl_->dev_mem_) {
      MBLOG_WARN << " MemAlloc " << size << " byte data failed";
      return STATUS_NOMEM;
    }

    bl_->buffer_list_.resize(data_size_list.size(), nullptr);

    size_t offset = 0;
    for (size_t i = 0; i < bl_->buffer_list_.size(); i++) {
      auto&& mem = bl_->dev_mem_->Cut(offset, data_size_list[i]);
      bl_->buffer_list_[i] = std::make_shared<TensorBuffer>(mem);
      offset += data_size_list[i];
    }

    bl_->is_contiguous_ = true;
    return SetShape<T>(shape_list);
  }

  template <typename T>
  Status BuildFromHost(const std::vector<std::vector<size_t>>& shape_list,
                       void* data, size_t data_size,
                       const DeleteFunction& func = nullptr) {
    std::vector<size_t> data_size_list(shape_list.size(), 0);
    std::transform(shape_list.begin(), shape_list.end(), data_size_list.begin(),
                   [](const std::vector<size_t>& shape) {
                     return Volume(shape) * sizeof(T);
                   });

    size_t size = std::accumulate(data_size_list.begin(), data_size_list.end(),
                                  (size_t)0, std::plus<size_t>());
    if (data_size < size) {
      MBLOG_WARN << "invalid data size. size: " << size
                 << " data_size: " << data_size;
      return STATUS_RANGE;
    }

    if (!bl_->dev_mem_) {
      return {STATUS_INVALID, "device memory must not be nullptr."};
    }

    auto device = bl_->dev_mem_->GetDevice();
    if (bl_->dev_mem_->IsHost() && func) {
      std::shared_ptr<void> data_ptr(data, func);
      bl_->dev_mem_ = device->MemAcquire(data_ptr, data_size);
    } else {
      bl_->dev_mem_ = device->MemWrite(data, data_size);
      if (!bl_->dev_mem_) {
        MBLOG_WARN << " device MemWrite failed.";
        return STATUS_NOMEM;
      }
    }

    bl_->buffer_list_.resize(data_size_list.size(), nullptr);

    size_t offset = 0;
    for (size_t i = 0; i < bl_->buffer_list_.size(); i++) {
      auto&& mem = bl_->dev_mem_->Cut(offset, data_size_list[i]);
      bl_->buffer_list_[i] = std::make_shared<TensorBuffer>(mem);
      offset += data_size_list[i];
    }

    bl_->is_contiguous_ = true;
    return SetShape<T>(shape_list);
  }

  size_t Size() const;
  size_t GetBytes() const;

  std::shared_ptr<TensorBuffer> operator[](size_t pos);
  std::shared_ptr<const TensorBuffer> operator[](size_t pos) const;
  std::shared_ptr<TensorBuffer> At(size_t idx);
  std::shared_ptr<const TensorBuffer> At(size_t idx) const;
  void PushBack(const std::shared_ptr<TensorBuffer>& buf);

  template <typename T>
  T* MutableBufferData(size_t idx) {
    if (idx > bl_->buffer_list_.size()) {
      MBLOG_WARN << "invalid idx: " << idx
                 << " buff_vec_view_.size(): " << bl_->buffer_list_.size();
      return nullptr;
    }

    auto tensor = std::dynamic_pointer_cast<TensorBuffer>(bl_->At(idx));
    return tensor->MutableData<T>();
  }

  template <typename T>
  const T* ConstBufferData(size_t idx) const {
    if (idx > bl_->buffer_list_.size()) {
      MBLOG_WARN << "invalid idx: " << idx
                 << " buff_vec_view_.size(): " << bl_->buffer_list_.size();
      return nullptr;
    }

    auto tensor = std::dynamic_pointer_cast<TensorBuffer>(bl_->At(idx));
    return tensor->ConstData<T>();
  }

  template <typename T>
  T* MutableData() {
    return static_cast<T*>(bl_->MutableData());
  }

  template <typename T>
  const T* ConstData() const {
    return static_cast<const T*>(bl_->ConstData());
  }

  Status CopyMeta(const std::shared_ptr<TensorList>& tl,
                  bool is_override = false);

  template <typename T>
  void Set(const std::string& key, T&& value) {
    bl_->Set(key, value);
  }

 private:
  std::shared_ptr<BufferList> bl_;
};
}  // namespace modelbox

#endif