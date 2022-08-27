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

#ifndef MODELBOX_TENSOR_H_
#define MODELBOX_TENSOR_H_

#include <modelbox/buffer.h>
#include <modelbox/type.h>

namespace modelbox {

/**
 * @brief Interface to access the data buffer with tensor API
 */
class TensorBuffer : public Buffer {
 public:
  /**
   * @brief Tensor buffer object
   */
  TensorBuffer();

  /**
   * @brief Create a new tensor buffer related with specific device
   * @param device related device
   */
  TensorBuffer(const std::shared_ptr<Device>& device);

  /**
   * @brief Create a new tensor buffer related with specific device memory
   * @param dev_mem related device memory
   */
  TensorBuffer(const std::shared_ptr<DeviceMemory>& dev_mem);

  /**
   * @brief Copy from another tensor buffer
   * @param other another tensor buffer
   */
  TensorBuffer(const TensorBuffer& other);

  ~TensorBuffer() override;

  /**
   * @brief Resize tensor
   * @param shape shape list to resize
   * @return resize result
   */
  template <typename T>
  Status Resize(const std::vector<size_t>& shape) {
    auto new_size = Volume(shape) * sizeof(T);
    auto status = Build(new_size);
    if (!status) {
      MBLOG_WARN << "Resize failed.";
      return status;
    }

    shape_.assign(shape.begin(), shape.end());
    type_ = TypeToDataType<T>::Value;
    return STATUS_OK;
  }

  /**
   * @brief Get tensor buffer shape
   * @return tensor buffer shape list.
   */
  const std::vector<size_t>& Shape() const;

  /**
   * @brief Set shape to tensor buffer
   * @param shape shape list
   * @return set result
   */
  template <typename T>
  Status SetShape(const std::vector<size_t>& shape) {
    auto type = TypeToDataType<T>::Value;
    if (MODELBOX_TYPE_INVALID == type_) {
      type_ = type;
    } else if (type_ != type) {
      return {STATUS_INVALID, "invalid data type."};
    }

    if (Volume(shape) * sizeof(T) != GetBytes()) {
      return {STATUS_INVALID, "tensor size must be equal device memory size."};
    }

    shape_.assign(shape.begin(), shape.end());
    return STATUS_OK;
  }

  /**
   * @brief Set tensor buffer data type
   * @param type data type
   */
  void SetType(ModelBoxDataType type);

  /**
   * @brief Get tensor buffer data type
   * @return type data type
   */
  ModelBoxDataType GetType();

  /**
   * @brief Get tensor buffer mutable raw data
   * @return raw data pointer to tensor buffer data
   */
  template <typename T>
  T* MutableData() {
    auto type = TypeToDataType<T>::Value;
    if (type_ != type) {
      MBLOG_WARN << "invalid data type.";
      return nullptr;
    }

    auto device_mem = GetDeviceMemory();
    if (!device_mem) {
      MBLOG_WARN << "device_mem is nullptr, may be exception buffer.";
      return nullptr;
    }

    auto&& data = device_mem->GetPtr<T>();
    if (!data) {
      return nullptr;
    }

    return data.get();
  }

  /**
   * @brief Get tensor buffer const raw data
   * @return raw data pointer to tensor buffer data
   */
  template <typename T>
  const T* ConstData() const {
    auto type = TypeToDataType<T>::Value;
    if (type_ != type) {
      MBLOG_WARN << "invalid data type.";
      return nullptr;
    }

    auto device_mem = GetDeviceMemory();
    if (!device_mem) {
      MBLOG_WARN << "dev_mem_ is nullptr, may be exception buffer.";
      return nullptr;
    }

    auto&& data = device_mem->GetConstPtr<T>();
    if (!data) {
      return nullptr;
    }

    return data.get();
  }

  /**
   * @brief Create a copy of buffer share same data buffer
   * @return new buffer object
   */
  std::shared_ptr<Buffer> Copy() const override;

  /**
   * @brief Create a copy of buffer with new data buffer
   * @return new buffer object
   */
  std::shared_ptr<Buffer> DeepCopy() const override;

 protected:
  /**
   * @brief Create a copy of buffer with new data buffer
   * @param other tensor buffer
   * @return copy result
   */
  Status DeepCopy(const TensorBuffer& other);

 private:
  friend BufferList;

  /// tensor shape
  std::vector<size_t> shape_;

  /// tensor data type
  ModelBoxDataType type_{MODELBOX_TYPE_INVALID};
};

}  // namespace modelbox

#endif  // MODELBOX_TENSOR_H_