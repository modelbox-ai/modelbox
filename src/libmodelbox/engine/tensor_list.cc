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


#include <modelbox/tensor_list.h>

namespace modelbox {
void TensorList::SetType(ModelBoxDataType type) {
  for (auto& buffer : bl_->buffer_list_) {
    auto tensor = std::dynamic_pointer_cast<TensorBuffer>(buffer);
    tensor->SetType(type);
  }
}

const std::vector<std::vector<size_t>> TensorList::GetShape() const {
  std::vector<std::vector<size_t>> shapes(bl_->buffer_list_.size());
  size_t i = 0;
  for (auto& buffer : bl_->buffer_list_) {
    if (buffer) {
      auto tensor = std::dynamic_pointer_cast<TensorBuffer>(buffer);
      const auto& shape = tensor->Shape();
      shapes[i++] = shape;
    }
  }

  return shapes;
}

size_t TensorList::Size() const { return bl_->Size(); }

size_t TensorList::GetBytes() const { return bl_->GetBytes(); }

std::shared_ptr<TensorBuffer> TensorList::operator[](size_t pos) {
  if (!bl_->At(pos)) {
    return nullptr;
  }

  return std::dynamic_pointer_cast<TensorBuffer>(bl_->At(pos));
}

std::shared_ptr<const TensorBuffer> TensorList::operator[](size_t pos) const {
  if (!bl_->At(pos)) {
    return nullptr;
  }

  return std::dynamic_pointer_cast<TensorBuffer>(bl_->At(pos));
};

std::shared_ptr<TensorBuffer> TensorList::At(size_t idx) {
  if (!bl_->At(idx)) {
    return nullptr;
  }

  return std::dynamic_pointer_cast<TensorBuffer>(bl_->At(idx));
}

std::shared_ptr<const TensorBuffer> TensorList::At(size_t idx) const {
  if (!bl_->At(idx)) {
    return nullptr;
  }

  return std::dynamic_pointer_cast<TensorBuffer>(bl_->At(idx));
}

void TensorList::PushBack(const std::shared_ptr<TensorBuffer>& buf) {
  bl_->PushBack(buf);
}

Status TensorList::CopyMeta(const std::shared_ptr<TensorList>& tl,
                            bool is_override) {
  if (!tl || Size() != tl->Size()) {
    return STATUS_FAULT;
  }

  return bl_->CopyMeta(tl->bl_, is_override);
}

}  // namespace modelbox