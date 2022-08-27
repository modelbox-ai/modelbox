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

#include "modelbox/buffer_type.h"

#include <utility>

#include "modelbox/base/log.h"
namespace modelbox {
BufferType::BufferType() = default;
BufferType::BufferType(std::string type) : type_(std::move(type)) {}
BufferType::~BufferType() = default;

std::shared_ptr<BufferTypeTree> BufferTypeTree::instance_(nullptr);

void BufferType::SetType(std::string type) { type_ = std::move(type); }

const std::string& BufferType::GetType() const { return type_; }

bool BufferType::AddParentType(const std::shared_ptr<BufferType>& parent) {
  if (parent == nullptr) {
    return false;
  }

  if (parent_ != nullptr) {
    return false;
  }

  parent_ = parent;
  return true;
}

bool BufferType::AddChildType(const std::shared_ptr<BufferType>& child) {
  bool reuslt = false;
  if (child == nullptr) {
    return false;
  }

  bool add_flag = true;
  auto type = child->GetType();
  for (const auto& own_child : children_) {
    if (own_child->GetType() == type) {
      add_flag = false;
      break;
    }
  }

  if (add_flag) {
    children_.push_back(child);
    reuslt = true;
  }

  return reuslt;
}

void BufferType::ClearChildType() { children_.clear(); }

void BufferType::ClearParentType() { parent_ = nullptr; }

void BufferType::RemoveType() {
  if (parent_ != nullptr) {
    auto children = parent_->GetChildrenType();
    std::vector<std::shared_ptr<BufferType>> keep_children;
    for (const auto& child : children) {
      if (child->GetType() != this->GetType()) {
        keep_children.push_back(child);
      }
    }

    parent_->ClearChildType();
    for (const auto& keep_child : keep_children) {
      parent_->AddChildType(keep_child);
    }
    parent_ = nullptr;
  }
  children_.clear();
}

bool BufferType::IsAncestor(const BufferType& other) {
  const auto& type = other.GetType();

  for (const auto& child : children_) {
    if (child->GetType() == type) {
      return true;
    }

    if (IsAncestor(*(child.get()))) {
      return true;
    }
  }

  return false;
}

bool BufferType::IsOffspring(const BufferType& other) {
  const auto& type = other.GetType();

  if (parent_ == nullptr) {
    return false;
  }

  if (parent_->GetType() == type) {
    return true;
  }

  return parent_->IsOffspring(other);
}

std::shared_ptr<BufferType> BufferType::GetParentType() { return parent_; }

std::vector<std::shared_ptr<BufferType>> BufferType::GetChildrenType() {
  return children_;
}

BufferTypeTree::BufferTypeTree() = default;

BufferTypeTree::~BufferTypeTree() = default;

bool BufferTypeTree::AddRootType(const std::string& root_type) {
  std::shared_ptr<BufferType> root_buffer_type_ptr = GetType(root_);

  if (root_buffer_type_ptr == nullptr) {
    auto* root_buffer_type = new BufferType();
    root_buffer_type->SetType(root_type);
    root_buffer_type_ptr.reset(root_buffer_type);
    nodes_.insert(std::make_pair(root_type, root_buffer_type_ptr));
    root_ = root_type;
    return true;
  }

  if (root_ == root_type) {
    return true;
  }

  return false;
}

bool BufferTypeTree::AddType(const std::string& type,
                             const std::string& parent_type) {
  std::shared_ptr<BufferType> child_buffer_type_ptr = GetType(type);
  std::shared_ptr<BufferType> parent_buffer_type_ptr = GetType(parent_type);

  if (parent_buffer_type_ptr == nullptr) {
    return false;
  }

  if (child_buffer_type_ptr != nullptr) {
    return false;
  }

  auto* child_buffer_type = new BufferType();
  child_buffer_type->SetType(type);
  child_buffer_type_ptr.reset(child_buffer_type);
  if (!parent_buffer_type_ptr->AddChildType(child_buffer_type_ptr)) {
    return false;
  }

  if (!child_buffer_type_ptr->AddParentType(parent_buffer_type_ptr)) {
    return false;
  }

  nodes_.insert(std::make_pair(type, child_buffer_type_ptr));
  return true;
}

std::shared_ptr<BufferType> BufferTypeTree::GetType(const std::string& type) {
  std::shared_ptr<BufferType> buffer_type = nullptr;
  for (const auto& node : nodes_) {
    if (node.first == type) {
      buffer_type = node.second;
      break;
    }
  }

  return buffer_type;
}

bool BufferTypeTree::RemoveType(const std::string& type) {
  std::shared_ptr<BufferType> buffer_type_ptr = GetType(type);

  if (buffer_type_ptr == nullptr) {
    return false;
  }

  for (const auto& child_type : buffer_type_ptr->GetChildrenType()) {
    RemoveType(child_type->GetType());
  }
  buffer_type_ptr->RemoveType();
  nodes_.erase(type);

  return true;
}

bool BufferTypeTree::IsCompatible(const std::string& type,
                                  const std::string& ancestor_type) {
  std::shared_ptr<BufferType> buffer_type_ptr = GetType(type);
  std::shared_ptr<BufferType> ancestor_buffer_type_ptr = GetType(ancestor_type);

  if (buffer_type_ptr == nullptr || ancestor_buffer_type_ptr == nullptr) {
    return false;
  }

  if (buffer_type_ptr == ancestor_buffer_type_ptr) {
    return true;
  }

  return ancestor_buffer_type_ptr->IsAncestor(*(buffer_type_ptr.get()));
}

BufferTypeTree* BufferTypeTree::GetInstance() {
  if (nullptr == instance_) {
    instance_.reset(new BufferTypeTree());
  }
  return instance_.get();
}

}  // namespace modelbox