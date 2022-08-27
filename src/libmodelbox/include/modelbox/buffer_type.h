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


#ifndef MODELBOX_BUFFER_TYPE_H_
#define MODELBOX_BUFFER_TYPE_H_

#include <modelbox/base/device.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace modelbox {

constexpr const char *ROOT_BUFFER_TYPE = "RAW";

class BufferType {
 public:
  BufferType();
  BufferType(std::string type);
  virtual ~BufferType();
  const std::string &GetType() const;
  std::shared_ptr<BufferType> GetParentType();
  std::vector<std::shared_ptr<BufferType>> GetChildrenType();

 private:
  void SetType(std::string type);
  bool AddChildType(const std::shared_ptr<BufferType> &child);
  bool AddParentType(const std::shared_ptr<BufferType> &parent);
  void RemoveType();
  void ClearChildType();
  void ClearParentType();
  bool IsAncestor(const BufferType &other);
  bool IsOffspring(const BufferType &other);
  std::string type_;
  std::shared_ptr<BufferType> parent_;
  std::vector<std::shared_ptr<BufferType>> children_;
  friend class BufferTypeTree;
};

class BufferTypeTree {
 public:
  bool AddRootType(const std::string &root_type);
  bool AddType(const std::string &type, const std::string &parent_type);
  bool RemoveType(const std::string &type);
  bool IsCompatible(const std::string &type, const std::string &ancestor_type);
  std::shared_ptr<BufferType> GetType(const std::string &type);
  static BufferTypeTree *GetInstance();

  virtual ~BufferTypeTree();
 private:
  BufferTypeTree();
  std::string root_;
  std::map<std::string, std::shared_ptr<BufferType>> nodes_;
  static std::shared_ptr<BufferTypeTree> instance_;
};
}  // namespace modelbox

#endif  // MODELBOX_BUFFER_TYPE_H_