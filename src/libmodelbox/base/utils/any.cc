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

#include <modelbox/base/any.h>

namespace modelbox {

Collection::Collection() = default;

Collection::~Collection() = default;

void Collection::Set(const std::string& key, const char* value) {
  entrys_[key] = Any(std::string(value));
}

std::tuple<Any*, bool> Collection::Get(const std::string& key) {
  auto iter = entrys_.find(key);
  if (iter != entrys_.end()) {
    return std::make_tuple(&(iter->second), true);
  }

  return std::make_tuple(nullptr, false);
}

void Collection::Merge(const Collection& other, bool is_override) {
  if (!is_override) {
    entrys_.insert(other.entrys_.begin(), other.entrys_.end());
    return;
  }

  for (auto& iter : entrys_) {
    entrys_[iter.first] = iter.second;
  }
}

bool Collection::CanConvert(size_t cast_code, size_t origin_code) {
  if (cast_code == origin_code) {
    return true;
  }

  auto iter = type_hash_code_map.find(origin_code);
  if (iter == type_hash_code_map.end()) {
    return false;
  }

  if (type_hash_code_map[origin_code] == cast_code) {
    MBLOG_DEBUG
        << "origin type is not match cast type, maybe loss the accuracy.";
    return true;
  }

  return false;
}

}  // namespace modelbox
