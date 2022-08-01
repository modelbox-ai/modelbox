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


#ifndef MODELBOX_COLLECTOR_H_
#define MODELBOX_COLLECTOR_H_

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "modelbox/base/log.h"
namespace modelbox {

template <typename T>
class Collector {
 public:
  Collector() = default;
  virtual ~Collector() = default;

  /**
   * @brief Add Object
   * @param name object key
   * @param obj object
   */
  void AddObject(const std::string &name, std::shared_ptr<T> obj) {
    std::unique_lock<std::mutex> lock(object_lock_);
    auto iter = objs_.find(name);
    if (iter != objs_.end()) {
      MBLOG_WARN << name << " is already in the map, overwrites it.";
    }
    objs_[name] = obj;
  }

  /**
   * @brief Remove Object
   * @param name object key
   */
  void RmvObject(const std::string &name) {
    std::unique_lock<std::mutex> lock(object_lock_);
    auto iter = objs_.find(name);
    if (iter == objs_.end()) {
      return;
    }

    objs_.erase(name);
  }

  /**
   * @brief Get object
   * @param name object key
   * @param obj object
   * @return get object success or not
   */
  bool GetObject(const std::string &name, std::shared_ptr<T> &obj) {
    auto iter = objs_.find(name);
    if (iter == objs_.end()) {
      return false;
    }
    obj = iter->second;
    return true;
  }

  /**
   * @brief Get All Objects
   * @return a vector of objects
   */
  std::vector<std::shared_ptr<T>> GetObjects() {
    std::vector<std::shared_ptr<T>> objs;
    for (auto &obj : objs_) {
      objs.push_back(obj.second);
    }
    return objs;
  }

  /**
   * @brief Get Object size
   * @return the object size
   */
  int GetObjectSize() { return objs_.size(); }

 private:
  std::mutex object_lock_;
  std::map<std::string, std::shared_ptr<T>> objs_;
};

}  // namespace modelbox

#endif