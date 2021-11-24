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


#ifndef MODELBOX_STREAM_H_
#define MODELBOX_STREAM_H_

#include <memory>
#include <unordered_map>

#include "modelbox/base/status.h"

namespace modelbox {

class DataMeta {
 public:
  DataMeta();
  virtual ~DataMeta();
  void SetMeta(const std::string &key, std::shared_ptr<void> meta);
  std::shared_ptr<void> GetMeta(const std::string &key);

 private:
  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;
};

class FlowUnitError {
 public:
  FlowUnitError(std::string desc);
  FlowUnitError(std::string node, std::string error_pos, Status error_status);
  virtual ~FlowUnitError();
  std::string GetDesc();

 private:
  std::string desc_;
  Status error_status_;
};

}  // namespace modelbox
#endif