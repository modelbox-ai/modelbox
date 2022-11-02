
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

#include <utility>

#include "modelbox/inner_event.h"

namespace modelbox {

FlowUnitEvent::FlowUnitEvent() = default;

FlowUnitEvent::~FlowUnitEvent() = default;

void FlowUnitEvent::SetPrivate(const std::string &key,
                               const std::shared_ptr<void> &private_content) {
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    private_map_.emplace(key, private_content);
  }
}
std::shared_ptr<void> FlowUnitEvent::GetPrivate(const std::string &key) {
  auto iter = private_map_.find(key);
  if (iter == private_map_.end()) {
    return nullptr;
  }
  return private_map_[key];
}

FlowUnitInnerEvent::FlowUnitInnerEvent(EventCode code) : code_(code), match_key_(nullptr){};

FlowUnitInnerEvent::~FlowUnitInnerEvent() = default;

int FlowUnitInnerEvent::GetPriority() { return priority_; }

void FlowUnitInnerEvent::SetDataCtxMatchKey(MatchKey *match_key) {
  match_key_ = match_key;
}

MatchKey *FlowUnitInnerEvent::GetDataCtxMatchKey() { return match_key_; }

FlowUnitInnerEvent::EventCode FlowUnitInnerEvent::GetEventCode() {
  return code_;
}

std::shared_ptr<FlowUnitEvent> FlowUnitInnerEvent::GetUserEvent() {
  return user_event_;
}
void FlowUnitInnerEvent::SetUserEvent(std::shared_ptr<FlowUnitEvent> event) {
  user_event_ = std::move(event);
}
}  // namespace modelbox
