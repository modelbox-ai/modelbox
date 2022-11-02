
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

#ifndef MODELBOX_INNER_EVENT_H_
#define MODELBOX_INNER_EVENT_H_
#include <memory>
#include <string>
#include <unordered_map>

namespace modelbox {

class MatchKey;

class FlowUnitEvent {
 public:
  FlowUnitEvent();
  virtual ~FlowUnitEvent();
  void SetPrivate(const std::string &key,
                  const std::shared_ptr<void> &private_content);
  std::shared_ptr<void> GetPrivate(const std::string &key);

  template <typename T>
  inline std::shared_ptr<T> GetPrivate(const std::string &key) {
    return std::static_pointer_cast<T>(GetPrivate(key));
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;
};

class FlowUnitInnerEvent {
 public:
  enum EventCode {
    EXPAND_UNFINISH_DATA = 0,
    EXPAND_NEXT_STREAM,
    COLLAPSE_NEXT_STREAM
  };
  FlowUnitInnerEvent(EventCode code);
  virtual ~FlowUnitInnerEvent();
  void SetDataCtxMatchKey(MatchKey *match_key);
  MatchKey *GetDataCtxMatchKey();
  std::shared_ptr<FlowUnitEvent> GetUserEvent();
  void SetUserEvent(std::shared_ptr<FlowUnitEvent> event);
  int GetPriority();
  EventCode GetEventCode();

 private:
  int priority_ = 0;
  EventCode code_ = EXPAND_UNFINISH_DATA;
  std::shared_ptr<FlowUnitEvent> user_event_;
  MatchKey *match_key_;
};
}  // namespace modelbox

#endif