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

#ifndef MODELBOX_SERVER_TIMER_H_
#define MODELBOX_SERVER_TIMER_H_

#include <modelbox/base/timer.h>

namespace modelbox {

class ServerTimer : public modelbox::Timer {
 public:
  /**
   * @brief Get server timer instance
   * @return pointer to server timer
   */
  static ServerTimer *GetInstance();

  /**
   * @brief Server timer start
   */
  void Start();

  /**
   * @brief Server timer run
   */
  void Run() override;

  /**
   * @brief Server timer stop
   */
  void Stop();

 private:
  ServerTimer() = default;
  ~ServerTimer() override = default;
};

/**
 * @brief Global server timer
 */
extern ServerTimer *kServerTimer;

}  // namespace modelbox

#endif  // MODELBOX_SERVER_TIMER_H_
