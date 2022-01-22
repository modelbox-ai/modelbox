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

#ifndef MODELBOX_COMMON_UTILS_H_
#define MODELBOX_COMMON_UTILS_H_

#include <signal.h>

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace modelbox {

/**
 * @brief Create pid file of current process
 * @param pid_file path of pid file
 * @return create result
 */
int modelbox_create_pid(const char *pid_file);

/**
 * @brief Handle process signal
 * @param sig_list signal list to handle
 * @param sig_num sig_list count
 * @param action signal handler
 * @return register result.
 */
int modelbox_sig_register(const int sig_list[], int sig_num,
                          void (*action)(int, siginfo_t *, void *));

/**
 * @brief Get cpu register data in string format
 * @param buf output message buffer
 * @param buf_size max size of buffer
 * @param ucontext signal context
 * @return result.
 */
int modelbox_cpu_register_data(char *buf, int buf_size, ucontext_t *ucontext);

/**
 * @brief Custom stream
 */
class OutStream {
 protected:
  using OStream = std::ostringstream;
  using Buffer_p = std::unique_ptr<OStream, std::function<void(OStream *)>>;
  virtual void ProcessStream(OStream *st) = 0;

 public:
  /**
   * @brief return stream
   */
  Buffer_p Stream() {
    return Buffer_p(new OStream, [=](OStream *st) {
      ProcessStream(st);
      delete st;
    });
  }
};

}  // namespace modelbox

#endif  // MODELBOX_COMMON_UTILS_H_