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

#include "modelbox/base/status.h"

namespace modelbox {

constexpr const char *MODELBOX_ROOT_VAR = "${MODELBOX_ROOT}";

/**
 * @brief Get current modelbox standalone directory
 * @return standalone root dir
 */
const std::string &modelbox_root_dir();

/**
 * @brief Get modelbox full path
 *
 * @param path
 * @return std::string
 */
std::string modelbox_full_path(const std::string &path);

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
 * @brief Split ip address and port from string
 * @param host host string
 * @param ip output ip address
 * @param port output port
 * @return result.
 */
Status SplitIPPort(const std::string &host, std::string &ip, std::string &port);

/**
 * @brief Get user id and gid by username
 * @param user username
 * @param uid user id
 * @param gid group id
 * @return result.
 */
Status GetUidGid(const std::string &user, uid_t &uid, gid_t &gid);

/**
 * @brief change user and group of path
 * @param user username
 * @param path path to change
 * @return result.
 */
Status ChownToUser(const std::string &user, const std::string &path);

/**
 * @brief run as user
 * @return result.
 */
Status RunAsUser(const std::string &user);

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