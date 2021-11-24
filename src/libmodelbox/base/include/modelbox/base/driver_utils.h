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

#ifndef MODELBOX_DRIVER_UTILS_H_
#define MODELBOX_DRIVER_UTILS_H_

#include <sys/wait.h>
#include <unistd.h>

#include "modelbox/base/log.h"
#include "modelbox/base/status.h"

namespace modelbox {
/**
 * @brief fork a process to Run func
 * @return func result
 */
template <typename func, typename... ts>
Status SubProcessRun(func &&fun, ts &&...params) {
  auto pid = fork();
  if (pid == 0) {
    Status ret = fun(params...);
    if (!ret) {
      _exit(0);
    }

    _exit(1);
  }

  if (pid == -1) {
    MBLOG_ERROR << "fork subprocess failed";
    return STATUS_FAULT;
  }

  MBLOG_INFO << "wait for subprocess " << pid << " process finished";
  int status;
  auto ret = waitpid(pid, &status, 0);
  if (ret < 0) {
    auto err_msg =
        "subprocess run failed, wait error, ret:" + std::to_string(errno) +
        ", msg: " + strerror(errno);
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  }

  if (WIFSIGNALED(status)) {
    auto err_msg = "killed by signal " + WTERMSIG(status);
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  } else if(WIFSTOPPED(status)) {
    auto err_msg = "stopped by signal " + WSTOPSIG(status);
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  }

  return STATUS_OK;
};

/**
 * @brief generate sha256 key from a check_sum
 * @param check_sum
 * @return sha256 result
 */
std::string GenerateKey(int64_t check_sum);

}  // namespace modelbox

#endif