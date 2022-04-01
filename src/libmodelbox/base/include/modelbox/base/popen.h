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

#ifndef MODELBOX_POPEN_H_
#define MODELBOX_POPEN_H_

#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "modelbox/base/status.h"

namespace modelbox {

/**
 * @brief safe pipe stream to or from a process, support command timeout,
 * support getting standard error output
 */
class Popen {
 public:
  Popen();
  virtual ~Popen();

  /**
   * @brief Opens a process by creating a pipe, forking.
   *
   * @param args parameter list, the first parameter is the program path
   * @param timeout command execution timeout, in milliseconds, When the command
   * times out, the child process will be killed
   * @param mode read-write mode,
   *    "w" for writing to standard input,
   *    "r" for reading standard output,
   *    "e" for reading standard error output.
   * @return Status operation result
   */
  Status Open(std::vector<std::string> args, int timeout = -1,
              const char *mode = "r");

  /**
   * @brief Opens a process by creating a pipe, forking.
   *
   * @param cmdline command line in string format.
   * @param timeout command execution timeout, in milliseconds, When the command
   * times out, the child process will be killed
   * @param mode read-write mode,
   *    "w" for writing to standard input,
   *    "r" for reading standard output,
   *    "e" for reading standard error output.
   * @return Status operation result
   */
  Status Open(const std::string &cmdline, int timeout = -1,
              const char *mode = "r");

  /**
   * @brief Close the command and get the command execution result
   *
   * @return int command execution result, Whether the command times out, you
   * can check whether the signal is SIGKILL.
   */
  int Close();

  /**
   * @brief Waiting to be read
   *
   * @param timeout wait period in milliseconds
   * @return 1: can read
   *         0: timeout
   *         -1: error
   */
  int WaitForLineRead(int timeout = -1);

  /**
   * @brief Read a line of stderr output
   *
   * @param line
   * @return 0: success
   *         -1: fail.
   */
  int ReadErrLine(std::string &line);

  /**
   * @brief Read a line of stdout output
   *
   * @param line
   * @return 0: success
   *         -1: fail.
   */
  int ReadOutLine(std::string &line);

  /**
   * @brief Read all outputs at once
   *
   * @param out standard output variable
   * @param err standard error output variable
   * @return 0: success
   *         -1: fail.
   */
  int ReadAll(std::string *out, std::string *err);

  /**
   * @brief Write string to child process
   *
   * @param in write message
   * @return 0: success
   *         -1: fail.
   */
  int WriteString(const std::string &in);

  /**
   * @brief Force stop child process
   *
   * @return OK success
   *         other: fail.
   */
  Status ForceStop();

  /**
   * @brief Keep command alive
   * 
   */
  void KeepAlive();

 private:
  struct stdfd {
    bool enable_{false};
    int fd_{-1};
    std::vector<char> buffer_;
    int newline_pos_{0};
    int iseof_{0};
  };

  int WaitForFds(std::vector<struct stdfd *> fds, int timeout,
                 std::function<int(struct stdfd *stdfd, int revents)> func);

  int ReadLineData(struct stdfd *stdfd);

  int WaitForFdsLineRead(std::vector<struct stdfd *> *fds, int timeout);

  bool DataReady(std::vector<struct stdfd *> *fds);

  void UpdateNewLinePos(struct stdfd *stdfd);

  int GetStringLine(struct stdfd *stdfd, std::string &line);

  int WaitChildTimeOut();

  int TimeOutLeft();

  void CloseStdFd();

  void CloseAllParentFds(int keep_fd);

  void SetupMode(const char *mode);

  int ParserArg(const std::string &cmd, std::vector<std::string> &args);

  struct stdfd fdout_;
  struct stdfd fderr_;
  struct stdfd fdin_;

  pid_t child_pid_{0};
  int timeout_{-1};
  std::chrono::high_resolution_clock::time_point start_tm_;
};

}  // namespace modelbox

#endif
