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

#include <fcntl.h>
#include <modelbox/base/popen.h>
#include <modelbox/base/utils.h>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

extern char **environ;

namespace modelbox {

constexpr int POPEN_ERROR = -1;
constexpr int POPEN_EOF = -2;
constexpr int POPEN_STOP_READ = -3;

int ParserArg(const std::string &cmd, std::vector<std::string> &args) {
  std::string arg;
  char quoteChar = 0;

  for (std::string::const_iterator it = cmd.begin(), it_end = cmd.end();
       it != it_end; ++it) {
    char ch = *it;
    if (quoteChar == '\\') {
      arg.push_back(ch);
      quoteChar = 0;
      continue;
    }

    if (quoteChar && ch != quoteChar) {
      arg.push_back(ch);
      continue;
    }

    switch (ch) {
      case '\'':
      case '\"':
      case '\\':
        quoteChar = quoteChar ? 0 : ch;
        break;
      case ' ':
      case '\t':
      case '\n':
        if (!arg.empty()) {
          args.push_back(arg);
          arg.clear();
        }
        break;
      default:
        arg.push_back(ch);
        break;
    }
  }

  if (!arg.empty()) {
    args.push_back(arg);
  }

  return 0;
}

Popen::Popen() {
  start_tm_ = std::chrono::high_resolution_clock::now();
  fdout_.buffer_.reserve(1024 * 16);
  fderr_.buffer_.reserve(1024 * 16);
}

Popen::~Popen() { Close(); }

void Popen::SetupMode(const char *mode) {
  if (strstr(mode, "r")) {
    fdout_.enable_ = true;
  }

  if (strstr(mode, "e")) {
    fderr_.enable_ = true;
  }

  if (strstr(mode, "w")) {
    fdin_.enable_ = true;
  }
}

Status Popen::Open(const std::string &cmdline, int timeout, const char *mode,
                   const PopenEnv &env) {
  std::vector<std::string> args;
  std::vector<std::string> envs;

  if (ParserArg(cmdline, args) != 0) {
    return {STATUS_INVALID, "command line is invalid"};
  }

  return Open(args, timeout, mode, env);
}

Status Popen::Open(std::vector<std::string> args, int timeout, const char *mode,
                   const PopenEnv &env) {
  pid_t child_pid;
  int fd_out[2] = {-1, -1};
  int fd_err[2] = {-1, -1};
  int fd_in[2] = {-1, -1};

  SetupMode(mode);

  if (child_pid_ > 0) {
    return STATUS_ALREADY;
  }

  if (fdin_.enable_ == true && pipe(fd_in) != 0) {
    goto errout;
  }

  if (pipe(fd_out) != 0) {
    goto errout;
  }

  if (fdout_.enable_ == true && pipe(fd_err) != 0) {
    goto errout;
  }

  child_pid = vfork();
  if (child_pid < 0) {
    return {STATUS_FAULT, StrError(errno)};
  } else if (child_pid == 0) {
    size_t i = 0;
    int fd_out_keep = fd_out[1];
    setsid();

    if (fdin_.enable_) {
      close(0);
      dup2(fd_in[0], 0);
      close(fd_in[1]);
    }

    if (fdout_.enable_) {
      close(1);
      dup2(fd_out_keep, 1);
      close(fd_out_keep);
      fd_out_keep = -1;
    }

    if (fderr_.enable_) {
      close(2);
      dup2(fd_err[1], 2);
      close(fd_err[1]);
    }

    // call readdir after vfork is not safe, for glibc only
    CloseAllParentFds(fd_out_keep);

    // args
    char *argv[args.size() + 1];
    for (i = 0; i < args.size(); i++) {
      argv[i] = (char *)args[i].c_str();
    }
    argv[args.size()] = 0;

    // env
    auto envs = env.GetEnvs();
    char *envp[envs.size() + 1];
    for (i = 0; i < envs.size(); i++) {
      envp[i] = (char *)envs[i].c_str();
    }
    envp[envs.size()] = 0;

    // exec command
    if (env.Changed()) {
      execvpe(argv[0], argv, envp);
    } else {
      execvp(argv[0], argv);
    }
    fprintf(stderr, "exec failed for %s, %s\n", argv[0],
            StrError(errno).c_str());
    _exit(1);
  }

  timeout_ = timeout;
  child_pid_ = child_pid;
  start_tm_ = std::chrono::high_resolution_clock::now();

  close(fd_out[1]);
  fdout_.fd_ = fd_out[0];
  fcntl(fdout_.fd_, F_SETFL, fcntl(fdout_.fd_, F_GETFL) | O_NONBLOCK);

  if (fdin_.enable_) {
    close(fd_in[0]);
    fdin_.fd_ = fd_in[1];
  }

  if (fderr_.enable_) {
    close(fd_err[1]);
    fderr_.fd_ = fd_err[0];
    fcntl(fderr_.fd_, F_SETFL, fcntl(fderr_.fd_, F_GETFL) | O_NONBLOCK);
  }

  return STATUS_OK;

errout:
  auto close_fd = [](int fd[2]) {
    if (fd[0] > 0) {
      close(fd[0]);
    }

    if (fd[1] > 0) {
      close(fd[1]);
    }
  };

  close_fd(fd_in);
  close_fd(fd_out);
  close_fd(fd_err);

  return {STATUS_FAULT, StrError(errno)};
}

void Popen::CloseAllParentFds(int keep_fd) {
  std::vector<std::string> files;
  ListFiles("/proc/self/fd", "*", &files);
  for (auto &file : files) {
    int port = std::stoi(GetBaseName(file));
    if (port == STDIN_FILENO || port == STDOUT_FILENO ||
        port == STDERR_FILENO || port == keep_fd) {
      continue;
    }
    close(port);
  }
}

bool Popen::DataReady(std::vector<struct stdfd *> *fds) {
  for (auto const &stdfd : *fds) {
    if (stdfd->newline_pos_ > 0) {
      return true;
    }

    if (stdfd->iseof_ == 1 && stdfd->buffer_.size() > 0) {
      return true;
    }
  }

  return false;
}

int Popen::ReadLineData(struct stdfd *stdfd) {
  char tmp[4096];

  stdfd->newline_pos_ = 0;
  while (true) {
    int len = read(stdfd->fd_, tmp, sizeof(tmp));
    if (len < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        UpdateNewLinePos(stdfd);
        return 0;
      }
      return POPEN_ERROR;
    } else if (len == 0) {
      stdfd->iseof_ = 1;
      UpdateNewLinePos(stdfd);
      return POPEN_STOP_READ;
    }

    if (stdfd->buffer_.size() + len >= stdfd->buffer_.capacity()) {
      stdfd->buffer_.erase(stdfd->buffer_.begin(),
                           stdfd->buffer_.begin() + len);
    }

    stdfd->buffer_.insert(stdfd->buffer_.end(), tmp, tmp + len);

    UpdateNewLinePos(stdfd);
    if (stdfd->newline_pos_ > 0) {
      return POPEN_STOP_READ;
    }
  }

  return 0;
}

int Popen::WaitForFdsLineRead(std::vector<struct stdfd *> *fds, int timeout) {
  if (DataReady(fds)) {
    return 1;
  }

  int ret = WaitForFds(*fds, timeout, [this](struct stdfd *stdfd, int revent) {
    return ReadLineData(stdfd);
  });

  if (DataReady(fds)) {
    return 1;
  }

  return ret;
}

int Popen::WaitForLineRead(int timeout) {
  std::vector<struct stdfd *> fds;
  if (fdout_.enable_) {
    fds.push_back(&fdout_);
  }

  if (fderr_.enable_) {
    fds.push_back(&fderr_);
  }

  return WaitForFdsLineRead(&fds, timeout);
}

void Popen::UpdateNewLinePos(struct stdfd *stdfd) {
  int len = stdfd->buffer_.size();

  for (int i = 0; i < len; i++) {
    if (stdfd->buffer_.data()[i] == '\n' || stdfd->buffer_.data()[i] == '\0') {
      stdfd->newline_pos_ = i + 1;
      return;
    }
  }

  stdfd->newline_pos_ = 0;
}

int Popen::GetStringLine(struct stdfd *stdfd, std::string &line) {
  if (stdfd->enable_ == false) {
    return -1;
  }

  if (stdfd->newline_pos_ <= 0) {
    std::vector<struct stdfd *> fds;
    fds.push_back(stdfd);
    WaitForFdsLineRead(&fds, -1);
  }

  if (stdfd->newline_pos_ <= 0) {
    if (stdfd->iseof_ && stdfd->buffer_.size() > 0) {
      line.assign(stdfd->buffer_.begin(),
                  stdfd->buffer_.begin() + stdfd->buffer_.size());
      stdfd->buffer_.clear();
      return 0;
    }

    return -1;
  }

  line.assign(stdfd->buffer_.begin(),
              stdfd->buffer_.begin() + stdfd->newline_pos_);
  stdfd->buffer_.erase(stdfd->buffer_.begin(),
                       stdfd->buffer_.begin() + stdfd->newline_pos_);

  UpdateNewLinePos(stdfd);

  return 0;
}

int Popen::ReadErrLine(std::string &line) {
  return GetStringLine(&fderr_, line);
}

int Popen::ReadOutLine(std::string &line) {
  return GetStringLine(&fdout_, line);
}

int Popen::ReadAll(std::string *out, std::string *err) {
  std::vector<struct stdfd *> fds;
  int ret = 0;

  if (fdout_.enable_) {
    fds.push_back(&fdout_);
  }

  if (fderr_.enable_) {
    fds.push_back(&fderr_);
  }

  while (true) {
    ret = WaitForFds(fds, -1, [this](struct stdfd *stdfd, int revent) {
      return ReadLineData(stdfd);
    });

    if (ret < 0) {
      break;
    }
  }

  if (out && fdout_.enable_) {
    out->assign(fdout_.buffer_.begin(),
                fdout_.buffer_.begin() + fdout_.buffer_.size());
    fdout_.buffer_.clear();
  }

  if (err && fderr_.enable_) {
    err->assign(fderr_.buffer_.begin(),
                fderr_.buffer_.begin() + fderr_.buffer_.size());
    fderr_.buffer_.clear();
  }

  return 0;
}

int Popen::WriteString(const std::string &in) {
  if (fdin_.enable_ == false) {
    return -1;
  }

  int len = 0;
  int total_len = in.size();

  struct sigaction act;
  struct sigaction old;
  Defer { sigaction(SIGPIPE, &old, NULL); };

  act.sa_handler = SIG_IGN;
  act.sa_flags = SA_RESTART;
  sigaction(SIGPIPE, &act, &old);

  do {
    int written_len = in.size() - total_len;
    len = write(fdin_.fd_, in.data() + written_len, in.size() - written_len);
    if (len < 0) {
      return -1;
    }

    total_len -= len;
  } while (total_len > 0);

  return 0;
}

int Popen::TimeOutLeft() {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fs = t1 - start_tm_;
  std::chrono::milliseconds diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(fs);

  int ret = timeout_ - diff.count();
  if (ret <= 0) {
    return -1;
  }

  return ret;
}

int Popen::WaitForFds(std::vector<struct stdfd *> fds, int timeout,
                      std::function<int(struct stdfd *, int)> func) {
  struct pollfd fdset[fds.size()];
  int fds_count = fds.size();
  int eof_count = fds.size();
  int i = 0;

  for (i = 0; i < fds_count; i++) {
    fdset[i].fd = fds[i]->fd_;
    fdset[i].events = POLLIN | POLLHUP;
  }

  while (true && fds_count > 0) {
    int polltimeout = TimeOutLeft();
    if (polltimeout < 0 && timeout_ != -1) {
      return POPEN_ERROR;
    }

    if ((timeout > polltimeout && timeout_ != -1) || timeout < 0) {
      timeout = polltimeout;
    }

    int ret = poll(fdset, fds_count, timeout);
    if (ret <= 0) {
      return ret;
    }

    timeout = 0;
    for (i = 0; i < fds_count; i++) {
      struct stdfd *stdfd = fds[i];
      if (stdfd->fd_ != fdset[i].fd) {
        raise(SIGSEGV);
        return POPEN_ERROR;
      }

      int func_ret = func(stdfd, fdset[i].revents);
      if (func_ret != 0 && func_ret != POPEN_STOP_READ) {
        return func_ret;
      }

      if (fdset[i].revents & POLLHUP || func_ret == POPEN_STOP_READ) {
        int j;
        for (j = i + 1; j < fds_count; j++) {
          fdset[i].fd = fdset[j].fd;
          fdset[i].events = fdset[j].events;
          fds.erase(fds.begin() + i);
        }

        if (fdset[i].revents & POLLHUP) {
          stdfd->iseof_ = 1;
          eof_count--;
        }
        fds_count--;
      }
    }
  }

  if (eof_count == 0) {
    return POPEN_EOF;
  }

  return 0;
}

int Popen::WaitChildTimeOut() {
  char buff[4096];
  std::vector<struct stdfd *> fds;
  fds.push_back(&fdout_);
  if (fderr_.enable_) {
    fds.push_back(&fderr_);
  }

  int ret = WaitForFds(fds, -1, [&buff](struct stdfd *stdfd, int revent) {
    int unused __attribute__((unused));
    if (!(revent & POLLIN)) {
      return 0;
    }

    unused = read(stdfd->fd_, buff, sizeof(buff));

    return 0;
  });

  return ret;
}

Status Popen::ForceStop() {
  if (child_pid_ <= 0) {
    return STATUS_NOTFOUND;
  }

  killpg(child_pid_, SIGKILL);

  return STATUS_OK;
}

void Popen::KeepAlive() {
  start_tm_ = std::chrono::high_resolution_clock::now();
}

void Popen::CloseStdFd() {
  auto closefd = [](struct stdfd *stdfd) {
    if (stdfd->fd_ > 0) {
      close(stdfd->fd_);
      stdfd->fd_ = -1;
    }

    stdfd->iseof_ = 0;
    stdfd->enable_ = false;
    stdfd->newline_pos_ = 0;
    stdfd->buffer_.clear();
  };

  closefd(&fdout_);
  closefd(&fderr_);
  closefd(&fdin_);
}

int Popen::Close() {
  int wstatus = 0;
  if (child_pid_ <= 0) {
    return 0;
  }

  if (timeout_ > 0) {
    int ret = WaitChildTimeOut();
    if (ret == 0 || ret == POPEN_ERROR) {
      killpg(child_pid_, SIGTERM);
      usleep(2000);
      killpg(child_pid_, SIGKILL);
    }
  }

  CloseStdFd();

  if (waitpid(child_pid_, &wstatus, 0) == -1) {
    return -1;
  }

  child_pid_ = 0;
  return wstatus;
}

PopenEnv::PopenEnv() {}

PopenEnv::~PopenEnv() {}

PopenEnv::PopenEnv(const std::string &item_list) { LoadEnvFromList(item_list); }

PopenEnv::PopenEnv(const char *item_list) { LoadEnvFromList(item_list); }

void PopenEnv::LoadEnvFromList(const std::string &item_list) {
  inherit_ = true;
  std::vector<std::string> envs;
  ParserArg(item_list, envs);
  if (envs.size() <= 0) {
    return;
  }

  LoadInherit();

  for (auto const &env : envs) {
    const char *envp = env.c_str();
    auto field = strstr(envp, "=");
    if (field == nullptr) {
      continue;
    }

    std::string item(envp, field - envp);
    std::string value = field + 1;
    Add(item, value);
  }
}

PopenEnv::PopenEnv(const std::string &item, const std::string &value) {
  inherit_ = true;
  Add(item, value);
}

void PopenEnv::LoadInherit() {
  char **ep;
  if (load_inherit_) {
    return;
  }

  load_inherit_ = true;
  for (ep = environ; *ep != NULL; ep++) {
    auto field = strstr(*ep, "=");
    if (field == nullptr) {
      continue;
    }

    std::string item(*ep, field);
    std::string value = field + 1;
    Add(item, value);
  }
}

PopenEnv &PopenEnv::Add(const std::string &item, const std::string &value) {
  LoadInherit();
  env_[item] = value;
  return *this;
}

PopenEnv &PopenEnv::Rmv(const std::string &item) {
  LoadInherit();
  env_.erase(item);
  return *this;
}

PopenEnv &PopenEnv::Clear() {
  env_.clear();
  load_inherit_ = true;
  return *this;
}

const bool PopenEnv::Changed() const { return load_inherit_; }

const std::vector<std::string> PopenEnv::GetEnvs() const {
  std::vector<std::string> envs;

  for (auto const &it : env_) {
    std::string value;
    value = it.first + "=" + it.second;
    envs.emplace_back(value);
  }

  return envs;
}

}  // namespace modelbox
