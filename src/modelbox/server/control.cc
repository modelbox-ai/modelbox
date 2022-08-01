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

#include "control.h"

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <iomanip>
#include <functional>

#include "config.h"
#include "modelbox/base/configuration.h"
#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/base/utils.h"
#include "modelbox/common/command.h"
#include "modelbox/common/control_msg.h"
#include "securec.h"

namespace modelbox {

Control::Control() = default;

Control::~Control() { Stop(); }

modelbox::Status Control::Init(
    std::shared_ptr<modelbox::Configuration> config) {
  config_ = config;
  auto ret = modelbox::STATUS_OK;
  DeferCond { return ret != modelbox::STATUS_OK; };

  if (config_ == nullptr) {
    return modelbox::STATUS_BADCONF;
  }

  if (config_->GetBool("control.enable", false) == false) {
    return modelbox::STATUS_OK;
  }

  listen_path_ = modelbox_full_path(
      config->GetString("control.listen", CONTROL_UNIX_PATH));

  struct sockaddr_un server_sockaddr;
  int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (fd <= 0) {
    std::string errmsg = "create socket: ";
    errmsg += modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    ret = {modelbox::STATUS_FAULT, errmsg};
    return ret;
  }
  DeferCondAdd { close(fd); };

  server_sockaddr.sun_family = AF_UNIX;
  strncpy_s(server_sockaddr.sun_path, sizeof(server_sockaddr.sun_path),
            listen_path_.c_str(), listen_path_.length());
  unlink(server_sockaddr.sun_path);
  int rc =
      bind(fd, (struct sockaddr *)&server_sockaddr, sizeof(server_sockaddr));
  if (rc != 0) {
    std::string errmsg =
        "bind socket: " + std::string(server_sockaddr.sun_path) + " failed, ";
    errmsg += modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    ret = {modelbox::STATUS_FAULT, errmsg};
    return ret;
  }
  chmod(server_sockaddr.sun_path, 0660);

  pool_ = std::make_shared<modelbox::ThreadPool>(0, 8);
  pool_->SetName("Control-Sender");

  server_fd_ = fd;
  return modelbox::STATUS_OK;
}

int Control::ProcessHelp(std::shared_ptr<ControlMsgHelp> msg,
                         MsgSendFunc reply_func) {
  auto cmds = modelbox::ToolCommandList::Instance()->GetAllCommands();
  if (cmds.size() == 0) {
    return 0;
  }

  auto out_msg = std::make_shared<ControlOutStream>();
  out_msg->SetReplyFunc(reply_func);

  *out_msg->Stream() << "Server command lists:\n";
  for (auto &cmd : cmds) {
    *out_msg->Stream() << "  " << std::left << std::setw(23)
                       << cmd->GetCommandName() << cmd->GetCommandDesc()
                       << "\n";
  }

  return 0;
}

int Control::ProcessCmd(std::shared_ptr<ControlMsgCmd> msg,
                        MsgSendFunc reply_func) {
  auto args = msg->GetArgv();
  int argc = args.size();
  char *argv[argc];

  for (int i = 0; i < argc; i++) {
    argv[i] = (char *)args[i].c_str();
  }

  if (argc <= 0) {
    return -1;
  }

  const char *action = argv[0];
  auto cmd = modelbox::ToolCommandList::Instance()->GetCommand(action);
  if (cmd == nullptr) {
    MBLOG_DEBUG << "command " << action << " not exists\n";
    return -1;
  }

  auto out_msg = std::make_shared<ControlOutStream>();
  auto err_msg = std::make_shared<ControlErrStream>();

  out_msg->SetReplyFunc(reply_func);
  err_msg->SetReplyFunc(reply_func);
  cmd->SetUp(out_msg, err_msg);

  return cmd->Run(argc, argv);
}

void Control::ProcessMsg(std::shared_ptr<ControlMsg> msg,
                         MsgSendFunc reply_func) {
  int process_ret = 0;
  modelbox::Status ret = modelbox::STATUS_OK;
  Defer {
    if (!ret) {
      ReplyMsgError(ret.Code(), ret.WrapErrormsgs(), reply_func);
    }
  };

  switch (msg->GetMsgType()) {
    case SERVER_CONTROL_MSG_TYPE_HELP: {
      auto help_msg = std::dynamic_pointer_cast<ControlMsgHelp>(msg);
      if (help_msg == nullptr) {
        const auto *errmsg = "message is invalid";
        ret = {modelbox::STATUS_FAULT, errmsg};
        return;
      }
      process_ret = ProcessHelp(help_msg, reply_func);
    } break;
    case SERVER_CONTROL_MSG_TYPE_CMD: {
      auto cmd_msg = std::dynamic_pointer_cast<ControlMsgCmd>(msg);
      if (cmd_msg == nullptr) {
        const auto *errmsg = "message is invalid";
        ret = {modelbox::STATUS_FAULT, errmsg};
        return;
      }
      process_ret = ProcessCmd(cmd_msg, reply_func);
    } break;
    default:
      const auto *errmsg = "command not found";
      ret = {modelbox::STATUS_NOTFOUND, errmsg};
      return;
      break;
  }

  ControlMsgResult ret_msg;
  ret_msg.SetResult(process_ret);
  ret = ret_msg.Serialize();
  if (!ret) {
    MBLOG_ERROR << "Serialize result msg failed." << ret;
    return;
  }

  int len = reply_func(ret_msg.GetData(), ret_msg.GetDataLen());
  if (len < 0) {
    const auto *errmsg = "send to client failed.";
    ret = {modelbox::STATUS_FAULT, errmsg};
    return;
  }
}

void Control::ReplyMsgError(int err_code, const std::string &err_msg,
                            MsgSendFunc reply_func) {
  ControlMsgError msg_err;
  msg_err.SetError(err_code, err_msg);
  if (msg_err.Serialize() != modelbox::STATUS_OK) {
    return;
  }
  reply_func(msg_err.GetData(), msg_err.GetDataLen());
}

modelbox::Status Control::RecvMsg(std::shared_ptr<ControlMsg> recv_msg) {
  struct sockaddr_un client;
  socklen_t client_len = sizeof(client);

  size_t len = sizeof(struct sockaddr);
  len = recvfrom(
      server_fd_, recv_msg->GetDataTail(), recv_msg->GetRemainSpace(), 0,
      (struct sockaddr *)((void *)&client), (socklen_t *)(&client_len));
  if (len < 0) {
    std::string errmsg = "recv from client failed.";
    errmsg += modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    return {modelbox::STATUS_FAULT, errmsg};
  }

  modelbox::Status ret = modelbox::STATUS_OK;

  int fd = server_fd_;
  auto reply_func = [fd, client, client_len](void *data, int len) -> int {
    int rc = sendto(fd, data, len, 0, (struct sockaddr *)&client, client_len);
    if (rc < 0) {
      MBLOG_ERROR << "send failed, " << client.sun_path
                  << ", error: " << modelbox::StrError(errno);
    }

    return rc;
  };

  Defer {
    if (!ret) {
      ReplyMsgError(ret.Code(), ret.WrapErrormsgs(), reply_func);
    }
  };

  ret = recv_msg->AppendDataLen(len);
  if (!ret) {
    MBLOG_ERROR << "update message len failed, " << ret;
    return ret;
  }

  ret = recv_msg->Unserialize();
  if (!ret) {
    return ret;
  }

  auto process_msg = ControlMsgBuilder::Build(recv_msg);
  if (process_msg == nullptr) {
    MBLOG_ERROR << "Invalid control message";
    ret = modelbox::STATUS_INVALID;
    return ret;
  }

  pool_->Submit(&Control::ProcessMsg, this, process_msg, reply_func);
  return ret;
}

void Control::ControlDaemon() {
  struct pollfd fds[1];
  std::shared_ptr<ControlMsg> msg = std::make_shared<ControlMsg>();

  int nfds = sizeof(fds) / sizeof(struct pollfd);

  memset_s(fds, sizeof(fds), 0, sizeof(fds));
  fds[0].fd = server_fd_;
  fds[0].events = POLLIN;

  modelbox::os->Thread->SetName("Control-Daemon");

  while (run_) {
    int rc = poll(fds, nfds, -1);
    if (rc <= 0) {
      continue;
    }

    if (run_ == false) {
      break;
    }

    if (fds[0].revents != POLLIN) {
      continue;
    }

    auto ret = RecvMsg(msg);
    if (ret == modelbox::STATUS_AGAIN || ret == modelbox::STATUS_OK) {
      continue;
    }

    msg->Reset();
  }
}

modelbox::Status Control::Start() {
  if (run_ == true) {
    return modelbox::STATUS_OK;
  }

  if (server_fd_ > 0) {
    run_ = true;
    daemon_thread_ = std::thread(&Control::ControlDaemon, this);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status Control::Stop() {
  if (run_ == false) {
    return modelbox::STATUS_OK;
  }

  run_ = false;
  if (server_fd_ > 0) {
    shutdown(server_fd_, SHUT_RDWR);
  }

  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }

  pool_ = nullptr;

  if (server_fd_ > 0) {
    close(server_fd_);
    if (listen_path_.length() > 0) {
      unlink(listen_path_.c_str());
      listen_path_.clear();
    }
    server_fd_ = -1;
  }

  return modelbox::STATUS_OK;
}

ControlStream::ControlStream() = default;
ControlStream::~ControlStream() = default;

bool ControlStream::HasError() { return has_error_; }

void ControlStream::SetReplyFunc(MsgSendFunc reply_func) {
  reply_func_ = reply_func;
}

ControlOutStream::ControlOutStream() = default;
ControlOutStream::~ControlOutStream() = default;

void ControlOutStream::ProcessStream(OStream *st) {
  ControlMsgStdout msg;
  msg.SetString(st->str());
  auto ret = msg.Serialize();
  if (!ret) {
    has_error_ = true;
    return;
  }

  int len = reply_func_(msg.GetData(), msg.GetDataLen());
  if (len <= 0) {
    has_error_ = true;
    return;
  }
}

ControlErrStream::ControlErrStream() = default;
ControlErrStream::~ControlErrStream() = default;

void ControlErrStream::ProcessStream(OStream *st) {
  ControlMsgErrout msg;
  msg.SetString(st->str());
  auto ret = msg.Serialize();
  if (!ret) {
    has_error_ = true;
    return;
  }

  int len = reply_func_(msg.GetData(), msg.GetDataLen());
  if (len <= 0) {
    has_error_ = true;
    return;
  }
}

}  // namespace modelbox