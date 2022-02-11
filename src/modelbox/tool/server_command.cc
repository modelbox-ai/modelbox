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

#include "server_command.h"

#include <errno.h>
#include <getopt.h>
#include <modelbox/base/crypto.h>
#include <modelbox/base/utils.h>
#include <netdb.h>
#include <openssl/evp.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <termios.h>
#include <unistd.h>

#include <iostream>

#include "modelbox/common/control_msg.h"
#include "modelbox/common/config.h"
#include "securec.h"

namespace modelbox {

REG_MODELBOX_TOOL_COMMAND(ToolCommandServer)

enum MODELBOX_TOOL_SERVER_COMMAND {
  MODELBOX_TOOL_SERVER_CONNECT,
  MODELBOX_TOOL_SERVER_INFO_FROM_CONF,
};

static struct option server_options[] = {
    {"conn", 1, 0, MODELBOX_TOOL_SERVER_CONNECT},
    {"conf", 1, 0, MODELBOX_TOOL_SERVER_INFO_FROM_CONF},
    {0, 0, 0, 0},
};

ToolCommandServer::ToolCommandServer() {
  char tmp_var[] = "/tmp/modelbox-tool.XXXXXXX";
  temp_fd_ = mkstemp(tmp_var);
  if (temp_fd_ < 0) {
    unix_path_ = "/tmp/modelbox-tool.sock";
  } else {
    unix_path_ = tmp_var;
  }
}

ToolCommandServer::~ToolCommandServer() {
  CloseClient();
  if (temp_fd_ > 0) {
    close(temp_fd_);
  }
  unlink(unix_path_.c_str());
}

std::string ToolCommandServer::GetHelp() {
  char help[] =
      "Server command option:\n"
      "  -conn\t\t\t  connect socket file, example: xxx.sock\n"
      "  -conf\t\t\t  server conf file\n";
  return help;
}

modelbox::Status ToolCommandServer::InitClient(const std::string &connect_url) {
  struct sockaddr_un client_sockaddr;
  auto ret = modelbox::STATUS_OK;
  DeferCond { return ret != modelbox::STATUS_OK; };

  struct stat stat_buf;
  if (stat(connect_url.c_str(), &stat_buf) < 0) {
    auto errmsg = "cannot access control file: " + connect_url +
                  ", error: " + modelbox::StrError(errno) + ". Maybe server is down";
    std::cout << errmsg << std::endl;
    return {modelbox::STATUS_PERMIT};
  }

  int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (fd <= 0) {
    std::string errmsg = "create socket: ";
    errmsg += modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    ret = {modelbox::STATUS_FAULT, errmsg};
    return ret;
  }
  DeferCondAdd { close(fd); };

  client_sockaddr.sun_family = AF_UNIX;
  auto err =
      strncpy_s(client_sockaddr.sun_path, sizeof(client_sockaddr.sun_path),
                unix_path_.c_str(), unix_path_.length());
  if (err != 0) {
    MBLOG_ERROR << "strncpy_s failed.";
    return modelbox::STATUS_FAULT;
  }
  unlink(client_sockaddr.sun_path);

  int rc =
      bind(fd, (struct sockaddr *)&client_sockaddr, sizeof(client_sockaddr));
  if (rc != 0) {
    std::string errmsg = "bind socket: ";
    errmsg += modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    ret = {modelbox::STATUS_FAULT, errmsg};
    return ret;
  }
  auto ss = chmod(client_sockaddr.sun_path, 0660);
  if (ss != 0) {
    MBLOG_ERROR << "ss chmod client ret: " << ss << ", " << modelbox::StrError(errno);
  }

  chown(client_sockaddr.sun_path, -1, stat_buf.st_gid);

  client_fd_ = fd;
  return modelbox::STATUS_OK;
}

void ToolCommandServer::CloseClient() {
  if (client_fd_ > 0) {
    close(client_fd_);
    client_fd_ = -1;
    unlink(unix_path_.c_str());
  }
}

modelbox::Status ToolCommandServer::SendMsg(std::shared_ptr<ControlMsg> msg,
                                            const std::string &connect_url) {
  modelbox::Status status = modelbox::STATUS_OK;
  std::string errmsg;
  struct sockaddr_un remote;
  remote.sun_family = AF_UNIX;

  auto ret = strncpy_s(remote.sun_path, sizeof(remote.sun_path),
                       connect_url.c_str(), connect_url.length());
  if (ret != 0) {
    MBLOG_ERROR << "strncpy_s failed.";
    return modelbox::STATUS_FAULT;
  }

  int rc = sendto(client_fd_, msg->GetData(), msg->GetDataLen(), 0,
                  (struct sockaddr *)&remote, sizeof(remote));
  if (rc <= 0) {
    errmsg = "send data to server failed, err: ";
    errmsg += modelbox::StrError(errno);
    if (errno == ENOENT) {
      errmsg += ", No such file or directory. Maybe server is down";
      status = {modelbox::STATUS_NOENT, errmsg};
    } else {
      status = {modelbox::STATUS_FAULT, errmsg};
    }

    std::cerr << errmsg;
  }

  msg->Flip();

  return status;
}

modelbox::Status ToolCommandServer::SendCommand(
    int argc, char *argv[], const std::string &connect_url) {
  std::shared_ptr<ControlMsg> msg;
  if (argc == 0) {
    std::cout << GetHelp();
    msg = std::make_shared<ControlMsgHelp>();
  } else {
    auto cmd_msg = std::make_shared<ControlMsgCmd>();
    cmd_msg->SetArgs(argc, argv);
    msg = cmd_msg;
  }

  auto ret = msg->Serialize();
  if (!ret) {
    return ret;
  }

  return SendMsg(msg, connect_url);
}

int ToolCommandServer::RecvCommand() {
  int result = 0;
  int len = 0;
  auto msg = std::make_shared<ControlMsg>();
  struct sockaddr_un remote;
  socklen_t addr_len = sizeof(remote);

  struct pollfd fds[1];
  int nfds = sizeof(fds) / sizeof(struct pollfd);

  memset_s(fds, sizeof(fds), 0, sizeof(fds));
  fds[0].fd = client_fd_;
  fds[0].events = POLLIN;

  while (true) {
    int rc = poll(fds, nfds, 30 * 1000);
    if (rc == 0) {
      return -1;
    } else if (rc < 0) {
      continue;
    }

    len = recvfrom(client_fd_, msg->GetDataTail(), msg->GetRemainSpace(), 0,
                   (sockaddr *)&remote, &addr_len);
    if (len <= 0) {
      return -1;
    }

    auto ret = msg->AppendDataLen(len);
    if (!ret) {
      return -1;
    }
    ret = msg->Unserialize();
    if (ret == modelbox::STATUS_AGAIN) {
      continue;
    } else if (ret == modelbox::STATUS_INVALID) {
      return -1;
    }

    int out_len = msg->GetMsgDataLen();
    switch (msg->GetMsgType()) {
      case SERVER_CONTROL_MSG_TYPE_OUTMSG:
        if (msg->GetMsgData()[out_len - 1] == '\0') {
          out_len -= 1;
        }
        write(STDOUT_FILENO, msg->GetMsgData(), out_len);
        break;
      case SERVER_CONTROL_MSG_TYPE_ERRMSG:
        if (msg->GetMsgData()[out_len - 1] == '\0') {
          out_len -= 1;
        }
        write(STDERR_FILENO, msg->GetMsgData(), out_len);
        break;
      case SERVER_CONTROL_MSG_TYPE_RESULT: {
        auto new_msg = std::dynamic_pointer_cast<ControlMsgResult>(
            ControlMsgBuilder::Build(msg));
        if (new_msg == nullptr) {
          return 1;
        }
        return new_msg->GetResult();
      } break;
      case SERVER_CONTROL_MSG_TYPE_ERR: {
        auto new_msg = std::dynamic_pointer_cast<ControlMsgError>(
            ControlMsgBuilder::Build(msg));
        if (new_msg == nullptr) {
          return 1;
        }

        MBLOG_ERROR << "server return error, code: " << new_msg->GetErrorMsg()
                    << ", message: " << new_msg->GetErrorMsg();
        return -1;
      } break;
      default:
        break;
    }

    msg->Flip();
  }

  return result;
}

modelbox::Status ToolCommandServer::GetSockFile(const std::string &conf_file,
                                                std::string &connect_url) {
#ifdef BUILD_TEST
  connect_url = CONTROL_UNIX_PATH;
  return modelbox::STATUS_OK;
#else
  std::shared_ptr<Configuration> config;
  if (conf_file.empty()) {
    config = LoadSubConfig(DEFAULT_MODELBOX_CONF);
  } else {
    config = LoadSubConfig(conf_file);
  }

  if (config == nullptr) {
    std::cout << "conf file is invalid." << std::endl;
    return modelbox::STATUS_INVALID;
  }

  if (config->GetBool("control.enable", false) == false) {
    std::cout << "server control function is disabled." << std::endl;
    return modelbox::STATUS_BADCONF;
  }

  connect_url = config->GetString("control.listen");
  if (connect_url.empty()) {
    std::cout << "control listen sock get failed." << std::endl;
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
#endif
}

int ToolCommandServer::Run(int argc, char *argv[]) {
  int cmdtype = 0;
  std::string connect_url;
  std::string conf_file;

  MODELBOX_COMMAND_GETOPT_BEGIN(cmdtype, server_options)
  switch (cmdtype) {
    case MODELBOX_TOOL_SERVER_CONNECT:
      connect_url = optarg;
      break;
    case MODELBOX_TOOL_SERVER_INFO_FROM_CONF:
      conf_file = optarg;
      break;
    default:
      break;
  }
  MODELBOX_COMMAND_GETOPT_END()
  modelbox::Status status = GetSockFile(conf_file, connect_url);
  if (status != modelbox::STATUS_OK) {
    return 1;
  }

  auto ret = InitClient(connect_url);
  if (!ret) {
    return 1;
  }
  Defer { CloseClient(); };

  ret = SendCommand(argc - optind, argv + optind, connect_url);
  if (!ret) {
    return 1;
  }

  return RecvCommand();
}

}  // namespace modelbox