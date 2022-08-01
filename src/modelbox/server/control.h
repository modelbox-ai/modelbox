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

#ifndef MODELBOX_CONTROL_H_
#define MODELBOX_CONTROL_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/base/thread_pool.h>

#include <iostream>
#include <memory>

#include "modelbox/common/control_msg.h"
#include "modelbox/common/utils.h"
#include "server_plugin.h"

namespace modelbox {

using MsgSendFunc = std::function<int(void *, int)>;

class Control {
 public:
  Control();
  virtual ~Control();
  modelbox::Status Init(std::shared_ptr<modelbox::Configuration> config);
  modelbox::Status Start();
  modelbox::Status Stop();

 private:
  void ControlDaemon();
  void ProcessMsg(std::shared_ptr<ControlMsg> msg, MsgSendFunc reply_func);
  void ReplyMsgError(int err_code, const std::string &err_msg,
                     MsgSendFunc reply_func);
  modelbox::Status RecvMsg(std::shared_ptr<ControlMsg> recv_msg);
  int ProcessCmd(std::shared_ptr<ControlMsgCmd> msg, MsgSendFunc reply_func);
  int ProcessHelp(std::shared_ptr<ControlMsgHelp> msg, MsgSendFunc reply_func);

  std::shared_ptr<modelbox::Configuration> config_;
  std::string listen_path_;
  int server_fd_{-1};
  bool run_{false};
  std::thread daemon_thread_;
  std::shared_ptr<modelbox::ThreadPool> pool_;
};

class ControlStream : public OutStream {
 public:
  ControlStream();
  virtual ~ControlStream();

  bool HasError();

  void SetReplyFunc(MsgSendFunc reply_func);

 protected:
  MsgSendFunc reply_func_;
  bool has_error_{false};
};

class ControlOutStream : public ControlStream {
 public:
  ControlOutStream();
  ~ControlOutStream() override;

 protected:
  void ProcessStream(OStream *st) override;
};

class ControlErrStream : public ControlStream {
 public:
  ControlErrStream();
  ~ControlErrStream() override;

 protected:
  void ProcessStream(OStream *st) override;
};

}  // namespace modelbox

#endif  // MODELBOX_CONTROL_H_
