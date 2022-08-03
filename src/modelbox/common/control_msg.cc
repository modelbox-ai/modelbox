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

#include "modelbox/common/control_msg.h"

#include "securec.h"

namespace modelbox {

ControlMsg::ControlMsg(size_t buffer_size) {
  data_buff_ = std::make_shared<std::vector<uint8_t>>();
  data_buff_->resize(buffer_size);
}

ControlMsg::ControlMsg() {
  data_buff_ = std::make_shared<std::vector<uint8_t>>();
  data_buff_->resize(CONTROL_MAX_MSG_LEN);
}

ControlMsg::~ControlMsg() = default;

size_t ControlMsg::GetRemainSpace() { return data_buff_->size() - data_len_; }

uint8_t *ControlMsg::GetData() { return data_buff_->data(); }
size_t ControlMsg::GetDataLen() { return data_len_; }

uint8_t *ControlMsg::GetDataTail() { return data_buff_->data() + data_len_; }

SERVER_CONTROL_MSG_TYPE ControlMsg::GetMsgType() {
  if (data_ready_ == false) {
    return msg_type_;
  }

  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  return (SERVER_CONTROL_MSG_TYPE)msg_head->type;
}

void ControlMsg::SetMsgType(SERVER_CONTROL_MSG_TYPE type) { msg_type_ = type; }

size_t ControlMsg::GetMsgLen() {
  if (data_ready_ == false) {
    return 0;
  }

  return sizeof(struct ControlMsgHead) + GetMsgDataLen();
}

size_t ControlMsg::GetMsgDataLen() {
  if (data_ready_ == false) {
    return 0;
  }

  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  return msg_head->len;
}

const uint8_t *ControlMsg::GetMsgData() {
  if (data_ready_ == false) {
    return nullptr;
  }

  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  return msg_head->msg;
}

struct ControlMsgHead *ControlMsg::GetControlMsgHead() {
  if (data_ready_ == false) {
    return nullptr;
  }

  return (struct ControlMsgHead *)data_buff_->data();
}

void ControlMsg::Flip() {
  if (data_ready_ == false) {
    return;
  }

  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  int data_msg_len = sizeof(*msg_head) + msg_head->len;
  int last_data_len = data_len_ - data_msg_len;
  if (last_data_len > 0) {
    auto ret = memmove_s(data_buff_->data(), data_buff_->size(),
                         data_buff_->data() + data_msg_len, last_data_len);
    if (ret != EOK) {
      MBLOG_ERROR << "memcpy_s failed";
    }
  }
  data_len_ = last_data_len;
  data_ready_ = false;

  Unserialize();
}

modelbox::Status ControlMsg::AppendDataLen(size_t len) {
  if (len > GetRemainSpace()) {
    return modelbox::STATUS_NOSPACE;
  }

  data_len_ += len;

  return modelbox::STATUS_OK;
}

modelbox::Status ControlMsg::AppendData(uint8_t *data, size_t data_len) {
  if (data_len > GetRemainSpace()) {
    return modelbox::STATUS_NOSPACE;
  }

  auto ret = memcpy_s(GetDataTail(), GetRemainSpace(), data, data_len);
  if (ret != EOK) {
    return modelbox::STATUS_FAULT;
  }

  AppendDataLen(data_len);
  return modelbox::STATUS_OK;
}

modelbox::Status ControlMsg::Unserialize() { return Unserialize(data_buff_); }

modelbox::Status ControlMsg::Unserialize(
    const std::shared_ptr<std::vector<uint8_t>> &data_buff) {
  auto *msg_head = (struct ControlMsgHead *)data_buff->data();

  if (data_ready_ == true) {
    return modelbox::STATUS_OK;
  }

  if (data_len_ < sizeof(struct ControlMsgHead)) {
    return modelbox::STATUS_AGAIN;
  }

  if (msg_head->magic != CONTROL_MAGIC) {
    return {modelbox::STATUS_INVALID, "magic is invalid"};
  }

  if (msg_head->len >= data_buff_->size() - sizeof(*msg_head)) {
    return {modelbox::STATUS_INVALID, "length is invalid"};
  }

  if (msg_head->type >= SERVER_CONTROL_MSG_TYPE_BUFF) {
    return {modelbox::STATUS_INVALID, "type is invalid"};
  }

  data_ready_ = true;
  SetMsgType((SERVER_CONTROL_MSG_TYPE)msg_head->type);
  return modelbox::STATUS_OK;
}

void ControlMsg::Reset() {
  data_ready_ = false;
  data_len_ = 0;
}

modelbox::Status ControlMsg::Serialize() {
  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  auto msg_len =
      SerializeMsg(msg_head->msg, data_buff_->size() - sizeof(*msg_head));
  if (msg_len < 0) {
    return modelbox::STATUS_NOBUFS;
  }

  msg_head->magic = CONTROL_MAGIC;
  msg_head->type = msg_type_;
  msg_head->len = msg_len;
  data_len_ = sizeof(*msg_head) + msg_len;
  data_ready_ = true;

  return modelbox::STATUS_OK;
}

modelbox::Status ControlMsg::BuildFromOtherMsg(ControlMsg *from_control_msg) {
  auto ret = modelbox::STATUS_OK;
  Defer {
    if (!ret) {
      Reset();
    }
  };

  data_len_ = from_control_msg->data_len_;
  ret = Unserialize(from_control_msg->data_buff_);
  if (!ret) {
    return ret;
  }

  size_t outer_msg_data_len =
      from_control_msg->GetDataLen() - from_control_msg->GetMsgLen();
  if (outer_msg_data_len) {
    auto rc = memcpy_s(
        data_buff_->data(), data_buff_->size(),
        from_control_msg->data_buff_->data() + from_control_msg->GetMsgLen(),
        outer_msg_data_len);
    if (rc != EOK) {
      return modelbox::STATUS_NOBUFS;
    }
  }

  data_len_ = from_control_msg->GetMsgLen();
  auto tmp = from_control_msg->data_buff_;
  from_control_msg->data_buff_ = data_buff_;
  data_buff_ = tmp;
  from_control_msg->Reset();
  from_control_msg->data_len_ = outer_msg_data_len;
  if (outer_msg_data_len > 0) {
    from_control_msg->Unserialize();
  }

  auto *msg_head = (struct ControlMsgHead *)data_buff_->data();
  return UnSerializeMsg(msg_head->msg, msg_head->len);
}

size_t ControlMsg::SerializeMsg(uint8_t *buff, size_t buff_max_len) {
  return 0;
}

modelbox::Status ControlMsg::UnSerializeMsg(uint8_t *buff, size_t buff_len) {
  return modelbox::STATUS_OK;
}

ControlMsgResult::ControlMsgResult() {
  SetMsgType(SERVER_CONTROL_MSG_TYPE_RESULT);
}

ControlMsgResult::~ControlMsgResult() = default;
void ControlMsgResult::SetResult(int result) { result_ = result; }

int ControlMsgResult::GetResult() { return result_; }

size_t ControlMsgResult::SerializeMsg(uint8_t *buff, size_t buff_max_len) {
  int *result = nullptr;
  if (buff_max_len < sizeof(int)) {
    return -1;
  }
  result = (int *)buff;
  *result = result_;
  return sizeof(*result);
}

modelbox::Status ControlMsgResult::UnSerializeMsg(uint8_t *buff,
                                                  size_t buff_len) {
  int *result;
  result = nullptr;
  if (buff_len < sizeof(*result)) {
    return modelbox::STATUS_NOBUFS;
  }

  result = (int *)buff;
  result_ = *result;

  return modelbox::STATUS_OK;
}

ControlMsgString::ControlMsgString() {
  SetMsgType(SERVER_CONTROL_MSG_TYPE_STRING);
}
ControlMsgString::~ControlMsgString() = default;

const std::string &ControlMsgString::GetString() { return str_; }

void ControlMsgString::SetString(const std::string &str) { str_ = str; }

size_t ControlMsgString::SerializeMsg(uint8_t *buff, size_t buff_max_len) {
  if (buff_max_len < str_.length() + 1) {
    return -1;
  }

  auto rc = memcpy_s(buff, buff_max_len, str_.c_str(), str_.length());
  if (rc != EOK) {
    return -1;
  }

  return str_.length() + 1;
}

modelbox::Status ControlMsgString::UnSerializeMsg(uint8_t *buff,
                                                  size_t buff_len) {
  str_.assign((char *)buff, buff_len);
  return modelbox::STATUS_OK;
}

ControlMsgHelp::ControlMsgHelp() { SetMsgType(SERVER_CONTROL_MSG_TYPE_HELP); }
ControlMsgHelp::~ControlMsgHelp() = default;

ControlMsgStdout::ControlMsgStdout() {
  SetMsgType(SERVER_CONTROL_MSG_TYPE_OUTMSG);
}
ControlMsgStdout::~ControlMsgStdout() = default;

ControlMsgErrout::ControlMsgErrout() {
  SetMsgType(SERVER_CONTROL_MSG_TYPE_ERRMSG);
}
ControlMsgErrout::~ControlMsgErrout() = default;

ControlMsgCmd::ControlMsgCmd() { SetMsgType(SERVER_CONTROL_MSG_TYPE_CMD); }

ControlMsgCmd::~ControlMsgCmd() = default;

void ControlMsgCmd::SetArgs(int argc, char *argv[]) {
  argv_.clear();
  for (int i = 0; i < argc; i++) {
    argv_.emplace_back(argv[i]);
  }
}

int ControlMsgCmd::GetArgc() { return argc_; }
std::vector<std::string> ControlMsgCmd::GetArgv() { return argv_; }

size_t ControlMsgCmd::SerializeMsg(uint8_t *buff, size_t buff_max_len) {
  auto *cmd_head = (struct MsgCmdHead *)buff;
  if (buff_max_len < sizeof(*cmd_head)) {
    return -1;
  }

  cmd_head->magic = CMD_MAGIC;
  size_t cmd_data_free_len = buff_max_len - sizeof(*cmd_head);
  char *cmd_data = cmd_head->args;

  for (auto &arg : argv_) {
    auto *cmd_arg = (struct MsgCmdArg *)cmd_data;
    if (cmd_data_free_len < sizeof(*cmd_arg)) {
      return -1;
    }
    cmd_data_free_len -= sizeof(*cmd_arg);
    cmd_data += sizeof(*cmd_arg);

    if (cmd_data_free_len < arg.length() + 1) {
      return -1;
    }
    auto ret =
        strncpy_s(cmd_arg->arg, cmd_data_free_len, arg.c_str(), arg.length());
    if (ret != 0) {
      MBLOG_ERROR << "strncpy_s failed.";
      return -1;
    }
    cmd_arg->len = arg.length() + 1;
    cmd_data_free_len -= arg.length() + 1;
    cmd_data += arg.length() + 1;
    cmd_arg->magic = CMD_MAGIC;
  }

  cmd_head->argc = argv_.size();
  return (uint8_t *)cmd_data - buff;
}

modelbox::Status ControlMsgCmd::UnSerializeMsg(uint8_t *buff, size_t buff_len) {
  auto *cmd_head = (struct MsgCmdHead *)buff;
  if (buff_len < sizeof(*cmd_head)) {
    return modelbox::STATUS_NOBUFS;
  }

  if (cmd_head->magic != CMD_MAGIC) {
    return modelbox::STATUS_INVALID;
  }

  argv_.clear();
  char *cmd_data = cmd_head->args;
  while (true) {
    size_t left_data_len = buff_len - ((uint8_t *)cmd_data - buff);
    if (left_data_len == 0) {
      break;
    }

    if (left_data_len < 0) {
      return modelbox::STATUS_NOBUFS;
    }

    auto *cmd_arg = (struct MsgCmdArg *)cmd_data;
    left_data_len -= sizeof(*cmd_arg);
    if (cmd_arg->len > left_data_len || cmd_arg->len <= 0) {
      return modelbox::STATUS_NOBUFS;
    }

    if (cmd_arg->magic != CMD_MAGIC) {
      return modelbox::STATUS_INVALID;
    }

    if (cmd_arg->arg[cmd_arg->len - 1] != 0) {
      return modelbox::STATUS_INVALID;
    }

    std::string arg;
    arg.assign(cmd_arg->arg, cmd_arg->len - 1);
    argv_.emplace_back(arg);
    cmd_data += sizeof(*cmd_arg) + cmd_arg->len;
  }

  if (cmd_head->argc != argv_.size() || cmd_head->argc <= 0) {
    return modelbox::STATUS_INVALID;
  }

  argc_ = cmd_head->argc;

  return modelbox::STATUS_OK;
}

ControlMsgError::ControlMsgError() { SetMsgType(SERVER_CONTROL_MSG_TYPE_ERR); }

ControlMsgError::~ControlMsgError() = default;

void ControlMsgError::SetError(int err_code, const std::string &err_msg) {
  err_code_ = err_code;
  err_msg_ = err_msg;
}

std::string ControlMsgError::GetErrorMsg() { return err_msg_; }

int ControlMsgError::GetErrorCode() { return err_code_; }

size_t ControlMsgError::SerializeMsg(uint8_t *buff, size_t buff_max_len) {
  auto *err_msg_head = (struct MsgErrorHead *)buff;
  if (buff_max_len < sizeof(*err_msg_head)) {
    return -1;
  }
  buff_max_len -= sizeof(*err_msg_head);
  if (buff_max_len < err_msg_.length() + 1) {
    return -1;
  }

  auto rc = memcpy_s(err_msg_head->err_msg, buff_max_len, err_msg_.c_str(),
                     err_msg_.length());
  if (rc != EOK) {
    return -1;
  }

  err_msg_head->err_msg_len = err_msg_.length();
  err_msg_head->err_code = err_code_;

  return sizeof(*err_msg_head) + err_msg_head->err_msg_len;
}

modelbox::Status ControlMsgError::UnSerializeMsg(uint8_t *buff,
                                                 size_t buff_len) {
  auto *err_msg_head = (struct MsgErrorHead *)buff;
  if (buff_len < sizeof(*err_msg_head)) {
    return modelbox::STATUS_NOBUFS;
  }

  buff_len -= sizeof(*err_msg_head);
  if (buff_len > (size_t)(err_msg_head->err_msg_len)) {
    return modelbox::STATUS_NOBUFS;
  }

  err_code_ = err_msg_head->err_code;
  err_msg_.assign((char *)err_msg_head->err_msg, err_msg_head->err_msg_len);
  return modelbox::STATUS_OK;
}

std::shared_ptr<ControlMsg> ControlMsgBuilder::Build(
    const std::shared_ptr<ControlMsg> &from_msg) {
  std::shared_ptr<ControlMsg> msg;
  switch (from_msg->GetMsgType()) {
    case SERVER_CONTROL_MSG_TYPE_STRING:
      msg = std::make_shared<ControlMsgString>();
      break;
    case SERVER_CONTROL_MSG_TYPE_CMD:
      msg = std::make_shared<ControlMsgCmd>();
      break;
    case SERVER_CONTROL_MSG_TYPE_RESULT:
      msg = std::make_shared<ControlMsgResult>();
      break;
    case SERVER_CONTROL_MSG_TYPE_OUTMSG:
      break;
    case SERVER_CONTROL_MSG_TYPE_ERR:
      msg = std::make_shared<ControlMsgError>();
      break;
    case SERVER_CONTROL_MSG_TYPE_HELP:
      msg = std::make_shared<ControlMsgHelp>();
      break;
    default:
      break;
  }

  if (msg == nullptr) {
    modelbox::StatusError = modelbox::STATUS_NOTFOUND;
    return nullptr;
  }

  auto ret = msg->BuildFromOtherMsg(from_msg.get());
  if (!ret) {
    modelbox::StatusError = ret;
    return nullptr;
  }

  return msg;
}

}  // namespace modelbox