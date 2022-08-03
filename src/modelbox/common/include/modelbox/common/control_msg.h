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

#ifndef MODELBOX_CONTROL_MSG_H_
#define MODELBOX_CONTROL_MSG_H_

#include <modelbox/base/config.h>
#include <modelbox/modelbox.h>

#ifdef BUILD_TEST
#include "test_config.h"
constexpr const char CONTROL_UNIX_PATH[] = "/tmp/modelbox.sock";
#else
constexpr const char CONTROL_UNIX_PATH[] = "/var/run/modelbox.sock";
#endif

namespace modelbox {
constexpr uint64_t CONTROL_MAGIC = 0x676d4c5443767273; /* srvCTLmg */
constexpr int CONTROL_MAX_MSG_LEN = 32 * 1024;

typedef enum SERVER_CONTROL_MSG_TYPE {
  SERVER_CONTROL_MSG_TYPE_NULL = 0,
  SERVER_CONTROL_MSG_TYPE_CMD = 1,
  SERVER_CONTROL_MSG_TYPE_RESULT = 2,
  SERVER_CONTROL_MSG_TYPE_OUTMSG = 3,
  SERVER_CONTROL_MSG_TYPE_ERRMSG = 4,
  SERVER_CONTROL_MSG_TYPE_HELP = 5,
  SERVER_CONTROL_MSG_TYPE_ERR = 6,
  SERVER_CONTROL_MSG_TYPE_STRING = 7,
  SERVER_CONTROL_MSG_TYPE_BUFF
} SERVER_CONTROL_MSG_TYPE;

struct ControlMsgHead {
  uint64_t magic;
  uint32_t type;
  uint32_t len;
  uint8_t msg[0];
};

class ControlMsg {
 public:
  ControlMsg(size_t buffer_size);
  ControlMsg();
  virtual ~ControlMsg();

  SERVER_CONTROL_MSG_TYPE GetMsgType();

  size_t GetMsgDataLen();
  size_t GetMsgLen();
  const uint8_t *GetMsgData();
  struct ControlMsgHead *GetControlMsgHead();

  modelbox::Status AppendDataLen(size_t len);
  modelbox::Status AppendData(uint8_t *data, size_t data_len);

  uint8_t *GetData();
  size_t GetDataLen();
  uint8_t *GetDataTail();
  size_t GetRemainSpace();

  modelbox::Status Unserialize();
  modelbox::Status Serialize();
  modelbox::Status BuildFromOtherMsg(ControlMsg *from_control_msg);
  void Flip();
  void Reset();

 protected:
  void SetMsgType(SERVER_CONTROL_MSG_TYPE type);
  modelbox::Status Unserialize(
      const std::shared_ptr<std::vector<uint8_t>> &data_buff);
  virtual modelbox::Status UnSerializeMsg(uint8_t *buff, size_t buff_len);
  virtual size_t SerializeMsg(uint8_t *buff, size_t buff_max_len);

  std::shared_ptr<std::vector<uint8_t>> data_buff_;
  size_t data_len_{0};
  SERVER_CONTROL_MSG_TYPE msg_type_{SERVER_CONTROL_MSG_TYPE_NULL};
  bool data_ready_{false};
};

class ControlMsgResult : public ControlMsg {
 public:
  ControlMsgResult();
  ~ControlMsgResult() override;

  void SetResult(int result);

  int GetResult();

 protected:
  size_t SerializeMsg(uint8_t *buff, size_t buff_max_len) override;
  modelbox::Status UnSerializeMsg(uint8_t *buff, size_t buff_len) override;

 private:
  int result_;
};

class ControlMsgString : public ControlMsg {
 public:
  ControlMsgString();
  ~ControlMsgString() override;

  void SetString(const std::string &str);
  const std::string &GetString();

 protected:
  size_t SerializeMsg(uint8_t *buff, size_t buff_max_len) override;
  modelbox::Status UnSerializeMsg(uint8_t *buff, size_t buff_len) override;

 private:
  std::string str_;
};

class ControlMsgHelp : public ControlMsgString {
 public:
  ControlMsgHelp();
  ~ControlMsgHelp() override;
};

class ControlMsgStdout : public ControlMsgString {
 public:
  ControlMsgStdout();
  ~ControlMsgStdout() override;
};

class ControlMsgErrout : public ControlMsgString {
 public:
  ControlMsgErrout();
  ~ControlMsgErrout() override;
};

class ControlMsgCmd : public ControlMsg {
 public:
  ControlMsgCmd();
  ~ControlMsgCmd() override;

  void SetArgs(int argc, char *argv[]);
  int GetArgc();
  std::vector<std::string> GetArgv();

 protected:
  uint32_t CMD_MAGIC = 0x5F446d43; /* CmD_*/
  struct MsgCmdHead {
    uint32_t argc;
    uint32_t magic;
    char args[0];
  };

  struct MsgCmdArg {
    uint32_t len;
    uint32_t magic;
    char arg[0];
  };
  size_t SerializeMsg(uint8_t *buff, size_t buff_max_len) override;
  modelbox::Status UnSerializeMsg(uint8_t *buff, size_t buff_len) override;

 private:
  int argc_;
  std::vector<std::string> argv_;
};

class ControlMsgError : public ControlMsg {
 public:
  ControlMsgError();
  ~ControlMsgError() override;

  void SetError(int err_code, const std::string &err_msg);

  std::string GetErrorMsg();

  int GetErrorCode();

 protected:
  struct MsgErrorHead {
    int err_code;
    int err_msg_len;
    char err_msg[0];
  };

  size_t SerializeMsg(uint8_t *buff, size_t buff_max_len) override;
  modelbox::Status UnSerializeMsg(uint8_t *buff, size_t buff_len) override;

 private:
  int err_code_;
  std::string err_msg_;
};

class ControlMsgBuilder {
 public:
  static std::shared_ptr<ControlMsg> Build(
      const std::shared_ptr<ControlMsg> &from_msg);
};

}  // namespace modelbox

#endif  // MODELBOX_CONTROL_MSG_H_
