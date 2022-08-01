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


#include "modelbox/base/status.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <string>
#include <thread>

#include "modelbox/base/utils.h"
#include "gtest/gtest.h"

namespace modelbox {

class StatusTest : public testing::Test {
 public:
  StatusTest() = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

TEST_F(StatusTest, OK) {
  EXPECT_EQ(STATUS_OK, STATUS_SUCCESS);
  const Status &result = STATUS_OK;
  const Status &result_OK1 = STATUS_OK;
  EXPECT_EQ(&result, &result_OK1);
  EXPECT_TRUE(result);
}

TEST_F(StatusTest, EqualNotEqual) {
  Status first(STATUS_SUCCESS);
  Status second(STATUS_OK);
  Status thrid(STATUS_ALREADY);

  EXPECT_EQ(first, second);
  EXPECT_NE(first, thrid);

  Status ret = STATUS_EXIST;
  EXPECT_TRUE(STATUS_EXIST == ret);
}

TEST_F(StatusTest, Message) {
  const char *msg = "this is message";
  Status result(STATUS_SUCCESS, msg);
  EXPECT_EQ(result.Errormsg(), msg);
}

TEST_F(StatusTest, WrapError) {
  const char *wrap_msg = "origin wrap msg.";
  const char *empty_msg = "";
  const char *middle_msg = "middle msg.";
  const char *msg_ret = "new msg.";
  Status origin(STATUS_EXIST, wrap_msg);
  Status middle(origin, middle_msg);
  Status empty(middle, empty_msg);
  Status ret(empty, msg_ret);
  std::string expect_msg = ret.StrCode() + ", " + ret.Errormsg() + " -> " +
                           middle.Errormsg() + " -> " + origin.Errormsg();
  EXPECT_EQ(ret.WrapErrormsgs(), expect_msg);
}

TEST_F(StatusTest, WrapErrorCodeOnly) {
  const char *wrap_msg = "";
  const char *empty_msg = "";
  const char *msg_ret = "";
  Status origin(STATUS_EXIST, wrap_msg);
  Status empty(origin, empty_msg);
  Status ret(empty, msg_ret);
  std::string expect_msg = origin.StrCode();
  EXPECT_EQ(ret.WrapErrormsgs(), expect_msg);
}

TEST_F(StatusTest, ToString) {
  std::string msg = "this is message";
  Status result(STATUS_SUCCESS, msg);
  EXPECT_EQ(result.Errormsg(), msg);
  EXPECT_EQ(result.ToString(), "code: Success, errmsg: " + msg);
}

TEST_F(StatusTest, OperationLogicalNot) {
  Status result_ok(STATUS_SUCCESS);
  EXPECT_FALSE(!result_ok);
  Status result_fault(STATUS_FAULT);
  EXPECT_TRUE(!result_fault);
}

TEST_F(StatusTest, GetAllMessage) {
  int num = STATUS_LASTFLAG;
  for (int i = 0; i < num; i++) {
    Status check((StatusCode)i);
    EXPECT_NE(check.StrCode(), "");
  }
}

}  // namespace modelbox