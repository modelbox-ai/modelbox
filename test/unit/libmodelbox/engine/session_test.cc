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

#include "modelbox/session.h"

#include <functional>
#include <future>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "modelbox/base/log.h"
#include "modelbox/device/mockdevice/device_mockdevice.h"

namespace modelbox {
class SessionTest : public testing::Test {};

TEST_F(SessionTest, SessionManage) {
  SessionManager sess_mgr;
  auto session = sess_mgr.CreateSession(nullptr);
  ASSERT_NE(session, nullptr);
  {
    auto session1 = sess_mgr.CreateSession(nullptr);
    ASSERT_NE(session1, nullptr);
    auto session2 = sess_mgr.CreateSession(nullptr);
    ASSERT_NE(session2, nullptr);
    EXPECT_EQ(sess_mgr.GetSessions().size(), 3);
  }
  EXPECT_EQ(sess_mgr.GetSessions().size(), 1);
}

class TestSessionIO : public SessionIO {
 public:
 public:
  virtual Status SetOutputMeta(const std::string &port_name,
                               std::shared_ptr<DataMeta> meta) {
    return STATUS_OK;
  }
  virtual Status Send(const std::string &port_name,
                      std::shared_ptr<BufferList> buffer_list) {
    return STATUS_OK;
  }
  virtual Status Recv(OutputBufferList &map_buffer_list, int timeout = 0) {
    return STATUS_OK;
  }
  virtual Status Close() { return STATUS_OK; }
  virtual Status Shutdown() { return STATUS_OK; }

  bool TestSessionEnd() { return session_end_; }

  std::shared_ptr<FlowUnitError> GetSessionError() { return error_; }

 protected:
  virtual void SessionEnd(std::shared_ptr<FlowUnitError> error = nullptr) {
    error_ = error;
    session_end_ = true;
  }

  bool session_end_{false};
  std::shared_ptr<FlowUnitError> error_;
};

TEST_F(SessionTest, SessionClose) {
  SessionManager sess_mgr;
  auto io1 = std::make_shared<TestSessionIO>();
  auto io2 = std::make_shared<TestSessionIO>();
  {
    auto session = sess_mgr.CreateSession(nullptr);
    session->SetSessionIO(io1);
  }
  {
    auto session = sess_mgr.CreateSession(nullptr);
    session->SetSessionIO(io2);
    session->Close();
  }
  EXPECT_TRUE(io1->TestSessionEnd());
  EXPECT_EQ(io1->GetSessionError(), nullptr);
  EXPECT_TRUE(io2->TestSessionEnd());
  ASSERT_NE(io2->GetSessionError(), nullptr);
  auto end_error = io2->GetSessionError();
  EXPECT_EQ(end_error->GetDesc(), "EOF");
}

}  // namespace modelbox