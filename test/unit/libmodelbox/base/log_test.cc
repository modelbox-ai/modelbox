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


#include "modelbox/base/log.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "securec.h"
namespace modelbox {

class LoggerTest : public Logger {
 public:
  LoggerTest() = default;
  ~LoggerTest() override = default;
  void Vprint(LogLevel level, const char *file, int lineno, const char *func,
              const char *format, va_list ap) override {
    char msg[1024];
    vsnprintf_s(msg, sizeof(msg), sizeof(msg), format, ap);
    std::unique_lock<std::mutex> lock(mutex_);
    log_msg_level_ = level;
    log_msg_file_ = file;
    log_msg_line_ = lineno;
    log_msg_ = msg;
    log_a_msg_ = true;
  };
  void SetLogLevel(LogLevel level) override { level_ = level; };
  LogLevel GetLogLevel() override { return level_; };

  std::string GetLogMsg() { return log_msg_; }
  void ClearLogMsg() {
    log_a_msg_ = false;
    return log_msg_.clear();
  }

  const char *GetLogMsgFile() { return log_msg_file_; }
  void ClearLogMsgFile() { log_msg_file_ = nullptr; }

  int GetLogMsgLine() { return log_msg_line_; }
  void ClearLogMsgLine() { log_msg_line_ = -1; }

  LogLevel GetLogMsgLevel() { return log_msg_level_; }
  void ClearLogMsgLevel() { log_msg_level_ = LOG_OFF; }

  bool IsLogMsg() { return log_a_msg_; }

  void Clear() {
    ClearLogMsg();
    ClearLogMsgFile();
    ClearLogMsgLine();
    ClearLogMsgLevel();
  }

 private:
  LogLevel level_ = LOG_DEBUG;
  std::mutex mutex_;
  std::string log_msg_;
  LogLevel log_msg_level_;
  const char *log_msg_file_;
  int log_msg_line_;
  bool log_a_msg_;
};

class LogTest : public testing::Test {
 public:
  LogTest() = default;

 protected:
  void SetUp() override {
    old_logger_ = ModelBoxLogger.GetLogger();
    old_level_ = ModelBoxLogger.GetLogger()->GetLogLevel();
  };
  void TearDown() override {
    ModelBoxLogger.SetLogger(old_logger_);
    ModelBoxLogger.GetLogger()->SetLogLevel(old_level_);
  };

 private:
  std::shared_ptr<Logger> old_logger_;
  LogLevel old_level_;
};

TEST_F(LogTest, LoggerConsole) {
  ModelBoxLogger.GetLogger()->SetLogLevel(LOG_DEBUG);
  MODELBOX_DEBUG("%s", "this is DEBUG");
  MODELBOX_INFO("%s", "this is INFO");
  MODELBOX_NOTICE("%s", "this is NOTICE");
  MODELBOX_ERROR("%s", "this is ERROR");
  MODELBOX_FATAL("%s", "this is FATAL");

  MBLOG_DEBUG << "this is DEBUG";
  MBLOG_INFO << "this is INFO";
  MBLOG_NOTICE << "this is NOTICE";
  MBLOG_ERROR << "this is ERROR";
  MBLOG_FATAL << "this is FATAL";
  MBLOG_STACKTRACE(LOG_INFO);
}

TEST_F(LogTest, LoggerWithID) {
  ModelBoxLogger.GetLogger()->SetLogLevel(LOG_DEBUG);
  auto logid = modelbox::LogSetLogID("LOGID");
  MBLOG_DEBUG << "this is DEBUG";
  MBLOG_INFO << "this is INFO";
  MBLOG_NOTICE << "this is NOTICE";
  MBLOG_ERROR << "this is ERROR";
  MBLOG_FATAL << "this is FATAL";
  MBLOG_STACKTRACE(LOG_INFO);
}

TEST_F(LogTest, LoggerCallBackPrint) {
  std::string origin_msg = "this is message";
  std::string expect_msg;
  RegLogPrint([&](LogLevel level, const char *file, int lineno,
                  const char *func, const char *msg) { expect_msg = msg; });
  MBLOG_ERROR << origin_msg;
  EXPECT_EQ(origin_msg, expect_msg);
}

TEST_F(LogTest, LoggerCallBackVprint) {
  std::string origin_msg = "this is message";
  std::string expect_msg;
  RegLogVprint([&](LogLevel level, const char *file, int lineno,
                   const char *func, const char *format, va_list ap) {
    char buff[4096];
    vsnprintf_s(buff, sizeof(buff), sizeof(buff), format, ap);
    expect_msg = buff;
  });
  MBLOG_ERROR << origin_msg;
  EXPECT_EQ(origin_msg, expect_msg);
}

TEST_F(LogTest, LoggerDataCheck) {
  std::string expect_msg = "this is a log";
  auto test_logger = std::make_shared<LoggerTest>();
  ModelBoxLogger.SetLogger(test_logger);

  test_logger->Clear();
  test_logger->SetLogLevel(LOG_DEBUG);
  int line = __LINE__ + 1;
  MBLOG_DEBUG << expect_msg;
  EXPECT_EQ(expect_msg, test_logger->GetLogMsg());
  EXPECT_EQ(LOG_DEBUG, test_logger->GetLogMsgLevel());
  EXPECT_EQ(BASE_FILE_NAME, test_logger->GetLogMsgFile());
  EXPECT_EQ(line, test_logger->GetLogMsgLine());
  EXPECT_TRUE(test_logger->IsLogMsg());

  test_logger->Clear();
  test_logger->SetLogLevel(LOG_OFF);

  MBLOG_DEBUG << expect_msg;
  EXPECT_EQ("", test_logger->GetLogMsg());
  EXPECT_EQ(LOG_OFF, test_logger->GetLogMsgLevel());
  EXPECT_FALSE(test_logger->IsLogMsg());
}

TEST_F(LogTest, LoggerMultiThread) {
  std::string expect_msg = "this is a log";
  std::vector<std::thread> threads;
  int loop = 100;
  auto test_logger = std::make_shared<LoggerTest>();
  ModelBoxLogger.SetLogger(test_logger);

  test_logger->Clear();
  test_logger->SetLogLevel(LOG_DEBUG);

  for (int i = 0; i < 10; i++) {
    std::thread t([&, i]() {
      for (int j = 0; j < loop; j++) {
        MODELBOX_LOGSTREAM(LogLevel(j % LOG_OFF))
            << "Thread" << i << ": Number:" << j;
      }
    });

    threads.emplace_back(std::move(t));
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

}  // namespace modelbox