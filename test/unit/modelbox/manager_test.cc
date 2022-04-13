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

#include "manager.h"

#include <dlfcn.h>
#include <stdio.h>

#include <fstream>
#include <future>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "manager_conf.h"
#include "manager_monitor.h"
#include "test_config.h"

namespace modelbox {

class ManagerTest : public testing::Test {
 public:
 protected:
  ManagerTest() {
    printf(
        "NOTICE: For valgrind memory leak check, please run with "
        "--child-silent-after-fork=yes to  skip child process check.\n");
  }
  virtual void SetUp(){

  };
  virtual void TearDown(){

  };
};

class ManagerTestServer {
 public:
  ManagerTestServer() {}
  virtual ~ManagerTestServer() { Stop(); }
  void Start() {
    manager_init_server();
    memset(&test_app, 0, sizeof(test_app));
    thread_ = std::thread(&ManagerTestServer::Run);
    running_ = true;
    usleep(1000);
  }

  void Stop() {
    int unused __attribute__((unused));
    if (running_ == false) {
      return;
    }

    running_ = false;

    manager_exit();
    if (thread_.joinable()) {
      thread_.join();
    }
    memset(&test_app, 0, sizeof(test_app));
    std::string delpid = "rm -f /tmp/modelbox_app_*";
    unused = system(delpid.c_str());
  }

  static int Run() { return manager_run(); }

 private:
  bool running_{false};
  std::thread thread_;
};

class ManagerTestApp {
 public:
  ManagerTestApp() {
    Reset();
    test_app.run = ManagerTestApp::Run;
    test_app.arg1 = this;
  }

  ~ManagerTestApp() {
    test_app.run = nullptr;
    test_app.arg1 = nullptr;
  }

  void Reset() {
    run_count_ = 0;
    pause_ = 0;
    pause_after_count_ = 0;
    ignore_segv_ = 0;
  }

  void SetRunCount(int n) { run_count_ = n; }

  void SetIgnoreSigSegv() { ignore_segv_ = 1; }

  void SetPause(int pause_time) { pause_ = pause_time; }

  void SetPauseAfterCount(int pause_time) { pause_after_count_ = pause_time; }

  static void SignalSegHandler(int sig) {
    printf("handle signal %d\n", sig);
    return;
  }

  static void SignalSegHandleExist(int sig) { _exit(1); }

  static int Run(struct Test_App *app, int count, const char *name) {
    class ManagerTestApp *test_app = (ManagerTestApp *)app->arg1;
    if (test_app->ignore_segv_) {
      signal(SIGSEGV, SignalSegHandler);
    } else {
      signal(SIGSEGV, SignalSegHandleExist);
    }

    while (test_app->pause_ > 0) {
      sleep(1);
      test_app->pause_--;
    }

    if (count <= test_app->run_count_ && test_app->run_count_ > 0) {
      return 0;
    }

    while (test_app->pause_after_count_ > 0) {
      sleep(1);
      test_app->pause_after_count_--;
    }

    return -1;
  }

 private:
  int run_count_{0};
  int ignore_segv_{0};
  int pause_{0};
  int pause_after_count_{0};
};

TEST_F(ManagerTest, DISABLED_Start) {
  ManagerTestServer server;
  server.Start();
  EXPECT_EQ(0, app_start("1", "sleep 1", nullptr, 0));
  sleep(1);
  int pid = app_getpid("1");
  EXPECT_GT(pid, 0);

  for (int i = 0; i < conf_watchdog_timeout * 2; i++) {
    EXPECT_EQ(0, app_alive("1"));
    EXPECT_EQ(pid, app_getpid("1"));
  }
}

TEST_F(ManagerTest, DISABLED_Start_dup) {
  ManagerTestServer server;
  server.Start();
  EXPECT_EQ(0, app_start("1", "test", NULL, 0));
  EXPECT_EQ(0, app_start("2", "test", NULL, 0));
  EXPECT_EQ(0, app_start("3", "test", NULL, 0));
  sleep(1);
  EXPECT_NE(0, app_start("2", "test", NULL, 0));
}

TEST_F(ManagerTest, DISABLED_Start_many) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(0, app_start(std::to_string(i).c_str(), "test", nullptr, 0));
  }
  sleep(1);
  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(0, app_alive(std::to_string(i).c_str()));
  }
}

TEST_F(ManagerTest, DISABLED_Start_stop_half) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(0, app_start(std::to_string(i).c_str(), "test", nullptr, 0));
  }
  sleep(1);
  for (int i = 0; i < 128; i++) {
    if (i % 2 == 0) {
      continue;
    }
    EXPECT_EQ(0, app_stop(std::to_string(i).c_str(), 0));
  }
  sleep(1);
  for (int i = 0; i < 128; i++) {
    if (i % 2 == 0) {
      EXPECT_EQ(0, app_alive(std::to_string(i).c_str()));
      continue;
    }
    EXPECT_NE(0, app_alive(std::to_string(i).c_str()));
  }
}

TEST_F(ManagerTest, DISABLED_Start_stop_all) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(0, app_start(std::to_string(i).c_str(), "test", nullptr, 0));
  }
  sleep(1);
  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(0, app_stop(std::to_string(i).c_str(), 0));
  }
  sleep(1);
  for (int i = 0; i < 128; i++) {
    EXPECT_NE(0, app_alive(std::to_string(i).c_str()));
  }
}

}  // namespace modelbox
