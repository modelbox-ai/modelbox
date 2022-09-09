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
  ManagerTest() = default;
  void SetUp() override{

  };
  void TearDown() override{

  };
};

class ManagerTestServer {
 public:
  ManagerTestServer() = default;
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

  static void SignalSegHandler(int sig) { printf("handle signal %d\n", sig); }

  static void SignalSegHandleExist(int sig) { _exit(1); }

  static int Run(struct Test_App *app, int count, const char *name) {
    auto *test_app = (ManagerTestApp *)app->arg1;
    printf("run %d %s\n", count, name);
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

TEST_F(ManagerTest, Start) {
  ManagerTestServer server;
  server.Start();
  struct app_start_info info;
  memset(&info, 0, sizeof(info));
  info.name = "1";
  info.cmdline = "sleep\0 1\0\0";
  info.cmd_max_len = PATH_MAX;
  info.check_alive = 1;
  info.keepalive_time = 60;
  info.heartbeat_interval = 5;
  EXPECT_EQ(0, app_start(&info));
  sleep(1);
  int pid = app_getpid("1");
  EXPECT_GT(pid, 0);

  for (int i = 0; i < conf_watchdog_timeout * 2; i++) {
    EXPECT_EQ(0, app_alive("1"));
    EXPECT_EQ(pid, app_getpid("1"));
  }
}

TEST_F(ManagerTest, Start_dup) {
  ManagerTestServer server;
  server.Start();
  struct app_start_info info;
  memset(&info, 0, sizeof(info));
  info.name = "1";
  info.cmdline = "test\0\0";
  info.cmd_max_len = PATH_MAX;
  info.check_alive = 1;
  info.keepalive_time = 60;
  info.heartbeat_interval = 5;
  EXPECT_EQ(0, app_start(&info));
  info.name = "2";
  EXPECT_EQ(0, app_start(&info));
  info.name = "3";
  EXPECT_EQ(0, app_start(&info));
  info.name = "2";
  EXPECT_NE(0, app_start(&info));
}

TEST_F(ManagerTest, Start_many) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 8; i++) {
    struct app_start_info info;
    memset(&info, 0, sizeof(info));
    info.name = std::to_string(i).c_str();
    info.cmdline = "test\0\0";
    info.cmd_max_len = PATH_MAX;
    info.check_alive = 1;
    info.keepalive_time = 60;
    info.heartbeat_interval = 5;
    EXPECT_EQ(0, app_start(&info));
  }
  sleep(1);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(0, app_alive(std::to_string(i).c_str()));
  }
}

TEST_F(ManagerTest, Start_stop_half) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 8; i++) {
    struct app_start_info info;
    memset(&info, 0, sizeof(info));
    info.name = std::to_string(i).c_str();
    info.cmdline = "test\0\0";
    info.cmd_max_len = PATH_MAX;
    info.check_alive = 1;
    info.keepalive_time = 60;
    info.heartbeat_interval = 5;
    EXPECT_EQ(0, app_start(&info));
  }
  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      continue;
    }
    EXPECT_EQ(0, app_stop(std::to_string(i).c_str(), 0));
  }
  sleep(1);
  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      EXPECT_EQ(0, app_alive(std::to_string(i).c_str()));
      continue;
    }
    EXPECT_NE(0, app_alive(std::to_string(i).c_str()));
  }
}

TEST_F(ManagerTest, Start_stop_all) {
  ManagerTestServer server;
  server.Start();
  for (int i = 0; i < 8; i++) {
    struct app_start_info info;
    memset(&info, 0, sizeof(info));
    info.name = std::to_string(i).c_str();
    info.cmdline = "test\0\0";
    info.cmd_max_len = PATH_MAX;
    info.check_alive = 1;
    info.keepalive_time = 60;
    info.heartbeat_interval = 5;
    EXPECT_EQ(0, app_start(&info));
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(0, app_stop(std::to_string(i).c_str(), 0));
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_NE(0, app_alive(std::to_string(i).c_str()));
  }
}

TEST_F(ManagerTest, monitor) {
  ManagerTestServer server;
  server.Start();
  ManagerTestApp app;
  app.SetPause(10);

  struct app_start_info info;
  memset(&info, 0, sizeof(info));
  info.name = "monitor";
  info.cmdline = "sleep\00900\0\0";
  info.cmd_max_len = PATH_MAX;
  info.check_alive = 1;
  info.keepalive_time = 2;
  info.heartbeat_interval = 1;
  EXPECT_EQ(0, app_start(&info));
  sleep(1);
  int pid = app_getpid("monitor");

  sleep(2);
  EXPECT_NE(pid, app_getpid("monitor"));
  EXPECT_NE(-1, app_getpid("monitor"));
  EXPECT_EQ(0, app_alive("monitor"));
  app_stop("monitor", 0);
}

TEST_F(ManagerTest, killcmd) {
  ManagerTestServer server;
  server.Start();
  ManagerTestApp app;
  app.SetPause(10);

  unlink("/tmp/killcmd");
  struct app_start_info info;
  memset(&info, 0, sizeof(info));
  info.name = "killcmd";
  info.cmdline = "sleep\00900\0\0";
  info.cmd_max_len = PATH_MAX;
  info.killcmd = "touch\0/tmp/killcmd\0\0";
  info.killcmd_max_len = PATH_MAX;
  info.check_alive = 1;
  info.keepalive_time = 2;
  info.heartbeat_interval = 1;
  EXPECT_EQ(0, app_start(&info));

  sleep(2);
  EXPECT_EQ(0, access("/tmp/killcmd", F_OK));
  app_stop("killcmd", 0);
  EXPECT_NE(0, app_alive("monitor"));
  unlink("/tmp/killcmd");
}

}  // namespace modelbox
