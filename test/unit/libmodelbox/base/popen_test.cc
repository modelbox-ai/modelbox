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

#include "modelbox/base/popen.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "modelbox/base/status.h"

namespace modelbox {

class PopenTest : public testing::Test {
 public:
  PopenTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){};
};

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

TEST_F(PopenTest, OpenCaptureOutput) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/bash");
  args.push_back("-c");
  args.push_back(
      "N=0;"
      "while [ $N -lt 1000 ]; do echo msg $N; echo err $N >&2; ((N=N+1)); "
      "done");
  p.Open(args, 1000, "re");

  int expect = 1000;
  int ret = 0;
  int stdoutcount = 0;
  int stderrcount = 0;
  while (true) {
    ret = p.WaitForLineRead(1000);
    if (ret < 0) {
      break;
    } else if (ret == 0) {
      continue;
    }

    std::string line;
    p.ReadOutLine(line);
    if (line.length() > 0) {
      stdoutcount++;
    }

    line.clear();
    p.ReadErrLine(line);
    if (line.length() > 0) {
      stderrcount++;
    }
  }
  ret = p.Close();
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(stdoutcount, expect);
  EXPECT_EQ(stderrcount, expect);
}

TEST_F(PopenTest, OpenTimeout) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/bash");
  args.push_back("-c");
  args.push_back(
      "N=0;"
      "while [ true ]; do echo msg $N; echo err $N >&2; ((N=N+1)); "
      "done");
  p.Open(args, 100, "re");

  int expect = 100;
  int ret = 0;
  int stdoutcount = 0;
  int stderrcount = 0;
  while (true) {
    ret = p.WaitForLineRead();
    if (ret < 0) {
      break;
    } else if (ret == 0) {
      continue;
    }

    std::string line;
    p.ReadOutLine(line);
    if (line.length() > 0) {
      stdoutcount++;
    }

    line.clear();
    p.ReadErrLine(line);
    if (line.length() > 0) {
      stderrcount++;
    }
  }
  ret = p.Close();
  EXPECT_EQ(WIFSIGNALED(ret), 1);
  EXPECT_EQ(WTERMSIG(ret), SIGTERM);
  EXPECT_GT(stdoutcount, expect);
  EXPECT_GT(stderrcount, expect);
}

TEST_F(PopenTest, OpenCaptureStdOutputOnly) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/bash");
  args.push_back("-c");
  args.push_back(
      "N=0;"
      "while [ $N -lt 10 ]; do echo msg $N; echo err $N >&2; ((N=N+1)); "
      "done");
  p.Open(args, 1000);

  int expect = 10;
  int ret = 0;
  int stdoutcount = 0;
  while (true) {
    std::string line;
    auto ret = p.ReadOutLine(line);
    if (ret < 0) {
      break;
    }

    if (line.length() > 0) {
      stdoutcount++;
    }
  }
  ret = p.Close();
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(stdoutcount, expect);
}

TEST_F(PopenTest, OpenCaptureNone) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/bash");
  args.push_back("-c");
  args.push_back(
      "N=0;"
      "while [ $N -lt 10 ]; do echo msg $N; echo err $N >&2; ((N=N+1)); "
      "done");
  p.Open(args, 1000, "");
  auto ret = p.Close();
  EXPECT_EQ(ret, 0);
}

TEST_F(PopenTest, OpenCaptureNoneTimeOut) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/sleep");
  args.push_back("10");
  p.Open(args, 100, "");
  auto ret = p.Close();
  EXPECT_EQ(WIFSIGNALED(ret), 1);
  EXPECT_EQ(WTERMSIG(ret), SIGTERM);
}

TEST_F(PopenTest, OpenWait) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/sleep");
  args.push_back("0.1");
  p.Open(args, -1, "");
  auto ret = p.Close();
  EXPECT_EQ(WIFSIGNALED(ret), 0);
  EXPECT_EQ(WEXITSTATUS(ret), 0);
}

TEST_F(PopenTest, OpenInput) {
  Popen p;
  std::vector<std::string> args;
  args.push_back("/bin/bash");
  args.push_back("-c");
  args.push_back(
      "N=0;\n"
      "read N\n"
      "echo $N\n"
      "exit $N\n");
  p.Open(args, 1000, "w");
  auto ret = p.WriteString("10\n");
  EXPECT_EQ(ret, 0);
  ret = p.Close();
  EXPECT_EQ(WEXITSTATUS(ret), 10);
}

TEST_F(PopenTest, OpenCmdLine) {
  Popen p;
  p.Open("/bin/bash -c \"ls -h /bin/sh \"", 100, "r");
  std::string line;
  p.ReadOutLine(line);
  EXPECT_EQ(line, "/bin/sh\n");
  auto ret = p.Close();
  EXPECT_EQ(WEXITSTATUS(ret), 0);
}

TEST_F(PopenTest, OpenEnvCheck) {
  Popen p;
  PopenEnv env ="TEST_ENV_1=a TEST_ENV_2=b";
  env.Rmv("USER");
  p.Open("/bin/bash -c \"echo $USER && echo $TEST_ENV_1; echo $TEST_ENV_2\"", 100, "r", env);
  std::string line;
  p.ReadOutLine(line);
  EXPECT_EQ(line, "\n");
  p.ReadOutLine(line);
  EXPECT_EQ(line, "a\n");
  p.ReadOutLine(line);
  EXPECT_EQ(line, "b\n");
  auto ret = p.Close();
  EXPECT_EQ(WEXITSTATUS(ret), 0);
}

TEST_F(PopenTest, OpenReadAll) {
  Popen p;
  std::string cmd =
      "/bin/bash -c \""
      "N=0;"
      "while [ $N -lt 100 ]; do echo msg $N; echo err $N >&2; ((N=N+1)); "
      "done\"";
  p.Open(cmd, 1000, "re");

  std::string out;
  std::string err;
  auto ret = p.ReadAll(&out, &err);
  EXPECT_EQ(ret, 0);

  auto count_line = [](const std::string &str) {
    size_t i = 0; 
    int count = 0;
    for (i = 0; i < str.length(); i++) {
      if (str.c_str()[i] == '\n' || str.c_str()[i] == '\0') {
        count++;
      }
    }

    return count;
  };
  
  ret = p.Close();
  EXPECT_EQ(count_line(out), 100);
  EXPECT_EQ(count_line(err), 100);
  EXPECT_EQ(ret, 0);
}

TEST_F(PopenTest, OpenNotExists) {
  Popen p;
  std::string cmd =
      "/NOT-EXIST";
  p.Open(cmd, 1000, "re");

  std::string out;
  std::string err;
  auto ret = p.ReadAll(&out, &err);
  EXPECT_EQ(ret, 0);

  EXPECT_GT(err.find_first_of(cmd), 0);
  
  ret = p.Close();
  EXPECT_EQ(ret, 256);
}

}  // namespace modelbox
