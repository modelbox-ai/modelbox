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

#include "modelbox/drivers/common/modelbox_fuse.h"

#include <dirent.h>
#include <securec.h>
#include <sys/types.h>

#include <functional>
#include <future>
#include <random>
#include <thread>

#include "driver_flow_test.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/buffer.h"
#include "test/mock/minimodelbox/mockflow.h"

using ::testing::_;

namespace modelbox {

const char *MOCK_FUSE_FILE = "/tmp/modelbox_fuse";

class ModelBoxFuseTest : public testing::Test {
 public:
  ModelBoxFuseTest() : driver_flow_(std::make_shared<MockFlow>()) {}

 protected:
  virtual void SetUp() {
    auto ret = AddMockFlowUnit();
    EXPECT_EQ(ret, STATUS_OK);
  };

  virtual void TearDown() { driver_flow_ = nullptr; };
  std::shared_ptr<MockFlow> GetDriverFlow();

 private:
  Status AddMockFlowUnit();
  std::shared_ptr<MockFlow> driver_flow_;
};

Status ModelBoxFuseTest::AddMockFlowUnit() { return STATUS_OK; }

class MockFuseFile : public modelbox::ModelBoxFuseFile {
 public:
  virtual ~MockFuseFile() = default;
  
  std::string msg{"Hello, world"};
  int Open(const std::string &path) { return 0; }
  int Release() { return 0; }
  int Read(char *buff, size_t size, off_t off) {
    if (off > (off_t)msg.length()) {
      return 0;
    }
    snprintf_s(buff, size, size, "%s", msg.c_str());
    return msg.length();
  }
  int Write(const char *buff, size_t size, off_t off) { return 0; }
  int FSync(int isdatasync) { return 0; }
  int Flush() { return 0; }
  int FileSize() { return msg.length(); }
  std::string GetMsg() { return msg; }
};

class MockFuseInode : public modelbox::ModelBoxFileInode {
 public:
  MockFuseInode(const std::string &path) { path_ = path; };
  virtual ~MockFuseInode(){};

  int FillStat(struct stat *stat) {
    auto inode = std::make_shared<MockFuseFile>();
    stat->st_size = inode->FileSize();
    return 0;
  }

  std::shared_ptr<modelbox::ModelBoxFuseFile> CreateFile() {
    return std::make_shared<MockFuseFile>();
  }

  std::string GetPath() { return path_; }

 private:
  std::string path_;
};

TEST_F(ModelBoxFuseTest, FuseStat) {
  auto fuse = modelbox::ModelBoxFuseOperation::CreateFuse(MOCK_FUSE_FILE);
  auto ret = fuse->Run();
  if (ret == STATUS_NOENT || ret == STATUS_PERMIT) {
    GTEST_SKIP();
  }

  struct stat stbuf;
  EXPECT_EQ(0, stat(MOCK_FUSE_FILE, &stbuf));
  EXPECT_EQ(2, stbuf.st_nlink);

  std::string name = "/file";
  auto inode = std::make_shared<MockFuseInode>(name);

  EXPECT_EQ(0, stat(MOCK_FUSE_FILE, &stbuf));
  EXPECT_EQ(2, stbuf.st_nlink);

  name = MOCK_FUSE_FILE;
  name += "/dir";
  mkdir(name.c_str(), 0755);
  EXPECT_EQ(0, stat(MOCK_FUSE_FILE, &stbuf));
  EXPECT_EQ(3, stbuf.st_nlink);
}

TEST_F(ModelBoxFuseTest, FuseMountCheckFile) {
  int expect_dir_num = 10;
  int expect_file_num = 10;
  int expect_total = expect_dir_num + expect_file_num + 2;

  auto fuse = modelbox::ModelBoxFuseOperation::CreateFuse(MOCK_FUSE_FILE);
  auto ret = fuse->Run();
  if (ret == STATUS_NOENT || ret == STATUS_PERMIT) {
    GTEST_SKIP();
  }
  ASSERT_EQ(ret, modelbox::STATUS_OK);
  for (int i = 0; i < expect_dir_num; i++) {
    std::string name = "/dir";
    name += std::to_string(i);
    EXPECT_EQ(fuse->MkDir(name.c_str(), 0755), 0);
  }

  for (int i = 0; i < expect_dir_num; i++) {
    std::string name = "/file";
    name += std::to_string(i);
    auto inode = std::make_shared<MockFuseInode>(name);
    fuse->AddFuseFile(inode);
  }

  DIR *dir;
  struct dirent *ent;
  int totalnum = 0;
  int dirnum = 0;
  int filenum = 0;
  if ((dir = opendir(MOCK_FUSE_FILE)) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      struct stat stbuf;
      std::string path = MOCK_FUSE_FILE;
      path += "/";
      path += ent->d_name;
      if (stat(path.c_str(), &stbuf) == -1) {
        continue;
      }
      totalnum++;

      if (strncmp(ent->d_name, ".", PATH_MAX) == 0 ||
          strncmp(ent->d_name, "..", PATH_MAX) == 0) {
        continue;
      }

      if (S_ISDIR(stbuf.st_mode) == 0) {
        dirnum++;
      }

      if (S_ISREG(stbuf.st_mode) == 0) {
        filenum++;
      }
    }
    closedir(dir);
  }

  EXPECT_EQ(totalnum, expect_total);
  EXPECT_EQ(filenum, expect_file_num);
  EXPECT_EQ(dirnum, expect_file_num);
}

TEST_F(ModelBoxFuseTest, FileOpen) {
  auto fuse = modelbox::ModelBoxFuseOperation::CreateFuse(MOCK_FUSE_FILE);
  auto ret = fuse->Run();
  if (ret == STATUS_NOENT || ret == STATUS_PERMIT) {
    GTEST_SKIP();
  }
  ASSERT_EQ(ret, modelbox::STATUS_OK);
  auto inode = std::make_shared<MockFuseInode>("/file");
  fuse->AddFuseFile(inode);

  std::string item_data;
  std::ifstream infile;
  infile.open(MOCK_FUSE_FILE + inode->GetPath());
  EXPECT_FALSE(infile.fail());
  Defer { infile.close(); };
  std::getline(infile, item_data);

  EXPECT_EQ(
      item_data,
      std::dynamic_pointer_cast<MockFuseFile>(inode->CreateFile())->GetMsg());
}

}  // namespace modelbox