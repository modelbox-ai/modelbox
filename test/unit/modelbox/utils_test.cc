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

/* clang-format off */
#include <modelbox/base/log.h>
#include <modelbox/server/utils.h>
#include <securec.h>

#include <list>
#include <toml.hpp>

#include "gtest/gtest.h"
#include "test_config.h"
#include <nlohmann/json.hpp>

/* clang-format on */
namespace modelbox {
class ServerUtilsTest : public testing::Test {
 public:
  ServerUtilsTest() {}
  virtual ~ServerUtilsTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(ServerUtilsTest, IPMatchClassC) {
  IPACL acl;
  acl.AddCidr("192.168.1.1/24");
  EXPECT_EQ(acl.IsMatch("192.168.1.2"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.1.0"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.1.127"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.1.255"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.2.0"), STATUS_NOTFOUND);
}

TEST_F(ServerUtilsTest, IPMatch) {
  IPACL acl;
  acl.AddCidr("192.168.1.1");
  EXPECT_EQ(acl.IsMatch("192.168.1.2"), STATUS_NOTFOUND);
  EXPECT_EQ(acl.IsMatch("192.168.1.0"), STATUS_NOTFOUND);
  EXPECT_EQ(acl.IsMatch("192.168.1.1"), STATUS_OK);
}

TEST_F(ServerUtilsTest, IPMatchAll) {
  IPACL acl;
  acl.AddCidr("0.0.0.0/0");
  EXPECT_EQ(acl.IsMatch("192.168.1.2"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.1.0"), STATUS_OK);
  EXPECT_EQ(acl.IsMatch("192.168.1.1"), STATUS_OK);
}

}  // namespace modelbox
