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


#include "modelbox/base/crypto.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "modelbox/base/log.h"
#include "gtest/gtest.h"

namespace modelbox {

class CryptoTest : public testing::Test {
 public:
  CryptoTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(CryptoTest, Base64) {
  char data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int len = sizeof(data);
  std::string base64_text;
  std::vector<unsigned char> in(&data[0], &data[len]);
  std::vector<unsigned char> out;
  EXPECT_TRUE(Base64Encode(in, &base64_text));
  EXPECT_TRUE(Base64Decode(base64_text, &out));
  MBLOG_INFO << "Base64: " << base64_text;
  for (unsigned int i = 0; i < out.size(); i++) {
    EXPECT_EQ(data[i], out[i]);
  }
}
TEST_F(CryptoTest, AesEncryptPass) {
  std::string str = "password";
  std::vector<char> pass(str.begin(), str.end());
  std::string rootkey;
  std::string enpass;

  EXPECT_EQ(PassEncrypt(pass, true, &rootkey, &enpass), STATUS_OK);

  std::vector<char> outpass;
  EXPECT_EQ(PassDecrypt(enpass, rootkey, &outpass), STATUS_OK);
  EXPECT_EQ(pass, outpass);
}
}  // namespace modelbox