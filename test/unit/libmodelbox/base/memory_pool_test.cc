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


#include "modelbox/base/memory_pool.h"
#include "gtest/gtest.h"
#include "modelbox/base/utils.h"
#include "securec.h"
#include <memory>
#include <thread>

using namespace modelbox;

class MemoryPoolTest : public testing::Test {
 public:
  MemoryPoolTest() = default;
  ~MemoryPoolTest() override = default;

 protected:
  void SetUp() override{};

  void TearDown() override{};
};


TEST_F(MemoryPoolTest, MemoryPool) {
  MemoryPoolBase p;
  p.InitSlabCache();
  unsigned int num = 0;
  for (int i = 0; i < 10; i++) {
    GetRandom((unsigned char *)&num, sizeof(num));
    int size = num % (1024 * 512);
    auto ptr = p.AllocSharedPtr(size);
    ASSERT_NE(ptr, nullptr);
    memset_s(ptr.get(), size, 0, size);
  }
}

TEST_F(MemoryPoolTest, MemoryPoolShrink) {
  MemoryPoolBase p;
  int slab_number = 4;
  int slab_expand_size = 1024 * 1024;
  int obj_size = 1024;
  int obj_number_per_slab = slab_expand_size / obj_size;
  int obj_number = obj_number_per_slab * slab_number;
  p.InitSlabCache(10, 10);
  std::vector<std::shared_ptr<void>> results;
  for (int i = 0; i < obj_number; i++) {
    auto size = obj_size;
    auto ptr = p.AllocSharedPtr(size);
    ASSERT_NE(ptr, nullptr);
    memset_s(ptr.get(), size, 0, size);
    results.push_back(ptr);
  }

  EXPECT_EQ(p.GetAllObjectNum(), obj_number);
  results.clear();
  EXPECT_EQ(p.GetAllActiveObjectNum(), 0);
  p.ShrinkSlabCache(3, 0);
  slab_number = 3;
  obj_number = obj_number_per_slab * slab_number;
  EXPECT_EQ(p.GetAllObjectNum(), obj_number);
  p.ShrinkSlabCache(0, 1);
  EXPECT_EQ(p.GetAllObjectNum(), obj_number);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  p.ShrinkSlabCache(0, 1);
  EXPECT_EQ(p.GetAllObjectNum(), 0);
}

TEST_F(MemoryPoolTest, MemoryPoolShrinkExpire) {
  MemoryPoolBase p;
  int slab_number = 4;
  int slab_expand_size = 1024 * 1024;
  int obj_size = 1024;
  int obj_number_per_slab = slab_expand_size / obj_size;
  int obj_number = obj_number_per_slab * slab_number;
  p.InitSlabCache(10, 10);
  std::vector<std::shared_ptr<void>> results;
  for (int i = 0; i < obj_number; i++) {
    auto size = obj_size;
    auto ptr = p.AllocSharedPtr(size);
    ASSERT_NE(ptr, nullptr);
    memset_s(ptr.get(), size, 0, size);
    results.push_back(ptr);
  }

  EXPECT_EQ(p.GetAllObjectNum(), obj_number);
  results.clear();
  EXPECT_EQ(p.GetAllActiveObjectNum(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  p.ShrinkSlabCache(4, 0, 1);
  EXPECT_EQ(p.GetAllObjectNum(), 0);
}
