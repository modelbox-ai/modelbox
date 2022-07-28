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


#include "modelbox/base/slab.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <string>
#include <thread>

#include "modelbox/base/list.h"
#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "gtest/gtest.h"
#include "securec.h"

namespace modelbox {

class SlabTest : public testing::Test {
 public:
  SlabTest() {}

 protected:
  virtual void SetUp(){

  };
  virtual void TearDown(){};
};

TEST_F(SlabTest, SlabMalloc) {
  Slab cache(nullptr, 128, 4096);
  EXPECT_TRUE(cache.Init());
  void *ptr = cache.Alloc();
  void *ptr1 = cache.Alloc();
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(ptr1, nullptr);
  EXPECT_EQ(2, cache.ActiveObjects());
  cache.Free(ptr);
  EXPECT_EQ(1, cache.ActiveObjects());
  cache.Free(ptr1);
  EXPECT_EQ(0, cache.ActiveObjects());
}

TEST_F(SlabTest, SlabMallocSharedPtrCheck) {
  int objsize = 128;
  int num = 1000 + 1;
  char mem_cmp[objsize];
  SlabCache cache(objsize, objsize * 4);
  std::vector<std::shared_ptr<void>> mem_ptr;
  for (int i = 0; i < num; i++) {
    auto ptr = cache.AllocSharedPtr();
    ASSERT_NE(ptr, nullptr);
    mem_ptr.push_back(ptr);
  }

  EXPECT_EQ(num, cache.GetActiveObjNumber());
  EXPECT_EQ(num / 4 + 1, cache.SlabNumber());

  for (int i = 0; i < num; i++) {
    auto ptr = mem_ptr[i];
    memset_s(ptr.get(), objsize, i, objsize);
  }

  for (int i = 0; i < num; i++) {
    auto ptr = mem_ptr[i];
    memset_s(mem_cmp, objsize, i, objsize);
    EXPECT_EQ(0, memcmp(ptr.get(), mem_cmp, objsize));
  }

  auto ptr = mem_ptr[0];
  memset_s(mem_cmp, objsize, 0, objsize);
  EXPECT_EQ(0, memcmp(ptr.get(), mem_cmp, objsize));
  ptr = nullptr;

  mem_ptr.pop_back();
  auto ptr_last = cache.AllocSharedPtr();
  memset_s(mem_cmp, objsize, num - 1, objsize);
  EXPECT_EQ(0, memcmp(ptr_last.get(), mem_cmp, objsize));
  ptr_last = nullptr;

  mem_ptr.clear();
  EXPECT_EQ(num / 4 + 1, cache.GetEmptySlabNumber());
  EXPECT_EQ(0, cache.GetActiveObjNumber());
}

TEST_F(SlabTest, SlabMallocSharedPtr) {
  SlabCache cache(128, 4096);
  auto ptr = cache.AllocSharedPtr();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(1, cache.GetActiveObjNumber());
  ptr = nullptr;
  EXPECT_EQ(0, cache.GetActiveObjNumber());
}

TEST_F(SlabTest, SlabCacheMallocSharedPtr) {
  SlabCache cache(128, 256);
  EXPECT_EQ(128, cache.ObjectSize());
  auto ptr = cache.AllocSharedPtr();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(256 / 128, cache.GetObjNumber());
  auto ptr1 = cache.AllocSharedPtr();
  ASSERT_NE(ptr1, nullptr);
  auto ptr2 = cache.AllocSharedPtr();
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(3, cache.GetActiveObjNumber());
  ptr = nullptr;
  EXPECT_EQ(2, cache.GetActiveObjNumber());
  ptr1 = nullptr;
  EXPECT_EQ(1, cache.GetActiveObjNumber());
  ptr2 = nullptr;
  EXPECT_EQ(0, cache.GetActiveObjNumber());
  EXPECT_EQ(2, cache.GetEmptySlabNumber());
}

TEST_F(SlabTest, SlabCacheActiveNumber) {
  SlabCache cache(128, 128);
  auto ptr = cache.AllocSharedPtr();
  auto ptr1 = cache.AllocSharedPtr();
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(ptr1, nullptr);
  EXPECT_EQ(2, cache.GetObjNumber());
  EXPECT_EQ(2, cache.GetActiveObjNumber());
  ptr = nullptr;
  EXPECT_EQ(2, cache.GetObjNumber());
  EXPECT_EQ(1, cache.GetActiveObjNumber());
  EXPECT_EQ(2, cache.GetObjNumber());
  ptr1 = nullptr;
  EXPECT_EQ(2, cache.GetEmptySlabNumber());
}

TEST_F(SlabTest, SlabShrink) {
  int obj_size = 128;
  int slab_size = 4096;
  int total = 0;
  int slab_count = 10;
  SlabCache cache(obj_size, slab_size);
  std::vector<std::shared_ptr<void>> results;
  for (int i = 0; i < slab_size / obj_size * slab_count; i++) {
    auto p = cache.AllocSharedPtr();
    ASSERT_NE(p, nullptr);
    results.push_back(p);
    total++;
  }

  cache.Shrink();
  EXPECT_EQ(cache.SlabNumber(), slab_count);
  EXPECT_EQ(total, cache.GetActiveObjNumber());
  EXPECT_GT(cache.GetObjNumber(), 0);
  results.clear();
  cache.Shrink(5);
  EXPECT_EQ(cache.SlabNumber(), 5);

  cache.Shrink(0, 1);
  EXPECT_EQ(cache.SlabNumber(), 5);

  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  cache.Shrink(0, 1);
  EXPECT_EQ(cache.SlabNumber(), 0);

  cache.Shrink();
  EXPECT_EQ(0, cache.GetActiveObjNumber());
  EXPECT_EQ(cache.GetObjNumber(), 0);
  EXPECT_EQ(cache.SlabNumber(), 0);
}

TEST_F(SlabTest, SlabCacheReclaim) {
  SlabCache cache(128, 128);
  std::vector<std::shared_ptr<void>> ptrs;
  int number = 100;
  for (int i = 0; i < number; i++) {
    auto ptr = cache.AllocSharedPtr();
    ptrs.emplace_back(ptr);
  }

  EXPECT_EQ(number, cache.GetObjNumber());
  EXPECT_EQ(number, cache.GetActiveObjNumber());
  ptrs.clear();
  EXPECT_EQ(number, cache.GetObjNumber());
  EXPECT_EQ(0, cache.GetActiveObjNumber());
  EXPECT_EQ(number, cache.GetEmptySlabNumber());
  cache.Reclaim(0);
  EXPECT_EQ(number * 10 / 100, cache.GetEmptySlabNumber());
  cache.Reclaim(0);
  EXPECT_EQ(1, cache.GetEmptySlabNumber());
}

TEST_F(SlabTest, Perf) {
  int obj_size = 4;
  SlabCache cache(obj_size, 4096);
  std::vector<std::thread> threads;
  std::atomic<unsigned long> number;
  bool stop = false;
  unsigned long begin;
  unsigned long end;
  int cpu_num = std::thread::hardware_concurrency() * 2;

  number = 0;
  begin = GetTickCount();
  for (int i = 0; i < cpu_num; i++) {
    auto t = std::thread([&, i]() {
      while (stop == false) {
        std::vector<std::shared_ptr<void>> ptrs;
        std::this_thread::sleep_for(std::chrono::milliseconds(i));
        for (int j = 0; j < 100000; j++) {
          if (stop == true) {
            break;
          }
          auto p = cache.AllocSharedPtr();
          if (p) {
            ptrs.push_back(p);
            *(int *)(p.get()) = i * j;
          }
        }

        for (int j = 0; j < (int)(ptrs.size()); j++) {
          EXPECT_EQ(i * j, *(int *)(ptrs[j].get()));
        }
        number += ptrs.size();
        ptrs.clear();
      }
    });
    threads.push_back(std::move(t));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stop = true;

  for (auto &t : threads) {
    t.join();
  }
  end = GetTickCount();

  MBLOG_INFO << "total: " << number;
  MBLOG_INFO << "ops: " << 1.0 * number / (end - begin) * 1000.0;
}

}  // namespace modelbox