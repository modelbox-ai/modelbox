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


#include <modelbox/base/log.h>
#include <modelbox/base/refcache.h>
#include <modelbox/base/utils.h>

#include <future>

#include "gtest/gtest.h"
#include "test_config.h"

namespace modelbox {
class RefCacheTest : public testing::Test {
 public:
  RefCacheTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

class Data {
 public:
  Data(){};
  virtual ~Data(){};
};

TEST_F(RefCacheTest, Get) {
  RefCache<Data> cache;
  std::shared_ptr<Data> data = std::make_shared<Data>();
  auto trans = cache.Insert("a");
  auto keep = trans->UpdateData(data);
  EXPECT_TRUE(cache.Get("a") != nullptr);
  keep = nullptr;
  EXPECT_TRUE(cache.Get("a") == nullptr);
}

TEST_F(RefCacheTest, Insert) {
  RefCache<Data> cache;
  std::shared_ptr<Data> data = std::make_shared<Data>();
  std::shared_ptr<Data> data2 = std::make_shared<Data>();
  auto trans1 = cache.Insert("a");
  auto trans2 = cache.Insert("a");
  auto keep = trans1->UpdateData(data);
  EXPECT_TRUE(keep != nullptr);
  EXPECT_TRUE(trans2 == nullptr);
}

TEST_F(RefCacheTest, InsertDelay) {
  RefCache<Data> cache;
  std::shared_ptr<Data> data = std::make_shared<Data>();
  auto trans1 = cache.Insert("a");
  auto result_future = std::async(std::launch::async, [&]() {
    auto data2 = cache.Get("a", true);
    EXPECT_TRUE(data2 != nullptr);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto data3 = trans1->UpdateData(data);
  EXPECT_TRUE(data3 != nullptr);
  data3 = nullptr;
  result_future.get();
  EXPECT_TRUE(cache.Get("a") == nullptr);
}

TEST_F(RefCacheTest, InsertAndGet) {
  RefCache<Data> cache;
  std::shared_ptr<Data> data;
  std::atomic_int count{0};
  int loop = 10;
  std::vector<std::future<void>> result;
  for (int i = 0; i < loop; i++) {
    auto result_future = std::async(std::launch::async, [&]() {
      auto trans = cache.InsertAndGet("a");
      if (trans == nullptr) {
        return;
      }
      ASSERT_TRUE(trans != nullptr);
      auto data = trans->GetData();
      if (data) {
        count++;
      } else {
        std::shared_ptr<Data> datain = std::make_shared<Data>();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        trans->UpdateData(datain);
      }
    });
    result.push_back(std::move(result_future));
  }

  for (auto &r : result) {
    r.get();
  }

  EXPECT_EQ(count, loop - 1);
}

TEST_F(RefCacheTest, Update) {
  RefCache<Data> cache;
  std::shared_ptr<Data> data = std::make_shared<Data>();
  std::shared_ptr<Data> data2 = std::make_shared<Data>();
  auto keep = cache.Update("a", data);
  auto keep2 = cache.Update("a", data2);
  EXPECT_TRUE(keep != nullptr);
  EXPECT_TRUE(keep2 != nullptr);
}

TEST_F(RefCacheTest, GetCacheData) {
  RefCacheData cache;
  std::shared_ptr<Data> data = std::make_shared<Data>();
  auto trans = cache.Insert("a");
  auto keep = trans->UpdateData(data);
  EXPECT_TRUE(cache.Get("a") != nullptr);
  keep = nullptr;
  EXPECT_TRUE(cache.Get("a") == nullptr);
}

}  // namespace modelbox
