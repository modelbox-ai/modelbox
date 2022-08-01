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


#include "modelbox/base/list.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <string>
#include <thread>

#include "gtest/gtest.h"

namespace modelbox {

class ListTest : public testing::Test {
 public:
  ListTest() = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

struct Item {
  ListHead list;
  int value;
};

TEST_F(ListTest, ForEach) {
  ListHead head;
  struct Item *item;
  struct Item *tmp;
  int i = 0;
  ListInit(&head);

  for (int i = 0; i < 10; i++) {
    item = (struct Item *)malloc(sizeof(struct Item));
    ListAddTail(&item->list, &head);
    item->value = i;
  }

  i = 0;
  ListForEachEntrySafe(item, tmp, &head, list) {
    EXPECT_EQ(item->value, i);
    i++;
    ListDel(&item->list);
    free(item);
  }
}

}  // namespace modelbox