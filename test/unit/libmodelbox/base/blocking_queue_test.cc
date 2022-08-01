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


#include "modelbox/base/blocking_queue.h"

#include <poll.h>
#include <sys/time.h>

#include <chrono>
#include <future>
#include <string>
#include <thread>

#include "modelbox/base/log.h"
#include "gtest/gtest.h"
namespace modelbox {
class BlockingQueueTest : public testing::Test {
 public:
  BlockingQueueTest() = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

class PriorityBlockingQueueTest : public testing::Test {
 public:
  PriorityBlockingQueueTest() = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

class TestNumber {
 public:
  TestNumber() = default;
  TestNumber(int n) { num_ = n; };
  virtual ~TestNumber() = default;
  int Get() { return num_; }
  void operator=(int n) { num_ = n; };
  bool operator==(int n) const { return n == num_; }
  bool operator==(const TestNumber& t) const { return t.num_ == num_; }
  bool operator<(const TestNumber& t) const { return num_ < t.num_; }
  bool operator>(const TestNumber& t) const { return num_ > t.num_; }
  bool operator<=(const TestNumber& t) const {
    if (num_ == t.num_) {
      return private_num_ >= t.private_num_;
    }
    return num_ <= t.num_;
  }
  std::string ToString() const {
    std::ostringstream oss;
    oss << num_;
    return oss.str();
  }
  void SetPrivate(int n) { private_num_ = n; };
  int GetPrivate() { return private_num_; };

 private:
  int num_ = 0;
  int private_num_ = 0;
};

std::ostream& operator<<(std::ostream& os, const TestNumber& t) {
  os << t.ToString();
  return os;
}

TEST_F(BlockingQueueTest, EnqueueDequeue) {
  const int queue_size = 12;
  BlockingQueue<TestNumber> queue(queue_size);

  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    queue.Push(value);
  }

  EXPECT_EQ(queue_size, queue.Size());
  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 0);

  for (int i = 0; i < queue_size; i++) {
    TestNumber value = -1;
    queue.Pop(&value);
    EXPECT_EQ(value, i * i);
  }
}

TEST_F(BlockingQueueTest, EnqueueDequeueSequence) {
  const int queue_size = 12;
  const int push_size = 6;
  std::vector<TestNumber> nums;
  BlockingQueue<TestNumber> queue(queue_size);

  for (int i = 0; i < push_size; i++) {
    TestNumber value = i * i;
    nums.push_back(value);
  }

  auto ret = queue.Push(&nums);
  EXPECT_EQ(ret, push_size);
  nums.clear();
  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    nums.push_back(value);
  }

  ret = queue.Push(&nums);
  EXPECT_EQ(queue_size, push_size + ret);
  EXPECT_EQ(push_size + ret, queue.Size());
  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 0);

  std::vector<TestNumber> out_nums;
  queue.Pop(&out_nums);
  EXPECT_EQ(out_nums.size(), queue_size);

  for (size_t i = 0; i < out_nums.size(); i++) {
    int v = i % push_size;
    EXPECT_EQ(out_nums[i], v * v);
  }
}

TEST_F(BlockingQueueTest, EnqueueDequeueSequenceBatchTimeout) {
  const int queue_size = 12;
  std::vector<TestNumber> second;
  std::vector<TestNumber> first;
  const int first_data_size = 6;
  BlockingQueue<TestNumber> queue(queue_size);

  for (int i = 0; i < first_data_size; i++) {
    TestNumber value = i * i;
    first.push_back(value);
  }

  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    second.push_back(value);
  }

  queue.Push(&first);
  auto ret = queue.PushBatch(&second, 10);
  EXPECT_FALSE(ret);

  EXPECT_EQ(first_data_size, queue.Size());
  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 0);

  std::vector<TestNumber> out_nums;
  queue.Pop(&out_nums);
  EXPECT_EQ(out_nums.size(), first_data_size);

  for (size_t i = 0; i < out_nums.size(); i++) {
    EXPECT_EQ(out_nums[i], i * i);
  }
}

TEST_F(BlockingQueueTest, EnqueueDequeueSequenceBatchBlock) {
  const int queue_size = 12;
  std::vector<TestNumber> second;
  std::vector<TestNumber> first;
  const int first_data_size = 6;
  BlockingQueue<TestNumber> queue(queue_size);

  for (int i = 0; i < first_data_size; i++) {
    TestNumber value = i * i;
    first.push_back(value);
  }

  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    second.push_back(value);
  }

  queue.Push(&first);
  EXPECT_EQ(first_data_size, queue.Size());

  auto start = std::chrono::high_resolution_clock::now();
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::vector<TestNumber> first_out;
    queue.Pop(&first_out);
    EXPECT_EQ(first_data_size, first_out.size());
    for (size_t i = 0; i < first_out.size(); i++) {
      EXPECT_EQ(first_out[i], i * i);
    }
  });

  auto ret = queue.PushBatch(&second);
  EXPECT_TRUE(ret);
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);

  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 0);

  std::vector<TestNumber> out_nums;
  queue.Pop(&out_nums);
  EXPECT_EQ(out_nums.size(), queue_size);

  for (size_t i = 0; i < out_nums.size(); i++) {
    EXPECT_EQ(out_nums[i], i * i);
  }
}

TEST_F(BlockingQueueTest, QueueSize) {
  const int queue_size = 12;
  BlockingQueue<int> queue(SIZE_MAX);

  for (int i = 0; i < queue_size; i++) {
    int value = i * i;
    queue.Push(value);
  }

  EXPECT_EQ(queue_size, queue.Size());

  int pop_num = 3;
  for (int i = 0; i < pop_num; i++) {
    int value = -1;
    queue.Pop(&value);
    EXPECT_EQ(value, i * i);
  }

  EXPECT_EQ(queue_size - pop_num, queue.Size());
}

TEST_F(BlockingQueueTest, QueueRemain) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  int push_num = 6;
  for (int i = 0; i < push_num; i++) {
    int value = i * i;
    queue.Push(value);
  }

  EXPECT_EQ(queue_size - push_num, queue.RemainCapacity());
}

TEST_F(BlockingQueueTest, QueueClear) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  int push_num = 6;
  for (int i = 0; i < push_num; i++) {
    int value = i * i;
    queue.Push(value);
  }

  queue.Clear();
  EXPECT_EQ(queue_size, queue.RemainCapacity());
  EXPECT_EQ(0, queue.Size());
}

TEST_F(BlockingQueueTest, QueueBlockClear) {
  const int queue_size = 12;
  BlockingQueue<TestNumber> queue(queue_size);

  int push_num = 6;
  for (int i = 0; i < push_num; i++) {
    TestNumber value = i * i;
    queue.Push(value);
  }

  EXPECT_EQ(push_num, queue.Size());
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queue.Close();
  });

  std::vector<TestNumber> in_nums;
  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    in_nums.push_back(value);
  }

  auto ret = queue.PushBatch(&in_nums);
  EXPECT_LE(ret, 0);
  EXPECT_EQ(queue_size, queue.RemainCapacity());
  EXPECT_EQ(0, queue.Size());
}

TEST_F(BlockingQueueTest, PopBatch) {
  const int queue_size = 12;
  BlockingQueue<TestNumber> queue(queue_size);

  std::vector<TestNumber> in_nums;
  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    in_nums.push_back(value);
  }

  std::vector<TestNumber> out_nums;
  auto ret_num = queue.PopBatch(&out_nums, -1);
  EXPECT_EQ(ret_num, 0);
  EXPECT_EQ(out_nums.size(), 0);

  auto ret = queue.PushBatch(&in_nums);
  EXPECT_EQ(ret, queue_size);
  EXPECT_EQ(queue_size, queue.Size());

  ret_num = queue.PopBatch(&out_nums);
  EXPECT_EQ(ret_num, ret);
  EXPECT_EQ(out_nums.size(), ret_num);
}

TEST_F(BlockingQueueTest, PopBatchMaxNum) {
  const int queue_size = 12;
  const int first_pop = 3;
  BlockingQueue<TestNumber> queue(queue_size);

  for (int i = 0; i < queue_size; i++) {
    TestNumber value = i * i;
    queue.Push(value);
  }

  std::vector<TestNumber> out_nums;

  auto ret_num = queue.PopBatch(&out_nums, -1, first_pop);
  EXPECT_EQ(ret_num, first_pop);
  EXPECT_EQ(out_nums.size(), ret_num);

  out_nums.clear();
  ret_num = queue.PopBatch(&out_nums, -1);
  EXPECT_EQ(queue_size - first_pop, ret_num);
  EXPECT_EQ(out_nums.size(), ret_num);
}

TEST_F(BlockingQueueTest, QueuePoll) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  int push_num = 6;
  for (int i = 0; i < push_num; i++) {
    int value = i * i;
    queue.Push(value);
  }

  for (int i = 0; i < queue_size; i++) {
    int value = -1;
    bool ret = queue.Poll(&value);
    if (i < push_num) {
      EXPECT_EQ(value, i * i);
    } else {
      EXPECT_FALSE(ret);
    }
  }

  queue.Clear();
  EXPECT_EQ(queue_size, queue.RemainCapacity());
  EXPECT_EQ(0, queue.Size());
}

TEST_F(BlockingQueueTest, QueueCapacity) {
  const int queue_size = 12;
  BlockingQueue<int> queue(SIZE_MAX);

  EXPECT_EQ(SIZE_MAX, queue.GetCapacity());
  queue.SetCapacity(queue_size);
  EXPECT_EQ(queue_size, queue.GetCapacity());
}

TEST_F(BlockingQueueTest, QueueFull) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  for (int i = 0; i < queue_size; i++) {
    int value = i * i;
    queue.Push(value);
  }

  /* None block */
  int value = 0;
  EXPECT_FALSE(queue.Push(value, -1));

  /* wait for 100ms */
  value = queue_size + 1;
  auto start = std::chrono::high_resolution_clock::now();
  EXPECT_FALSE(queue.Push(value, 100));
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);

  /* wait until wakeup */
  start = std::chrono::high_resolution_clock::now();
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int value = -1;
    queue.Pop(&value);
    EXPECT_EQ(0, value);
  });

  /* check wait time */
  EXPECT_TRUE(queue.Push(value, 0));
  finish = std::chrono::high_resolution_clock::now();
  elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);
}

TEST_F(BlockingQueueTest, QueueFront) {
  const int queue_size = 12;
  BlockingQueue<std::shared_ptr<TestNumber>> queue(queue_size);

  std::shared_ptr<TestNumber> value = nullptr;
  EXPECT_FALSE(queue.Front(&value));
  EXPECT_EQ(value, nullptr);

  int push_num = 6;
  for (int i = 0; i < push_num; i++) {
    std::shared_ptr<TestNumber> value = std::make_shared<TestNumber>(i * i);
    queue.Push(value);
  }

  value = nullptr;
  EXPECT_EQ(push_num, queue.Size());
  EXPECT_TRUE(queue.Front(&value));
  EXPECT_EQ(*value, 0);
  EXPECT_EQ(push_num, queue.Size());

  for (int i = 0; i < push_num; i++) {
    std::shared_ptr<TestNumber> value_front = nullptr;
    std::shared_ptr<TestNumber> value_pop = nullptr;
    EXPECT_TRUE(queue.Front(&value_front));
    EXPECT_EQ(*value_front, i * i);
    queue.Pop(&value_pop);
    EXPECT_EQ(*value_pop, i * i);
    EXPECT_EQ(*value_front, *value_pop);
    EXPECT_EQ(*value, 0);
  }
}

TEST_F(BlockingQueueTest, QueueEmpty) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  /* none blocking */
  int value = -1;
  bool ret = queue.Pop(&value, -1);
  EXPECT_FALSE(ret);
  EXPECT_TRUE(queue.Empty());

  /* wait until wakeup */
  auto start = std::chrono::high_resolution_clock::now();
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int value = 1;
    queue.Push(value);
  });
  EXPECT_TRUE(queue.Pop(&value, 0));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(queue.Empty());
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);

  /* wait timeout */
  value = -1;
  start = std::chrono::high_resolution_clock::now();
  EXPECT_FALSE(queue.Pop(&value, 100));
  EXPECT_EQ(errno, ETIMEDOUT);
  EXPECT_EQ(value, -1);
  EXPECT_TRUE(queue.Empty());
  finish = std::chrono::high_resolution_clock::now();
  elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);
}

TEST_F(BlockingQueueTest, ForcePush) {
  const int queue_size = 1;
  const int push_count = 12;
  BlockingQueue<int> queue(queue_size);

  for (int i = 0; i < push_count; i++) {
    int value = i * i;
    queue.PushForce(value);
  }

  EXPECT_EQ(queue.Size(), push_count);
  EXPECT_TRUE(queue.Full());

  for (int i = 0; i < push_count; i++) {
    int value;
    queue.Pop(&value);
    EXPECT_EQ(value, i * i);
  }
}

TEST_F(BlockingQueueTest, ForcePushBatch) {
  const int queue_size = 1;
  const int push_count = 12;
  BlockingQueue<int> queue(queue_size);
  std::vector<int> numbers;

  for (int i = 0; i < push_count; i++) {
    int value = i * i;
    numbers.push_back(value);
  }

  queue.PushBatchForce(&numbers);

  EXPECT_EQ(queue.Size(), push_count);

  for (int i = 0; i < push_count; i++) {
    int value;
    queue.Pop(&value);
    EXPECT_EQ(value, i * i);
  }
}

TEST_F(BlockingQueueTest, ForcePushBatchWait) {
  const int queue_size = 1;
  const int push_count = 12;
  BlockingQueue<int> queue(queue_size);
  std::vector<int> numbers;

  for (int i = 0; i < push_count; i++) {
    int value = i * i;
    numbers.push_back(value);
  }

  queue.PushBatchForce(&numbers);

  for (int i = push_count; i < push_count * 2; i++) {
    int value = i * i;
    numbers.push_back(value);
  }

  auto start = std::chrono::high_resolution_clock::now();
  queue.PushBatchForce(&numbers, true, 20);
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 20);
  EXPECT_EQ(queue.Size(), push_count);

  for (int i = 0; i < push_count; i++) {
    int value;
    queue.Pop(&value);
    EXPECT_EQ(value, i * i);
  }
}

TEST_F(BlockingQueueTest, QueueShutdown) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  /* none blocking */
  int value = -1;
  bool ret = queue.Pop(&value, -1);
  EXPECT_FALSE(ret);

  /* wait until wakeup */
  auto start = std::chrono::high_resolution_clock::now();
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queue.Shutdown();
  });

  /* wakup */
  EXPECT_FALSE(queue.Pop(&value, 0));
  EXPECT_EQ(value, -1);
  EXPECT_TRUE(queue.IsShutdown());
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);

  BlockingQueue<int> queueTwo(queue_size);
  for (int i = 0; i < queue_size; i++) {
    int value = i * i;
    queueTwo.Push(value);
  }

  /* wait until wakeup */
  start = std::chrono::high_resolution_clock::now();
  result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queueTwo.Shutdown();
  });

  /* wakup */
  value = queue_size + 1;
  EXPECT_FALSE(queueTwo.Push(value, 0));
  EXPECT_TRUE(queue.IsShutdown());
  finish = std::chrono::high_resolution_clock::now();
  elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  EXPECT_GE(elapsed.count(), 100);

  for (int i = 0; i < queue_size; i++) {
    int value = i * i;
    ret = queueTwo.Pop(&value);
    EXPECT_EQ(value, i * i);
  }

  value = -1;
  EXPECT_FALSE(queueTwo.Pop(&value));
  EXPECT_EQ(value, -1);
}

TEST_F(BlockingQueueTest, Wakeup) {
  const int queue_size = 12;
  BlockingQueue<int> queue(queue_size);

  /* wait until wakeup */
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queue.Wakeup();
  });

  auto func = [&]() {
    int value = -1;
    auto ret = queue.Pop(&value);
    EXPECT_FALSE(ret);
    EXPECT_EQ(errno, EINTR);
    EXPECT_EQ(value, -1);
  };

  std::thread Wait(func);

  func();
  Wait.join();
}

TEST_F(BlockingQueueTest, ConsumerProducer) {
  int queue_size = 10;
  int loop = 10000;
  int total_sum1 = 0;
  int total_sum2 = 0;
  int expect_sum1 = 0;
  int expect_sum2 = 0;

  BlockingQueue<TestNumber> queue(queue_size);

  std::thread producer1([&]() {
    for (int i = 0; i < loop; i++) {
      TestNumber value = i;
      queue.Push(value);
      expect_sum1 += i;
    }
  });

  std::thread producer2([&]() {
    for (int i = 0; i < loop; i++) {
      TestNumber value = i;
      queue.Push(value);
      expect_sum2 += i;
    }
  });

  std::thread consumer1([&]() {
    while (true) {
      TestNumber value = -1;
      auto ret = queue.Pop(&value);
      if (ret == false) {
        break;
      }

      total_sum1 += value.Get();
    }
  });

  std::thread consumer2([&]() {
    while (true) {
      TestNumber value = -1;
      auto ret = queue.Pop(&value);
      if (ret == false) {
        break;
      }

      total_sum2 += value.Get();
    }
  });

  producer1.join();
  producer2.join();
  queue.Shutdown();
  consumer1.join();
  consumer2.join();
  EXPECT_EQ(expect_sum1 + expect_sum2, total_sum1 + total_sum2);
}

TEST_F(PriorityBlockingQueueTest, PushPriorityCheck) {
  const int queue_size = 12;
  PriorityBlockingQueue<int> queue(queue_size);

  for (int i = 1; i <= queue_size; i++) {
    queue.Push(i);
  }
  int value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, queue_size);

  EXPECT_EQ(queue_size, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    int value = -1;
    queue.Pop(&value);
    EXPECT_EQ(i, value);
  }
}

TEST_F(PriorityBlockingQueueTest, PopBatch) {
  const int queue_size = 12;
  const int loop = 8;

  PriorityBlockingQueue<int> queue(queue_size * loop);

  for (int i = 1; i <= queue_size; i++) {
    for (int j = 0; j < loop; j++) {
      queue.Push(i);
    }
  }
  int value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 12);

  EXPECT_EQ(queue_size * loop, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    std::vector<int> out_nums;
    queue.PopBatch(&out_nums);
    EXPECT_EQ(loop, out_nums.size());
    EXPECT_EQ(i, out_nums[0]);
  }
}

TEST_F(PriorityBlockingQueueTest, PopBatchMaxNum) {
  const int queue_size = 12;
  const int loop = 8;
  const int first_pop_num = 3;

  PriorityBlockingQueue<int> queue(queue_size * loop);

  for (int i = 1; i <= queue_size; i++) {
    for (int j = 0; j < loop; j++) {
      queue.Push(i);
    }
  }
  int value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, 12);

  EXPECT_EQ(queue_size * loop, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    std::vector<int> out_nums;
    auto ret = queue.PopBatch(&out_nums, -1, first_pop_num);
    EXPECT_EQ(first_pop_num, ret);
    ret = queue.PopBatch(&out_nums, -1);
    EXPECT_EQ(loop - first_pop_num, ret);
    EXPECT_EQ(loop, out_nums.size());
    EXPECT_EQ(i, out_nums[0]);
  }
}

TEST_F(PriorityBlockingQueueTest, PushPriorityCustomCompare) {
  const int queue_size = 12;
  struct CustomCompare {
    auto operator()(TestNumber const& a, TestNumber const& b) const -> bool {
      return a <= b;
    }
  };

  PriorityBlockingQueue<TestNumber, CustomCompare> queue(queue_size);

  for (int i = 1; i <= queue_size; i++) {
    TestNumber value = i;
    queue.Push(value);
  }
  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, queue_size);

  EXPECT_EQ(queue_size, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    TestNumber value = -1;
    queue.Pop(&value);
    EXPECT_EQ(value, i);
  }
}

TEST_F(PriorityBlockingQueueTest, PopBatchCheckOrder) {
  const int queue_size = 2;
  const int loop = 4;
  PriorityBlockingQueue<TestNumber> queue(queue_size * loop);

  for (int i = 1; i <= queue_size; i++) {
    for (int j = 0; j < loop; j++) {
      TestNumber value = i;
      value.SetPrivate(j);
      queue.Push(value);
    }
  }
  TestNumber value = -1;
  queue.Front(&value);
  EXPECT_EQ(value, queue_size);

  EXPECT_EQ(queue_size * loop, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    std::vector<TestNumber> out_nums;
    queue.PopBatch(&out_nums);
    EXPECT_EQ(loop, out_nums.size());
    for (int j = 0; j < loop; j++) {
      EXPECT_EQ(out_nums[j].GetPrivate(), j);
    }
    EXPECT_EQ(out_nums[0], i);
  }
}

TEST_F(PriorityBlockingQueueTest, SharedPtrPopBatchCheckOrder) {
  const int queue_size = 2;
  const int loop = 4;

  struct CustomCompare {
    auto operator()(std::shared_ptr<TestNumber> const& a,
                    std::shared_ptr<TestNumber> const& b) const -> bool {
      return a->Get() < b->Get();
    }
  };

  PriorityBlockingQueue<std::shared_ptr<TestNumber>, CustomCompare> queue(
      queue_size * loop);

  for (int i = 1; i <= queue_size; i++) {
    for (int j = 0; j < loop; j++) {
      std::shared_ptr<TestNumber> value = std::make_shared<TestNumber>(i);
      value->SetPrivate(j);
      queue.Push(value);
    }
  }

  std::shared_ptr<TestNumber> value = std::make_shared<TestNumber>(-1);
  queue.Front(&value);
  EXPECT_EQ(*value, queue_size);

  EXPECT_EQ(queue_size * loop, queue.Size());
  for (int i = queue_size; i > 0; i--) {
    std::vector<std::shared_ptr<TestNumber>> out_nums;
    queue.PopBatch(&out_nums);
    EXPECT_EQ(loop, out_nums.size());
    for (int j = 0; j < loop; j++) {
      EXPECT_EQ(out_nums[j]->GetPrivate(), j);
    }
    EXPECT_EQ(*out_nums[0], i);
  }
}

TEST_F(PriorityBlockingQueueTest, ConsumerProducerBatch) {
  int queue_size = 5;
  int loop = 10;
  int total_sum1 = 0;
  int total_sum2 = 0;
  int expect_sum1 = 0;
  int expect_sum2 = 0;

  PriorityBlockingQueue<TestNumber> queue(queue_size);

  std::thread producer1([&]() {
    std::vector<TestNumber> in_nums;
    for (int i = 0; i < loop; i++) {
      TestNumber value = i;
      in_nums.push_back(value);
      expect_sum1 += i;
      if (i % queue_size == 0) {
        queue.PushBatch(&in_nums);
      }
    }
    queue.PushBatch(&in_nums);
  });

  std::thread producer2([&]() {
    for (int i = 0; i < loop; i++) {
      TestNumber value = i;
      queue.Push(value);
      expect_sum2 += i;
    }
  });

  std::thread consumer1([&]() {
    while (true) {
      TestNumber value = -1;
      auto ret = queue.Pop(&value);
      if (ret == false) {
        break;
      }

      total_sum1 += value.Get();
    }
  });

  std::thread consumer2([&]() {
    while (true) {
      std::vector<TestNumber> out_nums;
      auto ret = queue.PopBatch(&out_nums);
      if (ret == false) {
        break;
      }

      for (size_t i = 0; i < out_nums.size(); i++) {
        total_sum2 += out_nums[i].Get();
      }
    }
  });

  producer1.join();
  producer2.join();
  queue.Shutdown();
  consumer1.join();
  consumer2.join();
  EXPECT_EQ(expect_sum1 + expect_sum2, total_sum1 + total_sum2);
}

TEST_F(PriorityBlockingQueueTest, Perf) {
  int total_count = 0;
  int expect_count = 0;
  unsigned long begin;
  unsigned long end;
  bool stop = false;

  PriorityBlockingQueue<TestNumber> queue(8192);

  begin = GetTickCount();
  std::thread producer([&]() {
    std::vector<TestNumber> in_nums;
    int i = 0;
    while (stop == false) {
      TestNumber value = i++;
      in_nums.push_back(value);
      expect_count += 1;
      queue.PushBatch(&in_nums);
    }
  });

  std::thread consumer([&]() {
    while (true) {
      TestNumber value = -1;
      auto ret = queue.Pop(&value);
      if (ret == false) {
        break;
      }

      total_count += 1;
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stop = true;
  producer.join();
  queue.Shutdown();
  consumer.join();
  end = GetTickCount();
  EXPECT_EQ(expect_count, total_count);

  MBLOG_INFO << "total: " << total_count;
  MBLOG_INFO << "ops: " << 1.0 * total_count / (end - begin) * 1000.0;
}

}  // namespace modelbox