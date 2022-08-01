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

#include "modelbox/base/thread_pool.h"

#include "gtest/gtest.h"
class ThreadPoolTest : public testing::Test {
 public:
  ThreadPoolTest() = default;
  ~ThreadPoolTest() override = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

int compute(int a, int b) { return a + b; }

std::mutex coutMtx;

void short_task(int task_id) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void long_task(int consume_time) {
  std::this_thread::sleep_for(std::chrono::milliseconds(consume_time));
}

TEST_F(ThreadPoolTest, PoolCreate) {
  modelbox::ThreadPool pool(10);
  EXPECT_EQ(pool.GetThreadsNum(), 0);
}

TEST_F(ThreadPoolTest, SubmitTasks) {
  modelbox::ThreadPool pool;
  auto fut1 = pool.Submit(compute, 100, 100);
  auto fut2 = pool.Submit(compute, 100, 101);
  auto fut3 = pool.Submit(compute, 100, 102);
  auto fut4 = pool.Submit(compute, 100, 103);
  auto fut5 = pool.Submit(compute, 100, 104);

  EXPECT_EQ(fut1.get(), 200);
  EXPECT_EQ(fut2.get(), 201);
  EXPECT_EQ(fut3.get(), 202);
  EXPECT_EQ(fut4.get(), 203);
  EXPECT_EQ(fut5.get(), 204);
}

TEST_F(ThreadPoolTest, SubmitTasksMinTaskNumber) {
  modelbox::ThreadPool pool(0, 0, 0, 0);
  auto fut1 = pool.Submit(compute, 100, 100);
  auto fut2 = pool.Submit(compute, 100, 101);
  auto fut3 = pool.Submit(compute, 100, 102);
  auto fut4 = pool.Submit(compute, 100, 103);
  auto fut5 = pool.Submit(compute, 100, 104);

  EXPECT_EQ(fut1.get(), 200);
  EXPECT_EQ(fut2.get(), 201);
  EXPECT_EQ(fut3.get(), 202);
  EXPECT_EQ(fut4.get(), 203);
  EXPECT_EQ(fut5.get(), 204);
}

TEST_F(ThreadPoolTest, ThreadSize) {
  int thread_size = 4;
  modelbox::ThreadPool pool(thread_size, thread_size * 2, 2001);
  for (size_t i = 0; i < 2000; i++) {
    auto fut = pool.Submit(compute, 10, 100);
  }

  EXPECT_EQ(pool.GetThreadsNum(), thread_size);
}

TEST_F(ThreadPoolTest, SetThreadSize) {
  int thread_size = 4;
  modelbox::ThreadPool pool(thread_size, thread_size * 2, 2001);
  std::vector<std::future<int>> results;
  for (size_t i = 0; i < 2000; i++) {
    auto fut = pool.Submit(compute, 10, 100);
    results.push_back(std::move(fut));
  }

  results.clear();
  EXPECT_EQ(pool.GetThreadsNum(), thread_size);
  pool.SetThreadSize(1);
  pool.SetKeepAlive(10);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  if (pool.GetThreadsNum() > 1) {
    pool.SetThreadSize(1);
    pool.SetKeepAlive(10);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  EXPECT_EQ(pool.GetThreadsNum(), 1);
}

TEST_F(ThreadPoolTest, MaxThreadSize) {
  int thread_size = 4;
  int max_thread_size = 10;
  modelbox::ThreadPool pool(thread_size, max_thread_size, 1);
  for (size_t i = 0; i < 2000; i++) {
    auto fut = pool.Submit(compute, 10, 21000);
  }

  EXPECT_GT(pool.GetThreadsNum(), thread_size);
  EXPECT_LE(pool.GetThreadsNum(), max_thread_size);
}

TEST_F(ThreadPoolTest, Shutdown) {
  int thread_size = 4;
  int max_thread_size = 10;
  std::vector<std::future<int>> future_queue;
  modelbox::ThreadPool pool(thread_size, max_thread_size, 1);
  for (size_t i = 0; i < 20000; i++) {
    auto fut = pool.Submit(compute, 10, 21000);
    if (i == 1000) {
      pool.Shutdown();
    }

    if (i > 1000) {
      EXPECT_FALSE(fut.valid());
    } else {
      EXPECT_TRUE(fut.valid());
    }
    future_queue.emplace_back(std::move(fut));
  }

  future_queue.clear();
  EXPECT_EQ(pool.GetThreadsNum(), 0);
}

TEST_F(ThreadPoolTest, GetMaxThreadsNum) {
  int thread_size = 4;
  int max_thread_size = 10;
  modelbox::ThreadPool pool(thread_size, max_thread_size, 1);
  EXPECT_EQ(pool.GetMaxThreadsNum(), max_thread_size);
}

TEST_F(ThreadPoolTest, GetWaitingWorkCount) {
  int thread_size = 4;
  int max_thread_size = 10;
  modelbox::ThreadPool pool(thread_size, max_thread_size, 1);
  EXPECT_EQ(pool.GetWaitingWorkCount(), 0);
  for (size_t i = 0; i < 2000; i++) {
    auto fut = pool.Submit(compute, 10, 21000);
  }

  EXPECT_LE(pool.GetWaitingWorkCount(), 2000);
}

TEST_F(ThreadPoolTest, KeepAlive) {
  int thread_size = 4;
  int max_thread_size = 10;
  modelbox::ThreadPool pool(thread_size, max_thread_size, 1, -1);
  std::vector<std::future<int>> future_queue;
  for (size_t i = 0; i < 20000; i++) {
    auto fut = pool.Submit(compute, i, i);
    future_queue.push_back(std::move(fut));
  }

  for (size_t i = 0; i < future_queue.size(); ++i) {
    EXPECT_EQ(future_queue[i].get(), compute(i, i));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(pool.GetThreadsNum(), max_thread_size);
  std::this_thread::sleep_for(std::chrono::milliseconds(110));
  EXPECT_EQ(pool.GetThreadsNum(), thread_size);
  future_queue.clear();

  pool.SetKeepAlive(200);
  for (size_t i = 0; i < 1; i++) {
    auto fut = pool.Submit(compute, i, i);
    future_queue.push_back(std::move(fut));
  }

  for (size_t i = 0; i < future_queue.size(); ++i) {
    EXPECT_EQ(future_queue[i].get(), compute(i, i));
  }
  future_queue.clear();

  EXPECT_GE(pool.GetThreadsNum(), thread_size);
  EXPECT_LE(pool.GetThreadsNum(), max_thread_size);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  EXPECT_EQ(pool.GetThreadsNum(), thread_size);
}

TEST_F(ThreadPoolTest, Performance) {
  modelbox::ThreadPool pool(std::thread::hardware_concurrency());
  std::atomic<bool> is_stop_{false};
  std::atomic<bool> is_launch_print(true);
  std::atomic<bool> is_wait_print(true);

  std::mutex fut_mux;
  std::queue<std::future<int>> future_queue;
  std::condition_variable cv;

  size_t LAUNCH_TASK_COUNT = std::thread::hardware_concurrency();
  std::atomic<int64_t> launch_count{0};
  std::vector<std::future<void>> launch_list;
  for (size_t i = 0; i < LAUNCH_TASK_COUNT; ++i) {
    auto launch_task = std::async(std::launch::async, [&]() {
      auto begin_tick = modelbox::GetTickCount();
      while (!is_stop_) {
        auto fut = pool.Submit(compute, 1, 1);
        std::lock_guard<std::mutex> lock(fut_mux);
        future_queue.push(std::move(fut));
        cv.notify_all();
        launch_count++;
      }

      bool local_print = true;
      if (is_launch_print.compare_exchange_strong(local_print, false)) {
        MBLOG_INFO << "Submit rate: "
                   << ((float)(launch_count * 1000)) /
                          (modelbox::GetTickCount() - begin_tick)
                   << "/s";
      }
    });

    launch_list.push_back(std::move(launch_task));
  }

  size_t WAIT_TASK_COUNT = std::thread::hardware_concurrency();
  std::atomic<int64_t> wait_count{0};
  std::vector<std::future<void>> wait_list;
  for (size_t i = 0; i < WAIT_TASK_COUNT; ++i) {
    auto wait_task = std::async(std::launch::async, [&]() {
      auto begin_tick = modelbox::GetTickCount();
      while (!is_stop_) {
        std::unique_lock<std::mutex> lock(fut_mux);
        cv.wait(lock, [&]() { return !future_queue.empty() || is_stop_; });
        if (is_stop_) {
          break;
        }
        auto fut = std::move(future_queue.front());
        future_queue.pop();
        wait_count++;
        lock.unlock();

        EXPECT_TRUE(fut.valid());
        fut.wait();
      }

      bool local_print = true;
      if (is_wait_print.compare_exchange_strong(local_print, false)) {
        MBLOG_INFO << "Process rate: "
                   << ((float)(wait_count * 1000)) /
                          (modelbox::GetTickCount() - begin_tick)
                   << "/s";
      }
    });

    wait_list.push_back(std::move(wait_task));
  }

  for (size_t i = 0; i < LAUNCH_TASK_COUNT; ++i) {
    EXPECT_TRUE(launch_list[i].valid());
  }

  for (size_t i = 0; i < WAIT_TASK_COUNT; ++i) {
    EXPECT_TRUE(wait_list[i].valid());
  }

  auto status = wait_list[0].wait_for(std::chrono::milliseconds(1 * 500));
  if (status != std::future_status::ready) {
    is_stop_ = true;
    MBLOG_INFO << "set stop";
  }

  for (size_t i = 0; i < LAUNCH_TASK_COUNT; ++i) {
    launch_list[i].wait();
  }

  for (size_t i = 0; i < WAIT_TASK_COUNT; ++i) {
    wait_list[i].wait();
  }

  MBLOG_INFO << "launch count: " << launch_count
             << " wait_count: " << wait_count;
}