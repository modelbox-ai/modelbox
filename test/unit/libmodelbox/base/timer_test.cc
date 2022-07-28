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
#include <modelbox/base/timer.h>
#include <modelbox/base/utils.h>

#include <future>

#include "gtest/gtest.h"
#include "test_config.h"

namespace modelbox {
class TimerTest : public testing::Test {
 public:
  TimerTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(TimerTest, Empty) {
  {
    Timer tm;
    tm.Start();
  }
  EXPECT_TRUE(true);
}

TEST_F(TimerTest, Sched) {
  Timer tm;
  int count = 0;
  int loop = 2;
  uint64_t start = GetTickCount();
  uint64_t end = GetTickCount();
  const char *msg = "Hello";

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>(
      [&](const char *id) {
        EXPECT_LE(count, loop);
        count++;
        if (count == loop) {
          task->Stop();
          end = GetTickCount();
        }
        EXPECT_STREQ(msg, id);
      },
      msg);
  tm.Start();
  tm.Schedule(task, 0, 10);
  tm.Shutdown();
  EXPECT_GE(end - start, 20);
}

TEST_F(TimerTest, SchedMany) {
  Timer tm;
  int count = 10;
  uint64_t start = GetTickCount();
  const char *msg = "Hello";

  std::vector<std::shared_ptr<TimerTask>> taskset;
  std::vector<uint64_t> end_time;
  end_time.resize(count);
  taskset.resize(count);

  tm.Start(false);
  for (int i = count - 1; i >= 0; i--) {
    std::shared_ptr<TimerTask> task = std::make_shared<TimerTask>();
    task->Callback(
        [&, i](const char *id, TimerTask *task) {
          EXPECT_STREQ(msg, id);
          end_time[i] = GetTickCount();
          task->Stop();
        },
        msg, task.get());
    task->SetName(std::to_string(i));
    tm.Schedule(task, 0, 10 * (i + 1));
    taskset[i] = task;
  }
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  });

  tm.Shutdown();
  result_future.wait();
  for (int i = 0; i < count; i++) {
    auto end = end_time[i];
    EXPECT_GE(end - start, 10 * (i + 1));
    EXPECT_LE(end - start, 10 * (i + 1) + 10);
  }
}

TEST_F(TimerTest, Callback) {
  Timer tm;
  int count = 0;
  int loop = 2;
  const char *msg = "Hello";

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>();
  task->Callback(
      [&](const char *id) {
        EXPECT_LE(count, loop);
        count++;
        if (count == loop) {
          task->Stop();
        }
        EXPECT_STREQ(msg, id);
      },
      msg);
  tm.Start();
  tm.Schedule(task, 0, 10);
  tm.Shutdown();
}

TEST_F(TimerTest, ThreadLocalTaskGet) {
  Timer tm;
  int count = 0;
  int loop = 2;
  const char *msg = "Hello";

  tm.Start();
  {
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    auto *task_ptr = task.get();
    task->Callback(
        [&, task_ptr](const char *id) {
          EXPECT_LE(count, loop);
          count++;
          if (count == loop) {
            task->Stop();
          }
          EXPECT_STREQ(msg, id);
          EXPECT_EQ(task_ptr, Timer::CurrentTimerTask().get());
        },
        msg);
    tm.Schedule(task, 0, 10);
  }
  tm.Shutdown();
}

TEST_F(TimerTest, CallbackNoOwnerShip) {
  Timer tm;
  int count = 0;
  int loop = 2;
  const char *msg = "Hello";

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>();
  task->Callback(
      [&](const char *id) {
        EXPECT_LE(count, loop);
        count++;
        if (count == loop) {
          task->Stop();
        }
        EXPECT_STREQ(msg, id);
      },
      msg);
  tm.Start();
  tm.Schedule(task, 0, 10, true);
  tm.Shutdown();
}

TEST_F(TimerTest, NoCallbackNoOwnerShip) {
  Timer tm;
  int count = 0;
  int loop = 2;
  {
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    task->Callback(
        [&]() {
          EXPECT_LE(count, loop);
          count++;
          if (count == loop) {
            task->Stop();
          }
        });
    tm.Start();
    tm.Schedule(task, 0, 10);
    task = nullptr;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(count, 0);
  tm.Shutdown();
}

TEST_F(TimerTest, CallbackWithOwnerShip) {
  Timer tm;
  int count = 0;
  int loop = 2;
  const char *msg = "Hello";
  {
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    task->Callback(
        [&](const char *id, TimerTask *t) {
          EXPECT_LE(count, loop);
          count++;
          if (count == loop) {
            t->Stop();
          }
          EXPECT_STREQ(msg, id);
        },
        msg, task.get());
    tm.Start();
    tm.Schedule(task, 0, 10, true);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(count, 2);
  tm.Shutdown();
}

TEST_F(TimerTest, SchedDelay) {
  Timer tm;
  int count = 0;
  int loop = 2;
  uint64_t start = GetTickCount();
  uint64_t end = GetTickCount();
  const char *msg = "Hello";

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>(
      [&](const char *id) {
        EXPECT_LE(count, loop);
        count++;
        if (count == loop) {
          task->Stop();
          end = GetTickCount();
        }
        EXPECT_STREQ(msg, id);
      },
      msg);
  tm.Start();
  tm.Schedule(task, 100, 10);
  tm.Shutdown();
  EXPECT_GE(end - start, 120);
}

TEST_F(TimerTest, SchedOnce) {
  Timer tm;
  int count = 0;
  uint64_t start = GetTickCount();
  uint64_t end = GetTickCount();
  const char *msg = "Hello";

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>(
      [&](const char *id) {
        count++;
        EXPECT_EQ(count, 1);
        EXPECT_STREQ(msg, id);
        end = GetTickCount();
      },
      msg);
  tm.Start();
  tm.Schedule(task, 10, 0);
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    task->Stop();
  });

  tm.Shutdown();
  EXPECT_GE(end - start, 10);
  result_future.wait();
  EXPECT_EQ(count, 1);
}

TEST_F(TimerTest, StopBeforeHit) {
  Timer tm;
  int count = 0;
  int i = 0;
  uint64_t start = GetTickCount();
  uint64_t end = GetTickCount();
  tm.Start(false);

  std::vector<std::shared_ptr<TimerTask>> taskset;
  for (i = 0; i < 10; i++) {
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>([&]() { EXPECT_TRUE(false); });
    tm.Schedule(task, 0, 1000 * (i + 1));
    task->SetName(std::to_string(i));
    taskset.push_back(task);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  for (i = 0; i < 10; i++) {
    taskset[i]->Stop();
  }

  tm.Shutdown();
  end = GetTickCount();
  EXPECT_EQ(count, 0);
  EXPECT_LT(end - start, 30);
}

TEST_F(TimerTest, TakeOwnerShipStopBeforeHit) {
  Timer tm;
  int count = 0;
  tm.Start();
  {
    // trigger here
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    auto *task_ptr = task.get();
    task->Callback([&, task_ptr]() {
      count++;
      EXPECT_TRUE(true);
      task_ptr->Stop();
    });

    tm.Schedule(task, 0, 10, true);
  }

  {
    // no trigger
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    auto *task_ptr = task.get();
    task->Callback([&, task_ptr]() {
      count++;
      EXPECT_TRUE(true);
    });
    tm.Schedule(task, 0, 10, false);
  }

  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  result_future.wait();
  EXPECT_EQ(count, 1);
  tm.Stop();
}

TEST_F(TimerTest, SchedStopBeforeHit) {
  Timer tm;
  int count = 0;

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>([&]() {
    count++;
    EXPECT_FALSE(true);
  });

  tm.Start();
  tm.Schedule(task, 0, 1000);
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    task->Stop();
  });

  result_future.wait();

  EXPECT_EQ(count, 0);
  tm.Stop();
}

TEST_F(TimerTest, SchedStopAfterHit) {
  Timer tm;
  int count = 0;

  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>([&]() {
    count++;
    EXPECT_EQ(count, 1);
    task->Stop();
  });

  tm.Start();
  tm.Schedule(task, 0, 10);
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  tm.Shutdown();
  EXPECT_EQ(count, 1);
  result_future.wait();
}

TEST_F(TimerTest, SchedBatch) {
  Timer tm;
  std::atomic<uint32_t> count;
  int loop = 10;

  count = 0;
  std::vector<std::shared_ptr<TimerTask>> list;
  for (int i = 0; i < loop; i++) {
    std::shared_ptr<TimerTask> task;
    task = std::make_shared<TimerTask>();
    task->SetName(std::to_string(i));
    std::weak_ptr<TimerTask> task_weak = task;
    task->Callback(
        [&, task_weak](int index) {
          auto t = task_weak.lock();
          if (t == nullptr) {
            return;
          }

          count++;
          t->Stop();
        },
        i);
    list.push_back(task);
  }

  tm.Start();
  for (size_t i = 0; i < list.size(); i++) {
    tm.Schedule(list[i], 0, 10 * i);
  }

  tm.Shutdown();
  EXPECT_EQ(count, loop);
}

TEST_F(TimerTest, GlobalTimer) {
  int count = 0;
  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>([&]() {
    count++;
    EXPECT_EQ(count, 1);
    task->Stop();
  });

  TimerGlobal::Start();
  Defer { TimerGlobal::Stop(); };

  {
    TimerGlobal::Start();
    Defer { TimerGlobal::Stop(); };
  }

  TimerGlobal::Schedule(task, 0, 10);
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(count, 1);
  result_future.wait();
}

TEST_F(TimerTest, GlobalTimerStopBeforHit) {
  int count = 0;
  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>([&]() {
    count++;
  });

  TimerGlobal::Start();
  TimerGlobal::Schedule(task, 0, 10);
  auto result_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  });
  TimerGlobal::Stop(); 

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(count, 0);
  result_future.wait();
}

TEST_F(TimerTest, GlobalTimerTakeTooLong) {
  int count = 0;
  std::shared_ptr<TimerTask> task;
  task = std::make_shared<TimerTask>([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    count++;
    EXPECT_EQ(count, 1);
    task->Stop();
  });

  TimerGlobal::Start();
  Defer { TimerGlobal::Stop(); };
  TimerGlobal::Schedule(task, 0, 10);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(count, 1);
}

}  // namespace modelbox
