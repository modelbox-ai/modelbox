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


#ifndef MODELBOX_TIMER_H_
#define MODELBOX_TIMER_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace modelbox {

using TimerTaskFunction = std::function<void()>;
class Timer;
class TimerCompare;

static inline uint64_t GetTickDiff(uint64_t prev, uint64_t cur) {
  return ((prev) >= (cur)) ? ((prev) - (cur))
                           : ((~((uint64_t)(0)) - (cur)) + 1 + (prev));
}

/**
 * @brief Timer task.
 */
class TimerTask : public std::enable_shared_from_this<TimerTask> {
 public:
  /**
   * @brief Create a timer task
   * @param f f function
   * @param args args function arguments
   */
  template <typename Function, typename... Args>
  explicit TimerTask(Function &&f, Args &&...args) : is_running_(false) {
    Callback(f, args...);
  }

  /**
   * @brief Set timer routine
   * @param f function
   * @param args args function arguments
   */
  template <typename Function, typename... Args>
  void Callback(Function &&f, Args &&...args) {
    auto execute =
        std::bind(std::forward<Function>(f), std::forward<Args>(args)...);
    task_func_ = [execute]() { execute(); };

    if (task_name_.length() == 0) {
      task_name_ = GetCaller();
    }
  }

  /**
   * @brief Create a timer task
   */
  TimerTask();

  virtual ~TimerTask();

  /**
   * @brief Stop timer task
   */
  void Stop();

  /**
   * @brief Task run.
   */
  virtual void Run();

  /**
   * @brief Set task name.
   * @param name task name, default is caller function name
   */
  void SetName(const std::string &name);

  /**
   * @brief Get task name.
   * @return name task name
   */
  std::string GetName();

 protected:
  TimerTaskFunction task_func_;

  /**
   * @brief Get caller name
   * @return name caller name
   */
  std::string GetCaller();

  /**
   * @brief Get hit time
   * @return task hit time
   */
  uint64_t GetHitTime();

  /**
   * @brief Get task period
   * @return task period
   */
  uint64_t GetPeriod();

  /**
   * @brief Get task is running
   * @return task is running
   */
  bool IsRunning();

 private:
  friend class Timer;
  friend class TimerCompare;

  bool IsWeakPtrTimerTask();
  std::shared_ptr<TimerTask> MakeSchedWeakTimer();
  void SetHitTime(uint64_t time);
  void SetPeriod(uint64_t period);
  uint64_t GetDelay();
  void SetDelay(uint64_t delay);
  void SetTimerRunning(bool running);

  uint64_t hit_time_ = 0;
  uint64_t period_ = 0;
  uint64_t delay_ = 0;

  std::atomic_bool is_running_{false};
  std::string task_name_;
  bool is_weaktimer_{false};
  std::shared_ptr<TimerTask> sched_timer_{nullptr};
  std::weak_ptr<TimerTask> weak_timer_;
};

class TimerCompare {
 public:
  bool operator()(const std::shared_ptr<TimerTask> &lhs,
                  const std::shared_ptr<TimerTask> &rhs) {
    auto hit_time_lhs = lhs->GetHitTime();
    auto hit_time_rhs = rhs->GetHitTime();
    if (hit_time_lhs == hit_time_rhs) {
      return lhs->GetDelay() + lhs->GetPeriod() >
             rhs->GetDelay() + rhs->GetPeriod();
    }

    return hit_time_lhs > hit_time_rhs;
  }
};

/**
 * @brief Timer thread.
 */
class Timer {
 public:
  Timer();
  virtual ~Timer();

  /**
   * @brief Set timer thread priority
   * @param priority timer pritority
   */
  bool SetPriority(int priority);

  /**
   * @brief Shutdown main timer
   */
  void Shutdown();

  /**
   * @brief Set timer name
   */
  void SetName(const std::string &name);

  /**
   * @brief Start main timer, threading
   * @param lazy if true, will start thread when timer task is added.
   */
  void Start(bool lazy = true);

  /**
   * @brief Main timer run
   */
  virtual void Run();

  /**
   * @brief Stop main timer
   */
  void Stop();

  /**
   * @brief Schedule a timer task.
   * @param timer_task pointer to a timer task.
   * @param delay task for execution after the specified delay.
   * @param period schedule period, in millisecond.
   * @param take_owner_ship take ownership of shared_ptr timer_task.
   */
  void Schedule(const std::shared_ptr<TimerTask> &timer_task, uint64_t delay,
                uint64_t period, bool take_owner_ship = false);

  /**
   * @brief Get current tick
   * @return tick count
   */
  uint64_t GetCurrentTick();

  /**
   * @brief Get current timer task
   */
  static std::shared_ptr<TimerTask> CurrentTimerTask();

 protected:
  /**
   * @brief Run main timer
   */
  void RunTimer();

  /**
   * @brief Start main thread async
   */
  void StartAsync();

  /**
   * @brief Stop main timer async
   */
  void StopAsync();

 private:
  friend class TimerTask;
  void RunTimerTask(const std::shared_ptr<TimerTask> &timer,
                    const std::shared_ptr<TimerTask> &timer_call);

  void StartTimerThread();

  void InsertTimerTask(const std::shared_ptr<TimerTask> &timer_task,
                       uint64_t now);

  void RemoveTopTimerTask();

  void StopTimerTask(TimerTask *timer_task);

  void WaitTimerTask(std::unique_lock<std::mutex> &lock,
                     std::shared_ptr<TimerTask> &timer);

  bool GetTimerTask(std::unique_lock<std::mutex> &lock,
                    std::shared_ptr<TimerTask> &timer);

  bool RemoveStoppedTimer();

  thread_local static std::shared_ptr<TimerTask> current_timer_task_;
  bool is_shutdown_{false};
  uint64_t start_tick_{0};
  std::mutex lock_;
  std::thread thread_;
  bool timer_running_{false};
  bool thread_running_{false};
  std::string name_{"Timer"};
  std::condition_variable cond_;
  std::priority_queue<std::shared_ptr<TimerTask>,
                      std::vector<std::shared_ptr<TimerTask>>, TimerCompare>
      timer_queue_;
};

/**
 * @brief Global timer thread.
 */
class TimerGlobal {
 public:
  /**
   * @brief Start main timer, threading
   */
  static void Start();

  /**
   * @brief Stop main timer
   */
  static void Stop();

  /**
   * @brief Schedule a timer task.
   * @param timer_task pointer to a timer task.
   * @param delay task for execution after the specified delay.
   * @param period schedule period, in millisecond.
   * @param take_owner_ship take ownership of shared_ptr timer_task.
   */
  static void Schedule(const std::shared_ptr<TimerTask> &timer_task,
                       uint64_t delay, uint64_t period,
                       bool take_owner_ship = false);

 private:
  TimerGlobal();
  virtual ~TimerGlobal();

  static Timer timer_;
  static int refcnt_;
  static std::mutex lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_TIMER_H_