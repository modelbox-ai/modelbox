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
#include <modelbox/base/os.h>
#include <modelbox/base/timer.h>
#include <modelbox/base/utils.h>

namespace modelbox {

constexpr int TIMER_MAX_RUNNING_TIME = 50;

Timer TimerGlobal::timer_;
int TimerGlobal::refcnt_;
std::mutex TimerGlobal::lock_;

TimerTask::~TimerTask() { Stop(); }

TimerTask::TimerTask() { task_name_ = GetCaller(); }

void TimerTask::Stop() {
  if (sched_timer_ != nullptr) {
    sched_timer_->Stop();
  }

  is_running_ = false;
  weak_timer_.reset();
};

void TimerTask::Run() { task_func_(); }

void TimerTask::SetName(const std::string &name) {
  if (sched_timer_ != nullptr) {
    sched_timer_->SetName(name);
  }

  task_name_ = name;
}

std::string TimerTask::GetName() { return task_name_; }

std::string TimerTask::GetCaller() {
  std::stringstream str;

  str << "@" << std::hex << __builtin_return_address(0);
  return str.str();
}

std::shared_ptr<TimerTask> TimerTask::MakeSchedWeakTimer() {
  auto sched_timer = std::make_shared<TimerTask>();
  sched_timer->weak_timer_ = shared_from_this();
  sched_timer->is_weaktimer_ = true;
  sched_timer->task_name_ = task_name_;
  sched_timer->SetPeriod(period_);
  sched_timer->SetDelay(delay_);
  this->sched_timer_ = sched_timer;
  return sched_timer;
}

void TimerTask::SetHitTime(uint64_t time) { hit_time_ = time; }

uint64_t TimerTask::GetHitTime() { return hit_time_; }

void TimerTask::SetPeriod(uint64_t period) { period_ = period; }

uint64_t TimerTask::GetPeriod() { return period_; }

void TimerTask::SetDelay(uint64_t delay) { delay_ = delay; }

uint64_t TimerTask::GetDelay() { return delay_; }

void TimerTask::SetTimerRunning(bool running) {
  is_running_ = running;
  auto timer = weak_timer_.lock();
  if (timer) {
    timer->SetTimerRunning(running);
  }
}

bool TimerTask::IsWeakPtrTimerTask() { return is_weaktimer_; }

bool TimerTask::IsRunning() { return is_running_; }

Timer::Timer() {
  // make sure tick may not overflow for a long long time.
  start_tick_ = GetTickCount();
};

Timer::~Timer() { Stop(); };

thread_local std::shared_ptr<TimerTask> Timer::current_timer_task_ = nullptr;

std::shared_ptr<TimerTask> Timer::CurrentTimerTask() {
  return current_timer_task_;
}

void Timer::SetName(const std::string &name) {
  if (timer_running_) {
    return;
  }

  name_ = name;
}

void Timer::Start(bool lazy) {
  if (timer_running_) {
    return;
  }

  timer_running_ = true;
  is_shutdown_ = false;
  if (lazy == false) {
    StartTimerThread();
  }
}

void Timer::Shutdown() {
  if (timer_running_ == false) {
    return;
  }

  std::unique_lock<std::mutex> lock(lock_);
  is_shutdown_ = true;
  lock.unlock();

  cond_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Timer::Run() {
  if (name_.length() > 0) {
    os->Thread->SetName(name_);
  }

  if (thread_running_ == false) {
    return;
  }

  while (timer_running_) {
    RunTimer();
  }

  thread_running_ = false;
};

void Timer::StartAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  timer_running_ = true;
  is_shutdown_ = false;
  thread_running_ = true;
}

void Timer::StopAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  timer_running_ = false;
  cond_.notify_one();
}

void Timer::Stop() {
  if (timer_running_ == false) {
    return;
  }

  std::unique_lock<std::mutex> lock(lock_);
  timer_running_ = false;
  cond_.notify_all();
  lock.unlock();

  if (thread_.joinable()) {
    thread_.join();
  }

  lock.lock();
  while (!timer_queue_.empty()) {
    auto timer = timer_queue_.top();
    timer->Stop();
    timer_queue_.pop();
  }
};

void Timer::StartTimerThread() {
  if (thread_running_ == true) {
    return;
  }

  thread_running_ = true;
  thread_ = std::thread(&Timer::Run, this);
}

void Timer::Schedule(const std::shared_ptr<TimerTask> timer_task,
                     uint64_t delay, uint64_t period, bool take_owner_ship) {
  if (timer_running_ == false) {
    MBLOG_WARN << "Schedule timer failed, timer is not running.";
    return;
  }

  timer_task->SetPeriod(period);
  timer_task->SetDelay(delay);

  auto timer_task_sched = timer_task;
  if (take_owner_ship == false) {
    timer_task_sched = timer_task->MakeSchedWeakTimer();
  }

  std::unique_lock<std::mutex> lock(lock_);
  uint64_t now = GetCurrentTick();
  if (thread_running_ == false) {
    StartTimerThread();
  }

  InsertTimerTask(timer_task_sched, now);
  timer_task_sched->SetTimerRunning(true);
  auto top = timer_queue_.top();
  if (timer_task_sched.get() == top.get()) {
    cond_.notify_one();
  }

  return;
}

bool Timer::SetPriority(int priority) {
  MBLOG_WARN << "not support now";
  return false;
}

uint64_t Timer::GetCurrentTick() {
  return GetTickDiff(GetTickCount(), start_tick_);
}

void Timer::InsertTimerTask(std::shared_ptr<TimerTask> timer_task,
                            uint64_t now) {
  timer_task->SetHitTime(now + timer_task->GetPeriod() +
                         timer_task->GetDelay());
  timer_queue_.push(timer_task);
}

void Timer::RemoveTopTimerTask() {
  auto top = timer_queue_.top();
  timer_queue_.pop();
}

void Timer::StopTimerTask(TimerTask *timer_task) {
  std::unique_lock<std::mutex> lock(lock_);
  auto timer = timer_queue_.top();
  if (timer->IsRunning() == false) {
    cond_.notify_one();
  }
}

bool Timer::RemoveStoppedTimer() {
  bool removed = false;
  while (timer_queue_.size() > 0) {
    auto timer = timer_queue_.top();
    if (timer->IsRunning()) {
      return removed;
    }

    RemoveTopTimerTask();
    removed = true;
  }

  if (timer_queue_.size() == 0) {
    return true;
  }

  return removed;
}

void Timer::WaitTimerTask(std::unique_lock<std::mutex> &lock,
                          std::shared_ptr<TimerTask> &timer) {
  // wait for first timer task timeout
  timer = timer_queue_.top();
  uint64_t now = GetCurrentTick();
  uint64_t time_diff = GetTickDiff(timer->GetHitTime(), now);
  if (time_diff <= timer->GetPeriod() + timer->GetDelay()) {
    auto wait_time = std::chrono::milliseconds(time_diff);
    cond_.wait_for(lock, wait_time, [this, timer]() {
      // return true when timer stop, top timer stop, top timer changed.
      return timer_running_ == false || timer_queue_.size() == 0 ||
             (timer_queue_.top()->IsRunning() == false) ||
             timer_queue_.top().get() != timer.get();
    });

    if (timer_queue_.size() == 0 || timer_running_ == false) {
      timer = nullptr;
      return;
    }

    timer = timer_queue_.top();
  } else if (time_diff != 0) {
    // timer stall, force reset hit time
    timer = timer_queue_.top();
    time_diff = GetTickDiff(now, timer->GetHitTime());
    if (time_diff > (timer->GetPeriod() + timer->GetDelay()) * 5 &&
        timer->GetPeriod() + timer->GetDelay() > 0) {
      MBLOG_WARN << "timer stall too long, update timer task";
      MBLOG_WARN << "timer name: " << timer->GetName();
      MBLOG_WARN << "timer period: " << timer->GetPeriod();
      timer->SetHitTime(now);
    } else if (time_diff > timer->GetPeriod() + timer->GetDelay()) {
      MBLOG_DEBUG << "timer [" << timer->GetName() << "] stall for " << time_diff
                 << "ms";
    }
  }

  return;
}

bool Timer::GetTimerTask(std::unique_lock<std::mutex> &lock,
                         std::shared_ptr<TimerTask> &timer) {
  // wait for timer task
  if (timer_queue_.size() <= 0) {
    cond_.wait(lock, [this]() {
      return timer_queue_.size() > 0 || timer_running_ == false ||
             is_shutdown_ == true;
    });
    if (timer_running_ == false || is_shutdown_ == true) {
      return false;
    }
  }

  // skip timer task already stopped
  if (RemoveStoppedTimer() == true) {
    return false;
  }

  // wait timer task timeout
  WaitTimerTask(lock, timer);

  // stop, return.
  if (timer_running_ == false || timer_queue_.size() <= 0) {
    return false;
  }

  if (timer->IsRunning() == false) {
    RemoveTopTimerTask();
    return false;
  }

  // get a timer task
  uint64_t now = GetCurrentTick();
  uint64_t time_diff = GetTickDiff(timer->GetHitTime(), now);
  if ((time_diff <= timer->GetPeriod() + timer->GetDelay() && time_diff != 0) ||
      timer_running_ == false) {
    return false;
  }

  RemoveTopTimerTask();
  return true;
}

void Timer::RunTimerTask(std::shared_ptr<TimerTask> timer,
                         std::shared_ptr<TimerTask> timer_call) {
  try {
    uint64_t start = GetCurrentTick();
    current_timer_task_ = timer_call;
    timer_call->Run();
    current_timer_task_ = nullptr;
    uint64_t end = GetCurrentTick();

    auto elapsed = end - start;
    if (elapsed > TIMER_MAX_RUNNING_TIME) {
      std::string msg;
      if (name_.length() > 0) {
        MBLOG_WARN << name_ << ": timer '" << timer->GetName()
                   << "' run too long, take " << elapsed << "ms";
      } else {
        MBLOG_WARN << "timer '" << timer->GetName() << "' run too long, take "
                   << elapsed << "ms";
      }
    }
  } catch (const std::bad_function_call &ex) {
    MBLOG_WARN << "timer '" << timer->GetName()
               << "' is invalid, function is not set, disable";
    timer->SetTimerRunning(false);
  } catch (const std::exception &ex) {
    MBLOG_WARN << "timer '" << timer->GetName()
               << "'caght exception: " << ex.what();
  }
}

void Timer::RunTimer() {
  // get a timer
  std::shared_ptr<TimerTask> timer;
  std::shared_ptr<TimerTask> timer_call;

  std::unique_lock<std::mutex> lock(lock_);

  if (timer_queue_.size() == 0) {
    // reset tick
    start_tick_ = GetTickCount();
  }

  if (GetTimerTask(lock, timer) == false) {
    if (is_shutdown_ == true && timer_queue_.size() <= 0) {
      timer_running_ = false;
    }
    return;
  }

  lock.unlock();

  if (timer->IsWeakPtrTimerTask()) {
    timer_call = timer->weak_timer_.lock();
    if (timer_call == nullptr) {
      timer->SetTimerRunning(false);
      return;
    }
  } else {
    timer_call = timer;
  }

  // run timer
  RunTimerTask(timer, timer_call);

  if (timer->GetPeriod() == 0 || timer->IsRunning() == false) {
    timer->SetTimerRunning(false);
    return;
  }

  // reset delay time.
  if (timer->GetDelay() > 0) {
    timer->SetDelay(0);
  }

  // reschedue task
  lock.lock();
  InsertTimerTask(timer, timer->GetHitTime());
}

void TimerGlobal::Stop() {
  std::unique_lock<std::mutex> lock(lock_);
  refcnt_--;
  if (refcnt_ > 0) {
    return;
  }

  timer_.Stop();
}

void TimerGlobal::Start() {
  std::unique_lock<std::mutex> lock(lock_);
  refcnt_++;
  if (refcnt_ > 1) {
    return;
  }

  timer_.SetName("Global-Timer");
  timer_.Start();
}

void TimerGlobal::Schedule(const std::shared_ptr<TimerTask> timer_task,
                           uint64_t delay, uint64_t period,
                           bool take_owner_ship) {
  timer_.Schedule(timer_task, delay, period, take_owner_ship);
}

}  // namespace modelbox
