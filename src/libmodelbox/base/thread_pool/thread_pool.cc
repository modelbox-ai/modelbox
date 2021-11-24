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


#include <modelbox/base/os.h>
#include <modelbox/base/thread_pool.h>

#include <algorithm>
#include <cstring>

namespace modelbox {

constexpr int MIN_KEEP_ALIVE_TIME = 100;

ThreadWorker::ThreadWorker(ThreadPool *pool, int thread_id, bool core_worker) {
  pool_ = pool;
  is_core_worker_ = core_worker;
  is_joining_ = false;
  thread_id_ = thread_id;
}

ThreadWorker::~ThreadWorker() { Join(); }

void ThreadWorker::SetName(const std::string &name) {
  name_ = name;
  name_changed_.exchange(true);
}

void ThreadWorker::ChangeNameNow() {
  if (name_changed_.exchange(false) == false) {
    return;
  }

  if (name_.length() > 0) {
    os->Thread->SetName(name_);
  }
}

bool ThreadWorker::IsCore() { return is_core_worker_; }

void ThreadWorker::SetCore(bool is_core) { is_core_worker_ = is_core; }

void ThreadWorker::Run(ThreadWorker *worker) {
  while (worker->running_) {
    worker->ChangeNameNow();
    worker->pool_->RunWorker(worker);
  }

  auto pool = worker->pool_;
  worker->pool_ = nullptr;
  auto thread = worker->thread_;
  std::unique_lock<std::mutex> lock(worker->lock_);
  if (!worker->is_joining_ && worker->thread_) {
    worker->thread_->detach();
    worker->thread_ = nullptr;
  }

  lock.unlock();
  pool->ExitWorker(worker);
  // thread may be detached, leave nothing here.
}

void ThreadWorker::Start() {
  if (thread_) {
    return;
  }

  std::unique_lock<std::mutex> lock(lock_);
  running_ = true;
  thread_ = std::make_shared<std::thread>(&ThreadWorker::Run, this);
}

void ThreadWorker::Stop() {
  std::unique_lock<std::mutex> lock(lock_);
  running_ = false;
}

int ThreadWorker::Id() { return thread_id_; }

void ThreadWorker::Join() {
  Stop();
  std::shared_ptr<std::thread> thread = thread_;
  std::unique_lock<std::mutex> lock(lock_);
  if (thread_) {
    thread_ = nullptr;
    is_joining_ = true;
    lock.unlock();
    thread->join();
    thread = nullptr;
    lock.lock();
    is_joining_ = false;
  }
}

ThreadPool::ThreadPool(int thread_size, int max_thread_size, int queue_size,
                       int keep_alive) {
  if (thread_size < 0) {
    thread_size = std::thread::hardware_concurrency();
  }

  thread_size_ = thread_size;
  max_thread_size_ = max_thread_size;
  if (max_thread_size_ < thread_size_) {
    max_thread_size_ = thread_size_;
  }

  if (max_thread_size_ == 0) {
    max_thread_size_ = std::thread::hardware_concurrency();
  }

  if (queue_size < 0) {
    queue_size = thread_size;
    if (queue_size == 0) {
      queue_size = 1;
    }
  }

  if (keep_alive <= MIN_KEEP_ALIVE_TIME) {
    keep_alive = MIN_KEEP_ALIVE_TIME;
  }

  keep_alive_ = keep_alive;
  worker_num_ = 0;
  quit_ = false;
  work_queue_ = std::make_shared<BlockingQueue<ThreadFunction>>(queue_size);
}

ThreadPool::~ThreadPool() { Shutdown(); };

void ThreadPool::SetName(const std::string &name) { name_ = name; }

void ThreadPool::Shutdown(bool force) {
  work_queue_->Shutdown();
  if (force) {
    StopWokers();
  }

  std::unique_lock<std::mutex> lock(lock_);
  exit_cond_.wait(lock, [&]() { return workers_.size() <= 0; });
}

void ThreadPool::ExitWorker(ThreadWorker *worker) {
  RmvWorker(worker);
  // leave nothing here
}

bool ThreadPool::Park(ThreadWorker *worker, ThreadFunction &task) {
  auto wait_time = 0;
  if (worker->IsCore() == false) {
    int extend_thread_size = worker_num_ - thread_size_;
    if (extend_thread_size <= 0) {
      extend_thread_size = 1;
    }

    int wait_time_step = keep_alive_ / extend_thread_size;
    wait_time = wait_time_step * (worker_num_ - worker->Id() + 1);
    if (wait_time < MIN_KEEP_ALIVE_TIME) {
      wait_time = MIN_KEEP_ALIVE_TIME;
    } else if (wait_time > keep_alive_) {
      wait_time = keep_alive_;
    }
  }

  available_num_++;
  auto ret = work_queue_->Pop(&task, wait_time);
  available_num_--;
  if (ret == false) {
    if (errno == EINTR) {
      return false;
    }

    worker->Stop();
    return false;
  }

  return true;
}

void ThreadPool::RunWorker(ThreadWorker *worker) {
  ThreadFunction task;
  bool is_set_name = false;
  DeferCond { return is_set_name; };

  if (Park(worker, task) == false) {
    return;
  }

  if (task.name.length() > 0) {
    worker->SetName(task.name);
    worker->ChangeNameNow();
    is_set_name = true;
    DeferCondAdd { worker->SetName(name_); };
  }

  try {
    task.func();
  } catch (const std::exception &ex) {
    MBLOG_FATAL << "thread:  " << pthread_self() << " throw exception, "
                << ex.what();
  }
}

void ThreadPool::RmvWorker(ThreadWorker *worker) {
  worker_num_--;
  lock_.lock();
  for (auto iter = workers_.begin(); iter != workers_.end(); ++iter) {
    if ((*iter).get() == worker) {
      workers_.erase(iter);
      break;
    }
  }

  if (workers_.size() == 0) {
    exit_cond_.notify_one();
  }
  lock_.unlock();
  return;
}

Status ThreadPool::AddWorker(bool core_worker) {
  int id = worker_num_++;
  std::shared_ptr<ThreadWorker> worker =
      std::make_shared<ThreadWorker>(this, id, core_worker);
  worker->SetName(name_);
  lock_.lock();
  workers_.push_back(worker);
  lock_.unlock();
  worker->Start();
  worker->SetCore(core_worker);
  return STATUS_OK;
}

bool ThreadPool::SubmitTask(ThreadFunction &task) {
  bool is_queued = false;
  if (worker_num_++ < thread_size_) {
    AddWorker(true);
  }
  worker_num_--;

  auto ret = work_queue_->Push(task, -1);
  if (ret == true) {
    is_queued = true;
    if ((!work_queue_->Full() && thread_size_ > 0) || available_num_ > 0) {
      return ret;
    }
  }

  // expand extend thread pool
  auto num = worker_num_++;
  if (num < max_thread_size_) {
    bool create_core = false;
    if (num < thread_size_) {
      create_core = true;
    }
    AddWorker(create_core);
  }
  worker_num_--;

  if (is_queued) {
    return ret;
  }

  do {
    ret = work_queue_->Push(task, 0);
    if (ret == false && errno == EINTR) {
      continue;
    }
  } while (ret == false);

  return ret;
}

void ThreadPool::StopWokers() {
  work_queue_->Shutdown();
  std::unique_lock<std::mutex> lock(lock_);
  for (auto &workder : workers_) {
    workder->Stop();
  }
}

void ThreadPool::SetThreadSize(size_t size) {
  thread_size_ = size;
  if (max_thread_size_ < thread_size_) {
    max_thread_size_ = thread_size_;
  }

  int thread_num = 0;
  lock_.lock();
  for (size_t i = 0; i < workers_.size(); i++) {
    if (workers_[i]->IsCore() == false) {
      continue;
    }

    thread_num++;
    if (thread_num <= thread_size_) {
      continue;
    }

    workers_[i]->SetCore(false);
  }
  lock_.unlock();
  work_queue_->Wakeup();
}

void ThreadPool::SetMaxThreadSize(size_t size) {
  max_thread_size_ = size;
  if (max_thread_size_ < thread_size_) {
    max_thread_size_ = thread_size_;
  }
}

void ThreadPool::SetTaskQueueSize(size_t size) {
  work_queue_->SetCapacity(size);
}

void ThreadPool::SetKeepAlive(uint32_t timeout) {
  keep_alive_ = timeout;
  work_queue_->Wakeup();
}

int ThreadPool::GetThreadsNum() { return worker_num_; }

int ThreadPool::GetMaxThreadsNum() { return max_thread_size_; }

int ThreadPool::GetWaitingWorkCount() {
  return work_queue_ ? work_queue_->Size() : 0;
}

}  // namespace modelbox
