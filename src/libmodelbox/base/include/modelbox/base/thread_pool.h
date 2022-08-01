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


#ifndef MODELBOX_THREAD_POOL_H
#define MODELBOX_THREAD_POOL_H

#include <modelbox/base/blocking_queue.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <sched.h>
#include <unistd.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace modelbox {

struct ThreadFunction {
  std::string name;
  std::function<void()> func;
};

class ThreadPool;
class ThreadWorker {
 public:
  /**
   * @brief Init thread worker
   * @param pool thread pool.
   * @param thread_id thread id.
   * @param core_worker is core woker.
   * @return thread number.
   */
  ThreadWorker(ThreadPool *pool, int thread_id, bool core_worker = false);
  virtual ~ThreadWorker();

  /**
   * @brief Set thread pool name.
   * @param name thread pool name.
   * @return void
   */
  void SetName(const std::string &name);

  /**
   * @brief Is core thread.
   * @return is core thread.
   */
  bool IsCore();

  /**
   * @brief Start thread.
   */
  void Start();

  /**
   * @brief Stop thread.
   */
  void Stop();

  /**
   * @brief Wait for thread.
   */
  void Join();

  /**
   * @brief Get thread id
   */
  int Id();

 private:
  friend class ThreadPool;
  static void Run(ThreadWorker *worker);

  void ChangeNameNow();

  void SetCore(bool is_core);

  std::atomic<bool> running_{false};
  std::mutex lock_;
  bool is_joining_;
  std::shared_ptr<std::thread> thread_;
  ThreadPool *pool_;
  int thread_id_{0};
  bool is_core_worker_;
  std::string name_;
  std::atomic<bool> name_changed_{false};
};

class ThreadPool {
 public:
  /**
   * @brief Thread pool init
   * @param thread_size fixed thread size, default is cpu number.
   * @param max_thread_size max thread size, when queue is full, new thread will
   * be created.
   * @param queue_size task queue size, default equal thread size.
   * @param keep_alive non core thread keep alive time, minimum time is 100ms.
   */
  ThreadPool(int thread_size = -1, int max_thread_size = -1,
             int queue_size = -1, int keep_alive = 60000);

  virtual ~ThreadPool();

  /**
   * @brief Set thread pool name.
   * @param name thread pool name.
   * @return void
   */
  void SetName(const std::string &name);

  /**
   * @brief Set the size of core thread.
   * @param size queue size.
   */
  void SetThreadSize(size_t size);

  /**
   * @brief Set the size of max thread.
   * @param size queue size.
   */
  void SetMaxThreadSize(size_t size);

  /**
   * @brief Set the size of queue which task to submit in.
   * @param size queue size.
   */
  void SetTaskQueueSize(size_t size);

  /**
   * @brief Change none core thread keep alive time.
   * @param timeout
   */
  void SetKeepAlive(uint32_t timeout);

  /**
   * @brief Shutdown thread pool.
   * @param force force shutdown.
   * @return void
   */
  void Shutdown(bool force = false);

  /**
   * @brief Submit a task
   * @param fun function task to run
   * @param params function parameters.
   * @return task future
   */
  template <typename func, typename... ts>
  auto Submit(func &&fun, ts &&... params)
      -> std::future<typename std::result_of<func(ts...)>::type> {
    return Submit("", fun, params...);
  }

  /**
   * @brief Submit a task
   * @param name task name.
   * @param fun function task to run.
   * @param params function parameters.
   * @return task future
   */
  template <typename func, typename... ts>
  auto Submit(const std::string &name, func &&fun, ts &&... params)
      -> std::future<typename std::result_of<func(ts...)>::type> {
    auto execute =
        std::bind(std::forward<func>(fun), std::forward<ts>(params)...);
    using ReturnType = typename std::result_of<func(ts...)>::type;
    using PackagedTask = std::packaged_task<ReturnType()>;
    auto package_task = std::make_shared<PackagedTask>(std::move(execute));
    auto result = package_task->get_future();

    ThreadFunction task;
    task.func = [package_task]() { (*package_task)(); };
    if (name.length() > 0) {
      task.name = name;
    }

    if (SubmitTask(task) == false) {
      return std::future<typename std::result_of<func(ts...)>::type>();
    }

    return result;
  }

  /**
   * @brief Get running thread number.
   * @return thread number.
   */
  int GetThreadsNum();

  /**
   * @brief Get max thread number.
   * @return max thread number.
   */
  int GetMaxThreadsNum();

  /**
   * @brief Get waiting work.
   * @return waiting work number.
   */
  int GetWaitingWorkCount();

 private:
  friend class ThreadWorker;
  void ExitWorker(ThreadWorker *worker);

  void RunWorker(ThreadWorker *worker);

  bool Park(ThreadWorker *worker, ThreadFunction &task);

  void RmvWorker(ThreadWorker *worker);

  Status AddWorker(bool core_worker);

  void StopWokers();

  bool SubmitTask(ThreadFunction &task);

  std::shared_ptr<BlockingQueue<ThreadFunction>> work_queue_;
  bool quit_{false};
  std::list<std::shared_ptr<ThreadWorker>> workers_;
  int thread_size_{0};
  int max_thread_size_{1};
  int keep_alive_{60000};
  std::atomic<int> worker_num_{0};
  std::atomic<int> available_num_{0};
  std::mutex lock_;
  std::condition_variable exit_cond_;
  std::string name_;
};
}  // namespace modelbox

#endif  // MODELBOX_THREAD_POOL_H
