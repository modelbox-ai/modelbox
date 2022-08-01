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

#ifndef MODELBOX_BLOCKINGQUEUE_H_
#define MODELBOX_BLOCKINGQUEUE_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace modelbox {

/**
 * @brief QueueType: FifoQueue, PriorityQueue, StablePriorityQueue.
 *   FifoQueue: fisrt in, first out.
 *   PriorityQueue: priority queue.
 *   StablePriorityQueue: stable ordered queue, when priority same, first in,
 *                        first out.
 */

/**
 * @brief StableElement: to keep elements in order which have same priority.
 */
template <typename T, typename Compare = std::less<T>>
struct StableElement {
  /**
   * @brief StableElement
   */
  using StableElementT = StableElement<T, Compare>;

  /** @brief construct order element */
  StableElement(T&& o, std::size_t c) : object_(std::move(o)), order_(c) {}

  /** @brief construct order element, from element and size*/
  StableElement(const T& o, std::size_t c) : object_(o), order_(c) {}

  /** @brief destructor order element */
  virtual ~StableElement() = default;

  /** @brief override of () */
  explicit operator T() { return object_; }

  /** @brief override of < */
  bool operator<(const StableElementT& rhs) const {
    /* if element priority is same, compare with order */
    if (comp_(object_, rhs.object_) == false &&
        comp_(rhs.object_, object_) == false) {
      return order_ >= rhs.order_;
    }

    /* compare two objects */
    return comp_(object_, rhs.object_);
  }

  /// @brief object
  T object_;
  /// @brief order inserted
  std::size_t order_;
  /// @brief compare function
  Compare comp_;
};

/** @brief Stable priority queue */
template <typename T, typename Compare = std::less<T>>
class StablePriorityQueue
    : public std::priority_queue<StableElement<T, Compare>,
                                 std::vector<StableElement<T, Compare>>,
                                 std::less<StableElement<T, Compare>>> {
  using stableT = StableElement<T, Compare>;
  using std::priority_queue<
      stableT, std::vector<StableElement<T, Compare>>,
      std::less<StableElement<T, Compare>>>::priority_queue;

 public:
  /// @brief constructor of priority queue.
  StablePriorityQueue() = default;
  /// @brief destructor of priority queue.
  virtual ~StablePriorityQueue() = default;
  /// @brief front of the queue
  const T& front() { return this->c.front().object_; }
  /// @brief top of the queue
  const T& top() { return this->c.front().object_; }
  /// @brief push value into queue
  void push(const T& value) {
    /* push back and increase order */
    this->c.push_back(stableT(value, counter_++));
    std::push_heap(this->c.begin(), this->c.end(), this->comp);
  }

  /// @brief push value into queue
  void push(T&& value) {
    this->c.push_back(stableT(std::move(value), counter_++));
    std::push_heap(this->c.begin(), this->c.end(), this->comp);
  }

  /// @brief emplace value into queue
  template <class... Args>
  void emplace(Args&&... args) {
    /* emplace element */
    this->c.emplace_back(T(std::forward<Args>(args)...), counter_++);
    std::push_heap(this->c.begin(), this->c.end(), this->comp);
  }

  /// @brief pop from queue
  void pop() {
    /* pop element */
    std::pop_heap(this->c.begin(), this->c.end(), this->comp);
    this->c.pop_back();
    /* if queue is empty, reset counter, this will avoid counter overflow */
    if (this->empty()) {
      counter_ = 0;
    }
  }

 protected:
  /**
   * @brief counter, used as order.
   * counter will never overflow, as counter will reset when queue is empty.
   */
  std::size_t counter_ = 0;
};

/** @brief Std priority_queue wrap */
template <typename T, typename Sequence = std::vector<T>,
          typename Compare = std::less<typename Sequence::value_type>>
class PriorityQueue : public std::priority_queue<T, Sequence, Compare> {
 public:
  /**
   * @brief front of the queue
   */
  T front() const { return this->c.front(); }
};

/** @brief Std queue wrap */
template <typename T>
class FifoQueue : public std::queue<T> {};

/**
 * @brief Blocking queue, Blocking caller, when queue is empty, or full.
 */
template <typename T, typename Queue = FifoQueue<T>,
          typename Sequence = std::vector<T>>
class BlockingQueue {
 public:
  /**
   * @brief A blocking queue.
   * @param capacity capacity, default is SIZE_MAX
   */
  explicit BlockingQueue(size_t capacity = SIZE_MAX) : capacity_(capacity) {
    if (capacity <= 0) {
      capacity_ = SIZE_MAX;
    }
  }

  virtual ~BlockingQueue() { Close(); }

  /**
   * @brief Return element size
   * @return element size
   */
  size_t Size() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Set queue capacity
   * @param capacity queue capacity, langer than 0.
   */
  void SetCapacity(size_t capacity) {
    if (capacity <= 0) {
      return;
    }

    capacity_ = capacity;
  }

  /**
   * @brief Get queue capacity
   */
  size_t GetCapacity() const { return capacity_; }

  /**
   * @brief Get remain capacity
   */
  size_t RemainCapacity() {
    size_t queue_size = Size();
    if (queue_size > capacity_) {
      return 0;
    }

    return capacity_ - queue_size;
  }

  /**
   * @brief Clear queue
   */
  void Clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    Queue empty;
    std::swap(queue_, empty);
  }

  /**
   * @brief Close queue
   */
  void Close() {
    std::unique_lock<std::mutex> lock(mutex_);
    Queue empty;
    std::swap(queue_, empty);
    shutdown_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  /**
   * @brief Is queue full
   * @return true of false
   */
  bool Full() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size() >= capacity_;
  }

  /**
   * @brief Is queue empty
   * @return true of false
   */
  bool Empty() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /**
   * @brief Wake up waiters
   */
  void Wakeup() {
    std::unique_lock<std::mutex> lock(mutex_);
    need_wakeup_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  /**
   * @brief Push item into queue
   * @param elem item reference
   * @param timeout
   *   timeout > 0 if queue is full, blocking for timeout(ms), and return false.
   *   timeout = 0 if queue is full, blocking until queue is not full.
   *   timeout < 0 if queue is full, return immediately.
   * @return true or false
   */
  bool Push(const T& elem, int timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (PushQueue(lock, elem, timeout) == false) {
      return false;
    }

    not_empty_.notify_one();

    return true;
  }

  /**
   * @brief Push a sequence link vector into queue
   * @param elems sequence reference, return number of pushed elements.
   * @param timeout
   *   timeout > 0 if queue is full, blocking for timeout(ms), and return pushed
   * number.
   *   timeout = 0 if queue is full, blocking until queue is not full.
   *   timeout < 0 if queue is full, return immediately.
   * @return: number of pushed elems.
   */
  size_t Push(Sequence* elems, int timeout = 0) {
    size_t num = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    num = PushQueue(lock, elems, timeout);

    if (num <= 0) {
      return num;
    }

    not_empty_.notify_all();

    return num;
  }

  /**
   * @brief Push a sequence link vector into queue at once
   * @param elems sequence reference, return number of elems.
   * @param timeout
   *   timeout > 0 if queue is full, blocking for timeout(ms), and return false.
   *   timeout = 0 if queue is full, blocking until queue is not full.
   *   timeout < 0 if queue is full, return immediately.
   * @return: number of elems.
   */
  size_t PushBatch(Sequence* elems, int timeout = 0) {
    size_t ret = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    ret = PushQueue(lock, elems, timeout, elems->size());
    not_empty_.notify_all();
    return ret;
  }

  /**
   * @brief Force push a sequence link vector into queue at once, ignore
   * capacity
   * @param elems sequence reference, return number of elems.
   * @param wait_when_full wait when queue is full.
   * @param timeout
   *   timeout > 0 if queue is full, blocking for timeout(ms), and return false.
   *   timeout = 0 if queue is full, blocking until queue is not full.
   *   timeout < 0 if queue is full, return immediately.
   * @return: number of elems.
   */
  size_t PushBatchForce(Sequence* elems, bool wait_when_full = false,
                        int timeout = 0) {
    size_t ret = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    ret = PushQueueForce(lock, elems, wait_when_full, timeout);
    not_empty_.notify_all();
    return ret;
  }

  /**
   * @brief Push item into queue, blocking if queue is full.
   * @param elem item reference.
   * @return true or false
   */
  virtual bool Push(const T& elem) { return Push(elem, 0); }

  /**
   * @brief Force push item into queue.
   * @param elem item reference.
   * @return true or false
   */
  virtual bool PushForce(const T& elem) {
    bool ret = false;
    std::unique_lock<std::mutex> lock(mutex_);
    ret = PushQueueForce(elem);
    not_empty_.notify_one();

    return ret;
  }

  /**
   * @brief Get an item from queue, blocking if queue is empty.
   * @param elem item data.
   * @return true or false
   */
  virtual bool Pop(T* elem) { return Pop(elem, 0); }

  /**
   * @brief Get an item from queue, or return false if queue is empty, never
   * blocking.
   * @param elem item data.
   * @return true or false
   */
  bool Poll(T* elem) { return Pop(elem, -1); }

  /**
   * @brief Get many items from queue, or return false if queue is empty, never
   * blocking.
   * @param elems item data in vector.
   * @return number element returned.
   */
  size_t Poll(Sequence* elems) { return Pop(elems, -1); }

  /**
   * @brief Get an item from queue, if queue is empty, blocking for timeout(ms)
   * and return false
   * @param elem item
   * @param timeout
   *   timeout > 0 if queue is empty, blocking for timeout(ms) and return false.
   *   timeout = 0 if queue is empty, blocking until queue is not empty.
   *   timeout < 0 if queue is empty, return immediately.
   * @return is pop success
   */
  bool Pop(T* elem, int timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (PopQueue(lock, elem, timeout) == false) {
      return false;
    }
    /* wakeup waiter */
    not_full_.notify_one();

    return true;
  }

  /**
   * @brief Get an sequence from queue, if queue is empty, blocking for
   * timeout(ms) and return number of poped elems.
   * @param elems item
   * @param timeout
   *   timeout > 0 if queue is empty, blocking for timeout(ms) and return false.
   *   timeout = 0 if queue is empty, blocking until queue is not empty.
   *   timeout < 0 if queue is empty, return immediately. return: number of
   * poped elems.
   * @param maxsize max pop items number.
   * @return number of poped elemets.
   */
  virtual size_t Pop(Sequence* elems, int timeout = 0, size_t maxsize = 0) {
    size_t num = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    num = PopQueue(lock, elems, timeout, maxsize);
    if (num < 0) {
      return num;
    }

    /* wakeup waiter */
    not_full_.notify_all();

    return num;
  }

  /**
   * @brief Pop a sequence of elems from queue at once
   * @param elems sequence reference, return number of elems.
   * @param timeout
   *   timeout > 0 if queue is empty, blocking for timeout(ms), and return
   * false. timeout = 0 if queue is empty, blocking until queue is empty.
   *   timeout < 0 if queue is empty, return immediately.
   * @param max_elems max elements number returned.
   * @return: return number of elems.
   */
  size_t PopBatch(Sequence* elems, int timeout = 0, uint32_t max_elems = -1) {
    size_t num = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    num = PopQueueBatch(lock, elems, timeout, max_elems);

    /* wakeup waiter */
    not_full_.notify_all();

    return num;
  }

  /**
   * @brief Get item and not remove from queue, return false if queue is empty
   * @param elem element to save.
   * @return is get front success or not.
   */
  bool Front(T* elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (WaitQueue(lock, -1) == false) {
      return false;
    }

    *elem = queue_.front();
    return true;
  }

  /**
   * @brief Shutdown queue, push will wakeup and return false
   */
  void Shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }

  /**
   * @brief Queue is shutdown or not
   * @return is queue shutdown
   */
  bool IsShutdown() { return shutdown_; }

 protected:
  /**
   * @brief Wait queue.
   * @param cond condition
   * @param lock queue lock
   * @param timeout wait timeout
   * @param wait_cond wait function
   * @return wait success
   */
  bool Wait(std::condition_variable& cond, std::unique_lock<std::mutex>& lock,
            int timeout, const std::function<bool()> &wait_cond) {
    bool ret = false;
    auto cond_func = [&]() { return need_wakeup_ || wait_cond(); };

    waiter_number_++;
    if (timeout > 0) {
      /* if timeout is set, wait for timeout and return false */
      ret = cond.wait_for(lock, std::chrono::milliseconds(timeout), cond_func);
      if (ret == false) {
        errno = ETIMEDOUT;
      }
    } else if (timeout == 0) {
      /* if wait forever, do wait */
      cond.wait(lock, cond_func);
      ret = true;
    } else {
      /* do not wait */
      ret = cond_func();
    }

    waiter_number_--;
    /* if wake by Wakeup */
    if (need_wakeup_) {
      /* if all waiters have been woken up */
      if (waiter_number_ == 0) {
        need_wakeup_ = false;
      }
      errno = EINTR;
      ret = false;
    }

    /* wakeup */
    return ret;
  }

  /**
   * @brief Pop queue elements with batch
   * @param lock queue lock
   * @param elems elements list
   * @param timeout wait timeout
   * @param max_elems max pop elements number, -1 means all
   * @return poped elements number
   */
  size_t PopQueueBatch(std::unique_lock<std::mutex>& lock, Sequence* elems,
                       int timeout = 0, uint32_t max_elems = -1) {
    size_t num = 0;

    /* Loop and get same element */
    while (queue_.size() > 0 && num < max_elems) {
      auto time_wait = -1;
      if (num == 0) {
        time_wait = timeout;
      }

      if (WaitQueue(lock, time_wait) == false) {
        return num;
      }

      /* get and remove element */
      elems->emplace_back(std::move(queue_.front()));
      queue_.pop();
      num++;
    }

    return num;
  }

  /**
   * @brief Push one element into queue
   * @param lock queue lock
   * @param elem element
   * @param timeout wait timeout
   * @return push result
   */
  bool PushQueue(std::unique_lock<std::mutex>& lock, const T& elem,
                 int timeout) {
    /* shutdown or queue has enough space */
    auto wait_check = [&]() {
      return shutdown_ == true || capacity_ > queue_.size();
    };

    /* if queue already shutdown, return false */
    if (shutdown_) {
      errno = ESHUTDOWN;
      return false;
    }

    /* check if queue has space, or wait. */
    if (Wait(not_full_, lock, timeout, wait_check) == false) {
      return false;
    }

    /* if wakeup after shutdown, just return false */
    if (shutdown_) {
      errno = ESHUTDOWN;
      return false;
    }

    queue_.emplace(elem);

    return true;
  }

  /**
   * @brief Push element list into queue
   * @param lock queue lock
   * @param elems elements list
   * @param timeout wait timeout
   * @param expect_space push when free space greater than expect_space
   * @return pushed element number
   */
  size_t PushQueue(std::unique_lock<std::mutex>& lock, Sequence* elems,
                   int timeout = 0, size_t expect_space = 1) {
    size_t push_num = 0;

    /* shutdown or queue has enough space */
    auto wait_check = [&]() {
      return shutdown_ == true || queue_.size() == 0 ||
             capacity_ >= expect_space + queue_.size();
    };

    /* check if queue has space, or wait. */
    if (Wait(not_full_, lock, timeout, wait_check) == false) {
      return false;
    }

    for (auto it = elems->begin(); it != elems->end(); it++) {
      if (PushQueue(lock, *it, timeout) == false) {
        break;
      }

      push_num++;

      /* when get first element, stop waiting */
      if (timeout >= 0) {
        timeout = -1;
      }
    }

    /* remove elems from origin sequence */
    elems->erase(elems->begin(), elems->begin() + push_num);

    return push_num;
  }

  /**
   * @brief force Push element list into queue
   * @param lock queue lock
   * @param elems elements list
   * @param wait_when_full wait whether queue is full
   * @param timeout wait timeout
   * @return pushed element number
   */
  size_t PushQueueForce(std::unique_lock<std::mutex>& lock, Sequence* elems,
                        bool wait_when_full = false, int timeout = 0) {
    size_t push_num = 0;

    /* shutdown or queue is not full*/
    auto wait_check = [&]() {
      if (wait_when_full == true) {
        if (capacity_ >= 1 + queue_.size()) {
          return true;
        }
      }

      return wait_when_full == false || shutdown_ == true || queue_.size() == 0;
    };

    /* check if queue has space, or wait. */
    if (Wait(not_full_, lock, timeout, wait_check) == false) {
      return false;
    }

    for (auto it = elems->begin(); it != elems->end(); it++) {
      queue_.emplace(*it);
      push_num++;
    }

    /* remove elems from origin sequence */
    elems->erase(elems->begin(), elems->begin() + push_num);

    return push_num;
  }

  /**
   * @brief Force push item into queue.
   * @param elem item reference.
   * @return true or false
   */
  bool PushQueueForce(const T& elem) {
    /* if queue already shutdown, return false */
    if (shutdown_) {
      errno = ESHUTDOWN;
      return false;
    }

    queue_.emplace(elem);

    return true;
  }

  /**
   * @brief Get an element from queue
   * @param lock queue lock
   * @param elem element poped
   * @param timeout wait for timeout
   * @return pop result
   */
  bool PopQueue(std::unique_lock<std::mutex>& lock, T* elem, int timeout) {
    /* if queue is empty, try wait. */
    if (WaitQueue(lock, timeout) == false) {
      return false;
    }

    /* Get and remove element */
    *elem = std::move(queue_.front());
    queue_.pop();

    return true;
  }

  /**
   * @brief Get an sequence from queue, if queue is empty, blocking for
   * timeout(ms) and return number of poped elems.
   * @param lock queue lock
   * @param elems item
   * @param timeout wait for timeout
   * @param maxsize max pop items number.
   * @return number of poped elemets.
   */
  size_t PopQueue(std::unique_lock<std::mutex>& lock, Sequence* elems,
                  int timeout = 0, size_t maxsize = 0) {
    size_t num = 0;

    while (true) {
      if (WaitQueue(lock, timeout) == false) {
        return num;
      }

      /* remove element */
      elems->emplace_back(std::move(queue_.front()));
      queue_.pop();
      num++;
      if (maxsize > 0 && num >= maxsize) {
        return num;
      }

      /* when get first element, disable wait */
      if (timeout >= 0) {
        timeout = -1;
      }
    }

    return num;
  }

  /**
   * @brief wait for queue ready
   * @param lock queue lock
   * @param timeout wait for timeout
   * @return whether wait success.
   */
  bool WaitQueue(std::unique_lock<std::mutex>& lock, int timeout) {
    /* wait check has data */
    auto wait_check = [&]() { return shutdown_ == true || queue_.size() > 0; };

    /* return false when shutdown */
    if (shutdown_ == true && queue_.size() == 0) {
      errno = ESHUTDOWN;
      return false;
    }

    /* check if queue has data, or wait. */
    if (Wait(not_empty_, lock, timeout, wait_check) == false) {
      return false;
    }

    /* shutdown may be called, return false */
    if (queue_.size() == 0) {
      errno = ESHUTDOWN;
      return false;
    }

    return true;
  }

  /**
   * @brief Queue lock
   */
  std::mutex mutex_;

  /**
   * @brief Queue
   */
  Queue queue_;

  /**
   * @brief Wakeup when queue is full
   */
  std::condition_variable not_full_;

  /**
   * @brief Wakeup when queue is empty
   */
  std::condition_variable not_empty_;

 private:
  size_t capacity_ = 0;
  bool need_wakeup_ = false;
  int waiter_number_ = 0;
  bool shutdown_ = false;
};

/**
 * @brief Priority Blocking queue, blocking queue with stable priority.
 * Support geting all elements with same priority in one batch.
 */
template <typename T, typename Compare = std::less<T>,
          typename Sequence = std::vector<T>>
class PriorityBlockingQueue
    : public BlockingQueue<T, StablePriorityQueue<T, Compare>, Sequence> {
  using _BlockingQueue =
      BlockingQueue<T, StablePriorityQueue<T, Compare>, Sequence>;

 public:
  /**
   * @brief A priority blocking queue.
   * @param capacity queue capacity.
   */
  explicit PriorityBlockingQueue(size_t capacity = SIZE_MAX)
      : _BlockingQueue(capacity) {}

  ~PriorityBlockingQueue() override = default;

  /**
   * @brief Pop a sequence of elems which equal each other from queue at once
   * @param elems sequence reference, return number of elems.
   * @param timeout
   *   timeout > 0 if queue is full, blocking for timeout(ms), and return false.
   *   timeout = 0 if queue is full, blocking until queue is not full.
   *   timeout < 0 if queue is full, return immediately.
   * @param max_elems max elements number returned.
   * @return: return number of elems.
   */
  size_t PopBatch(Sequence* elems, int timeout = 0, uint32_t max_elems = -1) {
    size_t num = 0;
    {
      std::unique_lock<std::mutex> lock(_BlockingQueue::mutex_);

      num = PopQueueBatch(lock, elems, timeout, max_elems);
      if (num < 0) {
        return num;
      }
    }

    /* wakeup waiter */
    _BlockingQueue::not_full_.notify_all();

    return num;
  }

 private:
  size_t PopQueueBatch(std::unique_lock<std::mutex>& lock, Sequence* elems,
                       int timeout = 0, uint32_t max_elems = -1) {
    size_t num = 0;

    /* wait for elems first */
    if (num >= max_elems || _BlockingQueue::WaitQueue(lock, timeout) == false) {
      return num;
    }

    /* Get first element */
    auto first = _BlockingQueue::queue_.front();
    auto second = first;
    num++;
    elems->emplace_back(std::move(_BlockingQueue::queue_.front()));
    _BlockingQueue::queue_.pop();

    /* Loop and get same element */
    while (_BlockingQueue::queue_.size() > 0 && num < max_elems) {
      if (_BlockingQueue::WaitQueue(lock, -1) == false) {
        return num;
      }

      second = _BlockingQueue::queue_.front();

      /* try to get all same priority elements
       * break when priority is different
       */
      if (comp_(first, second) == true || comp_(second, first) == true) {
        break;
      }

      /* get and remove element */
      elems->emplace_back(std::move(_BlockingQueue::queue_.front()));
      _BlockingQueue::queue_.pop();
      num++;
    }

    return num;
  }

  Compare comp_;
};

}  // namespace modelbox
#endif
