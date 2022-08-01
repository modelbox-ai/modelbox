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


#ifndef MODELBOX_REFCACHE_H_
#define MODELBOX_REFCACHE_H_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace modelbox {

template <typename T, typename KEY>
class RefInsertTransaction;

template <typename T, typename KEY>
class RefCache;

/**
 * @brief Reference cache data container
 */
template <typename T, typename KEY>
class RefContainer {
 public:
  RefContainer() = default;
  virtual ~RefContainer() = default;

  /// reference count
  int refcount_{0};

  /// data pointer
  std::shared_ptr<T> data_ = nullptr;

  /// transaction pointer
  RefInsertTransaction<T, KEY> *trans_ = nullptr;
};

/**
 * @brief Reference cache insert transaction
 */
template <typename T, typename KEY = const std::string>
class RefInsertTransaction {
 public:
  /**
   * @brief Constructor of RefInsertTransaction
   * @param ref_cache Pointer to RefCache
   * @param container data container
   * @param data data to cache
   * @param key cache key
   */
  RefInsertTransaction(RefCache<T, KEY> *ref_cache,
                       std::shared_ptr<RefContainer<T, KEY>> container,
                       std::shared_ptr<T> data, const std::string &key) {
    ref_cache_ = ref_cache;
    ref_container_ = container;
    ref_data_ = data;
    key_ = key;
  };

  virtual ~RefInsertTransaction() {
    if (ref_container_) {
      if (ref_container_->trans_ != nullptr) {
        ref_container_->trans_ = nullptr;
        ref_cache_->NotifyAll();
      }

      if (ref_container_->data_ == nullptr) {
        ref_cache_->Release(key_);
      }
    }

    ref_data_ = nullptr;
  };

  /**
   * @brief End transaction for inserting data.
   * @param data insert.
   * @return return new reference data.
   */
  std::shared_ptr<T> UpdateData(std::shared_ptr<T> data) {
    ref_container_->data_ = data;
    ref_container_->trans_ = nullptr;
    auto ret = ref_cache_->Get(key_, false);
    ref_cache_->NotifyAll();

    return ret;
  }

  /**
   * @brief Get cache data.
   * @return return reference data.
   */
  std::shared_ptr<T> GetData() { return ref_data_; }

 private:
  friend class RefCache<T, KEY>;
  RefCache<T, KEY> *ref_cache_;
  std::shared_ptr<T> ref_data_;
  std::shared_ptr<RefContainer<T, KEY>> ref_container_;
  std::string key_;
};

/**
 * @brief Reference cache, support transactions to insert data.
 */
template <typename T, typename KEY = const std::string>
class RefCache {
 public:
  RefCache() = default;
  virtual ~RefCache() = default;

  /**
   * @brief Get reference data from key, when inserting, may blocking.
   * @param key reference key.
   * @return return data
   */
  std::shared_ptr<RefInsertTransaction<T, KEY>> InsertAndGet(KEY &key) {
    std::unique_lock<std::mutex> lock(lock_);
    auto itr = ref_data_set_.find(key);
    if (itr == ref_data_set_.end()) {
      return InsertLocked(&lock, key);
    }

    auto data = GetLocked(&lock, key, true);
    if (data == nullptr) {
      return nullptr;
    }

    auto reftrans = std::make_shared<RefInsertTransaction<T, KEY>>(
        this, nullptr, data, key);
    return reftrans;
  }

  /**
   * @brief Get reference data from key, when inserting, may blocking.
   * @param key reference key.
   * @param is_block should block when inserting data.
   * @return return data
   */
  std::shared_ptr<T> Get(KEY &key, bool is_block = false) {
    std::unique_lock<std::mutex> lock(lock_);
    return GetLocked(&lock, key, is_block);
  }

  /**
   * @brief Insert reference data.
   * @param key reference key.
   * @return return referenced transaction
   */
  std::shared_ptr<RefInsertTransaction<T, KEY>> Insert(KEY &key) {
    std::unique_lock<std::mutex> lock(lock_);
    return InsertLocked(&lock, key);
  }

  /**
   * @brief Update reference data.
   * @param key reference key.
   * @param data data.
   * @return return referenced datal
   */
  std::shared_ptr<T> Update(KEY &key, std::shared_ptr<T> data) {
    auto ref = std::make_shared<RefContainer<T, KEY>>();
    ref->data_ = data;
    ref->refcount_ = 0;
    std::unique_lock<std::mutex> lock(lock_);
    ref_data_set_[key] = ref;
    return GetLocked(&lock, key, false);
  }

 protected:
  friend class RefInsertTransaction<T, KEY>;

  /**
   * @brief insert data with lock hold
   * @param lock cache lock.
   * @param key reference key.
   * @return return transaction
   */
  std::shared_ptr<RefInsertTransaction<T, KEY>> InsertLocked(
      std::unique_lock<std::mutex> *lock, KEY &key) {
    auto itr = ref_data_set_.find(key);
    if (itr != ref_data_set_.end()) {
      return nullptr;
    }

    auto container = std::make_shared<RefContainer<T, KEY>>();
    auto reftrans = std::make_shared<RefInsertTransaction<T, KEY>>(
        this, container, nullptr, key);

    ref_data_set_[key] = container;
    container->trans_ = reftrans.get();
    container->data_ = nullptr;
    container->refcount_ = 0;

    return reftrans;
  }

  /**
   * @brief get data by with lock hold
   * @param lock cache lock.
   * @param key reference key.
   * @param is_block whether block when key is in transaction
   * @return return key value
   */
  std::shared_ptr<T> GetLocked(std::unique_lock<std::mutex> *lock, KEY &key,
                               bool is_block = false) {
    bool is_success = false;
    std::shared_ptr<bool> guard(&is_success, [&](bool *result) {
      if (is_success == false) {
        ReleaseLocked(key);
      }
    });

    auto itr = ref_data_set_.find(key);
    if (itr == ref_data_set_.end()) {
      return nullptr;
    }

    auto ref = itr->second;
    ref->refcount_ += 1;

    if (ref->data_ == nullptr) {
      if (is_block == false) {
        return nullptr;
      }

      if (ref->trans_ == nullptr) {
        return nullptr;
      }

      cond_.wait(*lock, [&]() {
        return ref->data_ != nullptr || ref->trans_ == nullptr;
      });

      itr = ref_data_set_.find(key);
      if (itr == ref_data_set_.end()) {
        return nullptr;
      }

      ref = itr->second;
      if (ref->data_ == nullptr) {
        return nullptr;
      }
    }

    is_success = true;
    std::shared_ptr<T> ret(ref->data_.get(),
                           [this, key](void *ptr) { Release(key); });
    return ret;
  }

  /**
   * @brief Update key data
   * @param key reference key.
   * @param data data.
   */
  void UpdateData(KEY &key, std::shared_ptr<T> data) {
    std::unique_lock<std::mutex> lock(lock_);
    auto itr = ref_data_set_.find(key);
    if (itr == ref_data_set_.end()) {
      return;
    }
    auto ref = itr->second;
    ref->data_ = data;
  }

  /**
   * @brief Wake up all blocking call
   */
  void NotifyAll() { cond_.notify_all(); }

  /**
   * @brief Release key with lock hold
   * @param key reference key
   */
  void ReleaseLocked(KEY &key) {
    auto itr = ref_data_set_.find(key);
    if (itr == ref_data_set_.end()) {
      return;
    }

    auto ref = itr->second;
    if (--ref->refcount_ > 0) {
      return;
    }

    ref_data_set_.erase(key);
  }

  /**
   * @brief Release key
   * @param key reference key
   */
  void Release(KEY &key) {
    std::unique_lock<std::mutex> lock(lock_);
    ReleaseLocked(key);
  }

 private:
  std::map<KEY, std::shared_ptr<RefContainer<T, KEY>>> ref_data_set_;
  std::mutex lock_;
  std::condition_variable cond_;
};

/**
 * @brief Default reference cache data for any data
 */
class RefCacheData : public RefCache<void> {
 public:
  RefCacheData();
  ~RefCacheData() override;
};

}  // namespace modelbox
#endif  // MODELBOX_REFCACHE_H_