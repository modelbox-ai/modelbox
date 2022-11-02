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

#ifndef MODELBOX_STATISTICS_H_
#define MODELBOX_STATISTICS_H_

#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "modelbox/base/any.h"
#include "modelbox/base/status.h"
#include "modelbox/base/thread_pool.h"
#include "modelbox/base/timer.h"

namespace modelbox {

constexpr const char* STATISTICS_ITEM_FLOW = "flow";

class StatisticsValue {
 public:
  StatisticsValue(std::shared_ptr<Any> val);

  virtual ~StatisticsValue();

  const std::type_info& GetType();

  template <typename T>
  bool IsSameTypeTo(T& val) {
    return typeid(T) == val_->type();
  }

  bool IsType(const std::type_info& type);

  template <typename T>
  bool GetValue(T& val) {
    if (!IsSameTypeTo(val)) {
      return false;
    }

    val = any_cast<T>(*val_);
    return true;
  }

  bool IsInt32();

  bool GetInt32(int32_t& val);

  bool IsUint32();

  bool GetUint32(uint32_t& val);

  bool IsInt64();

  bool GetInt64(int64_t& val);

  bool IsUint64();

  bool GetUint64(uint64_t& val);

  bool IsFloat();

  bool GetFloat(float& val);

  bool IsDouble();

  bool GetDouble(double& val);

  bool IsBool();

  bool GetBool(bool& val);

  bool IsString();

  bool GetString(std::string& val);

  std::string ToString();

 private:
  template <typename T>
  std::string ToString();

  std::shared_ptr<Any> val_;
};

template <typename T>
std::string StatisticsValue::ToString() {
  T val;
  if (GetValue(val) == false) {
    return "";
  }
  std::stringstream ss;
  ss << val;
  return ss.str();
}

enum class StatisticsNotifyType : uint32_t {
  CREATE = 1,
  DELETE = 2,
  CHANGE = 4,
  TIMER = 8
};

class StatisticsNotifyMsg {
 public:
  StatisticsNotifyMsg(std::string path, std::shared_ptr<StatisticsValue> value,
                      StatisticsNotifyType type);

  virtual ~StatisticsNotifyMsg();
  std::string path_;
  std::shared_ptr<StatisticsValue> value_;
  StatisticsNotifyType type_;
};

using StatisticsNotifyFunc =
    std::function<void(const std::shared_ptr<const StatisticsNotifyMsg>& msg)>;

class StatisticsItem;
class StatisticsNotifyConsumers;

constexpr const size_t minimum_notify_time = 10 * 1000;  // 10s

class StatisticsNotifyCfg {
  friend class StatisticsItem;
  friend class StatisticsNotifyConsumers;

 public:
  StatisticsNotifyCfg(std::string path_pattern, StatisticsNotifyFunc func,
                      const StatisticsNotifyType& type);

  StatisticsNotifyCfg(std::string path_pattern, StatisticsNotifyFunc func,
                      std::set<StatisticsNotifyType> types = {});

  StatisticsNotifyCfg(const StatisticsNotifyCfg& other);

  bool operator==(const StatisticsNotifyCfg& other);

  virtual ~StatisticsNotifyCfg();

  /**
   * @brief Set timer notify
   * @param delay Delay to run first, in second, >= 10s
   * @param interval Notify interval, in second, >= 10s
   */
  void SetNotifyTimer(size_t delay, size_t interval = 0);

 private:
  std::string GetRootPath() const;

  std::string GetSubPath() const;  // path without root

  void BindTimerTask(const std::shared_ptr<TimerTask>& timer_task);

  void RemoveTimerTask();

  std::string path_pattern_;
  StatisticsNotifyFunc func_;
  std::set<StatisticsNotifyType> type_set_;
  size_t delay_{minimum_notify_time};
  size_t interval_{minimum_notify_time};
  std::shared_ptr<TimerTask> timer_task_;
  uintptr_t id_;
};

class StatisticsNotifyTypeHash {
 public:
  size_t operator()(const StatisticsNotifyType& type) const {
    return (size_t)type;
  }
};

class StatisticsNotifyConsumers {
 public:
  StatisticsNotifyConsumers();

  virtual ~StatisticsNotifyConsumers();

  Status AddConsumer(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  Status DelConsumer(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  std::list<std::shared_ptr<StatisticsNotifyCfg>> GetConsumers(
      const StatisticsNotifyType& type);

  void Clear();

 private:
  std::unordered_map<StatisticsNotifyType, std::shared_ptr<std::mutex>,
                     StatisticsNotifyTypeHash>
      cfg_map_lock_;
  std::unordered_map<StatisticsNotifyType,
                     std::list<std::shared_ptr<StatisticsNotifyCfg>>,
                     StatisticsNotifyTypeHash>
      cfg_map_;
};

using StatisticsForEachFunc =
    std::function<Status(const std::shared_ptr<StatisticsItem>& item,
                         const std::string relative_path)>;

/**
 * @brief A statistics tree
 * only leaf node has value
 */
class StatisticsItem : public std::enable_shared_from_this<StatisticsItem> {
 public:
  /**
   * @brief Construct a root node for statistics
   * sub node can only create by parent
   */
  StatisticsItem();

  virtual ~StatisticsItem();

  /**
   * @brief Get parent path
   * @return Parent path
   */
  inline std::string GetParentPath() { return parent_path_; }

  /**
   * @brief Get name
   * @return Name
   */
  inline std::string GetName() { return name_; }

  /**
   * @brief Get full path
   * @return parent_path.name
   */
  inline std::string GetPath() { return path_; }

  /**
   * @brief Check item is leaf or not
   * @return check result
   */
  inline bool IsLeaf() { return is_leaf_; }

  /**
   * @brief Set value of this item
   * @return Result of set
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, bool>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, uint64_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, double>::value ||
          std::is_same<T, std::string>::value>::type>
  Status SetValue(const T& value);

  /**
   * @brief new_value = old_value + value
   * @param value Value to add
   * @return Result of Add
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value ||
          std::is_same<T, float>::value ||
          std::is_same<T, double>::value>::type>
  Status IncreaseValue(const T& value);

  /**
   * @brief new_value = old_value + value, will create new item if not exist
   * @param sub_item_name sub item name
   * @param value Value to add
   * @return Result of Add
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value ||
          std::is_same<T, float>::value ||
          std::is_same<T, double>::value>::type>
  Status IncreaseValue(const std::string& sub_item_name, const T& value);

  /**
   * @brief Get value
   * @param value Return value
   * @return Result of get
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, bool>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, uint64_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, double>::value ||
          std::is_same<T, std::string>::value>::type>
  Status GetValue(T& value);

  /**
   * @brief Get value
   * @return Wrap value
   */
  inline std::shared_ptr<StatisticsValue> GetValue() {
    if (!IsLeaf()) {
      StatusError = {STATUS_NOTSUPPORT,
                     "This is not a leaf node, get value failed."};
      return nullptr;
    }

    StatusError = STATUS_OK;
    return std::make_shared<StatisticsValue>(value_);
  }

  /**
   * @brief Get value
   * @return Result & value
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, bool>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, uint64_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, double>::value ||
          std::is_same<T, std::string>::value>::type>
  std::tuple<Status, T> GetValue();

  /**
   * @brief Add new item as child, it is not a leaf item, can not set value
   * @param name Name of new item
   * @return Status & new item
   */
  std::shared_ptr<StatisticsItem> AddItem(const std::string& name);

  /**
   * @brief Add new item as child with value, it is a leaf item, can not add
   * child
   * @param name Name of new item
   * @param value Value to set
   * @param override_val True: override value if item exist
   * @return Status & new item
   */
  template <
      typename T,
      typename = typename std::enable_if<
          std::is_same<T, bool>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, uint64_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, double>::value ||
          std::is_same<T, std::string>::value>::type>
  std::shared_ptr<StatisticsItem> AddItem(const std::string& name,
                                          const T& value,
                                          bool override_val = false);

  /**
   * @brief Get item with name
   * @param child_path Target item name
   * @return Target item, nullable
   */
  std::shared_ptr<StatisticsItem> GetItem(const std::string& child_path);

  /**
   * @brief Delete item with name
   * @param name Target item name
   */
  void DelItem(const std::string& name) noexcept;

  /**
   * @brief Clear all item
   */
  void ClearItem();

  /**
   * @brief Detach from parent and clear all item, this item should not use
   * again
   */
  void Dispose();

  /**
   * @brief Test has target item
   * @param name Target item name
   * @return Result of test
   */
  bool HasItem(const std::string& name);

  /**
   * @brief Get all sub item name
   * @return Name set
   */
  std::set<std::string> GetItemNames();

  /**
   * @brief Walk with dfs
   * @param func Func called on each item
   * @param recursive recursive for each
   * @return Status of func if not ok
   */
  Status ForEach(const StatisticsForEachFunc& func, bool recursive = false);

  /**
   * @brief Register notify for item, notify type {CREATE, DELETE, CHANGE,
   * TIMER}
   * @param cfg Config for notify
   * @return Result for register
   */
  Status RegisterNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  /**
   * @brief UnRegister notify with the cfg used in register
   * @param cfg Used in register
   */
  void UnRegisterNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  /**
   * @brief Notify the consumers of this item for specify type, async
   * @param type Notify type
   * @return Result for notify submit
   */
  Status Notify(const StatisticsNotifyType& type);

 private:
  StatisticsItem(std::string parent_path, std::string name,
                 std::weak_ptr<StatisticsItem> parent);

  Status AddNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  void DelNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  Status AddChildrenNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  void DelChildrenNotify(const std::shared_ptr<StatisticsNotifyCfg>& cfg);

  std::string GetRelativePath(const std::string& base_path);

  Status ForEachInner(const StatisticsForEachFunc& func, bool recursive,
                      const std::string& base_path);

  std::shared_ptr<StatisticsItem> AddItemInner(
      const std::string& name, const std::shared_ptr<Any>& value);

  std::string parent_path_;
  std::string name_;
  std::weak_ptr<StatisticsItem> parent_;
  std::string path_;  // full path : parent_path_ + "." + name_
  std::mutex value_lock_;
  std::shared_ptr<Any> value_;
  std::mutex children_lock_;
  std::map<std::string, std::shared_ptr<StatisticsItem>> children_;
  std::set<std::string> children_name_set_;

  std::mutex child_notify_cfg_lock_;
  std::map<std::string, std::list<std::shared_ptr<StatisticsNotifyCfg>>>
      children_notify_cfg_map_;  // For the child which has not been created.
                                 // <child_name, cfg_list>

  StatisticsNotifyConsumers consumers_;
  std::shared_ptr<ThreadPool> thread_pool_;
  std::shared_ptr<Timer> notify_timer_;
  std::chrono::steady_clock::time_point last_change_notify_time_;
  std::mutex last_change_notify_time_lock_;

  std::atomic_bool is_alive_{true};
  std::atomic_bool is_leaf_{false};
};

template <typename T, typename>
Status StatisticsItem::SetValue(const T& value) {
  if (!IsLeaf()) {
    return {STATUS_NOTSUPPORT, "This is not a leaf node, set value failed."};
  }

  std::lock_guard<std::mutex> lck(value_lock_);
  auto old_val = value_;
  value_ = std::make_shared<Any>(value);
  if (!(value_->type() == old_val->type() &&
        any_cast<T>(*value_) == any_cast<T>(*old_val))) {
    Notify(StatisticsNotifyType::CHANGE);
  }

  return STATUS_OK;
}

template <typename T, typename>
Status StatisticsItem::IncreaseValue(const T& value) {
  if (!IsLeaf()) {
    return {STATUS_NOTSUPPORT,
            "This is not a leaf node, increase value failed."};
  }

  std::lock_guard<std::mutex> lck(value_lock_);
  if (value_ == nullptr) {
    return STATUS_INVALID;
  }

  if (value_->type() != typeid(value)) {
    return STATUS_INVALID;
  }

  auto old_val = any_cast<T>(*value_);
  value_ = std::make_shared<Any>(old_val + value);
  Notify(StatisticsNotifyType::CHANGE);
  return STATUS_OK;
}

template <typename T, typename>
Status StatisticsItem::IncreaseValue(const std::string& sub_item_name,
                                     const T& value) {
  if (!is_alive_) {
    return {STATUS_FAULT, "This item is disposed"};
  }

  if (IsLeaf()) {
    return {STATUS_NOTSUPPORT, "This is a leaf node, has no child."};
  }

  std::lock_guard<std::mutex> lck(children_lock_);
  auto item = children_.find(sub_item_name);
  if (item != children_.end()) {
    return item->second->IncreaseValue(value);
  }

  auto value_ptr = std::make_shared<Any>(value);
  AddItemInner(sub_item_name, value_ptr);
  return StatusError;
}

template <typename T, typename>
Status StatisticsItem::GetValue(T& value) {
  if (!IsLeaf()) {
    return {STATUS_NOTSUPPORT, "This is not a leaf node, get value failed."};
  }

  std::lock_guard<std::mutex> lck(value_lock_);
  if (value_ == nullptr) {
    return STATUS_NODATA;
  }

  if (value_->type() != typeid(value)) {
    return STATUS_INVALID;
  }

  value = any_cast<T>(*value_);
  return STATUS_OK;
}

template <typename T, typename>
std::tuple<Status, T> StatisticsItem::GetValue() {
  T value;
  auto ret = GetValue(value);
  return std::make_tuple(ret, value);
}

template <typename T, typename>
std::shared_ptr<StatisticsItem> StatisticsItem::AddItem(const std::string& name,
                                                        const T& value,
                                                        bool override_val) {
  if (!is_alive_) {
    StatusError = {STATUS_FAULT, "This item is disposed"};
    return nullptr;
  }

  std::lock_guard<std::mutex> lck(children_lock_);
  auto item = children_.find(name);
  if (item != children_.end()) {
    StatusError = STATUS_EXIST;
    auto& target = item->second;
    if (override_val) {
      target->SetValue(value);
    }
    return target;
  }

  auto value_ptr = std::make_shared<Any>(value);
  return AddItemInner(name, value_ptr);
}

class Statistics {
 public:
  /**
   * @brief Get global statistics item
   */
  static std::shared_ptr<StatisticsItem> GetGlobalItem();

  static void ReleaseGlobalItem();

 private:
  static std::once_flag fix_item_init_flag_;
};

}  // namespace modelbox

#endif  // MODELBOX_STATISTICS_H_