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

#include "modelbox/statistics.h"

namespace modelbox {

/**
 * StatisticsValue
 */
StatisticsValue::StatisticsValue(const std::shared_ptr<Any>& val) : val_(val) {}

StatisticsValue::~StatisticsValue() {}

const std::type_info& StatisticsValue::GetType() { return val_->type(); }

bool StatisticsValue::IsType(const std::type_info& type) {
  return type == val_->type();
}

bool StatisticsValue::IsInt32() { return IsType(typeid(int32_t)); }

bool StatisticsValue::GetInt32(int32_t& val) { return GetValue(val); }

bool StatisticsValue::IsUint32() { return IsType(typeid(uint32_t)); }

bool StatisticsValue::GetUint32(uint32_t& val) { return GetValue(val); }

bool StatisticsValue::IsInt64() { return IsType(typeid(int64_t)); }

bool StatisticsValue::GetInt64(int64_t& val) { return GetValue(val); }

bool StatisticsValue::IsUint64() { return IsType(typeid(uint64_t)); }

bool StatisticsValue::GetUint64(uint64_t& val) { return GetValue(val); }

bool StatisticsValue::IsFloat() { return IsType(typeid(float)); }

bool StatisticsValue::GetFloat(float& val) { return GetValue(val); }

bool StatisticsValue::IsDouble() { return IsType(typeid(double)); }

bool StatisticsValue::GetDouble(double& val) { return GetValue(val); }

bool StatisticsValue::IsBool() { return IsType(typeid(bool)); }

bool StatisticsValue::GetBool(bool& val) { return GetValue(val); }

bool StatisticsValue::IsString() { return IsType(typeid(std::string)); }

bool StatisticsValue::GetString(std::string& val) { return GetValue(val); }

std::string StatisticsValue::ToString() {
  if (IsInt32()) {
    return ToString<int32_t>();
  } else if (IsUint32()) {
    return ToString<uint32_t>();
  } else if (IsInt64()) {
    return ToString<int64_t>();
  } else if (IsUint64()) {
    return ToString<uint64_t>();
  } else if (IsFloat()) {
    return ToString<float>();
  } else if (IsDouble()) {
    return ToString<double>();
  } else if (IsBool()) {
    return ToString<bool>();
  } else if (IsString()) {
    std::string val;
    GetValue(val);
    return val;
  }

  return "";
}

std::string StatisticsNotifyCfg::GetRootPath() const {
  auto pos = path_pattern_.find('.');
  if (pos == path_pattern_.npos) {
    return path_pattern_;
  }

  return path_pattern_.substr(0, pos);
}

std::string StatisticsNotifyCfg::GetSubPath() const {
  auto pos = path_pattern_.find('.');
  if (pos == path_pattern_.npos) {
    return "";
  }

  return path_pattern_.substr(pos + 1);
}

StatisticsNotifyConsumers::StatisticsNotifyConsumers() {
  std::vector<StatisticsNotifyType> all_types = {
      StatisticsNotifyType::CREATE, StatisticsNotifyType::DELETE,
      StatisticsNotifyType::CHANGE, StatisticsNotifyType::TIMER};
  for (auto& type : all_types) {
    cfg_map_lock_[type] = std::make_shared<std::mutex>();
  }
}

StatisticsNotifyConsumers::~StatisticsNotifyConsumers() { Clear(); }

modelbox::Status StatisticsNotifyConsumers::AddConsumer(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  for (auto& type : cfg->type_set_) {
    std::lock_guard<std::mutex> lck(*cfg_map_lock_[type]);
    cfg_map_[type].push_back(cfg);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status StatisticsNotifyConsumers::DelConsumer(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  for (auto& type : cfg->type_set_) {
    std::lock_guard<std::mutex> lck(*cfg_map_lock_[type]);
    auto& consumers_for_one_type = cfg_map_[type];
    consumers_for_one_type.remove_if(
        [cfg](const std::shared_ptr<StatisticsNotifyCfg>& val) {
          auto ret = (*cfg == *val);
          if (ret) {
            val->RemoveTimerTask();
          }

          return ret;
        });
  }

  return modelbox::STATUS_OK;
}

std::list<std::shared_ptr<StatisticsNotifyCfg>>
StatisticsNotifyConsumers::GetConsumers(const StatisticsNotifyType& type) {
  std::lock_guard<std::mutex> lck(*cfg_map_lock_[type]);
  return cfg_map_[type];
}

void StatisticsNotifyConsumers::Clear() {
  {
    auto type_timer = StatisticsNotifyType::TIMER;
    std::lock_guard<std::mutex> lck(*cfg_map_lock_[type_timer]);
    auto& consumers_for_timer = cfg_map_[type_timer];
    for (auto& consumer : consumers_for_timer) {
      consumer->RemoveTimerTask();
    }
  }

  for (auto& lock_item : cfg_map_lock_) {
    std::lock_guard<std::mutex> lck(*lock_item.second);
    cfg_map_[lock_item.first].clear();
  }
}

/**
 * StatisticsItem
 */
StatisticsItem::StatisticsItem() {
  thread_pool_ = std::make_shared<ThreadPool>(2, -1, 1000);
  thread_pool_->SetName("Stat-Notify");
  notify_timer_ = std::make_shared<Timer>();
  notify_timer_->SetName("Stat-Timer");
  notify_timer_->Start();
  last_change_notify_time_ = std::chrono::steady_clock::now();
}

StatisticsItem::StatisticsItem(const std::string& parent_path,
                               const std::string& name,
                               std::weak_ptr<StatisticsItem> parent)
    : parent_path_(parent_path), name_(name), parent_(parent) {
  if (!parent_path_.empty()) {
    path_ = parent_path_ + "." + name_;
  } else {
    path_ = name_;
  }

  last_change_notify_time_ = std::chrono::steady_clock::now();
}

StatisticsItem::~StatisticsItem() {
  consumers_.Clear();
  ClearItem();
}

std::shared_ptr<StatisticsItem> StatisticsItem::AddItem(
    const std::string& name) {
  if (!is_alive_) {
    StatusError = {STATUS_FAULT, "This item is disposed"};
    return nullptr;
  }

  std::lock_guard<std::mutex> lck(children_lock_);
  return AddItemInner(name, nullptr);
}

std::shared_ptr<StatisticsItem> StatisticsItem::AddItemInner(
    const std::string& name, std::shared_ptr<Any> value) {
  if (IsLeaf()) {
    StatusError = {STATUS_NOTSUPPORT, "This is a leaf node, can not add item."};
    return nullptr;
  }

  if (name.empty()) {
    StatusError = {STATUS_INVALID, "Add item failed, name is empty"};
    return nullptr;
  }

  if (name == "*") {
    StatusError = {STATUS_INVALID, "Item name should not be '*'"};
    return nullptr;
  }

  auto child_ptr = new StatisticsItem(path_, name, shared_from_this());
  std::shared_ptr<StatisticsItem> child(child_ptr);
  child->thread_pool_ = thread_pool_;
  child->notify_timer_ = notify_timer_;
  child->value_ = value;
  if (value != nullptr) {
    child->is_leaf_ = true;
  }
  // Delay register
  {
    std::lock_guard<std::mutex> lck(child_notify_cfg_lock_);
    auto all_child_notify_cfg = children_notify_cfg_map_["*"];
    for (auto& cfg : all_child_notify_cfg) {
      child->RegisterNotify(cfg);
    }

    auto specify_child_notify_cfg_item = children_notify_cfg_map_.find(name);
    if (specify_child_notify_cfg_item != children_notify_cfg_map_.end()) {
      auto& specify_child_notify_cfg = specify_child_notify_cfg_item->second;
      for (auto& cfg : specify_child_notify_cfg) {
        child->RegisterNotify(cfg);
      }
    }
  }

  child->Notify(StatisticsNotifyType::CREATE);
  children_[name] = child;
  children_name_set_.insert(name);
  StatusError = STATUS_OK;
  return child;
}

std::shared_ptr<StatisticsItem> StatisticsItem::GetItem(
    const std::string& child_path) {
  auto child_name = child_path;
  std::string sub_path;
  auto pos = child_path.find('.');
  if (pos != child_path.npos) {
    child_name = child_path.substr(0, pos);
    sub_path = child_path.substr(pos + 1);
  }

  std::shared_ptr<StatisticsItem> child;
  {
    std::lock_guard<std::mutex> lck(children_lock_);
    auto item = children_.find(child_name);
    if (item == children_.end()) {
      return nullptr;
    }

    child = item->second;
  }

  if (sub_path.empty()) {
    return child;
  }

  return child->GetItem(sub_path);
}

void StatisticsItem::DelItem(const std::string& name) noexcept {
  std::lock_guard<std::mutex> lck(children_lock_);
  auto item = children_.find(name);
  if (item == children_.end()) {
    return;
  }

  // Avoid that child has be captured by others
  auto& child = item->second;
  child->is_alive_ = false;
  child->ClearItem();
  child->Notify(StatisticsNotifyType::DELETE);
  child->consumers_.Clear();
  child->parent_.reset();
  children_name_set_.erase(name);
  children_.erase(name);
}

void StatisticsItem::ClearItem() {
  std::set<std::string> children_name_set;
  {
    std::lock_guard<std::mutex> lck(children_lock_);
    children_name_set = children_name_set_;
  }

  for (auto& name : children_name_set) {
    DelItem(name);
  }
}

void StatisticsItem::Dispose() {
  auto parent_ptr = parent_.lock();
  if (!parent_ptr) {
    MBLOG_WARN << "Parent for " << path_ << " not exist";
    return;
  }

  parent_ptr->DelItem(name_);
}

bool StatisticsItem::HasItem(const std::string& name) {
  std::lock_guard<std::mutex> lck(children_lock_);
  return children_.find(name) != children_.end();
}

std::set<std::string> StatisticsItem::GetItemNames() {
  return children_name_set_;
}

modelbox::Status StatisticsItem::ForEach(const StatisticsForEachFunc& func,
                                         bool recursive) {
  ForEachInner(func, recursive, path_);
  return modelbox::STATUS_OK;
}

std::string StatisticsItem::GetRelativePath(const std::string& base_path) {
  if (base_path.empty()) {
    return path_;
  }

  return path_.substr(base_path.size() + 1);
}

modelbox::Status StatisticsItem::ForEachInner(const StatisticsForEachFunc& func,
                                              bool recursive,
                                              const std::string& base_path) {
  std::map<std::string, std::shared_ptr<StatisticsItem>> childrens;
  {
    std::lock_guard<std::mutex> lck(children_lock_);
    childrens = children_;
  }

  for (auto& child_iter : childrens) {
    auto& child = child_iter.second;
    auto ret = func(child, child->GetRelativePath(base_path));
    if (!ret) {
      return ret;
    }

    if (recursive) {
      ret = child->ForEachInner(func, true, base_path);
      if (!ret) {
        return ret;
      }
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status StatisticsItem::RegisterNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  if (cfg == nullptr) {
    return modelbox::STATUS_INVALID;
  }

  if (cfg->path_pattern_.empty()) {
    return AddNotify(cfg);
  }

  return AddChildrenNotify(cfg);
}

void StatisticsItem::UnRegisterNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  if (cfg == nullptr) {
    return;
  }

  if (cfg->path_pattern_.empty()) {
    DelNotify(cfg);
    return;
  }

  DelChildrenNotify(cfg);
}

modelbox::Status StatisticsItem::Notify(const StatisticsNotifyType& type) {
  auto consumer_list = consumers_.GetConsumers(type);
  if (consumer_list.empty()) {
    return modelbox::STATUS_OK;
  }

  if (type == StatisticsNotifyType::CHANGE) {
    // Avoid lock frequently
    if ((std::chrono::steady_clock::now() - last_change_notify_time_) <
        std::chrono::seconds(1)) {
      return modelbox::STATUS_BUSY;
    }

    // Avoid data race
    std::lock_guard<std::mutex> lck(last_change_notify_time_lock_);
    auto now = std::chrono::steady_clock::now();
    if ((now - last_change_notify_time_) < std::chrono::seconds(1)) {
      return modelbox::STATUS_BUSY;
    }

    last_change_notify_time_ = now;
  }

  auto msg = std::make_shared<StatisticsNotifyMsg>(path_, GetValue(), type);
  if (thread_pool_ == nullptr) {
    MBLOG_ERROR << "Thread pool is nullptr, can not submit notify action";
    return modelbox::STATUS_INVALID;
  }

  auto notify_action = [consumer_list, msg]() {
    for (auto& cfg : consumer_list) {
      cfg->func_(msg);
    }
  };
  thread_pool_->Submit(notify_action);
  return modelbox::STATUS_OK;
}

modelbox::Status StatisticsItem::AddNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  consumers_.AddConsumer(cfg);
  if (cfg->type_set_.find(StatisticsNotifyType::TIMER) ==
      cfg->type_set_.end()) {
    return modelbox::STATUS_OK;
  }

  if (notify_timer_ == nullptr) {
    return modelbox::STATUS_INVALID;
  }

  auto timer_task = std::make_shared<TimerTask>();
  timer_task->SetName(path_);
  timer_task->Callback([this, cfg]() {
    auto msg = std::make_shared<StatisticsNotifyMsg>(
        path_, GetValue(), StatisticsNotifyType::TIMER);
    if (thread_pool_ == nullptr) {
      MBLOG_ERROR << "Thread pool is nullptr, can not submit notify action";
      return;
    }

    thread_pool_->Submit(cfg->func_, msg);
  });
  notify_timer_->Schedule(timer_task, cfg->delay_, cfg->interval_);
  cfg->BindTimerTask(timer_task);
  return modelbox::STATUS_OK;
}

void StatisticsItem::DelNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  consumers_.DelConsumer(cfg);
}

modelbox::Status StatisticsItem::AddChildrenNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  auto root_path = cfg->GetRootPath();
  auto child_cfg = std::make_shared<StatisticsNotifyCfg>(*cfg);
  child_cfg->path_pattern_ = cfg->GetSubPath();
  // Lock here to avoid one case:
  // 1.child created register cfg.
  // 2.new child added. finally.
  // 3.children_notify_cfg_map_ add the cfg.
  std::lock_guard<std::mutex> lck(children_lock_);
  // Register to the child created before
  if (root_path != "*") {
    auto item = children_.find(root_path);
    if (item != children_.end()) {
      item->second->RegisterNotify(child_cfg);
    }
  } else {
    for (auto& child : children_) {
      child.second->RegisterNotify(child_cfg);
    }
  }

  // Prepare for the child created after
  std::lock_guard<std::mutex> cfg_lck(child_notify_cfg_lock_);
  auto& cfg_list = children_notify_cfg_map_[root_path];
  cfg_list.push_back(child_cfg);
  return modelbox::STATUS_OK;
}

void StatisticsItem::DelChildrenNotify(
    const std::shared_ptr<StatisticsNotifyCfg>& cfg) {
  auto root_path = cfg->GetRootPath();
  auto child_cfg = std::make_shared<StatisticsNotifyCfg>(*cfg);
  child_cfg->path_pattern_ = cfg->GetSubPath();
  std::lock_guard<std::mutex> cfg_lck(child_notify_cfg_lock_);
  auto& cfg_list = children_notify_cfg_map_[root_path];
  cfg_list.remove_if([cfg](const std::shared_ptr<StatisticsNotifyCfg>& val) {
    return *cfg == *val;
  });

  std::lock_guard<std::mutex> lck(children_lock_);
  if (root_path != "*") {
    auto item = children_.find(root_path);
    if (item != children_.end()) {
      item->second->UnRegisterNotify(child_cfg);
    }
  } else {
    for (auto& child : children_) {
      child.second->UnRegisterNotify(child_cfg);
    }
  }
}

static std::shared_ptr<StatisticsItem> kGlobalRootStats;
std::mutex kGlobRootStatLock;

std::shared_ptr<StatisticsItem> Statistics::GetGlobalItem() {
  if (kGlobalRootStats) {
    return kGlobalRootStats;
  }

  std::lock_guard<std::mutex> lock(kGlobRootStatLock);
  if (kGlobalRootStats) {
    return kGlobalRootStats;
  }

  kGlobalRootStats = std::make_shared<StatisticsItem>();
  auto flow_item = kGlobalRootStats->AddItem(STATISTICS_ITEM_FLOW);

  if (flow_item == nullptr) {
    MBLOG_ERROR << "Add item " << STATISTICS_ITEM_FLOW << "failed";
  }

  return kGlobalRootStats;
}

void Statistics::ReleaseGlobalItem() { kGlobalRootStats = nullptr; }

}  // namespace modelbox