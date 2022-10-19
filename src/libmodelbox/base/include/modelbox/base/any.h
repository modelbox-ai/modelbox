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

#ifndef MODELBOX_ANY_H_
#define MODELBOX_ANY_H_

#include <modelbox/base/log.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

static std::map<size_t, size_t> type_hash_code_map = {
    {typeid(int).hash_code(), typeid(int64_t).hash_code()},
    {typeid(float).hash_code(), typeid(double).hash_code()},
    {typeid(int64_t).hash_code(), typeid(int).hash_code()},
    {typeid(double).hash_code(), typeid(float).hash_code()},
};

namespace modelbox {
class Any {
 public:
  // NOLINTNEXTLINE
  Any() noexcept {};

  virtual ~Any() noexcept { delete value_ptr_; }

  template <
      typename ValueType,
      typename = typename std::enable_if<
          !std::is_same<typename std::decay<ValueType>::type, Any>::value &&
          std::is_copy_constructible<
              typename std::decay<ValueType>::type>::value>::type>
  explicit Any(ValueType&& value)
      : value_ptr_(new AnyImpl<typename std::decay<ValueType>::type>(value)) {}

  Any(const Any& other)
      : value_ptr_(other.value_ptr_ ? other.value_ptr_->clone() : nullptr) {}

  Any(Any&& other) noexcept : value_ptr_(other.value_ptr_) {
    other.value_ptr_ = nullptr;
  }

  Any& swap(Any& rhs) noexcept {
    std::swap(value_ptr_, rhs.value_ptr_);
    return *this;
  }

  template <typename ValueType>
  Any& operator=(Any&& rhs) {
    *this = Any{std::forward<ValueType>(rhs)};
    return *this;
  }

  Any& operator=(const Any& rhs) {
    *this = Any{rhs};
    return *this;
  }

  Any& operator=(Any&& rhs) noexcept {
    reset();
    value_ptr_ = rhs.value_ptr_;
    rhs.value_ptr_ = nullptr;
    return *this;
  }

  template <typename ValueType>
  bool update(ValueType&& rhs) noexcept {
    if (type() != typeid(ValueType)) {
      return false;
    }

    *this = Any{std::forward<ValueType>(rhs)};
    return true;
  }

  bool has_value() const noexcept { return value_ptr_ != nullptr; }

  const std::type_info& type() const noexcept {
    return has_value() ? value_ptr_->type() : typeid(void);
  }

  void reset() noexcept {
    delete value_ptr_;
    value_ptr_ = nullptr;
  }

  template <typename ValueType>
  const ValueType* _Cast() const noexcept {
    if (has_value()) {
      return &(static_cast<Any::AnyImpl<ValueType>*>(value_ptr_)->value_);
    }
    return nullptr;
  }

  template <typename ValueType>
  ValueType* _Cast() noexcept {
    return const_cast<ValueType*>(
        const_cast<const Any*>(this)->_Cast<ValueType>());
  }

 protected:
  struct AnyImplBase {
    virtual ~AnyImplBase() noexcept = default;

    virtual const std::type_info& type() const noexcept = 0;

    virtual AnyImplBase* clone() const = 0;
  };

  template <typename ValueType>
  struct AnyImpl : public AnyImplBase {
    AnyImpl(ValueType value) : value_(std::move(value)) {}

    AnyImpl(ValueType&& value) : value_(std::move(value)) {}

    const std::type_info& type() const noexcept override {
      return typeid(ValueType);
    }

    AnyImplBase* clone() const override { return new AnyImpl(value_); }

    ValueType value_;
  };

 private:
  AnyImplBase* value_ptr_{nullptr};
};

template <typename ValueType>
ValueType* any_cast(Any* any) noexcept {
  return any->_Cast<ValueType>();
}

template <typename ValueType>
const ValueType* any_cast(const Any* any) noexcept {
  return any->_Cast<ValueType>();
}

template <typename ValueType>
ValueType any_cast(Any& any) {
  auto* const result = any_cast<typename std::decay<ValueType>::type>(&any);

  if (!result) {
    throw std::bad_cast();
  }

  return static_cast<ValueType>(*result);
}

class Collection {
 public:
  Collection();

  virtual ~Collection();

  template <typename T>
  void Set(const std::string& key, T&& value) {
    entrys_[key] = Any(value);
  }

  void Set(const std::string& key, const char* value);

  template <typename T>
  bool Get(const std::string& key, T&& value) {
    if (entrys_.find(key) == entrys_.end()) {
      // could be a normal condition
      MBLOG_DEBUG << "Key " << key << " not found";
      return false;
    }

    if (!CanConvert(typeid(T).hash_code(), entrys_[key].type().hash_code())) {
      // always a bad condition
      MBLOG_ERROR << "Get value for " << key
                  << " failed, type mismatch, param type " << typeid(T).name()
                  << ", stored value type " << entrys_[key].type().name();
      return false;
    }

    value = any_cast<T>(entrys_[key]);
    return true;
  }

  std::tuple<Any*, bool> Get(const std::string& key);

  void Merge(const Collection& other, bool is_override = false);
  bool CanConvert(size_t cast_code, size_t origin_code);

 private:
  std::map<std::string, Any> entrys_;
};

}  // namespace modelbox

#endif