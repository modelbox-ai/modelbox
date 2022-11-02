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

#ifndef MODELBOX_CONFIGURATION_H_
#define MODELBOX_CONFIGURATION_H_

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "modelbox/base/log.h"
#include "modelbox/base/status.h"

namespace modelbox {
// ETX (end of text) is used as a separator between values
constexpr const char *LIST_DELIMITER = "\003";
constexpr const uint32_t VALID_RANGE_OF_DOUBLE = 15;

class ConfigStore {
 public:
  virtual ~ConfigStore();

  void WriteProperty(const std::string &key, const std::string &property);

  Status ReadProperty(const std::string &key, std::string *property) const;

  size_t Size() const;

  std::set<std::string> GetKeys() const;

  bool Contain(const std::string &key) const;

  std::set<std::string> GetSubKeys(const std::string &prefix_key) const;

  std::unique_ptr<ConfigStore> GetSubConfigStore(
      const std::string &prefix_key) const;

  void Add(const ConfigStore &store);

  void Copy(const ConfigStore &store, const std::string &key);

  void SetExpandEnv(bool expand_env);

 private:
  std::map<std::string, std::string> properties_;
  std::map<std::string, std::set<std::string>> sub_key_index_;
  bool expand_env_{false};

  void AddSubConfig(const std::string &prefix_key, ConfigStore *store,
                    size_t key_offset) const;
};

class ConfigurationBuilder;

class Configuration {
  friend class ConfigurationBuilder;

 public:
  Configuration();
  Configuration(const Configuration &config) = delete;
  Configuration &operator=(const Configuration &config) = delete;
  Configuration(const Configuration &&config) = delete;
  Configuration &operator=(const Configuration &&config) = delete;

  virtual ~Configuration();

  static void Trim(std::string *value);

  static void StringSplit(const std::string &str, const std::string &delimiter,
                          std::vector<std::string> &sub_str_list);

  void Add(const Configuration &config);

  void Copy(const Configuration &config, const std::string &key);

  size_t Size() const;

  std::set<std::string> GetKeys() const;

  bool Contain(const std::string &key) const;

  std::set<std::string> GetSubKeys(const std::string &prefix_key) const;

  std::shared_ptr<Configuration> GetSubConfig(
      const std::string &prefix_key) const;

  template <class T>
  void SetProperty(const std::string &key, const T &prop);

  template <class T>
  void SetProperty(const std::string &key, const std::vector<T> &prop);

  template <class T>
  T GetProperty(const std::string &key, const T &default_prop) const;

  template <class T>
  std::vector<T> GetProperty(const std::string &key,
                             const std::vector<T> &default_prop) const;

  std::string GetString(const std::string &key,
                        const std::string &default_prop = "") const;

  bool GetBool(const std::string &key, bool default_prop = false) const;

  int8_t GetInt8(const std::string &key, int8_t default_prop = 0) const;

  uint8_t GetUint8(const std::string &key, uint8_t default_prop = 0) const;

  int16_t GetInt16(const std::string &key, int16_t default_prop = 0) const;

  uint16_t GetUint16(const std::string &key, uint16_t default_prop = 0) const;

  int32_t GetInt32(const std::string &key, int32_t default_prop = 0) const;

  uint32_t GetUint32(const std::string &key, uint32_t default_prop = 0) const;

  int64_t GetInt64(const std::string &key, int64_t default_prop = 0) const;

  uint64_t GetUint64(const std::string &key, uint64_t default_prop = 0) const;

  float GetFloat(const std::string &key, float default_prop = 0.0F) const;

  double GetDouble(const std::string &key, double default_prop = 0.0) const;

  std::vector<std::string> GetStrings(
      const std::string &key,
      const std::vector<std::string> &default_prop = {}) const;

  std::vector<bool> GetBools(const std::string &key,
                             const std::vector<bool> &default_prop = {}) const;

  std::vector<int8_t> GetInt8s(
      const std::string &key,
      const std::vector<int8_t> &default_prop = {}) const;

  std::vector<uint8_t> GetUint8s(
      const std::string &key,
      const std::vector<uint8_t> &default_prop = {}) const;

  std::vector<int16_t> GetInt16s(
      const std::string &key,
      const std::vector<int16_t> &default_prop = {}) const;

  std::vector<uint16_t> GetUint16s(
      const std::string &key,
      const std::vector<uint16_t> &default_prop = {}) const;

  std::vector<int32_t> GetInt32s(
      const std::string &key,
      const std::vector<int32_t> &default_prop = {}) const;

  std::vector<uint32_t> GetUint32s(
      const std::string &key,
      const std::vector<uint32_t> &default_prop = {}) const;

  std::vector<int64_t> GetInt64s(
      const std::string &key,
      const std::vector<int64_t> &default_prop = {}) const;

  std::vector<uint64_t> GetUint64s(
      const std::string &key,
      const std::vector<uint64_t> &default_prop = {}) const;

  std::vector<float> GetFloats(
      const std::string &key,
      const std::vector<float> &default_prop = {}) const;

  std::vector<double> GetDoubles(
      const std::string &key,
      const std::vector<double> &default_prop = {}) const;

 protected:
  Configuration(std::unique_ptr<ConfigStore> &store);

  template <class T>
  Status Convert(const std::string &property, T &convert_prop) const;

  std::unique_ptr<ConfigStore> store_;
};

template <class T>
void Configuration::SetProperty(const std::string &key, const T &prop) {
  std::stringstream ss;
  ss.precision(VALID_RANGE_OF_DOUBLE);
  ss << prop;
  store_->WriteProperty(key, ss.str());
}

template <class T>
void Configuration::SetProperty(const std::string &key,
                                const std::vector<T> &prop) {
  std::stringstream ss;
  ss.precision(VALID_RANGE_OF_DOUBLE);
  for (size_t i = 1; i < prop.size(); ++i) {
    ss << prop[i - 1] << LIST_DELIMITER;
  }

  if (!prop.empty()) {
    ss << prop.back();
  }

  store_->WriteProperty(key, ss.str());
}

template <class T>
T Configuration::GetProperty(const std::string &key,
                             const T &default_prop) const {
  std::string raw_prop;
  auto ret = store_->ReadProperty(key, &raw_prop);
  if (ret != STATUS_SUCCESS) {
    return default_prop;
  }

  T convert_prop{};
  ret = Convert<T>(raw_prop, convert_prop);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Convert [" << key << " : " << raw_prop << "] to "
                << ret.Errormsg();
    return default_prop;
  }

  return convert_prop;
};

template <class T>
std::vector<T> Configuration::GetProperty(
    const std::string &key, const std::vector<T> &default_prop) const {
  std::string raw_prop;
  auto ret = store_->ReadProperty(key, &raw_prop);
  if (ret != STATUS_SUCCESS) {
    return default_prop;
  }

  std::vector<std::string> raw_value_list;
  StringSplit(raw_prop, LIST_DELIMITER, raw_value_list);

  std::vector<T> value_list;
  T convert_prop{};
  for (const auto &raw_value : raw_value_list) {
    ret = Convert<T>(raw_value, convert_prop);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Convert [" << key << " : " << raw_prop << "]::["
                  << raw_value << "] to " << ret.Errormsg();
      return default_prop;
    }

    value_list.push_back(convert_prop);
  }

  return value_list;
}

template <class T>
Status Configuration::Convert(const std::string &property,
                              T &convert_prop) const {
  UNUSED_VAR(property);
  UNUSED_VAR(convert_prop);

  return STATUS_FAULT;
};

template <>
Status Configuration::Convert<std::string>(const std::string &property,
                                           std::string &convert_prop) const;

template <>
Status Configuration::Convert<bool>(const std::string &property,
                                    bool &convert_prop) const;

template <>
Status Configuration::Convert<int8_t>(const std::string &property,
                                      int8_t &convert_prop) const;

template <>
Status Configuration::Convert<uint8_t>(const std::string &property,
                                       uint8_t &convert_prop) const;

template <>
Status Configuration::Convert<int16_t>(const std::string &property,
                                       int16_t &convert_prop) const;

template <>
Status Configuration::Convert<uint16_t>(const std::string &property,
                                        uint16_t &convert_prop) const;

template <>
Status Configuration::Convert<int32_t>(const std::string &property,
                                       int32_t &convert_prop) const;

template <>
Status Configuration::Convert<uint32_t>(const std::string &property,
                                        uint32_t &convert_prop) const;

template <>
Status Configuration::Convert<int64_t>(const std::string &property,
                                       int64_t &convert_prop) const;

template <>
Status Configuration::Convert<uint64_t>(const std::string &property,
                                        uint64_t &convert_prop) const;

template <>
Status Configuration::Convert<float>(const std::string &property,
                                     float &convert_prop) const;

template <>
Status Configuration::Convert<double>(const std::string &property,
                                      double &convert_prop) const;

class ConfigParser {
 public:
  virtual Status Parse(const std::shared_ptr<Configuration> &config,
                       std::istream &is, const std::string &fname) = 0;
  virtual Status Parse(const std::shared_ptr<Configuration> &config,
                       const std::string &file) = 0;
};

enum class ConfigType { TOML };

class ConfigurationBuilder {
 public:
  ConfigurationBuilder();

  virtual ~ConfigurationBuilder();

  void AddProperty(const std::string &key, const std::string &property);

  void AddProperty(const std::string &key,
                   const std::vector<std::string> &properties);

  void AddProperties(const std::map<std::string, std::string> &properties);

  std::shared_ptr<Configuration> Build();

  std::shared_ptr<Configuration> Build(
      const std::string &file, const ConfigType &type = ConfigType::TOML,
      bool expand_env = false);

  std::shared_ptr<Configuration> Build(
      std::istream &is, const std::string &fname = "unknown file",
      const ConfigType &type = ConfigType::TOML, bool expand_env = false);

 private:
  std::shared_ptr<ConfigParser> CreateParser(
      const ConfigType &type = ConfigType::TOML);

  std::unique_ptr<ConfigStore> store_;
};

}  // namespace modelbox

#endif  // MODELBOX_CONFIGURATION_H_