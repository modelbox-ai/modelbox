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


#include "modelbox/base/configuration.h"

#include <toml.hpp>

namespace modelbox {

class TomlConfigParser : public ConfigParser {
 public:
  virtual ~TomlConfigParser() = default;
  Status Visit(
      const std::string &key, const toml::value &value,
      const std::function<void(const std::string key,
                               const std::string &basic_value)> &collector);

  Status Parse(const std::shared_ptr<Configuration> &config, std::istream &is,
               const std::string &fname = "unknown file") override;
  Status Parse(const std::shared_ptr<Configuration> &config,
               const std::string &file) override;
};

void ConfigStore::WriteProperty(const std::string &key,
                                const std::string &property) {
  properties_[key] = property;

  auto prefix_key = key;
  auto period_pos = prefix_key.find_last_of('.');
  while (period_pos != std::string::npos) {
    auto sub_key = prefix_key.substr(period_pos + 1);
    prefix_key = prefix_key.substr(0, period_pos);

    if (sub_key_index_.find(prefix_key) != sub_key_index_.end()) {
      period_pos = std::string::npos;
    } else {
      period_pos = prefix_key.find_last_of('.', period_pos);
    }

    sub_key_index_[prefix_key].insert(sub_key);
  }
}

Status ConfigStore::ReadProperty(const std::string &key,
                                 std::string *property) const {
  if (property == nullptr) {
    return STATUS_FAULT;
  }

  auto item = properties_.find(key);
  if (item == properties_.end()) {
    return STATUS_RANGE;
  }

  *property = item->second;
  return STATUS_SUCCESS;
}

std::set<std::string> ConfigStore::GetKeys() const {
  std::set<std::string> keys;
  for (const auto &propertie : properties_) {
    keys.insert(propertie.first);
  }

  return keys;
}

std::set<std::string> ConfigStore::GetSubKeys(
    const std::string &prefix_key) const {
  auto iter = sub_key_index_.find(prefix_key);
  if (iter == sub_key_index_.end()) {
    return {};
  }

  return iter->second;
}

std::unique_ptr<ConfigStore> ConfigStore::GetSubConfigStore(
    const std::string &prefix_key) const {
  std::unique_ptr<ConfigStore> sub_store(new ConfigStore());
  AddSubConfig(prefix_key, sub_store.get(), prefix_key.size() + 1);
  return sub_store;
}

void ConfigStore::AddSubConfig(const std::string &prefix_key,
                               ConfigStore *store, size_t key_offset) const {
  auto sub_keys = GetSubKeys(prefix_key);
  if (sub_keys.size() == 0) {
    StatusError = {STATUS_NOTFOUND, "sub config not found"};
    return;
  }

  for (const auto &sub_key : sub_keys) {
    auto new_prefix = prefix_key + ".";
    new_prefix += sub_key;
    auto item = properties_.find(new_prefix);
    if (item != properties_.end()) {
      store->WriteProperty(item->first.substr(key_offset), item->second);
    }
    AddSubConfig(new_prefix, store, key_offset);
  }

  if (sub_keys.size() > 0) {
    StatusError = STATUS_OK;
  }
}

Configuration::Configuration() {
  store_ = std::unique_ptr<ConfigStore>(new ConfigStore());
}

Configuration::Configuration(std::unique_ptr<ConfigStore> &store) {
  store_ = std::move(store);
}

void Configuration::Trim(std::string *value) {
  if (value == nullptr) {
    return;
  }

  value->erase(0, value->find_first_not_of(' '));
  value->erase(value->find_last_not_of(' ') + 1);
}

size_t Configuration::Size() const { return store_->Size(); }

std::set<std::string> Configuration::GetKeys() const {
  return store_->GetKeys();
}

bool Configuration::Contain(const std::string &key) const {
  return store_->Contain(key);
}

std::set<std::string> Configuration::GetSubKeys(
    const std::string &prefix_key) const {
  return store_->GetSubKeys(prefix_key);
}

std::shared_ptr<Configuration> Configuration::GetSubConfig(
    const std::string &prefix_key) const {
  auto sub_config_store = store_->GetSubConfigStore(prefix_key);
  std::shared_ptr<Configuration> sub_config(
      new Configuration(sub_config_store));
  return sub_config;
}

std::string Configuration::GetString(const std::string &key,
                                     const std::string &default_prop) const {
  return GetProperty(key, default_prop);
}

bool Configuration::GetBool(const std::string &key, bool default_prop) const {
  return GetProperty(key, default_prop);
}

int8_t Configuration::GetInt8(const std::string &key,
                              int8_t default_prop) const {
  return GetProperty(key, default_prop);
}

uint8_t Configuration::GetUint8(const std::string &key,
                                uint8_t default_prop) const {
  return GetProperty(key, default_prop);
}

int16_t Configuration::GetInt16(const std::string &key,
                                int16_t default_prop) const {
  return GetProperty(key, default_prop);
}

uint16_t Configuration::GetUint16(const std::string &key,
                                  uint16_t default_prop) const {
  return GetProperty(key, default_prop);
}

int32_t Configuration::GetInt32(const std::string &key,
                                int32_t default_prop) const {
  return GetProperty(key, default_prop);
}

uint32_t Configuration::GetUint32(const std::string &key,
                                  uint32_t default_prop) const {
  return GetProperty(key, default_prop);
}

int64_t Configuration::GetInt64(const std::string &key,
                                int64_t default_prop) const {
  return GetProperty(key, default_prop);
}

uint64_t Configuration::GetUint64(const std::string &key,
                                  uint64_t default_prop) const {
  return GetProperty(key, default_prop);
}

float Configuration::GetFloat(const std::string &key,
                              float default_prop) const {
  return GetProperty(key, default_prop);
}

double Configuration::GetDouble(const std::string &key,
                                double default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<std::string> Configuration::GetStrings(
    const std::string &key,
    const std::vector<std::string> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<bool> Configuration::GetBools(
    const std::string &key, const std::vector<bool> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<int8_t> Configuration::GetInt8s(
    const std::string &key, const std::vector<int8_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<uint8_t> Configuration::GetUint8s(
    const std::string &key, const std::vector<uint8_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<int16_t> Configuration::GetInt16s(
    const std::string &key, const std::vector<int16_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<uint16_t> Configuration::GetUint16s(
    const std::string &key, const std::vector<uint16_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<int32_t> Configuration::GetInt32s(
    const std::string &key, const std::vector<int32_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<uint32_t> Configuration::GetUint32s(
    const std::string &key, const std::vector<uint32_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<int64_t> Configuration::GetInt64s(
    const std::string &key, const std::vector<int64_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<uint64_t> Configuration::GetUint64s(
    const std::string &key, const std::vector<uint64_t> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<float> Configuration::GetFloats(
    const std::string &key, const std::vector<float> &default_prop) const {
  return GetProperty(key, default_prop);
}

std::vector<double> Configuration::GetDoubles(
    const std::string &key, const std::vector<double> &default_prop) const {
  return GetProperty(key, default_prop);
}

template <>
Status Configuration::Convert<std::string>(const std::string &property,
                                           std::string &convert_prop) const {
  convert_prop = property;
  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<bool>(const std::string &property,
                                    bool &convert_prop) const {
  if (property == "true" || property == "1") {
    convert_prop = true;
  } else if (property == "false" || property == "0") {
    convert_prop = false;
  } else {
    return {STATUS_FAULT, "bool failed, invalid"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<int8_t>(const std::string &property,
                                      int8_t &convert_prop) const {
  int value;
  size_t idx = 0;
  try {
    value = std::stoi(property, &idx);
    if (value > INT8_MAX || value < INT8_MIN) {
      return {STATUS_FAULT, "int8 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "int8 failed, invalid"};
    }

    convert_prop = static_cast<int8_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "int8 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "int8 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<uint8_t>(const std::string &property,
                                       uint8_t &convert_prop) const {
  int value;
  size_t idx = 0;
  try {
    value = std::stoi(property, &idx);
    if (value > UINT8_MAX || value < 0) {
      return {STATUS_FAULT, "uint8 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "uint8 failed, invalid"};
    }

    convert_prop = static_cast<uint8_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "uint8 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "uint8 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<int16_t>(const std::string &property,
                                       int16_t &convert_prop) const {
  int value;
  size_t idx = 0;
  try {
    value = std::stoi(property, &idx);
    if (value > INT16_MAX || value < INT16_MIN) {
      return {STATUS_FAULT, "int16 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "int16 failed, invalid"};
    }

    convert_prop = static_cast<int16_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "int16 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "int16 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<uint16_t>(const std::string &property,
                                        uint16_t &convert_prop) const {
  int value;
  size_t idx = 0;
  try {
    value = std::stoi(property, &idx);
    if (value > UINT16_MAX || value < 0) {
      return {STATUS_FAULT, "uint16 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "uint16 failed, invalid"};
    }

    convert_prop = static_cast<uint16_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "uint16 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "uint16 failed, out of range"};
  }
  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<int32_t>(const std::string &property,
                                       int32_t &convert_prop) const {
  long long value;
  size_t idx = 0;
  try {
    value = std::stoi(property, &idx);
    if (value > INT32_MAX || value < INT32_MIN) {
      return {STATUS_FAULT, "int32 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "int32 failed, invalid"};
    }

    convert_prop = static_cast<int32_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "int32 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "int32 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<uint32_t>(const std::string &property,
                                        uint32_t &convert_prop) const {
  long long value;
  size_t idx = 0;
  try {
    value = std::stoll(property, &idx);
    if (value > UINT32_MAX || value < 0) {
      return {STATUS_FAULT, "uint32 failed, out of range"};
    }

    if (idx != property.size()) {
      return {STATUS_FAULT, "uint32 failed, invalid"};
    }

    convert_prop = static_cast<uint32_t>(value);
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "uint32 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "uint32 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<int64_t>(const std::string &property,
                                       int64_t &convert_prop) const {
  size_t idx = 0;
  try {
    convert_prop = std::stoll(property, &idx);

    if (idx != property.size()) {
      return {STATUS_FAULT, "int64 failed, invalid"};
    }

  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "int64 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "int64 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<uint64_t>(const std::string &property,
                                        uint64_t &convert_prop) const {
  size_t idx = 0;
  try {
    convert_prop = std::stoull(property, &idx);

    if (idx != property.size()) {
      return {STATUS_FAULT, "uint64 failed, invalid"};
    }

    if (property[0] == '-') {
      return {STATUS_FAULT, "uint64 failed, out of range"};
    }

  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "uint64 failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "uint64 failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<float>(const std::string &property,
                                     float &convert_prop) const {
  size_t idx = 0;
  try {
    convert_prop = std::stof(property, &idx);

    if (idx != property.size()) {
      return {STATUS_FAULT, "float failed, invalid"};
    }
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "float failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "float failed, out of range"};
  }

  return STATUS_SUCCESS;
}

template <>
Status Configuration::Convert<double>(const std::string &property,
                                      double &convert_prop) const {
  size_t idx = 0;
  try {
    convert_prop = std::stod(property, &idx);

    if (idx != property.size()) {
      return {STATUS_FAULT, "double failed, invalid"};
    }
  } catch (std::invalid_argument &e) {
    return {STATUS_FAULT, "double failed, invalid"};
  } catch (std::out_of_range &e) {
    return {STATUS_FAULT, "double failed, out of range"};
  }

  return STATUS_SUCCESS;
}

void Configuration::StringSplit(const std::string &str,
                                const std::string &delimiter,
                                std::vector<std::string> &sub_str_list) {
  if (str.empty() || delimiter.empty()) {
    return;
  }

  auto str_with_delimiter = str + delimiter;
  auto begin = 0;
  auto end = str_with_delimiter.find(delimiter, begin);
  while (end != std::string::npos) {
    auto sub_str = str_with_delimiter.substr(begin, end - begin);
    sub_str_list.push_back(sub_str);
    begin = end + 1;
    end = str_with_delimiter.find(delimiter, begin);
  }
}

Status TomlConfigParser::Visit(
    const std::string &key, const toml::value &value,
    const std::function<void(const std::string key,
                             const std::string &basic_value)> &collector) {
  if (value.is_array()) {
    std::stringstream ss;
    auto array = value.as_array();
    int index = 0;
    for (const auto &v : array) {
      if (index > 0) {
        ss << LIST_DELIMITER;
      };
      ss << v.as_string().str;
      index++;
    }
    collector(key, ss.str());
    return STATUS_SUCCESS;
  }

  if (value.is_table()) {
    auto value_table = value.as_table();
    for (const auto &pair : value_table) {
      std::string sub_key;
      if (key.empty()) {
        sub_key = pair.first;
      } else {
        sub_key = key + "." + pair.first;
      }
      auto ret = Visit(sub_key, pair.second, collector);
      if (ret != STATUS_SUCCESS) {
        return ret;
      }
    }
    return STATUS_SUCCESS;
  }

  if (value.is_string()) {
    collector(key, value.as_string().str);
  } else {
    std::stringstream ss;
    ss << value;
    collector(key, ss.str());
  }

  return STATUS_SUCCESS;
}

Status TomlConfigParser::Parse(const std::shared_ptr<Configuration> &config,
                               std::istream &is, const std::string &fname) {
  toml::value data;
  try {
    data = toml::parse(is, fname);
  } catch (std::exception &e) {
    return {STATUS_FAULT, e.what()};
  }

  return Visit(
      "", data,
      [&config](const std::string &key, const std::string &basic_value) {
        config->SetProperty(key, basic_value);
      });
}

Status TomlConfigParser::Parse(const std::shared_ptr<Configuration> &config,
                               const std::string &file) {
  std::ifstream ifs(file.c_str(), std::ios_base::binary);
  return Parse(config, ifs, file);
}

ConfigurationBuilder::ConfigurationBuilder() = default;

void ConfigurationBuilder::AddProperty(const std::string &key,
                                       const std::string &property) {
  if (store_ == nullptr) {
    store_.reset(new ConfigStore());
  }

  store_->WriteProperty(key, property);
}

void ConfigurationBuilder::AddProperty(
    const std::string &key, const std::vector<std::string> &properties) {
  bool is_first = true;
  if (store_ == nullptr) {
    store_.reset(new ConfigStore());
  }

  std::string value;
  for (const auto &str : properties) {
    if (is_first == true) {
      is_first = false;
    } else {
      value += LIST_DELIMITER;
    }
    value += str;
  }

  store_->WriteProperty(key, value);
}

void ConfigurationBuilder::AddProperties(
    const std::map<std::string, std::string> &properties) {
  for (const auto &pair : properties) {
    AddProperty(pair.first, pair.second);
  }
}

std::shared_ptr<Configuration> ConfigurationBuilder::Build() {
  if (store_ == nullptr) {
    store_.reset(new ConfigStore());
  }

  StatusError = STATUS_SUCCESS;
  return std::shared_ptr<Configuration>(new Configuration(store_));
}

std::shared_ptr<Configuration> ConfigurationBuilder::Build(
    const std::string &file, const ConfigType &type) {
  std::ifstream ifs(file.c_str());
  if (!ifs.good()) {
    auto msg = "file open " + file + " error: " + StrError(errno);
    MBLOG_ERROR << msg;
    StatusError = {STATUS_INVALID, msg};
    return nullptr;
  }

  return Build(ifs, file, type);
}

std::shared_ptr<Configuration> ConfigurationBuilder::Build(
    std::istream &is, const std::string &fname, const ConfigType &type) {
  auto parser = CreateParser(type);
  if (parser == nullptr) {
    return nullptr;
  }

  store_.reset(new ConfigStore());
  std::shared_ptr<Configuration> config(new Configuration(store_));
  auto ret = parser->Parse(config, is, fname);
  if (ret != STATUS_SUCCESS) {
    StatusError = ret;
    return nullptr;
  }

  StatusError = STATUS_SUCCESS;
  return config;
}

std::shared_ptr<ConfigParser> ConfigurationBuilder::CreateParser(
    const ConfigType &type) {
  switch (type) {
    case ConfigType::TOML:
      return std::make_shared<TomlConfigParser>();
      break;

    default:
      StatusError = {STATUS_FAULT,
                     "Create unknow parser " + std::to_string((int32_t)type)};
      return nullptr;
  }
}

}  // namespace modelbox