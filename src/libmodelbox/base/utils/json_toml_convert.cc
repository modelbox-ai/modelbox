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

#include <fstream>
#include <nlohmann/json.hpp>
#include <toml.hpp>

#include "modelbox/base/utils.h"

namespace modelbox {

Status JsonToTomlProcess(
    std::list<std::tuple<std::shared_ptr<toml::value>, std::string,
                         nlohmann::json>> &json_obj_list,
    std::shared_ptr<toml::value> toml_root) {
  while (json_obj_list.size() > 0) {
    std::string key;
    nlohmann::json cur_value;
    std::shared_ptr<toml::value> toml_key;
    std::tie(toml_key, key, cur_value) = json_obj_list.back();
    json_obj_list.pop_back();

    switch (cur_value.type()) {
      case nlohmann::json::value_t::null:
        break;
      case nlohmann::json::value_t::number_integer:
        (*toml_key) = cur_value.get<int>();
        break;
      case nlohmann::json::value_t::number_unsigned:
        (*toml_key) = cur_value.get<unsigned int>();
        break;
      case nlohmann::json::value_t::number_float:
        (*toml_key) = cur_value.get<double>();
        break;
      case nlohmann::json::value_t::boolean:
        (*toml_key) = cur_value.get<bool>();
        break;
      case nlohmann::json::value_t::string:
        (*toml_key) = cur_value.get<std::string>();
        break;
      case nlohmann::json::value_t::object: {
        for (nlohmann::json::iterator obj = cur_value.begin();
             obj != cur_value.end(); obj++) {
          auto *value = new toml::value;
          std::string key = obj.key();
          std::shared_ptr<toml::value> toml_new(value, [=](toml::value *value) {
            if (value->is_uninitialized()) {
              delete value;
              return;
            }
            (*toml_key)[key] = *value;
            delete value;
          });
          json_obj_list.push_front(
              std::make_tuple(toml_new, obj.key(), obj.value()));
        }
        break;
      }
      case nlohmann::json::value_t::array: {
        auto *array = new toml::array;
        std::shared_ptr<toml::array> array_new(array, [=](toml::array *array) {
          (*toml_key) = *array;
          delete array;
        });

        for (auto &item : cur_value) {
          auto *value = new toml::value;
          std::shared_ptr<toml::value> toml_new(value, [=](toml::value *value) {
            (*array_new).push_back(*value);
            delete value;
          });
          json_obj_list.push_front(std::make_tuple(toml_new, key, item));
        }
        break;
      }
      default:
        MBLOG_ERROR << "Process json to toml failed, " << key << ":"
                    << cur_value;
        return {modelbox::STATUS_BADCONF};
        break;
    }
  }

  return modelbox::STATUS_OK;
}

Status JsonToToml(const std::string &json_data, std::string *toml_data) {
  nlohmann::json root;
  auto toml_root = std::make_shared<toml::value>();

  if (toml_data == nullptr) {
    return {modelbox::STATUS_INVALID};
  }

  try {
    root = nlohmann::json::parse(json_data);

    std::list<
        std::tuple<std::shared_ptr<toml::value>, std::string, nlohmann::json>>
        json_obj_list;
    json_obj_list.emplace_back(std::make_tuple(toml_root, "", root));

    auto ret = JsonToTomlProcess(json_obj_list, toml_root);
    if (!ret) {
      return ret;
    }
  } catch (std::exception &e) {
    MBLOG_ERROR << "parse json failed, " << e.what();
    return {modelbox::STATUS_BADCONF, e.what()};
  }

  std::ostringstream os;
  os << *toml_root;
  *toml_data = os.str();
  return modelbox::STATUS_OK;
}

struct JsonSerializer {
  JsonSerializer(bool indent) : indent_(indent) {}

  void operator()(toml::boolean v) { oss_ << toml::value(v); }
  void operator()(toml::integer v) { oss_ << toml::value(v); }
  void operator()(toml::floating v) { oss_ << toml::value(v); }
  void operator()(const toml::string &v) {
    // since toml11 automatically convert string to multiline string that is
    // valid only in TOML, we need to format the string to make it valid in
    // JSON.
    oss_ << "\"" << this->escape_string(v.str) << "\"";
  }
  void operator()(const toml::local_time &v) { oss_ << toml::value(v); }
  void operator()(const toml::local_date &v) { oss_ << toml::value(v); }
  void operator()(const toml::local_datetime &v) { oss_ << toml::value(v); }
  void operator()(const toml::offset_datetime &v) { oss_ << toml::value(v); }
  void operator()(const toml::array &v) {
    bool has_data = false;
    if (!v.empty() && v.front().is_table()) {
      oss_ << '[';
      IndentIn();
      bool is_first = true;
      for (const auto &elem : v) {
        if (!is_first) {
          oss_ << ",";
        }
        is_first = false;
        has_data = true;
        toml::visit(*this, elem);
      }
      IndentOut(has_data);
      oss_ << ']';
    } else {
      oss_ << "[";
      IndentIn();
      bool is_first = true;
      for (const auto &elem : v) {
        if (!is_first) {
          oss_ << ",";
        }
        IndentSpace();
        is_first = false;
        has_data = true;
        toml::visit(*this, elem);
      }
      IndentOut(has_data);
      oss_ << "]";
    }
  }
  void operator()(const toml::table &v) {
    oss_ << '{';
    bool has_data = false;
    IndentIn();
    bool is_first = true;
    for (const auto &elem : v) {
      if (!is_first) {
        oss_ << ",";
      }
      has_data = true;
      is_first = false;
      IndentSpace();
      oss_ << this->format_key(elem.first) << ": ";
      toml::visit(*this, elem.second);
    }
    IndentOut(has_data);
    oss_ << '}';
  }

  std::string escape_string(const std::string &s) const {
    std::string retval;
    for (const char c : s) {
      switch (c) {
        case '\\': {
          retval += "\\\\";
          break;
        }
        case '\"': {
          retval += "\\\"";
          break;
        }
        case '\b': {
          retval += "\\b";
          break;
        }
        case '\t': {
          retval += "\\t";
          break;
        }
        case '\f': {
          retval += "\\f";
          break;
        }
        case '\n': {
          retval += "\\n";
          break;
        }
        case '\r': {
          retval += "\\r";
          break;
        }
        default: {
          retval += c;
          break;
        }
      }
    }
    return retval;
  }

  std::string format_key(const std::string &s) const {
    const auto *quote = "\"";
    return quote + escape_string(s) + quote;
  }

  std::string GetJsonData() { return oss_.str(); }

 private:
  void IndentIn() {
    if (indent_ == false) {
      return;
    }

    space_.append("    ");
  }

  void IndentSpace() {
    if (indent_ == false) {
      return;
    }

    oss_ << std::endl << space_;
  }

  void IndentOut(bool need_newline) {
    if (indent_ == false) {
      return;
    }

    if (need_newline == true) {
      oss_ << std::endl;
    }
    space_.pop_back();
    space_.pop_back();
    space_.pop_back();
    space_.pop_back();

    if (need_newline == true) {
      oss_ << space_;
    }
  }
  std::ostringstream oss_;
  std::string space_;
  bool indent_{false};
};

Status TomlToJson(const std::string &toml_data, std::string *json_data,
                  bool readable) {
  if (json_data == nullptr) {
    return {modelbox::STATUS_INVALID};
  }

  try {
    std::istringstream instring(toml_data);
    auto json_serialize = JsonSerializer(readable);
    auto toml = toml::parse(instring);
    toml::visit(json_serialize, toml);
    *json_data = json_serialize.GetJsonData();
  } catch (const std::exception &e) {
    MBLOG_ERROR << "parse toml failed, " << e.what();
    return {modelbox::STATUS_BADCONF, e.what()};
  }

  return modelbox::STATUS_OK;
}

}  // namespace modelbox
