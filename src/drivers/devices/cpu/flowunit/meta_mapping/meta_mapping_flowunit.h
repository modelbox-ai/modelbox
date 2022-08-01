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

#ifndef MODELBOX_FLOWUNIT_META_MAP_CPU_H_
#define MODELBOX_FLOWUNIT_META_MAP_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <functional>
#include <map>
#include <string>

#include "modelbox/base/any.h"
#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "buff_meta_mapping";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Modify the input buffer meta field name and value according to custom "
    "rules. \n"
    "\t@Port parameter: The input port and the output buffer type are binary. \n"
    "\t@Constraint: ";
constexpr const char *INPUT_DATA = "in_data";
constexpr const char *OUTPUT_DATA = "out_data";

using MappingRules = std::map<std::string, std::string>;
using AnyToStringCaster =
    std::function<void(std::stringstream &, modelbox::Any *)>;
using BufferSetter = std::function<void(std::shared_ptr<modelbox::Buffer> &,
                                        const std::string &)>;
class MetaMappingFlowUnit : public modelbox::FlowUnit {
 public:
  MetaMappingFlowUnit();
  ~MetaMappingFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  /* run when processing data */
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

 private:
  void InitToStringCasters();

  void InitBufferMetaSetters();

  modelbox::Status ParseRules(const std::vector<std::string> &rules);

  modelbox::Status ToString(modelbox::Any *any, std::string &val);

  modelbox::Status SetValue(std::shared_ptr<modelbox::Buffer> &buffer,
                            std::string &str, const std::type_info &type);

  MappingRules mapping_rules_;
  std::string src_meta_name_;
  std::string dest_meta_name_;

  std::map<size_t, AnyToStringCaster> to_string_casters_;
  std::map<size_t, BufferSetter> buffer_meta_setters_;
};

#endif  // MODELBOX_FLOWUNIT_META_MAP_CPU_H_
