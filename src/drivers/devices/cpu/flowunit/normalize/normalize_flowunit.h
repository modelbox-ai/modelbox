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

#ifndef MODELBOX_FLOWUNIT_NORMALIZE_H_
#define MODELBOX_FLOWUNIT_NORMALIZE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <normalize_flowunit_base.h>

constexpr const char *FLOWUNIT_NAME = "normalize";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: The operator is used to normalize for tensor data, "
    "for example the image(RGB/BGR). \n"
    "\t@Port parameter: The input port and the output buffer type are tensor. \n"
    "\t  The tensor type buffer contain the following meta fields:\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint: ";

class NormalizeFlowUnit : public NormalizeFlowUnitBase {
 public:
  NormalizeFlowUnit();
  ~NormalizeFlowUnit() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  template <typename T>
  void Process(const T *input_data,
               const std::shared_ptr<modelbox::Buffer> &input_buf,
               const std::shared_ptr<modelbox::Buffer> &out_buff);
};

#endif  // MODELBOX_FLOWUNIT_NORMALIZE_H_