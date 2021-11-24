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


#ifndef MODELBOX_FLOWUNIT_NORMALIZE_BASE_H_
#define MODELBOX_FLOWUNIT_NORMALIZE_BASE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr uint32_t SHAPE_SIZE = 3;
constexpr uint32_t CHANNEL_NUM = 3;

class NormalizeParams {
 public:
  std::vector<double> normalizes_;
};

class NormalizeFlowUnitBase : public modelbox::FlowUnit {
 public:
  NormalizeFlowUnitBase();
  virtual ~NormalizeFlowUnitBase();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();

  /* run when processing data */
  virtual modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) = 0;

  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataGroupPre(std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
  };

 protected:
  bool CheckBufferListValid(
      const std::shared_ptr<modelbox::BufferList> &buffer_list);
  NormalizeParams params_;
};

bool BuildOutputBufferList(
    const std::shared_ptr<modelbox::BufferList> &input_bufs,
    std::shared_ptr<modelbox::BufferList> &output_bufs);
#endif  // MODELBOX_FLOWUNIT_NORMALIZE_BASE_H_