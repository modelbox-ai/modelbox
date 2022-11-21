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

#ifndef MODELBOX_MINDSPRORE_INFERENCE_H_
#define MODELBOX_MINDSPRORE_INFERENCE_H_

#include "include/api/context.h"
#include "include/api/model.h"
#include "modelbox/base/configuration.h"
#include "modelbox/data_context.h"
#include "modelbox/flowunit.h"

struct MindSporeIOList {
  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  std::vector<std::string> input_type_list;
  std::vector<std::string> output_type_list;
  std::vector<std::string> input_device_list;
};

constexpr const char *INFERENCE_TYPE = "mindspore";

class MindSporeInference {
 public:
  MindSporeInference(const std::shared_ptr<modelbox::Device> &flowunit_device,
                     const std::shared_ptr<mindspore::Context> &context);
  virtual ~MindSporeInference();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts,
                        std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc);
  modelbox::Status Infer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  modelbox::Status Init(const std::string &model_entry,
                        std::shared_ptr<modelbox::Configuration> &config,
                        const std::shared_ptr<modelbox::Drivers> &drivers_ptr);
  modelbox::Status GetFlowUnitIO(
      std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc);
  modelbox::Status GetModelType(const std::string &model_entry,
                                mindspore::ModelType &model_type);
  modelbox::Status CheckMindSporeInfo(
      const std::vector<mindspore::MSTensor> &tensor_list,
      const std::vector<std::string> &name_list);
  modelbox::Status CheckMindSporeIO();
  void PrepareInputTensor(
      std::vector<mindspore::MSTensor> &ms_inputs,
      std::vector<std::vector<int64_t>> &new_shapes,
      const std::shared_ptr<modelbox::DataContext> &data_ctx);
  modelbox::Status PrepareOutputTensor(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<mindspore::MSTensor> &ms_outputs,
      std::vector<std::shared_ptr<modelbox::BufferList>> &model_output_lists);
  modelbox::Status PrepareOutputBufferList(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<mindspore::MSTensor> &ms_outputs);

 private:
  std::shared_ptr<modelbox::Device> flowunit_device_;
  std::shared_ptr<mindspore::Context> context_;
  std::shared_ptr<mindspore::Model> model_{nullptr};
  int64_t batch_size_{0};
  struct MindSporeIOList io_list_;
  std::string config_file_;
  std::set<mindspore::DeviceType> device_type_;
  bool model_need_padding_{false};
  size_t padding_batch_size_{0};
  uint32_t config_batch_size_{1};
};

#endif
