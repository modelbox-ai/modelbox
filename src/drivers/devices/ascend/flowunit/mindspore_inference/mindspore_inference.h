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


#ifndef MODELBOX_MINDSPRORE_ASCEND_INFERENCE_H_
#define MODELBOX_MINDSPRORE_ASCEND_INFERENCE_H_

#include <acl/acl.h>

#include "modelbox/base/configuration.h"
#include "modelbox/data_context.h"
#include "include/api/model.h"

class MindSporeInference {
 public:
  MindSporeInference() = default;
  ~MindSporeInference() = default;

  modelbox::Status Init(const std::string &model_entry,
                      std::shared_ptr<modelbox::Configuration> &config,
                      const std::vector<std::string> &input_name_list,
                      const std::vector<std::string> &output_name_list,
                      const std::vector<std::string> &input_type_list,
                      const std::vector<std::string> &output_type_list,
                      const std::shared_ptr<modelbox::Drivers>& drivers_ptr);
  modelbox::Status Infer(std::shared_ptr<modelbox::DataContext> data_ctx);
  int64_t GetBatchSize() { return batch_size_; };

 private:
  void SetDevice(std::shared_ptr<modelbox::Configuration> &config);
  modelbox::Status GetModelType(const std::string &model_entry,
                              mindspore::ModelType &model_type);
  modelbox::Status CheckMindSporeInfo(
      const std::vector<mindspore::MSTensor> &tensor_list,
      const std::vector<std::string> &name_list,
      const std::vector<std::string> &type_list);
  modelbox::Status CheckMindSporeIO(
      const std::vector<std::string> &input_name_list,
      const std::vector<std::string> &output_name_list,
      const std::vector<std::string> &input_type_list,
      const std::vector<std::string> &output_type_list);
  std::shared_ptr<mindspore::Model> model_;
  int64_t batch_size_{0};
};

#endif
