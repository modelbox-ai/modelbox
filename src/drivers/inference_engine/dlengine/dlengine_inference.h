/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_DLENGINE_INFERENCE_H_
#define MODELBOX_DLENGINE_INFERENCE_H_

#include "dlengine.h"
#include "modelbox/base/status.h"
#include "modelbox/data_context.h"
#include "modelbox/flowunit.h"
#include "nlohmann/json.hpp"
#include "virtualdriver_inference.h"

class TensorShapeParam {
 public:
  modelbox::Status Init(const std::string &shape);

  modelbox::Status Init(const std::string &min_shape,
                        const std::string &opt_shape,
                        const std::string &max_shape);

  void GenTensorConfig(nlohmann::json &tensor_config);

 private:
  modelbox::Status Parse(const std::string &shape_str,
                         std::vector<size_t> &shape_value);

  bool fix_shape_{false};
  std::vector<size_t> shape_;
  std::vector<size_t> min_shape_;
  std::vector<size_t> opt_shape_;
  std::vector<size_t> max_shape_;
};

class DLEngineInference {
 public:
  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &unit_cfg,
      const std::shared_ptr<modelbox::FlowUnitDesc> &desc,
      const std::string &device_type, int32_t device_id);

  modelbox::Status Infer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  modelbox::Status InitInferInfo(
      const std::shared_ptr<VirtualInferenceFlowUnitDesc> &desc);

  modelbox::Status InitInputInfo(modelbox::FlowUnitInput &input);

  modelbox::Status LoadModel(
      const std::shared_ptr<modelbox::Configuration> &cfg);

  modelbox::Status GenModelConfig(
      const std::shared_ptr<modelbox::Configuration> &cfg,
      nlohmann::json &model_config);

  modelbox::Status PrepareInput(
      dlengine::IInferContext *infer_context,
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      size_t batch_size);

  modelbox::Status PrepareOutput(
      dlengine::IInferContext *infer_context,
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      size_t batch_size);

  bool CheckDataType(dlengine::DataType data_type);

  void SetBufferInfo(const std::shared_ptr<modelbox::BufferList> &buffer_list,
                     dlengine::DataType data_type,
                     const dlengine::DimSize &shape);

  modelbox::Status SetUpTensorShape(dlengine::DimSize &cur_shape,
                                    const dlengine::DimSize &origin_shape,
                                    size_t batch_size);

  size_t SingleTensorSize(const dlengine::DimSize &shape,
                          dlengine::DataType data_type);

  std::string device_type_;
  int32_t device_id_{0};

  std::string model_entry_;
  std::vector<std::string> input_name_list_;
  std::vector<std::shared_ptr<TensorShapeParam>> input_tensor_shape_param_list_;
  std::vector<std::string> output_name_list_;

  std::string backend_zoo_;
  dlengine::IInferer *inferer_{nullptr};
  dlengine::DeviceType infer_device_{dlengine::DeviceType::UNKNOWN};
};

#endif  // MODELBOX_DLENGINE_INFERENCE_H_
