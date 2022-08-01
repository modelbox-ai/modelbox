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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_H_
#define MODELBOX_FLOWUNIT_INFERENCE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/refcache.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <modelbox/tensor.h>
#include <modelbox/tensor_list.h>
#include <torch/script.h>

#include <typeinfo>

constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *INFERENCE_TYPE = "torch";
constexpr const char *TENSORLIST = "tensorlist";

class TorchInferenceFlowUnitDesc : public modelbox::FlowUnitDesc {
  friend class TorchInferenceFlowUnit;

 public:
  TorchInferenceFlowUnitDesc() = default;
  virtual ~TorchInferenceFlowUnitDesc() = default;

  void SetModelEntry(const std::string model_entry);
  const std::string GetModelEntry();

  std::string model_entry_;
};

class TorchInferenceParam {
 public:
  std::vector<std::string> input_name_list_, output_name_list_;
  std::vector<std::string> input_type_list_, output_type_list_;
  std::vector<modelbox::FlowUnitInput> input_list_;
  std::vector<modelbox::FlowUnitOutput> output_list_;
};

class TorchInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  TorchInferenceFlowUnit();
  virtual ~TorchInferenceFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  bool skip_first_dim_;
  TorchInferenceParam params_;
  torch::jit::script::Module model_;
  modelbox::Status ConvertType(const std::string &type,
                               c10::ScalarType &torch_type);
  modelbox::Status CreateOutputBufferList(
      std::shared_ptr<modelbox::BufferList> &output_buffer_list,
      torch::Tensor &output_tensor, size_t input_size);
  modelbox::Status CreateOutputBufferListFromVector(
      std::shared_ptr<modelbox::BufferList> &output_buffer_list,
      std::vector<torch::Tensor> &output_tensor, size_t input_size);
  modelbox::Status PreProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                              std::vector<torch::jit::IValue> &inputs);
  modelbox::Status SetOutputBufferListMeta(
      const std::vector<torch::Tensor> &output,
      std::shared_ptr<modelbox::BufferList> &output_buf);
  modelbox::Status PostProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               torch::jit::IValue &outputs);

  modelbox::Status InitConfig(
      const std::shared_ptr<modelbox::Configuration> &config);
  modelbox::Status LoadModel(
      const std::string &model_path,
      const std::shared_ptr<modelbox::Configuration> &config);
  void FillInput(
      const std::vector<modelbox::FlowUnitInput> &flowunit_input_list);
  void FillOutput(
      const std::vector<modelbox::FlowUnitOutput> &flowunit_output_list);
  modelbox::Status ChunkTensors(
      const std::vector<torch::Tensor> &output_tensor,
      std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>>
          &chunk_buffers,
      size_t input_size);
  modelbox::Status CreateTorchTensor(
      const std::shared_ptr<modelbox::BufferList> &input_buf,
      const torch::TensorOptions &option, torch::Tensor &input_tensor);
  modelbox::Status CreateTorchTensorList(
      const std::shared_ptr<modelbox::BufferList> &input_buf,
      const torch::TensorOptions &option,
      std::vector<torch::Tensor> &tensor_vec);
  modelbox::Status GetOutputTensorVec(torch::jit::IValue &outputs,
                                      std::vector<torch::Tensor> &output_vector,
                                      int index);
};

class TorchInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  TorchInferenceFlowUnitFactory() = default;
  virtual ~TorchInferenceFlowUnitFactory() = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type);

  std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };
  std::string GetVirtualType() { return INFERENCE_TYPE; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_H_
