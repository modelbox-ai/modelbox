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

#ifndef MODELBOX_TENSORFLOW_INFERENCE_COMMON_H_
#define MODELBOX_TENSORFLOW_INFERENCE_COMMON_H_

#include <dlfcn.h>
#include <modelbox/base/device.h>
#include <modelbox/base/refcache.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <modelbox/tensor.h>
#include <modelbox/tensor_list.h>

#include <typeinfo>

#include "tensorflow/c/c_api.h"
#include "tensorflow_inference_plugin.h"

constexpr const char *INFERENCE_TYPE = "tensorflow";
constexpr const char *TAGS = "serve";

class InferenceTensorflowParams {
 public:
  InferenceTensorflowParams()
      : graph(nullptr), session(nullptr), options(nullptr), status(nullptr){};
  virtual ~InferenceTensorflowParams(){};

  modelbox::Status Clear();

  std::vector<std::string> input_name_list_, output_name_list_;
  std::vector<std::string> input_type_list_, output_type_list_;
  std::vector<TF_Output> input_op_list, output_op_list;

  int device{0};

  // Tensorflow Options
  TF_Graph *graph;
  TF_Session *session;
  TF_SessionOptions *options;
  TF_Status *status;
  std::vector<uint8_t> config_proto_binary_ = {
      0x32, 0xe,  0x9,  0xcd, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
      0xec, 0x3f, 0x20, 0x1,  0x2a, 0x1,  0x30, 0x38, 0x1};
};

using TensorflowProcess = std::function<modelbox::Status(
    std::shared_ptr<modelbox::DataContext>, std::vector<TF_Tensor *> &)>;

class InferenceTensorflowFlowUnitDesc : public modelbox::FlowUnitDesc {
  friend class InferenceTensorflowFlowUnit;

 public:
  InferenceTensorflowFlowUnitDesc() = default;
  virtual ~InferenceTensorflowFlowUnitDesc() = default;

  void SetModelEntry(const std::string model_entry);
  const std::string GetModelEntry();

  std::string model_entry_;
};

class InferenceTensorflowFlowUnit : public modelbox::FlowUnit {
 public:
  InferenceTensorflowFlowUnit();
  virtual ~InferenceTensorflowFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status PreProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                              std::vector<TF_Tensor *> &input_tf_tensor_list);

  modelbox::Status PostProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               std::vector<TF_Tensor *> &output_tf_tensor_list);
  modelbox::Status SetUpInferencePlugin(
      std::shared_ptr<modelbox::Configuration> config);
  modelbox::Status SetUpDynamicLibrary(
      std::shared_ptr<modelbox::Configuration> config);

  modelbox::Status ReadBufferFromFile(const std::string &file, TF_Buffer *buf);
  modelbox::Status InitConfig(
      const std::shared_ptr<modelbox::Configuration> &fu_config);
  modelbox::Status LoadGraph(const std::string &model_path);
  modelbox::Status Inference(const std::vector<TF_Tensor *> &input_tensor_list,
                             std::vector<TF_Tensor *> &output_tensor_list);
  modelbox::Status ConvertType(const std::string &type, TF_DataType &TFType);
  modelbox::Status ClearTensor(std::vector<TF_Tensor *> &input_tensor_list,
                               std::vector<TF_Tensor *> &output_tensor_list);
  modelbox::Status CreateOutputBufferList(
      std::shared_ptr<modelbox::BufferList> &output_buffer_list,
      const std::vector<size_t> &shape_vector, void *tensor_data,
      size_t tensor_byte, int index);
  modelbox::Status GetTFOperation(const std::string &name, TF_Output &op);
  modelbox::Status FillInput(
      const std::vector<modelbox::FlowUnitInput> &flowunit_input_list);
  modelbox::Status FillOutput(
      const std::vector<modelbox::FlowUnitOutput> &flowunit_output_list);
  modelbox::Status NewSession(bool is_save_model,
                              const std::string &model_entry);
  bool IsSaveModelType(const std::string &model_path);
  InferenceTensorflowParams params_;
  std::string plugin_;
  void *driver_handler_{nullptr};

  std::shared_ptr<InferencePlugin> inference_plugin_{nullptr};
  TensorflowProcess pre_process_{nullptr};
  TensorflowProcess post_process_{nullptr};
}; 

#endif  // MODELBOX_TENSORFLOW_INFERENCE_COMMON_H_
