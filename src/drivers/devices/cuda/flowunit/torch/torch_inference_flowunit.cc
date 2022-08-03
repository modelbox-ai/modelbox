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

#include "torch_inference_flowunit.h"

#include <cuda_runtime.h>
#include <model_decrypt.h>
#include <modelbox/base/crypto.h>

#include <fstream>
#include <mutex>

#include "modelbox/device/cuda/device_cuda.h"
#include "modelbox/type.h"
#include "virtualdriver_inference.h"

static std::mutex torch_load_mutex;

static std::map<std::string, c10::ScalarType> type_map = {
    {"FLOAT", torch::kFloat32},  {"DOUBLE", torch::kFloat64},
    {"INT", torch::kInt32},      {"UINT8", torch::kUInt8},
    {"LONG", torch::kInt64},     {"INT64", torch::kInt64},
    {"FLOAT16", torch::kFloat16}};

static std::map<c10::ScalarType, modelbox::ModelBoxDataType> t2a_map = {
    {torch::kFloat32, modelbox::MODELBOX_FLOAT},
    {torch::kFloat64, modelbox::MODELBOX_DOUBLE},
    {torch::kInt32, modelbox::MODELBOX_INT32},
    {torch::kUInt8, modelbox::MODELBOX_UINT8},
    {torch::kInt64, modelbox::MODELBOX_INT64},
    {torch::kFloat16, modelbox::MODELBOX_HALF}};

TorchInferenceFlowUnit::TorchInferenceFlowUnit() = default;
TorchInferenceFlowUnit::~TorchInferenceFlowUnit() = default;

void TorchInferenceFlowUnit::FillInput(
    const std::vector<modelbox::FlowUnitInput> &flowunit_input_list) {
  for (auto const &input_item : flowunit_input_list) {
    auto input_name = input_item.GetPortName();
    auto input_type = input_item.GetPortType();
    params_.input_name_list_.push_back(input_name);
    params_.input_type_list_.push_back(input_type);
    params_.input_list_.push_back(input_item);
  }
}

void TorchInferenceFlowUnit::FillOutput(
    const std::vector<modelbox::FlowUnitOutput> &flowunit_output_list) {
  for (auto const &output_item : flowunit_output_list) {
    auto output_name = output_item.GetPortName();
    auto output_type = output_item.GetPortType();
    params_.output_name_list_.push_back(output_name);
    params_.output_type_list_.push_back(output_type);
    params_.output_list_.push_back(output_item);
  }
}

modelbox::Status TorchInferenceFlowUnit::LoadModel(
    const std::string &model_path,
    const std::shared_ptr<modelbox::Configuration> &config) {
  std::lock_guard<std::mutex> lck(torch_load_mutex);
  try {
    MBLOG_DEBUG << "model_path: " << model_path;
    auto drivers_ptr = GetBindDevice()->GetDeviceManager()->GetDrivers();

    c10::Device device(c10::kCUDA, dev_id_);
    ModelDecryption torch_decrypt;
    torch_decrypt.Init(model_path, drivers_ptr, config);
    // use GetModelState to check err, so donot need check Init ret
    if (torch_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
      int64_t model_len = 0;
      std::shared_ptr<uint8_t> modelBuf =
          torch_decrypt.GetModelSharedBuffer(model_len);
      if (!modelBuf) {
        return {modelbox::STATUS_FAULT, "Decrypt model fail"};
      }
      std::stringstream modelStream;
      modelStream.rdbuf()->pubsetbuf((char *)modelBuf.get(), model_len);
      model_ = torch::jit::load(modelStream, device);
    } else if (torch_decrypt.GetModelState() ==
               ModelDecryption::MODEL_STATE_PLAIN) {
      model_ = torch::jit::load(model_path, device);
    } else {
      return {modelbox::STATUS_FAULT, "open onnx model file fail"};
    }
  } catch (const c10::Error &e) {
    auto err_msg = "loading model " + model_path +
                   " failed, c10 error: " + e.msg() +
                   "\n backtrace: " + e.backtrace();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  } catch (const std::exception &e) {
    auto err_msg = "other loading error, " + std::string(e.what());
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
  MBLOG_DEBUG << "model loads success.";
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::InitConfig(
    const std::shared_ptr<modelbox::Configuration> &config) {
  auto inference_desc_ =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
          this->GetFlowUnitDesc());
  const std::vector<modelbox::FlowUnitInput> &flowunit_input_list =
      inference_desc_->GetFlowUnitInput();
  const std::vector<modelbox::FlowUnitOutput> &flowunit_output_list =
      inference_desc_->GetFlowUnitOutput();

  std::string model_path = inference_desc_->GetModelEntry();

  auto status = LoadModel(model_path, config);
  if (modelbox::STATUS_OK != status) {
    auto err_msg =
        "could not load inference graph, err: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  FillInput(flowunit_input_list);
  FillOutput(flowunit_output_list);

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  skip_first_dim_ = opts->GetBool("skip_first_dim", false);

  auto inference_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto config = inference_desc->GetConfiguration();
  if (config == nullptr) {
    return {modelbox::STATUS_BADCONF, "inference config is invalid."};
  }

  auto merge_config = std::make_shared<modelbox::Configuration>();
  merge_config->Add(*config);
  merge_config->Add(*opts);
  modelbox::Status status = InitConfig(merge_config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "init config failed: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::ConvertType(
    const std::string &type, c10::ScalarType &torch_type) {
  if (type_map.find(type) == type_map.end()) {
    return {modelbox::STATUS_FAULT, "unsupported type " + type};
  }

  torch_type = type_map[type];
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::CreateTorchTensor(
    const std::shared_ptr<modelbox::BufferList> &input_buf,
    const torch::TensorOptions &option, torch::Tensor &input_tensor) {
  std::vector<size_t> buffer_shape;
  auto result = input_buf->At(0)->Get("shape", buffer_shape);
  if (!result) {
    auto err_msg = "the input buffer don't have meta shape.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_OK, err_msg};
  }

  std::vector<int64_t> shape_vec;
  if (!skip_first_dim_) {
    shape_vec.emplace_back(static_cast<int64_t>(input_buf->Size()));
  }
  copy(buffer_shape.begin(), buffer_shape.end(), back_inserter(shape_vec));

  at::IntArrayRef shape(shape_vec);
  input_tensor = torch::from_blob(const_cast<void *>(input_buf->ConstData()),
                                  shape, option);
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status TorchInferenceFlowUnit::CreateTorchTensorList(
    const std::shared_ptr<modelbox::BufferList> &input_buf,
    const torch::TensorOptions &option,
    std::vector<torch::Tensor> &tensor_vec) {
  std::vector<std::vector<size_t>> buffer_shape_vec;
  modelbox::ModelBoxDataType buffer_type;
  auto result = input_buf->At(0)->Get("shape", buffer_shape_vec);
  if (!result) {
    auto err_msg = "the input buffer don't have meta shape.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  result = input_buf->At(0)->Get("type", buffer_type);
  if (!result) {
    auto err_msg = "the input buffer don't have meta type.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::vector<size_t> bytes{0};
  size_t acc_bytes = 0;
  for (auto &buffer_shape : buffer_shape_vec) {
    auto byte = std::accumulate(buffer_shape.begin(), buffer_shape.end(),
                                (size_t)0, std::multiplies<size_t>()) *
                modelbox::GetDataTypeSize(buffer_type);
    acc_bytes += byte;
    bytes.emplace_back(acc_bytes);
  }

  for (size_t i = 0; i < buffer_shape_vec.size(); i++) {
    std::vector<torch::Tensor> concat_tensor_vec;
    for (size_t j = 0; j < input_buf->Size(); ++j) {
      std::vector<int64_t> shape_vec;
      if (!skip_first_dim_) {
        shape_vec.emplace_back(static_cast<int64_t>(input_buf->Size()));
      }
      copy(buffer_shape_vec[i].begin(), buffer_shape_vec[i].end(),
           back_inserter(shape_vec));

      at::IntArrayRef shape(shape_vec);
      torch::Tensor tensor_item = torch::from_blob(
          (char *)(const_cast<void *>(input_buf->At(j)->ConstData())) +
              bytes[i],
          shape, option);
      concat_tensor_vec.emplace_back(tensor_item);
    }
    torch::TensorList concat_tensorlist(concat_tensor_vec);
    auto concat_tensor = torch::cat(concat_tensorlist);
    tensor_vec.emplace_back(concat_tensor);
    concat_tensor_vec.clear();
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::PreProcess(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<torch::jit::IValue> &inputs) {
  int index = 0;
  modelbox::Status status;
  for (const auto &input_name : params_.input_name_list_) {
    const auto input_buf = data_ctx->Input(input_name);

    std::string type = params_.input_type_list_[index];
    std::string torch_set_type =
        params_.input_list_[index++].GetProperity("torch_type");
    std::transform(type.begin(), type.end(), type.begin(), ::toupper);
    c10::ScalarType torch_type;
    status = ConvertType(type, torch_type);
    if (status != modelbox::STATUS_OK) {
      return {status, "input type convert failed."};
    }

    torch::TensorOptions option = torch::TensorOptions()
                                      .device(torch::kCUDA, dev_id_)
                                      .layout(torch::kStrided)
                                      .dtype(torch_type);

    if (torch_set_type.empty()) {
      torch::Tensor input_tensor;
      status = CreateTorchTensor(input_buf, option, input_tensor);
      if (status != modelbox::STATUS_SUCCESS) {
        auto err_msg =
            "create torch tensor failed, err: " + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {modelbox::STATUS_FAULT, err_msg};
      }

      inputs.emplace_back(input_tensor);
      continue;
    }

    if (torch_set_type == TENSORLIST) {
      std::vector<torch::Tensor> tensor_vec;
      status = CreateTorchTensorList(input_buf, option, tensor_vec);
      if (status != modelbox::STATUS_SUCCESS) {
        auto err_msg =
            "create torch tensor list failed, err: " + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {modelbox::STATUS_FAULT, err_msg};
      }

      torch::TensorList input_tensorlist(tensor_vec);
      inputs.emplace_back(input_tensorlist);
      continue;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::ChunkTensors(
    const std::vector<torch::Tensor> &output_tensor,
    std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>> &chunk_buffers,
    size_t input_size) {
  modelbox::Status status = modelbox::STATUS_SUCCESS;
  for (size_t tensor_index = 0; tensor_index < output_tensor.size();
       tensor_index++) {
    auto chunk_tensors =
        torch::chunk(output_tensor[tensor_index], input_size, 1);

    for (size_t chunk_index = 0; chunk_index < chunk_tensors.size();
         chunk_index++) {
      std::shared_ptr<modelbox::Buffer> buffer =
          std::make_shared<modelbox::Buffer>(GetBindDevice());
      auto tensor = chunk_tensors[chunk_index];
      status = buffer->Build(tensor.data_ptr(), tensor.nbytes(),
                             [tensor](void *ptr) {});
      if (status != modelbox::STATUS_OK) {
        auto err_msg = "output buffer builds error: " + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {modelbox::STATUS_FAULT, err_msg};
      }

      if (tensor_index == 0) {
        chunk_buffers.emplace_back();
      }

      chunk_buffers[chunk_index].push_back(buffer);
    }
  }

  return status;
}

modelbox::Status TorchInferenceFlowUnit::CreateOutputBufferList(
    std::shared_ptr<modelbox::BufferList> &output_buffer_list,
    torch::Tensor &output_tensor, size_t input_size) {
  // TODO: build buffer from flowunit device memory by trasfer device ptr
  torch::Tensor output = output_tensor;
  if (output_tensor.is_cuda()) {
    output = output_tensor.cpu();
  }
  auto tensor_byte = output.nbytes();
  auto tensor_data = output.data_ptr();

  std::vector<size_t> shape_vector;
  if (skip_first_dim_) {
    shape_vector.emplace_back(tensor_byte);
  } else {
    if (input_size == 0) {
      return {modelbox::STATUS_FAULT, "Divisor is zero"};
    }
    auto single_byte = tensor_byte / input_size;
    for (size_t i = 0; i < input_size; ++i) {
      shape_vector.emplace_back(single_byte);
    }
  }

  modelbox::Status status;
  status =
      output_buffer_list->BuildFromHost(shape_vector, tensor_data, tensor_byte);

  if (status != modelbox::STATUS_OK) {
    auto err_msg = "output buffer list builds error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::CreateOutputBufferListFromVector(
    std::shared_ptr<modelbox::BufferList> &output_buffer_list,
    std::vector<torch::Tensor> &output_tensor, size_t input_size) {
  MBLOG_DEBUG << "output_tensor size: " << output_tensor.size();

  std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>>
      chunk_torch_tensors;
  auto status = ChunkTensors(output_tensor, chunk_torch_tensors, input_size);
  if (status != modelbox::STATUS_SUCCESS) {
    auto err_msg = "chunk tensors failed, err: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  for (auto &chunk_tensor : chunk_torch_tensors) {
    auto buffer_list = std::make_shared<modelbox::BufferList>(chunk_tensor);
    auto ret = buffer_list->MakeContiguous();
    if (!ret) {
      MBLOG_ERROR << "buffer list merge failed: " << ret;
      return modelbox::STATUS_FAULT;
    }

    auto dev_mem = buffer_list->GetDeviceMemory();
    auto merge = std::make_shared<modelbox::Buffer>(dev_mem);
    output_buffer_list->PushBack(merge);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::SetOutputBufferListMeta(
    const std::vector<torch::Tensor> &output,
    std::shared_ptr<modelbox::BufferList> &output_buf) {
  modelbox::ModelBoxDataType modelbox_type = modelbox::MODELBOX_TYPE_INVALID;
  std::vector<std::vector<size_t>> output_shape_vec;
  for (auto &item : output) {
    auto torch_type = item.scalar_type();
    auto iter = t2a_map.find(torch_type);
    if (iter == t2a_map.end()) {
      return {modelbox::STATUS_NOTSUPPORT, "unsupport output type."};
    }
    modelbox_type = t2a_map[torch_type];

    auto sizes = item.sizes();
    std::vector<size_t> output_shape;
    for (long size : sizes) {
      output_shape.push_back((size_t)size);
    }
    output_shape_vec.emplace_back(output_shape);
  }

  output_buf->Set("type", modelbox_type);
  if (output_shape_vec.size() == 1) {
    output_buf->Set("shape", output_shape_vec[0]);
    return modelbox::STATUS_OK;
  }

  output_buf->Set("shape", output_shape_vec);
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::GetOutputTensorVec(
    torch::jit::IValue &outputs, std::vector<torch::Tensor> &output_vector,
    int index) {
  if (!outputs.isTensor() && !outputs.isTuple()) {
    auto err_msg = "unsupported torch inference output type.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (outputs.isTensor()) {
    // single output
    output_vector.push_back(outputs.toTensor());
    return modelbox::STATUS_OK;
  }

  // multi outputs
  auto tmp_output = outputs.toTuple()->elements()[index];
  if (tmp_output.isTensor()) {
    output_vector.push_back(tmp_output.toTensor());
  } else if (tmp_output.isTensorList()) {
    // one of the outputs is tensorlist
    output_vector = tmp_output.toTensorVector();
  } else if (tmp_output.isTuple()) {
    // one of the outputs is also tuple, only fall into next layer.
    auto tmp_output_value = tmp_output.toTuple()->elements();
    MBLOG_DEBUG << "size: " << tmp_output_value.size();
    for (auto &output_value : tmp_output_value) {
      if (output_value.isTensorList()) {
        output_vector = output_value.toTensorVector();
      }
    }
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::PostProcess(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    torch::jit::IValue &outputs) {
  int index = 0;
  for (const auto &output_name : params_.output_name_list_) {
    std::vector<torch::Tensor> output_vector;
    MBLOG_DEBUG << "output name:\t" << output_name;

    auto status = GetOutputTensorVec(outputs, output_vector, index);
    if (status != modelbox::STATUS_OK) {
      auto err_msg =
          "get output tensor vect failed, err: " + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto output_buf = data_ctx->Output(output_name);
    auto size = data_ctx->Input(params_.input_name_list_[0])->Size();
    if (size == 0) {
      auto err_msg = "input size is 0 bytes";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    MBLOG_DEBUG << "input size: " << size;
    if (output_vector.size() == 1) {
      auto status = CreateOutputBufferList(output_buf, output_vector[0], size);
      if (status != modelbox::STATUS_OK) {
        auto err_msg =
            "CreateOutputBufferList single failed." + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {modelbox::STATUS_FAULT, err_msg};
      }
    } else {
      auto status =
          CreateOutputBufferListFromVector(output_buf, output_vector, size);
      if (status != modelbox::STATUS_OK) {
        auto err_msg =
            "CreateOutputBufferList vector failed." + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {modelbox::STATUS_FAULT, err_msg};
      }
    }

    status = SetOutputBufferListMeta(output_vector, output_buf);
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "SetOutputBufferListMeta failed." + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
    index++;
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::vector<torch::jit::IValue> inputs;
  modelbox::Status status = PreProcess(data_ctx, inputs);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "torch inference preprocess failed, error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  torch::jit::IValue outputs;
  try {
    outputs = model_.forward(inputs);
    MBLOG_DEBUG << "output data: " << outputs;
  } catch (const c10::Error &e) {
    auto err_msg = "model inference error, " + e.msg();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  } catch (const std::exception &e) {
    auto err_msg = "other inference error, " + std::string(e.what());
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = PostProcess(data_ctx, outputs);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "torch inference postprocess failed, error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TorchInferenceFlowUnit::Close() { return modelbox::STATUS_OK; }

void TorchInferenceFlowUnitDesc::SetModelEntry(const std::string &model_entry) {
  model_entry_ = model_entry;
}

const std::string TorchInferenceFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}

std::shared_ptr<modelbox::FlowUnit>
TorchInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<TorchInferenceFlowUnit>();
};
