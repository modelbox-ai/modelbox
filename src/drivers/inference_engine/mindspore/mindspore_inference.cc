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

#include "mindspore_inference.h"

#include <cstdint>
#include <map>
#include <utility>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "model_decrypt.h"
#include "modelbox/base/status.h"
#include "virtualdriver_inference.h"

static std::map<std::string, mindspore::ModelType> model_type_map{
    {"mindir", mindspore::ModelType::kMindIR},
    {"air", mindspore::ModelType::kAIR},
    {"om", mindspore::ModelType::kOM},
    {"ms", mindspore::ModelType::kMindIR},
    {"onnx", mindspore::ModelType::kONNX}};

static std::map<mindspore::DataType, std::string> data_type_map{
    {mindspore::DataType::kNumberTypeFloat32, "float"},
    {mindspore::DataType::kNumberTypeFloat16, "float16"},
    {mindspore::DataType::kNumberTypeFloat64, "float64"},
    {mindspore::DataType::kNumberTypeInt8, "int8"},
    {mindspore::DataType::kNumberTypeInt32, "int"},
    {mindspore::DataType::kNumberTypeInt16, "int16"},
    {mindspore::DataType::kNumberTypeInt64, "int64"},
    {mindspore::DataType::kNumberTypeUInt8, "uint8"},
    {mindspore::DataType::kNumberTypeUInt16, "uint16"},
    {mindspore::DataType::kNumberTypeUInt32, "uint32"},
    {mindspore::DataType::kNumberTypeUInt64, "uint64"},
    {mindspore::DataType::kNumberTypeBool, "bool"},
    {mindspore::DataType::kObjectTypeString, "str"}};

static std::map<mindspore::DataType, modelbox::ModelBoxDataType>
    data_type_flow_map{
        {mindspore::DataType::kNumberTypeFloat32, modelbox::MODELBOX_FLOAT},
        {mindspore::DataType::kNumberTypeFloat16, modelbox::MODELBOX_HALF},
        {mindspore::DataType::kNumberTypeFloat64, modelbox::MODELBOX_DOUBLE},
        {mindspore::DataType::kNumberTypeInt8, modelbox::MODELBOX_INT8},
        {mindspore::DataType::kNumberTypeInt32, modelbox::MODELBOX_INT32},
        {mindspore::DataType::kNumberTypeInt16, modelbox::MODELBOX_INT16},
        {mindspore::DataType::kNumberTypeInt64, modelbox::MODELBOX_INT64},
        {mindspore::DataType::kNumberTypeUInt8, modelbox::MODELBOX_UINT8},
        {mindspore::DataType::kNumberTypeUInt16, modelbox::MODELBOX_UINT16},
        {mindspore::DataType::kNumberTypeUInt32, modelbox::MODELBOX_UINT32},
        {mindspore::DataType::kNumberTypeUInt64, modelbox::MODELBOX_UINT64},
        {mindspore::DataType::kObjectTypeString, modelbox::MODELBOX_STRING},
        {mindspore::DataType::kNumberTypeBool, modelbox::MODELBOX_BOOL}};

MindSporeInference::~MindSporeInference() {
  model_ = nullptr;
  context_ = nullptr;
}

modelbox::Status MindSporeInference::GetFlowUnitIO(
    struct MindSporeIOList &io_list,
    std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc) {
  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  auto input_desc = unit_desc->GetFlowUnitInput();
  auto output_desc = unit_desc->GetFlowUnitOutput();
  for (auto &input : input_desc) {
    io_list.input_name_list.push_back(input.GetPortName());
    io_list.input_type_list.push_back(input.GetPortType());
    io_list.input_device_list.push_back(input.GetDeviceType());
  }

  for (auto &output : output_desc) {
    io_list.output_name_list.push_back(output.GetPortName());
    io_list.output_type_list.push_back(output.GetPortType());
  }

  if (io_list.input_name_list.empty() || io_list.output_name_list.empty()) {
    MBLOG_ERROR << "Wrong input name [" << io_list.input_name_list.size()
                << "] or output name [" << io_list.output_name_list.size()
                << "] number";
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::GetModelType(
    const std::string &model_entry, mindspore::ModelType &model_type) {
  auto type_vec = modelbox::StringSplit(model_entry, '.');
  if (type_vec.size() == 0) {
    return {modelbox::STATUS_BADCONF, "model entry format is not suitable."};
  }

  auto iter = model_type_map.find(type_vec.back());
  if (iter == model_type_map.end()) {
    model_type = mindspore::ModelType::kUnknownType;
    return {modelbox::STATUS_BADCONF, ""};
  }

  model_type = model_type_map[type_vec.back()];
  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::CheckMindSporeInfo(
    const std::vector<mindspore::MSTensor> &tensor_list,
    const std::vector<std::string> &name_list) {
  if (tensor_list.size() != name_list.size()) {
    auto err_msg = "model port size " + std::to_string(tensor_list.size()) +
                   " does not match for config file port name size " +
                   std::to_string(name_list.size());
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_BADCONF, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::CheckMindSporeIO(
    struct MindSporeIOList &io_list) {
  auto input_tensors = model_->GetInputs();
  auto ret = CheckMindSporeInfo(input_tensors, io_list.input_name_list);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg = "check ms input failed " + ret.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return ret;
  }

  auto output_tensors = model_->GetOutputs();
  ret = CheckMindSporeInfo(output_tensors, io_list.output_name_list);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg = "check ms output failed " + ret.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::Init(
    const std::shared_ptr<mindspore::Context> &mindspore_context,
    const std::string &model_entry,
    std::shared_ptr<modelbox::Configuration> &config,
    struct MindSporeIOList &io_list,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr) {
  context_ = mindspore_context;
  auto device_info_list = context_->MutableDeviceInfo();
  for (const auto &device_info : device_info_list) {
    device_type_.insert(device_info->GetDeviceType());
  }
  context_->SetInterOpParallelNum(std::thread::hardware_concurrency());
  MBLOG_INFO << "set interopparalle num: " << context_->GetInterOpParallelNum();

  mindspore::ModelType mindspore_type = mindspore::ModelType::kMindIR;
  auto ret = GetModelType(model_entry, mindspore_type);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg = "get model type failed " + ret.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return ret;
  }

  mindspore::Status ms_status{mindspore::kSuccess};
  ModelDecryption model_decrypt;
  ret = model_decrypt.Init(model_entry, drivers_ptr, config);
  if (ret != modelbox::STATUS_SUCCESS) {
    return {ret, "init model fail"};
  }

  model_ = std::make_shared<mindspore::Model>();
  if (!config_file_.empty()) {
    ms_status = model_->LoadConfig(config_file_);
    if (ms_status != mindspore::kSuccess) {
      std::string err_msg = "load model config:" + config_file_ +
                            " failed, ret: " + ms_status.ToString();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }

  if (model_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        model_decrypt.GetModelSharedBuffer(model_len);
    if (!modelBuf) {
      return {modelbox::StatusError, "Decrypt model fail"};
    }

    ms_status = model_->Build((const void *)modelBuf.get(), (size_t)model_len,
                              mindspore_type, context_);
  } else if (model_decrypt.GetModelState() ==
             ModelDecryption::MODEL_STATE_PLAIN) {
    ms_status = model_->Build(model_entry, mindspore_type, context_);
  }
  if (ms_status != mindspore::kSuccess) {
    std::string err_msg =
        "model init failed, code: " + std::to_string(ms_status.StatusCode()) +
        ", msg: " + ms_status.ToString();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  ret = CheckMindSporeIO(io_list);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg = "input or output info got error, " + ret.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_BADCONF, err_msg};
  }

  for (auto &input_tensor : model_->GetInputs()) {
    std::stringstream ss;
    ss << "input name:" << input_tensor.Name() << ", shape: [";
    for (size_t i = 0; i < input_tensor.Shape().size(); ++i) {
      ss << input_tensor.Shape()[i];
      if (i != input_tensor.Shape().size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    MBLOG_INFO << ss.str();
  }

  io_list_ = io_list;
  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::Open(
    const std::shared_ptr<modelbox::Configuration> &opts,
    std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    std::shared_ptr<mindspore::Context> context) {
  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  auto config = unit_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  merge_config->Add(*config);
  merge_config->Add(*opts);

  struct MindSporeIOList iolist;

  auto ret = GetFlowUnitIO(iolist, flowunit_desc);
  if (ret != modelbox::STATUS_OK) {
    return ret;
  }

  std::string config_file_ = merge_config->GetString("config.config_file");
  if (!modelbox::IsAbsolutePath(config_file_)) {
    auto relpath =
        modelbox::GetDirName(unit_desc->GetDriverDesc()->GetFilePath());
    config_file_ = relpath + "/" + config_file_;
  }

  ret = Init(context, unit_desc->GetModelEntry(), merge_config, iolist,
             drivers_ptr);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init inference failed, " << ret;
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::Infer(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto ms_inputs = model_->GetInputs();
  std::vector<std::vector<int64_t>> new_shapes;
  PrepareInputTensor(ms_inputs, new_shapes, data_ctx);

  auto ms_ret = model_->Resize(ms_inputs, new_shapes);
  if (ms_ret != mindspore::kSuccess) {
    auto err_msg = "mindspore resize failed, ret " +
                   std::to_string(ms_ret.StatusCode()) +
                   " err_msg: " + ms_ret.ToString();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  for (const auto &input : ms_inputs) {
    MBLOG_DEBUG << "input portname: " << input.Name()
                << ", batch size: " << input.Shape()[0]
                << ", data size: " << input.DataSize()
                << ", element num: " << input.ElementNum();
  }

  auto ms_outputs = model_->GetOutputs();
  auto ret = PrepareOutputTensor(data_ctx, ms_outputs);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg = "prepare output tensor failed, err_msg: " + ret.Errormsg();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  ms_ret = model_->Predict(ms_inputs, &ms_outputs);
  if (ms_ret != mindspore::kSuccess) {
    auto err_msg = "mindspore inference failed, ret " +
                   std::to_string(ms_ret.StatusCode()) +
                   " err_msg: " + ms_ret.ToString();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  ret = PrepareOutputBufferList(data_ctx, ms_outputs);
  if (ret != modelbox::STATUS_OK) {
    auto err_msg =
        "prepare output bufferlist failed, err_msg: " + ret.Errormsg();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

void MindSporeInference::PrepareInputTensor(
    std::vector<mindspore::MSTensor> &ms_inputs,
    std::vector<std::vector<int64_t>> &new_shapes,
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  for (size_t i = 0; i < ms_inputs.size(); ++i) {
    auto name = ms_inputs[i].Name();
    auto input_shape = ms_inputs[i].Shape();
    auto portname = io_list_.input_name_list[i];
    auto input_buffer_list = data_ctx->Input(portname);
    MBLOG_DEBUG << "input_buffer_list: " << portname << ", model port: " << name
                << ", size: " << input_buffer_list->Size()
                << ", bytes:" << input_buffer_list->GetBytes();
    // cpu is host data
    if (io_list_.input_device_list[i] == "cpu") {
      ms_inputs[i].SetData(const_cast<void *>(input_buffer_list->ConstData()),
                           false);
    } else {
      ms_inputs[i].SetDeviceData(
          const_cast<void *>(input_buffer_list->ConstData()));
    }
    // set current batch size
    input_shape[0] = input_buffer_list->Size();
    MBLOG_DEBUG << "input name: " << name << " shape: ";
    for (auto &item : input_shape) {
      MBLOG_DEBUG << item;
    }
    new_shapes.push_back(input_shape);
  }
}

modelbox::Status MindSporeInference::PrepareOutputTensor(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<mindspore::MSTensor> &ms_outputs) {
  if (device_type_.find(mindspore::DeviceType::kGPU) == device_type_.end()) {
    // only gpu support set output device data
    return modelbox::STATUS_OK;
  }

  // set output mem
  for (size_t i = 0; i < ms_outputs.size(); ++i) {
    auto name = ms_outputs[i].Name();
    auto portname = io_list_.output_name_list[i];
    if (ms_outputs[i].Shape()[0] == 0) {
      auto err_msg = "output_tensor " + portname + " first dim is zero";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto output_buffer_list = data_ctx->Output(portname);
    output_buffer_list->Build(std::vector<size_t>(
        ms_outputs[i].Shape()[0],
        ms_outputs[i].DataSize() / ms_outputs[i].Shape()[0]));
    ms_outputs[i].SetDeviceData(output_buffer_list->MutableData());
    MBLOG_DEBUG << "output port name: " << portname
                << ", batch size: " << ms_outputs[i].Shape()[0]
                << ", data size: " << ms_outputs[i].DataSize()
                << ", element num: " << ms_outputs[i].ElementNum()
                << ", datatype: " << (int)ms_outputs[i].DataType();

    auto tensor_shape = ms_outputs[i].Shape();
    std::vector<size_t> output_shape;
    tensor_shape[0] = 1;
    MBLOG_DEBUG << "output name:" << name << ", shape: ";
    for (auto &item : tensor_shape) {
      output_shape.push_back(item);
      MBLOG_DEBUG << item;
    }
    output_buffer_list->Set("shape", output_shape);
    output_buffer_list->Set("type",
                            data_type_flow_map[ms_outputs[i].DataType()]);
  }
  return modelbox::STATUS_OK;
}

modelbox::Status MindSporeInference::PrepareOutputBufferList(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<mindspore::MSTensor> &ms_outputs) {
  if (device_type_.find(mindspore::DeviceType::kGPU) != device_type_.end()) {
    // gpu infer has been set output buffer list
    return modelbox::STATUS_OK;
  }

  for (size_t i = 0; i < ms_outputs.size(); ++i) {
    auto portname = io_list_.output_name_list[i];
    auto output_buffer_list = data_ctx->Output(portname);
    MBLOG_DEBUG << "output port name: " << portname
                << ", batch size: " << ms_outputs[i].Shape()[0]
                << ", data size: " << ms_outputs[i].DataSize()
                << ", element num: " << ms_outputs[i].ElementNum()
                << ", datatype: " << (int)ms_outputs[i].DataType();
    if (ms_outputs[i].Shape()[0] == 0) {
      auto err_msg = "ms_outputs " + portname + " first dim is zero";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    std::vector<size_t> shape_size(
        ms_outputs[i].Shape()[0],
        ms_outputs[i].DataSize() / ms_outputs[i].Shape()[0]);
    auto status = output_buffer_list->BuildFromHost(
        shape_size, ms_outputs[i].MutableData(), ms_outputs[i].DataSize());
    if (status != modelbox::STATUS_OK) {
      auto err_msg =
          "output buffer list build from host failed " + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto tensor_shape = ms_outputs[i].Shape();
    std::vector<size_t> output_shape;
    tensor_shape[0] = 1;
    MBLOG_DEBUG << "output shape: ";
    for (const auto &item : tensor_shape) {
      output_shape.push_back(item);
      MBLOG_DEBUG << item;
    }
    output_buffer_list->Set("shape", output_shape);
    output_buffer_list->Set("type",
                            data_type_flow_map[ms_outputs[i].DataType()]);
  }
  return modelbox::STATUS_OK;
}
