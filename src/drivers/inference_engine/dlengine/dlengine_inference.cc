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

#include "dlengine_inference.h"

#include <unordered_map>
#include <utility>

static const std::unordered_map<std::string, dlengine::DeviceType>
    g_device_type_map{{"cpu", dlengine::kCPU},
                      {"cuda", dlengine::kCUDA},
                      {"ascend", dlengine::kASCEND}};

static const std::unordered_map<dlengine::DataType, modelbox::ModelBoxDataType>
    g_dlengine_to_mb_type_map{
        {dlengine::DataType::FLOAT, modelbox::MODELBOX_FLOAT},
        {dlengine::DataType::INT8, modelbox::MODELBOX_INT8},
        {dlengine::DataType::INT32, modelbox::MODELBOX_INT32}};

static const std::unordered_map<modelbox::ModelBoxDataType, dlengine::DataType>
    g_mb_to_dlengine_type_map{
        {modelbox::MODELBOX_FLOAT, dlengine::DataType::FLOAT},
        {modelbox::MODELBOX_INT8, dlengine::DataType::INT8},
        {modelbox::MODELBOX_INT32, dlengine::DataType::INT32}};

static const std::unordered_map<dlengine::DataType, size_t>
    g_dlengine_type_size_map{{dlengine::DataType::FLOAT, sizeof(float)},
                             {dlengine::DataType::INT8, sizeof(int8_t)},
                             {dlengine::DataType::INT32, sizeof(int32_t)}};

modelbox::Status TensorShapeParam::Init(const std::string &shape) {
  fix_shape_ = true;
  return Parse(shape, shape_);
}

modelbox::Status TensorShapeParam::Init(const std::string &min_shape,
                                        const std::string &opt_shape,
                                        const std::string &max_shape) {
  fix_shape_ = false;
  auto ret = Parse(min_shape, min_shape_);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "parse min shape failed";
    return ret;
  }

  ret = Parse(opt_shape, opt_shape_);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "parse opt shape failed";
    return ret;
  }

  ret = Parse(max_shape, max_shape_);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "parse max shape failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TensorShapeParam::Parse(const std::string &shape_str,
                                         std::vector<size_t> &shape_value) {
  auto format_shape_str = shape_str;
  std::transform(format_shape_str.begin(), format_shape_str.end(),
                 format_shape_str.begin(),
                 [](int c) { return std::tolower(c); });
  auto dims = modelbox::StringSplit(format_shape_str, 'x');
  if (dims.empty()) {
    MBLOG_ERROR << "shape [" << shape_str
                << "] format error, it should be like nxcxhxw";
    return modelbox::STATUS_BADCONF;
  }

  for (const auto &dim : dims) {
    try {
      auto trim_dim = dim;
      trim_dim.erase(0, trim_dim.find_first_not_of(' '));
      trim_dim.erase(trim_dim.find_last_not_of(' ') + 1);
      auto v = std::stoul(trim_dim);
      shape_value.push_back(v);
    } catch (const std::exception &e) {
      MBLOG_ERROR << "shape [" << shape_str
                  << "] format error, it should be like nxcxhxw, detail: "
                  << e.what();
      return modelbox::STATUS_BADCONF;
    }
  }

  return modelbox::STATUS_OK;
}

void TensorShapeParam::GenTensorConfig(nlohmann::json &tensor_config) {
  if (fix_shape_) {
    tensor_config["shape"] = shape_;
    return;
  }

  tensor_config["min_shape"] = min_shape_;
  tensor_config["opt_shape"] = opt_shape_;
  tensor_config["max_shape"] = max_shape_;
}

modelbox::Status DLEngineInference::Init(
    const std::shared_ptr<modelbox::Configuration> &unit_cfg,
    const std::shared_ptr<modelbox::FlowUnitDesc> &desc,
    const std::string &device_type, int32_t device_id) {
  device_type_ = device_type;
  auto device_type_item = g_device_type_map.find(device_type);
  if (device_type_item == g_device_type_map.end()) {
    MBLOG_ERROR << "not support device type " << device_type;
    return modelbox::STATUS_BADCONF;
  }

  infer_device_ = device_type_item->second;
  device_id_ = device_id;

  auto infer_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(desc);
  if (infer_desc == nullptr) {
    MBLOG_ERROR << "cast virtual inference desc failed, flowunit desc is null";
    return modelbox::STATUS_FAULT;
  }

  auto infer_cfg = infer_desc->GetConfiguration();
  if (infer_cfg == nullptr) {
    MBLOG_ERROR << "infer description get config failed";
    return modelbox::STATUS_FAULT;
  }

  auto merge_config = std::make_shared<modelbox::Configuration>();
  merge_config->Add(*infer_cfg);
  merge_config->Add(*unit_cfg);

  auto ret = InitInferInfo(infer_desc);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "init infer info failed, ret " << ret;
    return ret;
  }

  ret = LoadModel(merge_config);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "load model failed, ret " << ret;
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::Infer(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto infer_context = inferer_->GetInferContext(infer_device_);
  Defer { delete infer_context; };

  auto batch_size = data_ctx->Input(input_name_list_[0])->Size();

  auto ret = PrepareInput(infer_context, data_ctx, batch_size);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "prepare input for dlengine failed, ret " << ret;
    return ret;
  }

  ret = PrepareOutput(infer_context, data_ctx, batch_size);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "prepare output for dlengine failed, ret " << ret;
    return ret;
  }

  auto dl_ret = inferer_->Run(infer_context);
  if (!dl_ret) {
    MBLOG_ERROR << "dlengine infer failed, see log for detail";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::InitInferInfo(
    const std::shared_ptr<VirtualInferenceFlowUnitDesc> &desc) {
  model_entry_ = desc->GetModelEntry();
  if (model_entry_.empty()) {
    MBLOG_ERROR << "model entry is empty";
    return modelbox::STATUS_BADCONF;
  }

  auto input_desc_list = desc->GetFlowUnitInput();
  auto output_desc_list = desc->GetFlowUnitOutput();
  for (auto &input : input_desc_list) {
    auto ret = InitInputInfo(input);
    if (ret != modelbox::STATUS_OK) {
      MBLOG_ERROR << "init input " << input.GetPortName() << " failed";
      return ret;
    }
  }

  if (input_name_list_.empty()) {
    MBLOG_ERROR << "input name list is empty for model " << model_entry_;
    return modelbox::STATUS_BADCONF;
  }

  for (auto &output : output_desc_list) {
    output_name_list_.push_back(output.GetPortName());
  }

  if (output_name_list_.empty()) {
    MBLOG_ERROR << "output name list is empty for model " << model_entry_;
    return modelbox::STATUS_OK;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::InitInputInfo(
    modelbox::FlowUnitInput &input) {
  input_name_list_.push_back(input.GetPortName());
  auto shape_str = input.GetProperity("shape");
  if (!shape_str.empty()) {
    auto tensor_shape_param = std::make_shared<TensorShapeParam>();
    auto ret = tensor_shape_param->Init(shape_str);
    if (ret != modelbox::STATUS_OK) {
      MBLOG_ERROR << "input port " << input.GetPortName()
                  << " config wrong shape";
      return modelbox::STATUS_BADCONF;
    }

    input_tensor_shape_param_list_.push_back(tensor_shape_param);
    return modelbox::STATUS_OK;
  }

  auto min_shape_str = input.GetProperity("min_shape");
  auto opt_shape_str = input.GetProperity("opt_shape");
  auto max_shape_str = input.GetProperity("max_shape");
  if (min_shape_str.empty() && opt_shape_str.empty() && max_shape_str.empty()) {
    // no shape config
    return modelbox::STATUS_OK;
  }

  auto tensor_shape_param = std::make_shared<TensorShapeParam>();
  auto ret =
      tensor_shape_param->Init(min_shape_str, opt_shape_str, max_shape_str);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "input port " << input.GetPortName()
                << " config wrong shape";
    return modelbox::STATUS_BADCONF;
  }

  input_tensor_shape_param_list_.push_back(tensor_shape_param);
  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::LoadModel(
    const std::shared_ptr<modelbox::Configuration> &cfg) {
  nlohmann::json model_config;
  auto ret = GenModelConfig(cfg, model_config);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "generate model config failed, ret " << ret;
    return ret;
  }

  auto model_config_str = model_config.dump();
  inferer_ = dlengine::API::Compile(
      model_entry_.c_str(), model_config_str.c_str(), DLENGINE_BACKEND_ZOO);
  if (inferer_ != nullptr) {
    return modelbox::STATUS_OK;
  }

  inferer_ = dlengine::API::GetInferer(model_entry_.c_str());
  if (inferer_ != nullptr) {
    return modelbox::STATUS_OK;
  }

  return {modelbox::STATUS_FAULT, "compile model " + model_entry_ + " failed"};
}

modelbox::Status DLEngineInference::GenModelConfig(
    const std::shared_ptr<modelbox::Configuration> &cfg,
    nlohmann::json &model_config) {
  model_config["model_type"] = cfg->GetString("config.model_type");
  model_config["backend_type"] = cfg->GetString("config.backend_type");
  if (cfg->Contain("config.precision")) {
    model_config["precision"] = cfg->GetString("config.precision");
  }

  if (input_tensor_shape_param_list_.empty()) {
    return modelbox::STATUS_OK;
  }

  auto inputs_config = nlohmann::json::array();
  for (size_t i = 0; i < input_name_list_.size(); ++i) {
    nlohmann::json input_config;
    input_config["name"] = input_name_list_[i];
    auto shape_param = input_tensor_shape_param_list_[i];
    shape_param->GenTensorConfig(input_config);
    inputs_config.push_back(input_config);
  }

  model_config["inputs"] = inputs_config;
  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::PrepareInput(
    dlengine::IInferContext *infer_context,
    const std::shared_ptr<modelbox::DataContext> &data_ctx, size_t batch_size) {
  for (size_t i = 0; i < input_name_list_.size(); ++i) {
    auto &input_name = input_name_list_[i];
    auto in_buffer_list = data_ctx->Input(input_name);
    if (in_buffer_list == nullptr) {
      MBLOG_ERROR << "input name " << input_name << " not found in data ctx";
      return modelbox::STATUS_FAULT;
    }

    const auto &origin_shape = inferer_->GetOriginalInputShape(i);
    auto data_type = inferer_->GetInputDataType(i);

    if (!CheckDataType(data_type)) {
      MBLOG_ERROR << "not support model data type " << (int32_t)data_type;
      return modelbox::STATUS_FAULT;
    }

    auto single_tensor_size = SingleTensorSize(origin_shape, data_type);
    if (single_tensor_size * batch_size != in_buffer_list->GetBytes()) {
      MBLOG_ERROR << "process batch " << batch_size << ", input bytes "
                  << in_buffer_list->GetBytes() << " != model input size "
                  << single_tensor_size * batch_size << ", input name "
                  << input_name;
      return modelbox::STATUS_FAULT;
    }

    dlengine::DimSize cur_input_shape;
    SetUpTensorShape(cur_input_shape, origin_shape, batch_size);

    auto tensor = infer_context->GetInputTensor(i);
    if (tensor == nullptr) {
      MBLOG_ERROR << "tensor index " << i << " is out of range for model "
                  << model_entry_;
      return modelbox::STATUS_FAULT;
    }

    tensor->Resize(cur_input_shape);
    tensor->SetPtr((std::intptr_t)in_buffer_list->ConstData());
  }

  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineInference::PrepareOutput(
    dlengine::IInferContext *infer_context,
    const std::shared_ptr<modelbox::DataContext> &data_ctx, size_t batch_size) {
  for (size_t i = 0; i < output_name_list_.size(); ++i) {
    auto &output_name = output_name_list_[i];
    auto out_buffer_list = data_ctx->Output(output_name);
    if (out_buffer_list == nullptr) {
      MBLOG_ERROR << "output name " << out_buffer_list
                  << " not found in data ctx";
      return modelbox::STATUS_FAULT;
    }

    const auto &origin_shape = inferer_->GetOriginalOutputShape(i);
    auto data_type = inferer_->GetOutputDataType(i);

    if (!CheckDataType(data_type)) {
      MBLOG_ERROR << "not support model data type " << (int32_t)data_type;
      return modelbox::STATUS_FAULT;
    }

    auto single_tensor_size = SingleTensorSize(origin_shape, data_type);
    auto ret = out_buffer_list->Build(
        std::vector<size_t>(batch_size, single_tensor_size));
    if (ret != modelbox::STATUS_OK) {
      MBLOG_ERROR << "build output buffer " << output_name << " failed, count "
                  << batch_size << ", size " << single_tensor_size << ", err "
                  << ret;
      return ret;
    }

    dlengine::DimSize cur_output_shape;
    SetUpTensorShape(cur_output_shape, origin_shape, out_buffer_list->Size());

    auto tensor = infer_context->GetOutputTensor(i);
    if (tensor == nullptr) {
      MBLOG_ERROR << "tensor index " << i << " is out of range for model "
                  << model_entry_;
      return modelbox::STATUS_FAULT;
    }

    tensor->Resize(cur_output_shape);
    tensor->SetPtr((std::intptr_t)out_buffer_list->ConstData());

    SetBufferInfo(out_buffer_list, data_type, origin_shape);
  }

  return modelbox::STATUS_OK;
}

bool DLEngineInference::CheckDataType(dlengine::DataType data_type) {
  auto item = g_dlengine_to_mb_type_map.find(data_type);
  if (item == g_dlengine_to_mb_type_map.end()) {
    return false;
  }

  return true;
}

void DLEngineInference::SetBufferInfo(
    const std::shared_ptr<modelbox::BufferList> &buffer_list,
    dlengine::DataType data_type, const dlengine::DimSize &shape) {
  // data_type has been checked
  auto mb_data_type = g_dlengine_to_mb_type_map.at(data_type);
  std::vector<size_t> out_shape;
  out_shape.reserve(shape.num_dims);
  const size_t n = 1;
  out_shape.push_back(n);  // in modelbox, for each buffer n must be 1
  for (size_t i = 1; i < shape.num_dims; ++i) {
    out_shape.push_back(shape.dims[i]);
  }

  for (const auto &buffer : *buffer_list) {
    buffer->Set("type", mb_data_type);
    buffer->Set("shape", out_shape);
  }
}

modelbox::Status DLEngineInference::SetUpTensorShape(
    dlengine::DimSize &cur_shape, const dlengine::DimSize &origin_shape,
    size_t batch_size) {
  cur_shape.num_dims = origin_shape.num_dims;
  for (size_t i = 0; i < origin_shape.num_dims; ++i) {
    cur_shape.dims[i] = origin_shape.dims[i];
  }

  if (batch_size > INT_MAX) {
    MBLOG_ERROR << "batch size " << batch_size << " is too big";
    return modelbox::STATUS_FAULT;
  }

  cur_shape.dims[0] = (int32_t)batch_size;
  return modelbox::STATUS_OK;
}

size_t DLEngineInference::SingleTensorSize(const dlengine::DimSize &shape,
                                           dlengine::DataType data_type) {
  size_t tensor_size = 1;
  for (size_t i = 1; i < shape.num_dims; ++i) {
    tensor_size *= shape.dims[i];
  }

  // data_type has been checked
  return tensor_size * g_dlengine_type_size_map.at(data_type);
}
