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

#include "tensorrt_inference_flowunit.h"

#include <dlfcn.h>
#include <model_decrypt.h>

#include <cstddef>
#include <fstream>
#include <random>

#include "common_util.h"
#include "modelbox/base/status.h"
#include "modelbox/device/cuda/device_cuda.h"
#ifndef TENSORRT8
#include "nvplugin/plugin_factory.h"
#endif
#include "virtualdriver_inference.h"

TensorRTInferenceFlowUnit::TensorRTInferenceFlowUnit(){};
TensorRTInferenceFlowUnit::~TensorRTInferenceFlowUnit() {
  context_ = nullptr;
  engine_ = nullptr;
  plugin_factory_ = nullptr;

  pre_process_ = nullptr;
  post_process_ = nullptr;
  data_pre_ = nullptr;
  data_post_ = nullptr;
  inference_plugin_ = nullptr;

  if (driver_handler_ != nullptr) {
    dlclose(driver_handler_);
    driver_handler_ = nullptr;
  }
};

RndInt8Calibrator::RndInt8Calibrator(
    int total_samples, std::string cache_file,
    std::map<std::string, nvinfer1::Dims3>& input_dims)
    : total_samples_(total_samples),
      current_sample_(0),
      cache_file_(cache_file) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (auto& elem : input_dims) {
    int elemCount = Volume(elem.second);

    std::vector<float> rnd_data(elemCount);
    for (auto& val : rnd_data) val = distribution(generator);

    void* data = nullptr;
    if (cudaMalloc(&data, elemCount * sizeof(float)) != 0) {
      MBLOG_WARN << "Cuda failure: cudaMalloc";
      continue;
    }
    if (cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float),
                   cudaMemcpyHostToDevice) != 0) {
      MBLOG_WARN << "Cuda failure: cudaMemcpy";
      cudaFree(data);
      continue;
    }

    input_device_buffers_.insert(std::make_pair(elem.first, data));
  }
}

RndInt8Calibrator::~RndInt8Calibrator() {
  for (auto& elem : input_device_buffers_)
    if (cudaFree(elem.second) != 0) {
      MBLOG_WARN << "Cuda failure: cudaFree";
    }
}

int RndInt8Calibrator::getBatchSize() const noexcept { return 1; }

bool RndInt8Calibrator::getBatch(void* bindings[], const char* names[],
                                 int nbBindings) noexcept {
  if (current_sample_ >= total_samples_) {
    return false;
  }

  for (int i = 0; i < nbBindings; ++i) {
    bindings[i] = input_device_buffers_[names[i]];
  }

  ++current_sample_;
  return true;
}

const void* RndInt8Calibrator::readCalibrationCache(size_t& length) noexcept {
  calibration_cache_.clear();
  std::ifstream input(cache_file_, std::ios::binary);
  input >> std::noskipws;
  if (input.good()) {
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(calibration_cache_));
  }

  length = calibration_cache_.size();
  return length ? &calibration_cache_[0] : nullptr;
}

void RndInt8Calibrator::writeCalibrationCache(const void*, size_t) noexcept {}

modelbox::Status TensorRTInferenceFlowUnit::PreProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::PostProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  if (!data_pre_) {
    return modelbox::STATUS_OK;
  }

  return data_pre_(data_ctx);
}

modelbox::Status TensorRTInferenceFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  if (!data_post_) {
    return modelbox::STATUS_OK;
  }

  return data_post_(data_ctx);
}

void TensorRTInferenceFlowUnit::SetUpOtherConfig(
    std::shared_ptr<modelbox::Configuration> config) {
  params_.calibration_cache =
      config->GetString("calibration_cache", "CalibrationTable");

  params_.plugin = config->GetString("plugin");
  params_.dynamic_batch_contain = config->Contain("dynamic_batch");
  params_.dynamic_batch = config->GetBool("dynamic_batch", false);
  params_.onnx_opt_batch_size = config->GetInt64("onnx_opt_batch_size", 1);
  params_.onnx_max_batch_size = config->GetInt64("onnx_max_batch_size", 1);
  params_.workspace_size = config->GetInt64("workspace_size", 16);
  params_.use_DLACore = config->GetInt64("use_dla_core", -1);
  params_.fp16 = config->GetBool("fp16", false);
  params_.int8 = config->GetBool("int8", false);
  params_.verbose = config->GetBool("verbose", false);
  params_.allow_GPUFallback = config->GetBool("allow_gpu_fallback", false);
  params_.pct = config->GetFloat("pct", 99);
}

modelbox::Status TensorRTInferenceFlowUnit::SetUpModelFile(
    std::shared_ptr<modelbox::Configuration> config,
    const std::string& model_file) {
  std::string suffix_str = model_file.substr(model_file.find_last_of('.') + 1);

  if (suffix_str == SUFFIX_UFF) {
    params_.uff_file = model_file;
  }

  if (!params_.uff_file.empty()) {
    std::vector<std::string> uff_inputs_string =
        config->GetStrings("uff_input");
    if (uff_inputs_string.empty()) {
      return {modelbox::STATUS_BADCONF,
              "uff file need to config uffInput, please configure uffInput "
              "like 'name, c, h ,w'."};
    }

    for (size_t i = 0; i < uff_inputs_string.size(); ++i) {
      // TODO wait for configure adjust for ',' in string
      std::vector<std::string> split_uff_inputs_string =
          modelbox::StringSplit(uff_inputs_string[i], '.');
      std::string name = split_uff_inputs_string[0];
      std::shared_ptr<nvinfer1::Dims> dims = nullptr;

      switch (split_uff_inputs_string.size()) {
        case 1:
        case 2:
          MBLOG_ERROR << "invalid uffInputs";
          break;
        case 3:
          dims = std::make_shared<nvinfer1::Dims2>(
              atoi(split_uff_inputs_string[1].c_str()),
              atoi(split_uff_inputs_string[2].c_str()));
          break;
        case 4:
          dims = std::make_shared<nvinfer1::Dims3>(
              atoi(split_uff_inputs_string[1].c_str()),
              atoi(split_uff_inputs_string[2].c_str()),
              atoi(split_uff_inputs_string[3].c_str()));
          break;
        case 5:
          dims = std::make_shared<nvinfer1::Dims4>(
              atoi(split_uff_inputs_string[1].c_str()),
              atoi(split_uff_inputs_string[2].c_str()),
              atoi(split_uff_inputs_string[3].c_str()),
              atoi(split_uff_inputs_string[4].c_str()));
          break;
        default:
          MBLOG_ERROR << "invalid uffInputs";
          break;
      }

      for (int i = 0; i < dims->nbDims; ++i) {
        if (dims->d[i] != 0) {
          continue;
        }
        MBLOG_ERROR << "invalid uffInputs";
        return {modelbox::STATUS_BADCONF, "invalid uffInputs."};
      }
      params_.uff_input_list.push_back(std::make_pair(name, dims));
    }

    return modelbox::STATUS_OK;
  }

  if (suffix_str == SUFFIX_ONNX) {
    params_.onnx_model_file = model_file;
    return modelbox::STATUS_OK;
  }

  if (suffix_str == SUFFIX_PROTXT) {
    params_.deploy_file = model_file;
    params_.model_file = config->GetString("model_file");
    return modelbox::STATUS_OK;
  }

  params_.engine = model_file;

  return modelbox::STATUS_OK;
}

void TensorRTInferenceFlowUnit::configureBuilder(
    std::shared_ptr<IBuilder> builder, RndInt8Calibrator& calibrator) {
#ifndef TENSORRT8
  builder->setMaxWorkspaceSize(static_cast<unsigned int>(params_.workspace_size)
                               << 20);
  builder->setFp16Mode(params_.fp16);

  if (!params_.fp16 && params_.int8) {
    builder->setInt8Mode(true);
    builder->setInt8Calibrator(&calibrator);
  }

  if (params_.use_DLACore >= 0) {
    builder->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    builder->setDLACore(params_.use_DLACore);
    if (params_.allow_GPUFallback) builder->allowGPUFallback(true);
  }
#endif
}

void TensorRTInferenceFlowUnit::PrintModelBindInfo(
    const std::vector<std::string>& name_list) {
  for (auto& bind_name : name_list) {
    auto bind_index = engine_->getBindingIndex(bind_name.c_str());
    auto bind_dims = engine_->getBindingDimensions(bind_index);
    std::stringstream dim_info;
    dim_info << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << ", bind name: " << bind_name << ", dims: [";
    for (int dim_index = 0; dim_index < bind_dims.nbDims; ++dim_index) {
      dim_info << bind_dims.d[dim_index];
      if (dim_index != bind_dims.nbDims - 1) {
        dim_info << ", ";
      }
    }
    dim_info << "]";
    MBLOG_INFO << dim_info.str();

#ifdef TENSORRT8
    params_.use_enqueue_v2 = true;
#else
    params_.use_enqueue_v2 = (bind_dims.d[0] == -1);
#endif
  }

  if (params_.dynamic_batch_contain) {
    params_.use_enqueue_v2 = params_.dynamic_batch;
  }
}

modelbox::Status TensorRTInferenceFlowUnit::CaffeToTRTModel(
    const std::shared_ptr<modelbox::Configuration>& config,
    std::shared_ptr<IBuilder>& builder,
    std::shared_ptr<INetworkDefinition>& network) {
#ifndef TENSORRT8
  // parse the caffe model to populate the network, then set the outputs
  std::shared_ptr<ICaffeParser> parser =
      TensorRTInferObject(createCaffeParser());
  if (parser == nullptr) {
    auto err_msg = "create parser from caffe model failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  const IBlobNameToTensor* blobNameToTensor = nullptr;
  auto drivers_ptr = GetBindDevice()->GetDeviceManager()->GetDrivers();

  ModelDecryption deploy_decrypt;
  auto ret = deploy_decrypt.Init(params_.deploy_file, drivers_ptr, config);
  if (ret != modelbox::STATUS_SUCCESS) {
    return {modelbox::STATUS_FAULT, "open caffe deploy failed."};
  }
  ModelDecryption model_decrypt;
  ret = model_decrypt.Init(params_.model_file, drivers_ptr, config);
  if (ret != modelbox::STATUS_SUCCESS && !params_.model_file.empty()) {
    return {modelbox::STATUS_FAULT, "open caffe model failed."};
  }

  if (deploy_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT ||
      model_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t deploy_len = 0;
    std::shared_ptr<uint8_t> deployBuf =
        deploy_decrypt.GetModelSharedBuffer(deploy_len);
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        model_decrypt.GetModelSharedBuffer(model_len);
    if (!deployBuf || (!modelBuf && !params_.model_file.empty())) {
      return {modelbox::STATUS_FAULT, "Decrypt model fail"};
    }
    blobNameToTensor = parser->parseBuffers(
        (const char*)deployBuf.get(), (size_t)deploy_len,
        modelBuf ? (const char*)modelBuf.get() : nullptr, (size_t)deploy_len,
        *network, params_.fp16 ? DataType::kHALF : DataType::kFLOAT);
  } else {
    blobNameToTensor = parser->parse(
        params_.deploy_file.c_str(),
        params_.model_file.empty() ? 0 : params_.model_file.c_str(), *network,
        params_.fp16 ? DataType::kHALF : DataType::kFLOAT);
  }
  if (!blobNameToTensor) {
    return {modelbox::STATUS_FAULT, "parser caffe model failed."};
  }

  for (int i = 0, n = network->getNbInputs(); i < n; i++) {
    auto input = network->getInput(i);
    if (input == nullptr) {
      MBLOG_ERROR << "input " << i << "is invalid";
      return {modelbox::STATUS_FAULT, "get input failed"};
    }

    Dims3 dims = static_cast<Dims3&&>(input->getDimensions());
    input_dims_.insert(std::make_pair(input->getName(), dims));
  }

  // specify which tensors are outputs
  for (auto& output_item : params_.outputs_name_list) {
    if (blobNameToTensor->find(output_item.c_str()) == nullptr) {
      auto err_msg = "could not find output blob, " + output_item;
      return {modelbox::STATUS_FAULT, err_msg};
    }
    network->markOutput(*blobNameToTensor->find(output_item.c_str()));
  }

  // Build the engine
  RndInt8Calibrator calibrator(1, params_.calibration_cache, input_dims_);
  configureBuilder(builder, calibrator);

  engine_ = TensorRTInferObject(builder->buildCudaEngine(*network));
  if (engine_ == nullptr) {
    auto err_msg = "build engine from caffe model failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  context_ = TensorRTInferObject(engine_->createExecutionContext());
  if (context_ == nullptr) {
    auto err_msg = "build context from caffe model engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
#endif  // TENSORRT8
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::UffToTRTModel(
    const std::shared_ptr<modelbox::Configuration>& config,
    std::shared_ptr<IBuilder>& builder,
    std::shared_ptr<INetworkDefinition>& network) {
#ifndef TENSORRT8
  // parse the uff model to populate the network, then set the outputs
  std::shared_ptr<IUffParser> parser = TensorRTInferObject(createUffParser());
  if (parser == nullptr) {
    auto err_msg = "create parser from uff model engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  // specify which tensors are outputs
  for (auto& output_item : params_.outputs_name_list) {
    if (!parser->registerOutput(output_item.c_str())) {
      auto err_msg =
          "Failed to register output " + output_item + " in uff file.";
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }

  // specify which tensors are inputs (and their dimensions)
  // TODO set nhwc or nchw
  for (auto& input_item : params_.uff_input_list) {
    if (!parser->registerInput(input_item.first.c_str(), *(input_item.second),
                               UffInputOrder::kNCHW)) {
      auto err_msg =
          "Failed to register input " + input_item.first + " in uff file.";
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }

  bool parseRet = false;
  auto drivers_ptr = GetBindDevice()->GetDeviceManager()->GetDrivers();
  ModelDecryption uff_decrypt;
  // do not need to check return , just use GetModelState
  uff_decrypt.Init(params_.uff_file, drivers_ptr, config);
  if (uff_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        uff_decrypt.GetModelSharedBuffer(model_len);
    if (modelBuf) {
      parseRet = parser->parseBuffer(
          (const char*)modelBuf.get(), (size_t)model_len, *network,
          params_.fp16 ? DataType::kHALF : DataType::kFLOAT);
    }
  } else if (uff_decrypt.GetModelState() ==
             ModelDecryption::MODEL_STATE_PLAIN) {
    parseRet = parser->parse(params_.uff_file.c_str(), *network,
                             params_.fp16 ? DataType::kHALF : DataType::kFLOAT);
  }
  if (!parseRet) {
    return {modelbox::STATUS_FAULT, "parser uff model failed."};
  }

  for (int i = 0, n = network->getNbInputs(); i < n; i++) {
    auto input = network->getInput(i);
    if (input == nullptr) {
      MBLOG_ERROR << "input " << i << "is invalid";
      return {modelbox::STATUS_FAULT, "get input failed"};
    }

    Dims3 dims = static_cast<Dims3&&>(input->getDimensions());
    input_dims_.insert(std::make_pair(input->getName(), dims));
  }

  // Build the engine
  RndInt8Calibrator calibrator(1, params_.calibration_cache, input_dims_);
  configureBuilder(builder, calibrator);

  engine_ = TensorRTInferObject(builder->buildCudaEngine(*network));
  if (engine_ == nullptr) {
    auto err_msg = "build engine from uff model failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  context_ = TensorRTInferObject(engine_->createExecutionContext());
  if (context_ == nullptr) {
    auto err_msg = "build context from uff model engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
#endif  // TENSORRT8
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::OnnxToTRTModel(
    const std::shared_ptr<modelbox::Configuration>& config,
    std::shared_ptr<IBuilder>& builder,
    std::shared_ptr<INetworkDefinition>& network) {
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

  // parse the onnx model to populate the network, then set the outputs
  std::shared_ptr<IParser> parser =
      TensorRTInferObject(nvonnxparser::createParser(*network, gLogger));
  if (parser == nullptr) {
    auto err_msg = "create parser from onnx model engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  bool parseRet = false;
  auto drivers_ptr = GetBindDevice()->GetDeviceManager()->GetDrivers();

  ModelDecryption onnx_decrypt;
  onnx_decrypt.Init(params_.onnx_model_file, drivers_ptr, config);
  // do not need to check return , just use GetModelState
  if (onnx_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        onnx_decrypt.GetModelSharedBuffer(model_len);
    if (modelBuf) {
      parseRet = parser->parse((void const*)modelBuf.get(), (size_t)model_len);
    }
  } else if (onnx_decrypt.GetModelState() ==
             ModelDecryption::MODEL_STATE_PLAIN) {
    parseRet =
        parser->parseFromFile(params_.onnx_model_file.c_str(), verbosity);
  }
  if (!parseRet) {
    return {modelbox::STATUS_FAULT, "failed to parse onnex file."};
  }

#if defined(TENSORRT7) || defined(TENSORRT8)
  auto builder_config = builder->createBuilderConfig();
  auto profile = builder->createOptimizationProfile();
  for (int i = 0, n = network->getNbInputs(); i < n; i++) {
    auto input = network->getInput(i);
    nvinfer1::Dims dims = input->getDimensions();
    if (dims.d[0] == -1) {
      dims.d[0] = 1;
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, dims);
      dims.d[0] = params_.onnx_opt_batch_size;
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, dims);
      dims.d[0] = params_.onnx_max_batch_size;
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, dims);
    } else {
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, dims);
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, dims);
    }
  }
  builder_config->addOptimizationProfile(profile);
#ifdef TENSORRT8
  auto serialized_engine = TensorRTInferObject(
      builder->buildSerializedNetwork(*network, *builder_config));
  std::shared_ptr<IRuntime> infer =
      TensorRTInferObject(createInferRuntime(gLogger));
  if (infer == nullptr) {
    auto err_msg = "create runtime from model_file engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
  engine_ = TensorRTInferObject(infer->deserializeCudaEngine(
      serialized_engine->data(), serialized_engine->size()));
#else
  engine_ = TensorRTInferObject(
      builder->buildEngineWithConfig(*network, *builder_config));
#endif
#else
  for (int i = 0, n = network->getNbInputs(); i < n; i++) {
    auto input = network->getInput(i);
    if (input == nullptr) {
      MBLOG_ERROR << "input " << i << "is invalid";
      return {modelbox::STATUS_FAULT, "get input failed"};
    }

    Dims3 dims = static_cast<Dims3&&>(input->getDimensions());
    input_dims_.insert(std::make_pair(input->getName(), dims));
  }

  // Build the engine
  RndInt8Calibrator calibrator(1, params_.calibration_cache, input_dims_);
  configureBuilder(builder, calibrator);

  engine_ = TensorRTInferObject(builder->buildCudaEngine(*network));
#endif
  if (engine_ == nullptr) {
    auto err_msg = "build engine from onnx model failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << ", max batch size: " << engine_->getMaxBatchSize();
  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << " model inputs num:" << params_.inputs_name_list.size();
  PrintModelBindInfo(params_.inputs_name_list);
  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << " model outputs num:" << params_.outputs_name_list.size();
  PrintModelBindInfo(params_.outputs_name_list);

  context_ = TensorRTInferObject(engine_->createExecutionContext());
  if (context_ == nullptr) {
    auto err_msg = "build context from onnx model engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::EngineToModel(
    const std::shared_ptr<modelbox::Configuration>& config) {
  MBLOG_INFO << "engines: " << params_.engine;
  std::shared_ptr<IRuntime> infer =
      TensorRTInferObject(createInferRuntime(gLogger));
  if (infer == nullptr) {
    auto err_msg = "create runtime from model_file engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (params_.use_DLACore >= 0) {
    infer->setDLACore(params_.use_DLACore);
  }

  SetPluginFactory(params_.plugin);

  auto drivers_ptr = GetBindDevice()->GetDeviceManager()->GetDrivers();
  ModelDecryption engine_decrypt;
  auto ret = engine_decrypt.Init(params_.engine, drivers_ptr, config);
  if (ret != modelbox::STATUS_SUCCESS) {
    auto err_msg = "open engine deploy failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
  // do not need to check return , just use GetModelState
  auto modelState = engine_decrypt.GetModelState();
  if (modelState == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        engine_decrypt.GetModelSharedBuffer(model_len);
    if (modelBuf == nullptr) {
      auto err_msg =
          "failed to decrypt model, the model file " + params_.engine;
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_BADCONF, err_msg};
    }
#ifdef TENSORRT8
    engine_ = TensorRTInferObject(
        infer->deserializeCudaEngine(modelBuf.get(), model_len));
#else
    engine_ = TensorRTInferObject(infer->deserializeCudaEngine(
        modelBuf.get(), model_len, plugin_factory_.get()));
#endif
  } else if (modelState == ModelDecryption::MODEL_STATE_PLAIN) {
    std::vector<char> trtModelStream;
    size_t size{0};
    std::ifstream file(params_.engine, std::ios::binary);
    if (!file.good()) {
      auto err_msg = "read model file failed, the model file " + params_.engine;
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(size);
    file.read(trtModelStream.data(), size);
    file.close();
#ifdef TENSORRT8
    engine_ = TensorRTInferObject(
        infer->deserializeCudaEngine(trtModelStream.data(), size));
#else
    engine_ = TensorRTInferObject(infer->deserializeCudaEngine(
        trtModelStream.data(), size, plugin_factory_.get()));
#endif

  } else {
    auto err_msg = "model state error.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (engine_ == nullptr) {
    auto err_msg = "build engine from model_file failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << ", max batch size: " << engine_->getMaxBatchSize();
  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << " model inputs num:" << params_.inputs_name_list.size();
  PrintModelBindInfo(params_.inputs_name_list);
  MBLOG_INFO << "flowunit: " << GetFlowUnitDesc()->GetFlowUnitName()
             << " model outputs num:" << params_.outputs_name_list.size();
  PrintModelBindInfo(params_.outputs_name_list);

  context_ = TensorRTInferObject(engine_->createExecutionContext());
  if (context_ == nullptr) {
    auto err_msg = "build context from model_file engine failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::CreateEngine(
    const std::shared_ptr<modelbox::Configuration>& config) {
  modelbox::Status status;
  // load directly from serialized engine file if deploy not specified
  if (!params_.engine.empty()) {
    return EngineToModel(config);
  }

  std::shared_ptr<IBuilder> builder =
      TensorRTInferObject(createInferBuilder(gLogger));
  if (builder == nullptr) {
    auto err_msg = "create builder failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  // parse the caffe model to populate the network, then set the outputs
#if defined(TENSORRT7) || defined(TENSORRT8)
  std::shared_ptr<INetworkDefinition> network =
      TensorRTInferObject(builder->createNetworkV2(
          1U << static_cast<int>(
              NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#else
  std::shared_ptr<INetworkDefinition> network =
      TensorRTInferObject(builder->createNetwork());
#endif
  if (network == nullptr) {
    auto err_msg = "creat network failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (!params_.deploy_file.empty()) {
    return CaffeToTRTModel(config, builder, network);
  }

  if (!params_.uff_file.empty()) {
    return UffToTRTModel(config, builder, network);
  }

  if (!params_.onnx_model_file.empty()) {
    return OnnxToTRTModel(config, builder, network);
  }

  return modelbox::STATUS_OK;
}

void TensorRTInferenceFlowUnit::SetPluginFactory(std::string pluginName) {
  if (pluginName.empty()) {
    return;
  }
#ifndef TENSORRT8
  if (pluginName == "yolo") {
    plugin_factory_ = std::make_shared<YoloPluginFactory>();
    return;
  }
#endif
  MBLOG_DEBUG << "The plugin " << pluginName.c_str() << " is not supported";
  return;
}

modelbox::Status TensorRTInferenceFlowUnit::InitConfig(
    const std::shared_ptr<modelbox::Configuration>& fu_config) {
  auto inference_desc_ =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
          this->GetFlowUnitDesc());
  const std::vector<modelbox::FlowUnitInput>& flowunit_input_list =
      inference_desc_->GetFlowUnitInput();
  const std::vector<modelbox::FlowUnitOutput>& flowunit_output_list =
      inference_desc_->GetFlowUnitOutput();

  auto model_file = inference_desc_->GetModelEntry();

  auto inner_config = fu_config->GetSubConfig("config");
  SetUpOtherConfig(inner_config);
  auto status = SetUpModelFile(inner_config, model_file);
  if (status != modelbox::STATUS_OK) {
    return {modelbox::STATUS_BADCONF,
            "parser config failed, " + status.WrapErrormsgs()};
  }

  for (const auto& output_item : flowunit_output_list) {
    params_.outputs_name_list.push_back(output_item.GetPortName());
    params_.outputs_type_list.push_back(output_item.GetPortType());
  }

  for (const auto& input_item : flowunit_input_list) {
    params_.inputs_name_list.push_back(input_item.GetPortName());
  }

  status = CreateEngine(fu_config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "engine create failed." + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::SetUpDynamicLibrary(
    std::shared_ptr<modelbox::Configuration> config) {
  typedef std::shared_ptr<TensorRTInferencePlugin> (*PluginObject)();
  auto status = modelbox::STATUS_OK;
  void* driver_handler = dlopen(plugin_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    auto dl_errmsg = dlerror();
    auto err_msg = "dlopen " + plugin_ + " failed";
    if (dl_errmsg) {
      err_msg += ", error: " + std::string(dl_errmsg);
    }

    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  DeferCond { return !status; };
  DeferCondAdd {
    if (driver_handler != nullptr) {
      dlclose(driver_handler);
      driver_handler = nullptr;
    }
  };

  auto create_plugin =
      reinterpret_cast<PluginObject>(dlsym(driver_handler, "CreatePlugin"));
  if (create_plugin == nullptr) {
    auto dlerr_msg = dlerror();
    std::string err_msg = "dlsym CreatePlugin failed";
    if (dlerr_msg) {
      err_msg += " error: ";
      err_msg += dlerr_msg;
    }

    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  std::shared_ptr<TensorRTInferencePlugin> inference_plugin = create_plugin();
  if (inference_plugin == nullptr) {
    auto err_msg = "CreatePlugin failed";
    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  status = inference_plugin->PluginInit(config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "plugin init failed, error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  driver_handler_ = std::move(driver_handler);
  inference_plugin_ = std::move(inference_plugin);

  pre_process_ = std::bind(&TensorRTInferencePlugin::PreProcess,
                           inference_plugin_, std::placeholders::_1);
  post_process_ = std::bind(&TensorRTInferencePlugin::PostProcess,
                            inference_plugin_, std::placeholders::_1);
  data_pre_ = std::bind(&TensorRTInferencePlugin::DataPre, inference_plugin_,
                        std::placeholders::_1);
  data_post_ = std::bind(&TensorRTInferencePlugin::DataPost, inference_plugin_,
                         std::placeholders::_1);

  return status;
}

modelbox::Status TensorRTInferenceFlowUnit::SetUpInferencePlugin(
    std::shared_ptr<modelbox::Configuration> config) {
  if (plugin_.empty()) {
    pre_process_ = std::bind(&TensorRTInferenceFlowUnit::PreProcess, this,
                             std::placeholders::_1);
    post_process_ = std::bind(&TensorRTInferenceFlowUnit::PostProcess, this,
                              std::placeholders::_1);
    return modelbox::STATUS_OK;
  }

  if (!modelbox::IsAbsolutePath(plugin_)) {
    auto relpath = modelbox::GetDirName(plugin_);
    plugin_ = relpath + "/" + plugin_;
  }

  return SetUpDynamicLibrary(config);
}

modelbox::Status TensorRTInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration>& opts) {
  auto inference_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  auto config = inference_desc->GetConfiguration();

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  merge_config->Add(*config);
  merge_config->Add(*opts);

  params_.device = dev_id_;
  auto status = InitConfig(merge_config);
  if (status != modelbox::STATUS_OK) {
    MBLOG_ERROR << status.WrapErrormsgs();
    return {modelbox::STATUS_BADCONF, status.WrapErrormsgs()};
  }

  plugin_ = merge_config->GetString("plugin");
  status = SetUpInferencePlugin(merge_config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "setup preprocess and postprocess failed: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::BindMemory(
    std::vector<void*>& buffers, const std::string& name, const void* mem,
    size_t mem_size, size_t size) {
  int data_type_size = 0;

  const int binding_index = engine_->getBindingIndex(name.c_str());
  if (binding_index >= (int)buffers.size() || binding_index < 0) {
    auto err_msg = name + " not found in network";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  nvinfer1::DataType data_type =
      engine_->getBindingDataType((int)binding_index);
  switch (data_type) {
    case DataType::kFLOAT:
      data_type_size = sizeof(float);
      break;
    case DataType::kHALF:
      data_type_size = sizeof(short);
      break;
    case DataType::kINT8:
      data_type_size = sizeof(char);
      break;
    case DataType::kINT32:
      data_type_size = sizeof(int);
      break;
    default:
      break;
  }

  const nvinfer1::Dims dims = engine_->getBindingDimensions((int)binding_index);
  const size_t expect_size = Volume(dims) * size * data_type_size;
  if (expect_size != mem_size) {
    auto err_msg = "the input buffer size " + std::to_string(mem_size) +
                   " is not equal tensorrt input real size " +
                   std::to_string(expect_size) +
                   ", batch size: " + std::to_string(size) +
                   ", input name: " + name +
                   ", input tensorrt type: " + std::to_string(int(data_type));
    err_msg += ", input dims: [";
    for (int idx = 0; idx < dims.nbDims; ++idx) {
      err_msg += std::to_string(dims.d[idx]) + ", ";
    }
    err_msg += "]";

    return {modelbox::STATUS_FAULT, err_msg};
  }

  buffers[binding_index] = const_cast<void*>(mem);
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::CreateMemory(
    std::vector<void*>& buffers, const std::string& name,
    const std::string& type, std::shared_ptr<modelbox::BufferList>& output_buf,
    size_t size) {
  int data_type_size = 0;
  modelbox::Status status;

  const int binding_index = engine_->getBindingIndex(name.c_str());
  if (binding_index >= (int)buffers.size() || binding_index < 0) {
    auto err_msg = name + " not found in network";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  const nvinfer1::Dims dims = engine_->getBindingDimensions((int)binding_index);
  const nvinfer1::DataType data_type =
      engine_->getBindingDataType((int)binding_index);
  switch (data_type) {
    case DataType::kFLOAT:
      data_type_size = sizeof(float);
      break;
    case DataType::kHALF:
      data_type_size = sizeof(short);
      break;
    case DataType::kINT8:
      data_type_size = sizeof(char);
      break;
    case DataType::kINT32:
      data_type_size = sizeof(int);
      break;
    default:
      break;
  }

  auto single_bytes = Volume(dims) * data_type_size;
  std::vector<size_t> output_shape;
  for (int i = 0; i < dims.nbDims; ++i) {
    output_shape.push_back(dims.d[i]);
  }
  std::vector<size_t> shape_vector(size, single_bytes);
  status = output_buf->Build(shape_vector);
  if (type == "float") {
    output_buf->Set("type", modelbox::MODELBOX_FLOAT);
  } else if (type == "double") {
    output_buf->Set("type", modelbox::MODELBOX_DOUBLE);
  } else if (type == "int") {
    output_buf->Set("type", modelbox::MODELBOX_INT32);
  } else if (type == "uint8") {
    output_buf->Set("type", modelbox::MODELBOX_UINT8);
  } else if (type == "long") {
    output_buf->Set("type", modelbox::MODELBOX_INT16);
  } else
    return {modelbox::STATUS_NOTSUPPORT, "unsupport output type."};

  output_buf->Set("shape", output_shape);
  buffers[binding_index] = output_buf->MutableData();
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::PrePareInput(
    std::shared_ptr<modelbox::DataContext>& data_ctx,
    std::vector<void*>& memory) {
  for (const auto& input_name : params_.inputs_name_list) {
    auto input_buf = data_ctx->Input(input_name);
    auto data = input_buf->ConstData();
    auto status = BindMemory(memory, input_name, data, input_buf->GetBytes(),
                             input_buf->Size());
    if (status != modelbox::STATUS_OK) {
      auto err_msg =
          "bindMemory " + input_name + " failed." + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::PrePareOutput(
    std::shared_ptr<modelbox::DataContext>& data_ctx,
    std::vector<void*>& memory) {
  int index = 0;
  size_t size = data_ctx->Input(params_.inputs_name_list[0])->Size();
  for (const auto& output_name : params_.outputs_name_list) {
    auto output_buf = data_ctx->Output(output_name);
    auto output_type = params_.outputs_type_list[index++];
    auto status =
        CreateMemory(memory, output_name, output_type, output_buf, size);
    if (status != modelbox::STATUS_OK) {
      auto err_msg =
          "createMemory " + output_name + " failed." + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  modelbox::Status status;
  std::vector<void*> memory(params_.inputs_name_list.size() +
                            params_.outputs_name_list.size());

  status = pre_process_(data_ctx);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "pre_process failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = PrePareInput(data_ctx, memory);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "prepare input failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = PrePareOutput(data_ctx, memory);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "prepare output failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  bool enqueue_res;
#if defined(TENSORRT7) || defined(TENSORRT8)
  if (params_.use_enqueue_v2) {
    for (auto& input_name : params_.inputs_name_list) {
      auto bind_index = engine_->getBindingIndex(input_name.c_str());
      auto bind_dims = engine_->getBindingDimensions(bind_index);
      auto input_batch_size = data_ctx->Input(input_name)->Size();
      bind_dims.d[0] = input_batch_size;
      context_->setBindingDimensions(bind_index, bind_dims);
    }

    enqueue_res = context_->enqueueV2(&memory[0], stream, nullptr);
  } else
#endif
  {
    size_t size = data_ctx->Input(params_.inputs_name_list[0])->Size();
    if (engine_->getMaxBatchSize() < (int)size) {
      auto err_msg = "engine max batch size is " +
                     std::to_string(engine_->getMaxBatchSize()) +
                     ", less than batch_size: " + std::to_string(size);
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    enqueue_res = context_->enqueue(size, &memory[0], stream, nullptr);
  }

  if (!enqueue_res) {
    auto err_msg = "enqueue failed.";
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Cuda stream synchronize failed, gpu "
                << " cuda ret " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  status = post_process_(data_ctx);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "post_process failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTParams::Clear() {
  uff_input_list.clear();
  inputs_name_list.clear();
  outputs_name_list.clear();
  outputs_type_list.clear();
  return modelbox::STATUS_OK;
}

modelbox::Status TensorRTInferenceFlowUnit::Close() {
  input_dims_.clear();
  return params_.Clear();
}

std::shared_ptr<modelbox::FlowUnit>
TensorRTInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string& unit_name, const std::string& unit_type,
    const std::string& virtual_type) {
  auto inference_flowunit = std::make_shared<TensorRTInferenceFlowUnit>();
  return inference_flowunit;
};
