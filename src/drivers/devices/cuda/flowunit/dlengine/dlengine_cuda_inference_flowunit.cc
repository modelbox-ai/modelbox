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

#include "dlengine_cuda_inference_flowunit.h"

constexpr const char *BACKEND_TYPE = "PBJAFgZcNjg=";

DLEngineCUDAInferenceFlowUnit::DLEngineCUDAInferenceFlowUnit()
    : inference_(std::make_shared<DLEngineInference>()) {}

DLEngineCUDAInferenceFlowUnit::~DLEngineCUDAInferenceFlowUnit() = default;

modelbox::Status DLEngineCUDAInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &config) {
  if (!config->Contain("config.model_type")) {
    config->SetProperty("config.model_type", "onnx");
  }

  // fix backend on gpu
  config->SetProperty("config.backend_type", BACKEND_TYPE);

  return inference_->Init(config, GetFlowUnitDesc(), GetBindDevice()->GetType(),
                          dev_id_);
}

modelbox::Status DLEngineCUDAInferenceFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status DLEngineCUDAInferenceFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  auto ret = cudaStreamSynchronize(stream);
  if (ret != cudaSuccess) {
    MBLOG_ERROR << "cuda stream sync failed, err " << ret;
    return modelbox::STATUS_FAULT;
  }

  return inference_->Infer(data_ctx);
}

std::shared_ptr<modelbox::FlowUnit>
DLEngineCUDAInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<DLEngineCUDAInferenceFlowUnit>();
}

std::string DLEngineCUDAInferenceFlowUnitFactory::GetFlowUnitFactoryType() {
  return FLOWUNIT_TYPE;
}

std::string DLEngineCUDAInferenceFlowUnitFactory::GetVirtualType() {
  return INFERENCE_TYPE;
}
