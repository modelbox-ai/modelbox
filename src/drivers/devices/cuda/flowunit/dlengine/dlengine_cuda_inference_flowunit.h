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

#ifndef MODELBOX_FLOWUNIT_DLENGINE_CPU_INFERENCE_H_
#define MODELBOX_FLOWUNIT_DLENGINE_CPU_INFERENCE_H_

#include "dlengine_inference_flowunit.h"
#include "modelbox/device/cuda/device_cuda.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_TYPE = "cuda";

class DLEngineCUDAInferenceFlowUnit : public modelbox::CudaFlowUnit {
 public:
  DLEngineCUDAInferenceFlowUnit();

  ~DLEngineCUDAInferenceFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &config) override;

  modelbox::Status Close() override;

  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               cudaStream_t stream) override;

 private:
  std::shared_ptr<DLEngineInference> inference_;
};

class DLEngineCUDAInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override;

  std::string GetVirtualType() override;
};

#endif  // MODELBOX_FLOWUNIT_DLENGINE_CPU_INFERENCE_H_
