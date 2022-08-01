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

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>
#include <modelbox/buffer.h>
#include <modelbox/device/cuda/device_cuda.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <modelbox/tensor.h>
#include <modelbox/tensor_list.h>

#include <typeinfo>

#include "tensorrt_inference_plugin.h"

constexpr const char* FLOWUNIT_TYPE = "cuda";
constexpr const char* INFERENCE_TYPE = "tensorrt";
const std::string SUFFIX_ENGINE = "engine";
const std::string SUFFIX_UFF = "uff";
const std::string SUFFIX_ONNX = "onnx";
const std::string SUFFIX_PROTXT = "prototxt";

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;
using TensorRTProcess =
    std::function<modelbox::Status(std::shared_ptr<modelbox::DataContext>)>;

class RndInt8Calibrator;

class iLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // suppress info-level messages
    if (severity < Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};

extern iLogger gLogger;

class TensorRTParams {
 public:
  TensorRTParams() = default;
  virtual ~TensorRTParams() = default;

  modelbox::Status Clear();
  // caffe model file
  // .prototxt net file
  std::string deploy_file;
  // .caffemodel weight file
  std::string model_file;
  // uff model file
  std::string uff_file;
  std::vector<std::pair<std::string, std::shared_ptr<nvinfer1::Dims>>>
      uff_input_list;
  // onnx model file
  std::string onnx_model_file;
  // tensorrt engine file
  std::string engine;

  std::vector<std::string> inputs_name_list, outputs_name_list;
  std::vector<std::string> outputs_type_list;
  std::string calibration_cache{"CalibrationTable"};
  std::string plugin;
  int device{0};
  int onnx_opt_batch_size{1};
  int onnx_max_batch_size{1};
  int workspace_size{16};
  int use_DLACore{-1};
  bool dynamic_batch_contain{false};
  bool dynamic_batch{false};
  bool use_enqueue_v2{false};
  bool fp16{false};
  bool int8{false};
  bool verbose{false};
  bool allow_GPUFallback{false};
  float pct{99};
};

class TensorRTInferenceFlowUnit : public modelbox::CudaFlowUnit {
 public:
  TensorRTInferenceFlowUnit();
  ~TensorRTInferenceFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration>& opts) override;

  modelbox::Status Close() override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  /* run when processing data */
  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               cudaStream_t stream) override;

 private:
  void SetUpOtherConfig(std::shared_ptr<modelbox::Configuration> config);
  modelbox::Status InitConfig(
      const std::shared_ptr<modelbox::Configuration>& fu_config);
  modelbox::Status CreateEngine(
      const std::shared_ptr<modelbox::Configuration>& config);
  modelbox::Status SetUpModelFile(
      std::shared_ptr<modelbox::Configuration> config,
      const std::string& model_file);
  modelbox::Status SetUpDynamicLibrary(
      std::shared_ptr<modelbox::Configuration> config);
  modelbox::Status SetUpInferencePlugin(
      std::shared_ptr<modelbox::Configuration> config);
  void configureBuilder(std::shared_ptr<IBuilder> builder,
                        RndInt8Calibrator& calibrator);
  modelbox::Status PrePareOutput(
      std::shared_ptr<modelbox::DataContext>& data_ctx,
      std::vector<void*>& memory);
  modelbox::Status PrePareInput(
      std::shared_ptr<modelbox::DataContext>& data_ctx,
      std::vector<void*>& memory);
  modelbox::Status PreProcess(std::shared_ptr<modelbox::DataContext> data_ctx);
  modelbox::Status PostProcess(std::shared_ptr<modelbox::DataContext> data_ctx);
  modelbox::Status CreateMemory(
      std::vector<void*>& buffers, const std::string& name,
      const std::string& type,
      std::shared_ptr<modelbox::BufferList>& output_buf, size_t size);
  modelbox::Status BindMemory(std::vector<void*>& buffers,
                              const std::string& name, const void* mem,
                              size_t mem_size, size_t size);
  modelbox::Status EngineToModel(
      const std::shared_ptr<modelbox::Configuration>& config);
  void PrintModelBindInfo(const std::vector<std::string>& name_list);
  modelbox::Status UffToTRTModel(
      const std::shared_ptr<modelbox::Configuration>& config,
      std::shared_ptr<IBuilder>& builder,
      std::shared_ptr<INetworkDefinition>& network);
  modelbox::Status OnnxToTRTModel(
      const std::shared_ptr<modelbox::Configuration>& config,
      std::shared_ptr<IBuilder>& builder,
      std::shared_ptr<INetworkDefinition>& network);
  modelbox::Status CaffeToTRTModel(
      const std::shared_ptr<modelbox::Configuration>& config,
      std::shared_ptr<IBuilder>& builder,
      std::shared_ptr<INetworkDefinition>& network);
  void SetPluginFactory(std::string pluginName);

  TensorRTProcess pre_process_{nullptr}, post_process_{nullptr};
  TensorRTProcess data_pre_{nullptr}, data_post_{nullptr};
  TensorRTParams params_;
  std::string plugin_;
  void* driver_handler_{nullptr};
  std::shared_ptr<TensorRTInferencePlugin> inference_plugin_{nullptr};

  std::shared_ptr<ICudaEngine> engine_{nullptr};
  std::shared_ptr<IExecutionContext> context_{nullptr};
  std::shared_ptr<nvinfer1::IPluginFactory> plugin_factory_{nullptr};
  std::map<std::string, nvinfer1::Dims3> input_dims_;
};

class RndInt8Calibrator : public IInt8EntropyCalibrator {
 public:
  RndInt8Calibrator(int total_samples, std::string cache_file,
                    std::map<std::string, nvinfer1::Dims3>& input_dims);

  ~RndInt8Calibrator() override;
  int getBatchSize() const noexcept override;
  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override;
  const void* readCalibrationCache(size_t& length) noexcept override;
  void writeCalibrationCache(const void* /*ptr*/,
                             size_t /*length*/) noexcept override;

 private:
  int total_samples_{0};
  int current_sample_{0};
  std::string cache_file_;
  std::map<std::string, void*> input_device_buffers_;
  std::vector<char> calibration_cache_;
};

class TensorRTInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  TensorRTInferenceFlowUnitFactory() = default;
  ~TensorRTInferenceFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string& unit_name, const std::string& unit_type,
      const std::string& virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; };
  std::string GetVirtualType() override { return INFERENCE_TYPE; };

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  };
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_H_
