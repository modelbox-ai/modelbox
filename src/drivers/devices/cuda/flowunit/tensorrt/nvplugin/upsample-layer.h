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


#ifndef MODELBOX_FLOWUNIT_INFERENCE_UPSAMPLE_LAYER_H
#define MODELBOX_FLOWUNIT_INFERENCE_UPSAMPLE_LAYER_H

#include <NvInfer.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string.h>

#include <cmath>
#include <iostream>

namespace nvinfer1 {
class UpsampleLayerPlugin2 : public IPluginExt {
 public:
  explicit UpsampleLayerPlugin2(float scale, float cudaThread = 512);
  // create the plugin at runtime from a byte stream
  UpsampleLayerPlugin2(const void *data, size_t length);

  ~UpsampleLayerPlugin2() override;

  int getNbOutputs() const override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override;

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return (type == DataType::kFLOAT || type == DataType::kHALF ||
            type == DataType::kINT8) &&
           format == PluginFormat::kNCHW;
  }

  void configureWithFormat(const Dims *inputDims, int nbInputs,
                           const Dims *outputDims, int nbOutputs, DataType type,
                           PluginFormat format, int maxBatchSize) override;

  int initialize() override;

  void terminate() override{};

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;

  size_t getSerializationSize() override {
    return sizeof(nvinfer1::Dims) + sizeof(mDataType) + sizeof(mScale) +
           sizeof(mOutputWidth) + sizeof(mOutputHeight) + sizeof(mThreadCount);
  }

  void serialize(void *buffer) override;

  template <typename Dtype>
  void forwardGpu(const Dtype *input, Dtype *outputint, int N, int C, int H,
                  int W, cudaStream_t stream);

 private:
  size_t type2size(DataType type) {
    return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half);
  }

  nvinfer1::Dims mCHW = {0};
  DataType mDataType{DataType::kFLOAT};
  float mScale = 0.0;
  int mOutputWidth = 0;
  int mOutputHeight = 0;
  int mThreadCount = 0;

  void *mInputBuffer{nullptr};  // host
  void *mOutputBuffer{nullptr};
};
};  // namespace nvinfer1

#endif  // MODELBOX_FLOWUNIT_INFERENCE_UPSAMPLE_LAYER_H