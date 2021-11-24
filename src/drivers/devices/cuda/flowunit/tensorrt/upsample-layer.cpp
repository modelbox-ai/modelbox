
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


#include "upsample-layer.h"

#include <modelbox/device/cuda/device_cuda.h>

#include "modelbox/base/log.h"

namespace nvinfer1 {
template <typename T>
void write(char *&buffer, const T &val) {
  *reinterpret_cast<T *>(buffer) = val;
  buffer += sizeof(T);
}

template <typename T>
void read(const char *&buffer, T &val) {
  val = *reinterpret_cast<const T *>(buffer);
  buffer += sizeof(T);
}

UpsampleLayerPlugin2::UpsampleLayerPlugin2(const float scale,
                                           const float cudaThread /*= 512*/)
    : mScale(scale), mThreadCount(cudaThread) {
  mScale = 0.0;
  mOutputWidth = 0;
  mOutputHeight = 0;
  mThreadCount = 0;
}

UpsampleLayerPlugin2::~UpsampleLayerPlugin2() {}

// create the plugin at runtime from a byte stream
UpsampleLayerPlugin2::UpsampleLayerPlugin2(const void *data, size_t length) {
  const char *d = reinterpret_cast<const char *>(data);
  const char *a = d;
  read(d, mCHW);
  read(d, mDataType);
  read(d, mScale);
  read(d, mOutputWidth);
  read(d, mOutputHeight);
  read(d, mThreadCount);
  if (d != a + length) {
    MBLOG_ERROR << "create plugin from byte stream error.";
  }
}

void UpsampleLayerPlugin2::serialize(void *buffer) {
  char *d = static_cast<char *>(buffer);
  char *a = d;
  write(d, mCHW);
  write(d, mDataType);
  write(d, mScale);
  write(d, mOutputWidth);
  write(d, mOutputHeight);
  write(d, mThreadCount);
  if (d != a + getSerializationSize()) {
    MBLOG_ERROR << "create plugin from byte serialization data error.";
  }
}

int UpsampleLayerPlugin2::initialize() {
  int inputHeight = mCHW.d[1];
  int inputWidth = mCHW.d[2];

  mOutputHeight = inputHeight * mScale;
  mOutputWidth = inputWidth * mScale;

  int totalElems = mCHW.d[0] * mCHW.d[1] * mCHW.d[2];
  cudaHostAlloc(&mInputBuffer, totalElems * type2size(mDataType),
                cudaHostAllocDefault);

  totalElems = mCHW.d[0] * mOutputHeight * mOutputWidth;
  cudaHostAlloc(&mOutputBuffer, totalElems * type2size(mDataType),
                cudaHostAllocDefault);

  return 0;
}

void UpsampleLayerPlugin2::configureWithFormat(
    const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
    DataType type, PluginFormat format, int maxBatchSize) {
  if (type != DataType::kFLOAT && type != DataType::kHALF) {
    MBLOG_ERROR << "unsupport data type.";
    return;
  }

  if (format != PluginFormat::kNCHW) {
    MBLOG_ERROR << "unsupport data format.";
    return;
  }

  mDataType = type;
}

// it is called prior to any call to initialize().
Dims UpsampleLayerPlugin2::getOutputDimensions(int index, const Dims *inputs,
                                               int nbInputDims) {
  mCHW = inputs[0];
  mOutputHeight = inputs[0].d[1] * mScale;
  mOutputWidth = inputs[0].d[2] * mScale;
  return Dims3(mCHW.d[0], mOutputHeight, mOutputWidth);
}
}  // namespace nvinfer1