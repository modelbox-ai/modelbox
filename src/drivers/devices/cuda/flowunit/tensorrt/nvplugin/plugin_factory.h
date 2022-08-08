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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_PLUGIN_FACTORY_H
#define MODELBOX_FLOWUNIT_INFERENCE_PLUGIN_FACTORY_H

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include "upsample-layer.h"

static constexpr float NEG_SLOPE2 = 0.1;
static constexpr float UPSAMPLE_SCALE2 = 2.0;
static constexpr int CUDA_THREAD_NUM2 = 512;

// Integration for serialization.
class YoloPluginFactory : public nvinfer1::IPluginFactory,
                          public nvcaffeparser1::IPluginFactoryExt {
 public:
  // NOLINTNEXTLINE
  virtual ~YoloPluginFactory() = default;
  inline bool isLeakyRelu(const char *layerName) {
    std::string src(layerName);
    bool LeakyRelu_flag = src.find("leaky", 0) != std::string::npos and
                          src.find("layer", 0) != std::string::npos;
    return LeakyRelu_flag;
  }

  inline bool isUpsample(const char *layerName) {
    std::string src(layerName);
    bool Upsample_flag = src.find("upsample", 0) != std::string::npos and
                         src.find("layer", 0) != std::string::npos;
    return Upsample_flag;
  }

  nvinfer1::IPlugin *createPlugin(const char *layerName,
                                  const nvinfer1::Weights *weights,
                                  int nbWeights) override {
    if (isPlugin(layerName) == false) {
      MBLOG_ERROR << "plugin layername is null";
      return nullptr;
    }

#ifndef TENSORRT7
    if (isLeakyRelu(layerName)) {
      if (nbWeights != 0 || weights != nullptr) {
        MBLOG_ERROR << "weights error.";
        return nullptr;
      }

      auto plugin = std::unique_ptr<nvinfer1::plugin::INvPlugin,
                                    void (*)(nvinfer1::plugin::INvPlugin *)>(
          nvinfer1::plugin::createPReLUPlugin(NEG_SLOPE2), nvPluginDeleter);
      mPluginLeakyRelu.emplace_back(std::move(plugin));
      return mPluginLeakyRelu.back().get();
    }
#endif

    if (isUpsample(layerName)) {
      if (nbWeights != 0 || weights != nullptr) {
        MBLOG_ERROR << "weights error.";
        return nullptr;
      }

      auto plugin = std::unique_ptr<nvinfer1::UpsampleLayerPlugin2>(
          new nvinfer1::UpsampleLayerPlugin2(UPSAMPLE_SCALE2,
                                             CUDA_THREAD_NUM2));
      mPluginUpsample.emplace_back(std::move(plugin));
      return mPluginUpsample.back().get();
    }

    return nullptr;
  }

  nvinfer1::IPlugin *createPlugin(const char *layerName, const void *serialData,
                                  size_t serialLength) override {
    if (isPlugin(layerName) == false) {
      MBLOG_ERROR << "plugin layername is null";
      return nullptr;
    }

#ifndef TENSORRT7
    if (isLeakyRelu(layerName)) {
      auto plugin = std::unique_ptr<nvinfer1::plugin::INvPlugin,
                                    void (*)(nvinfer1::plugin::INvPlugin *)>(
          nvinfer1::plugin::createPReLUPlugin(serialData, serialLength),
          nvPluginDeleter);
      mPluginLeakyRelu.emplace_back(std::move(plugin));
      return mPluginLeakyRelu.back().get();
    }
#endif

    if (isUpsample(layerName)) {
      auto plugin = std::unique_ptr<nvinfer1::UpsampleLayerPlugin2>(
          new nvinfer1::UpsampleLayerPlugin2(serialData, serialLength));
      mPluginUpsample.emplace_back(std::move(plugin));
      return mPluginUpsample.back().get();
    }

    return nullptr;
  }

  bool isPlugin(const char *name) override { return isPluginExt(name); }

  bool isPluginExt(const char *name) override {
#ifndef TENSORRT7
    return (isLeakyRelu(name) or isUpsample(name));
#else
    return isUpsample(name);
#endif
  }

  // The application has to destroy the plugin when it knows it's safe to do so.
  void destroyPlugin() {
    for (auto &item : mPluginUpsample) {
      item.reset();
    }
  }

  void (*nvPluginDeleter)(nvinfer1::plugin::INvPlugin *){
      [](nvinfer1::plugin::INvPlugin *ptr) { ptr->destroy(); }};

  std::vector<std::unique_ptr<nvinfer1::plugin::INvPlugin,
                              void (*)(nvinfer1::plugin::INvPlugin *)>>
      mPluginLeakyRelu{};
  std::vector<std::unique_ptr<nvinfer1::UpsampleLayerPlugin2>>
      mPluginUpsample{};
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_PLUGIN_FACTORY_H
