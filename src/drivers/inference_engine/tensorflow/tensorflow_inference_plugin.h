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

#ifndef MODELBOX_INFER_PLUGIN_H_
#define MODELBOX_INFER_PLUGIN_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer_list.h>
#include <modelbox/data_context.h>

#include <vector>

#include "tensorflow/c/c_api.h"

class InferencePlugin {
 public:
  InferencePlugin() = default;
  virtual ~InferencePlugin() = default;

  virtual modelbox::Status PluginInit(
      std::shared_ptr<modelbox::Configuration> config) = 0;

  virtual modelbox::Status PreProcess(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<TF_Tensor *> &input_tf_tensor_list) = 0;

  virtual modelbox::Status PostProcess(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<TF_Tensor *> &output_tf_tensor_list) = 0;
};

extern "C" {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

MODELBOX_DLL_PUBLIC std::shared_ptr<InferencePlugin> CreatePlugin();

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

#endif