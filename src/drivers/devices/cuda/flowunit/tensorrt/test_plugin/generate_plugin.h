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


#ifndef MODELBOX_SAMPLE_INFER_PLUGIN_H_
#define MODELBOX_SAMPLE_INFER_PLUGIN_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer_list.h>
#include <modelbox/data_context.h>

#include "tensorrt_inference_plugin.h"

class OriginInferencePlugin : public TensorRTInferencePlugin {
 public:
  OriginInferencePlugin() = default;
  virtual ~OriginInferencePlugin() = default;

  modelbox::Status PreProcess(std::shared_ptr<modelbox::DataContext> ctx) override;

  modelbox::Status PostProcess(std::shared_ptr<modelbox::DataContext> ctx) override;

  modelbox::Status PluginInit(
      std::shared_ptr<modelbox::Configuration> config) override;

  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> ctx) override;
  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  modelbox::Status SetUpInputOutput(std::shared_ptr<modelbox::Configuration> config,
                                  const std::string &type,
                                  std::vector<std::string> &names,
                                  std::vector<std::string> &types);
  std::vector<std::string> input_name_list_, output_name_list_;
  std::vector<std::string> input_type_list_, output_type_list_;
};

#endif