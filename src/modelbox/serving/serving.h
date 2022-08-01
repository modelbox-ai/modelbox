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

#ifndef MODELBOX_SERVING_H_
#define MODELBOX_SERVING_H_

#include <vector>

#include "modelbox/base/configuration.h"
#include "modelbox/base/status.h"
#ifdef BUILD_TEST
constexpr const char *DEFAULT_GRAPTH_PATH = "/tmp";
constexpr const char *DEFAULT_FLOWUNIT_PATH = "/tmp";
#else
constexpr const char *DEFAULT_GRAPTH_PATH = "/usr/local/etc/modelbox/graph";
constexpr const char *DEFAULT_FLOWUNIT_PATH = "/tmp";
#endif

class ModelServingConfig {
 public:
  ModelServingConfig() = default;
  virtual ~ModelServingConfig() {
    input_names_.clear();
    output_names_.clear();
    input_types_.clear();
    output_types_.clear();
  }

  std::string GetModelEntry() { return model_entry_; }
  std::string GetModelEngine() { return model_engine_; }
  int64_t GetMaxBatchSize() { return max_batch_size_; }
  std::vector<std::string> GetDevices() { return devices_; }
  std::string GetMode() { return mode_; }
  std::vector<std::string> GetInputNames() { return input_names_; }
  std::vector<std::string> GetOutputNames() { return output_names_; }
  std::vector<std::string> GetInputTypes() { return input_types_; }
  std::vector<std::string> GetOutputTypes() { return output_types_; }

  void SetModelEntry(const std::string &model_entry) {
    model_entry_ = model_entry;
  }

  void SetModelEngine(const std::string &model_engine) {
    model_engine_ = model_engine;
  }

  void SetMaxBatchSize(int64_t max_batch_size) {
    max_batch_size_ = max_batch_size;
  }

  void SetDevices(const std::vector<std::string> &devices) {
    devices_ = devices;
  }

  void SetMode(const std::string &mode) {
    mode_ = mode;
  }

  void SetInputNames(const std::vector<std::string> &input_names) {
    input_names_ = input_names;
  }

  void SetOutputNames(const std::vector<std::string> &output_names) {
    output_names_ = output_names;
  }

  void SetInputTypes(const std::vector<std::string> &input_types) {
    input_types_ = input_types;
  }

  void SetOutputTypes(const std::vector<std::string> &output_types) {
    output_types_ = output_types;
  }

  std::string model_entry_;
  std::string model_engine_;
  int64_t max_batch_size_;
  std::vector<std::string> devices_;
  std::string mode_;

  std::vector<std::string> input_names_, output_names_;
  std::vector<std::string> input_types_, output_types_;
};

class ModelServing {
 public:
  ModelServing() = default;
  virtual ~ModelServing() = default;

  modelbox::Status GenerateTemplate(const std::string &model_name,
                                    const std::string &model_path, int port);

 private:
  modelbox::Status CheckConfigFiles(const std::string &model_path);
  modelbox::Status ParseModelToml();
  modelbox::Status FillModelItem(const std::string &type);

  std::string GetDeviceType(const std::string &model_engine);
  modelbox::Status GenerateModelServingTemplate(const std::string &model_name,
                                                int port);
  modelbox::Status GenerateDefaultGraphConfig(const std::string &model_name,
                                              int port);
  modelbox::Status GenerateInferConfig(const std::string &default_flowunit_path,
                                       const std::string &model_name);
  modelbox::Status GeneratePrePostConfig(
      const std::string &default_flowunit_path, const std::string &type);
  modelbox::Status GeneratePrePostFlowUnit(const std::string &default_file_path,
                                           const std::string &type);
  modelbox::Status GenerateDefaultPrePostFlowUnit(
      const std::string &default_file_path, const std::string &type);
  modelbox::Status UpdateGraphTemplateByToml(const std::string &model_name);

  modelbox::Status UpdatePreFlowUnit(const std::string &model_name);
  modelbox::Status UpdatePostFlowUnit(const std::string &model_name);
  modelbox::Status UpdateGraphToml(const std::string &model_name);

  std::string model_toml_;
  std::string model_custom_service_file_;
  std::string graph_toml_file_;
  bool custom_service_{false};
  ModelServingConfig model_serving_config_;
  std::shared_ptr<modelbox::Configuration> config_;
};

#endif  // MODELBOX_SERVING_H_
