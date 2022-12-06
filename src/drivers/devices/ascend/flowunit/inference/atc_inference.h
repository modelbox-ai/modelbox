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

#ifndef MODELBOX_FLOWUNIT_ATC_INFERENCE_H_
#define MODELBOX_FLOWUNIT_ATC_INFERENCE_H_

#include <acl/acl.h>
#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>

#include <string>
#include <vector>

class AtcInference {
 public:
  modelbox::Status Init(const std::string &model_file,
                        const std::shared_ptr<modelbox::Configuration> &config,
                        const std::vector<std::string> &unit_input_list,
                        const std::vector<std::string> &unit_output_list,
                        const std::shared_ptr<modelbox::Drivers> &drivers_ptr);

  modelbox::Status Infer(std::shared_ptr<modelbox::DataContext> &data_ctx,
                         aclrtStream stream);

  modelbox::Status Deinit();

 private:
  modelbox::Status ParseConfig(
      const std::shared_ptr<modelbox::Configuration> &config);

  modelbox::Status LoadModel(
      const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
      const std::shared_ptr<modelbox::Configuration> &config);

  modelbox::Status GetModelDesc();

  modelbox::Status CheckModelIO(
      const std::vector<std::string> &unit_input_list,
      const std::vector<std::string> &unit_output_list);

  void ReadModelInfo();

  void SaveOutputShape(const aclmdlIODims &dims);

  void SaveBatchInfo(aclmdlDesc *desc_ptr, std::stringstream &model_info,
                     size_t &max_batch_size);

  void LogTensorInfo(aclmdlDesc *desc_ptr, size_t index, aclmdlIODims &dims,
                     size_t size, aclFormat format, aclDataType data_type,
                     std::stringstream &model_info);

  std::string GetFormatStr(aclFormat format);

  std::string GetDataTypeStr(aclDataType data_type);

  modelbox::ModelBoxDataType GetModelBoxDataType(aclDataType data_type);

  modelbox::Status PrepareOutput(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      const size_t &current_batch_size);

  std::shared_ptr<aclmdlDataset> CreateDataSet(
      const std::shared_ptr<modelbox::BufferListMap> &buffer_list_map,
      std::vector<std::string> &name_list, const size_t &current_batch_size);

  modelbox::Status GetCurrentBatchSize(
      std::shared_ptr<modelbox::DataContext> &data_ctx, size_t &batch_size);

  int32_t device_id_{0};
  std::string model_file_;
  int32_t dynamic_batch_tensor_index_{-1};
  void *dynamic_batch_mem_ptr_{nullptr};
  std::set<size_t> dynamic_batch_set_;

  uint32_t model_id_{0};
  bool is_model_load_{false};
  std::shared_ptr<aclmdlDesc> model_desc_{nullptr};
  std::vector<std::string> model_input_list_;
  std::vector<std::string> model_output_list_;
  std::vector<size_t> model_input_size_;
  std::vector<size_t> model_output_size_;
  std::vector<std::vector<size_t>> output_shape_;
  std::vector<modelbox::ModelBoxDataType> output_data_type_;

  std::map<aclFormat, std::string> format_str_{
      {ACL_FORMAT_UNDEFINED, "UNDEFINED"}, {ACL_FORMAT_NCHW, "NCHW"},
      {ACL_FORMAT_NHWC, "NHWC"},           {ACL_FORMAT_ND, "ND"},
      {ACL_FORMAT_NC1HWC0, "NC1HWC0"},     {ACL_FORMAT_FRACTAL_Z, "FRACTAL_Z"}};
  std::map<aclDataType, std::string> data_type_str_{
      {ACL_DT_UNDEFINED, "UNDEFINED"},
      {ACL_FLOAT, "FLOAT"},
      {ACL_FLOAT16, "FLOAT16"},
      {ACL_INT8, "INT8"},
      {ACL_INT32, "INT32"},
      {ACL_UINT8, "UINT8"},
      {ACL_INT16, "INT16"},
      {ACL_UINT16, "UINT16"},
      {ACL_UINT32, "UINT32"},
      {ACL_INT64, "INT64"},
      {ACL_UINT64, "UINT64"},
      {ACL_DOUBLE, "DOUBLE"},
      {ACL_BOOL, "BOOL"}};
  std::map<aclDataType, modelbox::ModelBoxDataType> data_type_flow_{
      {ACL_FLOAT, modelbox::MODELBOX_FLOAT},
      {ACL_FLOAT16, modelbox::MODELBOX_HALF},
      {ACL_INT8, modelbox::MODELBOX_INT8},
      {ACL_INT32, modelbox::MODELBOX_INT32},
      {ACL_UINT8, modelbox::MODELBOX_UINT8},
      {ACL_INT16, modelbox::MODELBOX_INT16},
      {ACL_UINT16, modelbox::MODELBOX_UINT16},
      {ACL_UINT32, modelbox::MODELBOX_UINT32},
      {ACL_INT64, modelbox::MODELBOX_INT64},
      {ACL_UINT64, modelbox::MODELBOX_UINT64},
      {ACL_DOUBLE, modelbox::MODELBOX_DOUBLE},
      {ACL_BOOL, modelbox::MODELBOX_BOOL}};
};

#endif