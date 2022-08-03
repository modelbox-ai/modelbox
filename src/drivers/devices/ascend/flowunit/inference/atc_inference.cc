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

#include "atc_inference.h"

#include <acl/acl.h>
#include <model_decrypt.h>
#include <modelbox/base/log.h>

#include <cstdint>
#include <utility>

modelbox::Status AtcInference::Init(
    const std::string &model_file,
    const std::shared_ptr<modelbox::Configuration> &config,
    const std::vector<std::string> &unit_input_list,
    const std::vector<std::string> &unit_output_list,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr) {
  model_file_ = model_file;
  auto ret = ParseConfig(config);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  auto acl_ret = aclrtSetDevice(device_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclrtSetDevice failed, ret " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  ret = LoadModel(drivers_ptr, config);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = GetModelDesc();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ReadModelInfo();
  ret = CheckModelIO(unit_input_list, unit_output_list);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status AtcInference::ParseConfig(
    const std::shared_ptr<modelbox::Configuration> &config) {
  device_id_ = config->GetInt32("deviceid");
  batch_size_ = config->GetInt32("batch_size", 1);
  MBLOG_INFO << "Model batch size " << batch_size_;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status AtcInference::LoadModel(
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    const std::shared_ptr<modelbox::Configuration> &config) {
  aclError ret = ACL_ERROR_INVALID_FILE;
  ModelDecryption model_decrypt;
  if (modelbox::STATUS_SUCCESS !=
      model_decrypt.Init(model_file_, drivers_ptr, config)) {
    MBLOG_ERROR << "init model fail";
    return modelbox::STATUS_FAULT;
  }

  if (model_decrypt.GetModelState() == ModelDecryption::MODEL_STATE_ENCRYPT) {
    int64_t model_len = 0;
    std::shared_ptr<uint8_t> modelBuf =
        model_decrypt.GetModelSharedBuffer(model_len);
    if (!modelBuf) {
      MBLOG_ERROR << "GetDecryptModelBuffer fail";
      return modelbox::STATUS_FAULT;
    }
    ret = aclmdlLoadFromMem((char *)(modelBuf.get()), model_len, &model_id_);
  } else if (model_decrypt.GetModelState() ==
             ModelDecryption::MODEL_STATE_PLAIN) {
    ret = aclmdlLoadFromFile(model_file_.c_str(), &model_id_);
  }
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclmdlLoadFromFile failed, ret " << ret
                << ", model:" << model_file_;
    return modelbox::STATUS_FAULT;
  }

  is_model_load_ = true;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status AtcInference::GetModelDesc() {
  auto *desc = aclmdlCreateDesc();
  if (desc == nullptr) {
    MBLOG_ERROR << "aclmdlCreateDesc failed";
    return modelbox::STATUS_FAULT;
  }

  model_desc_.reset(desc, [](aclmdlDesc *desc) { aclmdlDestroyDesc(desc); });
  auto ret = aclmdlGetDesc(model_desc_.get(), model_id_);
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclmdlGetDesc failed, model:" << model_file_;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status AtcInference::CheckModelIO(
    const std::vector<std::string> &unit_input_list,
    const std::vector<std::string> &unit_output_list) {
  std::set<std::string> unit_input_set(unit_input_list.begin(),
                                       unit_input_list.end());
  std::set<std::string> unit_output_set(unit_output_list.begin(),
                                        unit_output_list.end());
  if (model_input_list_.empty() || model_output_list_.empty() ||
      model_input_list_.size() != unit_input_list.size() ||
      model_output_list_.size() != unit_output_list.size()) {
    MBLOG_ERROR << "Model input[" << model_input_list_.size() << "], output["
                << model_output_list_.size() << "], FlowUnit input["
                << unit_input_list.size() << "], output["
                << unit_output_list.size() << "], these io count is bad";
    return modelbox::STATUS_BADCONF;
  }

  for (auto &model_input_name : model_input_list_) {
    if (unit_input_set.find(model_input_name) == unit_input_set.end()) {
      MBLOG_ERROR << "Model miss input [" << model_input_name
                  << "] in graph config";
      return modelbox::STATUS_BADCONF;
    }
  }

  for (auto &model_output_name : model_output_list_) {
    if (unit_output_set.find(model_output_name) == unit_output_set.end()) {
      MBLOG_ERROR << "Model miss output [" << model_output_name
                  << "] in graph config";
      return modelbox::STATUS_BADCONF;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

void AtcInference::ReadModelInfo() {
  auto input_num = aclmdlGetNumInputs(model_desc_.get());
  auto output_num = aclmdlGetNumOutputs(model_desc_.get());
  std::stringstream model_info;
  model_info << "Model:" << model_file_ << std::endl;
  auto *desc_ptr = model_desc_.get();
  LogBatchInfo(desc_ptr, model_info);
  model_info << "Input:" << std::endl;
  aclmdlIODims dims;
  for (size_t i = 0; i < input_num; ++i) {
    auto ret = aclmdlGetInputDims(desc_ptr, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "Get model info for input [" << i << "] failed, ret "
                  << ret;
      continue;
    }

    std::string name = aclmdlGetInputNameByIndex(desc_ptr, i);
    model_input_list_.push_back(name);
    auto size = aclmdlGetInputSizeByIndex(desc_ptr, i);
    model_input_size_.push_back(size);
    auto format = aclmdlGetInputFormat(desc_ptr, i);
    auto data_type = aclmdlGetInputDataType(desc_ptr, i);
    LogTensorInfo(desc_ptr, i, dims, size, format, data_type, model_info);
  }

  model_info << "Output:" << std::endl;
  for (size_t i = 0; i < output_num; ++i) {
    auto ret = aclmdlGetOutputDims(desc_ptr, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "Get model info for output [" << i << "] failed, ret "
                  << ret;
      continue;
    }

    SaveOutputShape(dims);
    std::string name = aclmdlGetOutputNameByIndex(desc_ptr, i);
    model_output_list_.push_back(name);
    auto size = aclmdlGetOutputSizeByIndex(desc_ptr, i);
    model_output_size_.push_back(size);
    auto format = aclmdlGetOutputFormat(desc_ptr, i);
    auto data_type = aclmdlGetOutputDataType(desc_ptr, i);
    output_data_type_.push_back(GetModelBoxDataType(data_type));
    LogTensorInfo(desc_ptr, i, dims, size, format, data_type, model_info);
  }

  MBLOG_INFO << model_info.str();
}

void AtcInference::SaveOutputShape(const aclmdlIODims &dims) {
  std::vector<size_t> shape;
  for (size_t i = 0; i < dims.dimCount; ++i) {
    shape.push_back(dims.dims[i]);
  }

  output_shape_.push_back(shape);
}

void AtcInference::LogBatchInfo(aclmdlDesc *desc_ptr,
                                std::stringstream &model_info) {
  aclmdlBatch batch;
  auto ret = aclmdlGetDynamicBatch(desc_ptr, &batch);
  if (ret != ACL_ERROR_NONE) {
    model_info << "Get dynamic batch failed, ret " << ret;
  } else {
    model_info << "Dynamic batch:[";
    for (size_t i = 0; i < batch.batchCount; ++i) {
      model_info << batch.batch[i];
      if (i + 1 == batch.batchCount) {
        model_info << "]";
      } else {
        model_info << ",";
      }
    }

    if (batch.batchCount == 0) {
      model_info << "]";
    }
  }
  model_info << std::endl;
}

void AtcInference::LogTensorInfo(aclmdlDesc *desc_ptr, size_t index,
                                 aclmdlIODims &dims, size_t size,
                                 aclFormat format, aclDataType data_type,
                                 std::stringstream &model_info) {
  model_info << "index:" << index;
  model_info << ",name:" << dims.name;
  model_info << ",dim:[";
  for (size_t j = 0; j < dims.dimCount; ++j) {
    model_info << dims.dims[j];
    if (j + 1 == dims.dimCount) {
      model_info << "]";
    } else {
      model_info << ",";
    }
  }
  if (dims.dimCount == 0) {
    model_info << "]";
  }

  model_info << ",size:" << size;
  model_info << ",format:" << GetFormatStr(format);
  model_info << ",data type:" << GetDataTypeStr(data_type);
  model_info << std::endl;
}

std::string AtcInference::GetFormatStr(aclFormat format) {
  auto item = format_str_.find(format);
  if (item != format_str_.end()) {
    return item->second;
  }

  return std::to_string(format);
}

std::string AtcInference::GetDataTypeStr(aclDataType data_type) {
  auto item = data_type_str_.find(data_type);
  if (item != data_type_str_.end()) {
    return item->second;
  }

  return std::to_string(data_type);
}

modelbox::ModelBoxDataType AtcInference::GetModelBoxDataType(
    aclDataType data_type) {
  auto item = data_type_flow_.find(data_type);
  if (item != data_type_flow_.end()) {
    return item->second;
  }

  return modelbox::ModelBoxDataType::MODELBOX_TYPE_INVALID;
}

modelbox::Status AtcInference::Infer(
    std::shared_ptr<modelbox::DataContext> &data_ctx, aclrtStream stream) {
  auto acl_ret = aclrtSetDevice(device_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclrtSetDevice failed, device_id " << device_id_ << ",ret "
                << acl_ret;
    return {modelbox::STATUS_FAULT, "Set device failed"};
  }

  auto input = CreateDataSet(data_ctx->Input(), model_input_list_);
  if (input == nullptr) {
    MBLOG_ERROR << "Create input for infer failed";
    return {modelbox::STATUS_FAULT, "Create input failed"};
  }

  auto ret = PrepareOutput(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Prepare output failed";
    return {modelbox::STATUS_FAULT, "Prepare output failed"};
  }

  auto output = CreateDataSet(data_ctx->Output(), model_output_list_);
  if (output == nullptr) {
    MBLOG_ERROR << "Create output for infer failed";
    return {modelbox::STATUS_FAULT, "Create output failed"};
  }

  acl_ret = ACL_ERROR_NONE;
  if (stream == nullptr) {
    acl_ret = aclmdlExecute(model_id_, input.get(), output.get());
  } else {
    acl_ret = aclmdlExecuteAsync(model_id_, input.get(), output.get(), stream);
    aclrtSynchronizeStream(stream);
  }

  if (acl_ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclmdlExecute failed, ret " << acl_ret;
    return {modelbox::STATUS_FAULT, "Execute acl infer failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status AtcInference::PrepareOutput(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto output_count = model_output_list_.size();
  for (size_t i = 0; i < output_count; ++i) {
    auto &name = model_output_list_[i];
    auto buffer_list = data_ctx->Output(name);
    auto &size = model_output_size_[i];
    std::vector<size_t> shape(batch_size_, size);
    buffer_list->Build(shape);
    buffer_list->Set("shape", output_shape_[i]);
    buffer_list->Set("type", output_data_type_[i]);
  }

  return modelbox::STATUS_SUCCESS;
}

std::shared_ptr<aclmdlDataset> AtcInference::CreateDataSet(
    const std::shared_ptr<modelbox::BufferListMap> &buffer_list_map,
    std::vector<std::string> &name_list) {
  auto *data_set_ptr = aclmdlCreateDataset();
  if (data_set_ptr == nullptr) {
    MBLOG_ERROR << "aclmdlCreateDataset return null";
    return nullptr;
  }

  std::shared_ptr<aclmdlDataset> data_set(data_set_ptr, [](aclmdlDataset *ptr) {
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(ptr); ++i) {
      aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(ptr, i);
      (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(ptr);
  });

  for (auto &tensor_name : name_list) {
    auto buffer_list = buffer_list_map->at(tensor_name);
    if (buffer_list == nullptr) {
      MBLOG_ERROR << "Create data set for tensor " << tensor_name
                  << " failed, buffer list is null";
      return nullptr;
    }

    auto *data_buffer = aclCreateDataBuffer(
        const_cast<void *>(buffer_list->ConstData()), buffer_list->GetBytes());
    if (data_buffer == nullptr) {
      MBLOG_ERROR << "Create data set buffer for tensor " << tensor_name
                  << "failed";
      return nullptr;
    }

    auto ret = aclmdlAddDatasetBuffer(data_set_ptr, data_buffer);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "Add data buffer to set failed for tensor " << tensor_name;
      aclDestroyDataBuffer(data_buffer);
      return nullptr;
    }
  }

  return data_set;
}

modelbox::Status AtcInference::Deinit() {
  if (!is_model_load_) {
    return modelbox::STATUS_SUCCESS;
  }

  auto acl_ret = aclrtSetDevice(device_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "aclrtSetDevice failed, ret " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "Unload model failed, model id is " << model_id_;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}