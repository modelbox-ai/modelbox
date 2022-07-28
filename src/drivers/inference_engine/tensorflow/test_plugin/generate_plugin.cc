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

#include "generate_plugin.h"

static std::map<std::string, TF_DataType> map = {
    {"FLOAT", TF_FLOAT}, {"DOUBLE", TF_DOUBLE}, {"INT", TF_INT32},
    {"UINT8", TF_UINT8}, {"LONG", TF_INT64},    {"STRING", TF_STRING}};

std::shared_ptr<InferencePlugin> CreatePlugin() {
  return std::make_shared<OriginInferencePlugin>();
}

modelbox::Status OriginInferencePlugin::ConvertType(const std::string &type,
                                                    TF_DataType &TFType) {
  auto iter = map.find(type);
  if (iter == map.end()) {
    MBLOG_ERROR << "unsupported type " << type;
    return {modelbox::STATUS_BADCONF, "unsuppored type"};
  }

  TFType = map[type];
  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::CreateOutputBufferList(
    std::shared_ptr<modelbox::BufferList> &output_buffer_list,
    const std::vector<size_t> &shape_vector, void *tensor_data,
    size_t tensor_byte, int index) {
  auto type_output_temp = output_type_list_[index];
  auto status =
      output_buffer_list->BuildFromHost(shape_vector, tensor_data, tensor_byte);
  if (type_output_temp == "float") {
    output_buffer_list->Set("type", modelbox::MODELBOX_FLOAT);
  } else if (type_output_temp == "double") {
    output_buffer_list->Set("type", modelbox::MODELBOX_DOUBLE);
  } else if (type_output_temp == "int") {
    output_buffer_list->Set("type", modelbox::MODELBOX_INT32);
  } else if (type_output_temp == "uint8") {
    output_buffer_list->Set("type", modelbox::MODELBOX_UINT8);
  } else if (type_output_temp == "long") {
    output_buffer_list->Set("type", modelbox::MODELBOX_INT16);
  } else {
    return {modelbox::STATUS_NOTSUPPORT, "unsupport output type."};
  }

  if (status != modelbox::STATUS_OK) {
    auto err_msg = "output buffer list builds error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::PluginInit(
    std::shared_ptr<modelbox::Configuration> config) {
  modelbox::Status status = modelbox::STATUS_OK;
  std::vector<std::string> names, types;
  status = SetUpInputOutput(config, "input", names, types);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "set up input failed, error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  input_name_list_.swap(names);
  input_type_list_.swap(types);

  status = SetUpInputOutput(config, "output", names, types);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "set up output failed, error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  output_name_list_.swap(names);
  output_type_list_.swap(types);

  return status;
}

modelbox::Status OriginInferencePlugin::SetUpInputOutput(
    std::shared_ptr<modelbox::Configuration> config, const std::string &type,
    std::vector<std::string> &names, std::vector<std::string> &types) {
  auto keys = config->GetSubKeys(type);
  for (unsigned int i = 1; i <= keys.size(); ++i) {
    std::string inner_name, inner_type;
    auto key = type + "." + type + std::to_string(i);
    auto item_table = config->GetSubKeys(key);
    if (item_table.empty()) {
      auto err_msg = "the key " + key + " is not found in config file.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto name_index = key + ".name";
    inner_name = config->GetString(name_index);
    if (inner_name.empty()) {
      auto err_msg = "the key " + key + " should have key name.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    auto type_index = key + ".type";
    inner_type = config->GetString(type_index);
    if (inner_type.empty()) {
      auto err_msg = "the key " + key + " should have key type.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    names.push_back(inner_name);
    types.push_back(inner_type);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::PreProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<TF_Tensor *> &input_tf_tensor_list) {
  int index = 0;
  modelbox::Status status;
  for (const auto &input_name : input_name_list_) {
    const auto input_buf = data_ctx->Input(input_name);

    std::string type = input_type_list_[index++];
    std::transform(type.begin(), type.end(), type.begin(), ::toupper);
    TF_DataType tf_type;
    status = ConvertType(type, tf_type);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "input type convert failed. " << status.WrapErrormsgs();
      return {status, "input type convert failed."};
    }

    std::vector<size_t> buffer_shape;
    auto result = input_buf->At(0)->Get("shape", buffer_shape);
    if (!result) {
      MBLOG_ERROR << "the input buffer don't have meta shape.";
      return {modelbox::STATUS_FAULT,
              "the input buffer don't have meta shape."};
    }

    if (std::any_of(input_buf->begin(), input_buf->end(),
                    [&](const std::shared_ptr<modelbox::Buffer> &buffer) {
                      std::vector<size_t> shape;
                      buffer->Get("shape", shape);
                      return shape != buffer_shape;
                    })) {
      MBLOG_ERROR << "the input shapes are not the same.";
      return {modelbox::STATUS_FAULT, "the input shapes are not the same."};
    }

    std::vector<int64_t> tf_dims{static_cast<int64_t>(input_buf->Size())};
    copy(buffer_shape.begin(), buffer_shape.end(), back_inserter(tf_dims));

    TF_Tensor *input_tensor = TF_NewTensor(
        tf_type, tf_dims.data(), tf_dims.size(),
        const_cast<void *>(input_buf->ConstData()), input_buf->GetBytes(),
        [](void *data, size_t length, void *arg) {}, nullptr);
    if (nullptr == input_tensor) {
      auto err_msg = "TF_NewTensor " + std::string(input_name) + " failed.";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
    input_tf_tensor_list.push_back(input_tensor);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OriginInferencePlugin::PostProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<TF_Tensor *> &output_tf_tensor_list) {
  int index = 0;
  for (const auto &output_name : output_name_list_) {
    auto tensor_byte = TF_TensorByteSize(output_tf_tensor_list[index]);
    auto tensor_data = TF_TensorData(output_tf_tensor_list[index]);
    std::vector<size_t> output_shape;

    int64_t num_dims = TF_NumDims(output_tf_tensor_list[index]);
    if (0 == num_dims) {
      auto err_msg = "the size of the " + std::string(output_name) + "is null.";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    for (int i = 1; i < num_dims; ++i) {
      output_shape.push_back(TF_Dim(output_tf_tensor_list[index], i));
    }

    auto num = TF_Dim(output_tf_tensor_list[index], 0);
    if (num == 0) {
      return {modelbox::STATUS_INVALID, "output tensor dim is zero"};
    }

    auto output_buf = data_ctx->Output(output_name);
    auto single_bytes = tensor_byte / num;
    std::vector<size_t> shape_vector(num, single_bytes);
    auto status = CreateOutputBufferList(output_buf, shape_vector, tensor_data,
                                         tensor_byte, index);
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "postProcess failed." + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    output_buf->Set("shape", output_shape);

    index++;
  }

  return modelbox::STATUS_OK;
}