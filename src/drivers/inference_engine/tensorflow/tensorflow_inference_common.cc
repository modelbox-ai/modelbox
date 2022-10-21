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

#include "tensorflow_inference_common.h"

#include <model_decrypt.h>
#include <modelbox/base/crypto.h>

#include <utility>

#include "modelbox/base/status.h"
#include "virtualdriver_inference.h"

static std::map<std::string, TF_DataType> type_map = {
    {"FLOAT", TF_FLOAT}, {"DOUBLE", TF_DOUBLE}, {"INT", TF_INT32},
    {"UINT8", TF_UINT8}, {"LONG", TF_INT64},    {"STRING", TF_STRING}};

static std::map<TF_DataType, modelbox::ModelBoxDataType> tftype_mbtype_map = {
    {TF_FLOAT, modelbox::MODELBOX_FLOAT},
    {TF_DOUBLE, modelbox::MODELBOX_DOUBLE},
    {TF_INT32, modelbox::MODELBOX_INT32},
    {TF_UINT8, modelbox::MODELBOX_UINT8},
    {TF_INT64, modelbox::MODELBOX_INT64},
    {TF_STRING, modelbox::MODELBOX_STRING}};

static std::map<modelbox::ModelBoxDataType, TF_DataType> mbtype_tftype_map = {
    {modelbox::MODELBOX_FLOAT, TF_FLOAT},
    {modelbox::MODELBOX_DOUBLE, TF_DOUBLE},
    {modelbox::MODELBOX_INT32, TF_INT32},
    {modelbox::MODELBOX_UINT8, TF_UINT8},
    {modelbox::MODELBOX_INT64, TF_INT64},
    {modelbox::MODELBOX_STRING, TF_STRING}};

modelbox::Status ConvertTFTypeToModelBoxType(
    TF_DataType tf_type, modelbox::ModelBoxDataType &modelbox_type) {
  auto iter = tftype_mbtype_map.find(tf_type);
  if (iter == tftype_mbtype_map.end()) {
    return {modelbox::STATUS_NOTSUPPORT,
            "covert Tensorflow Type to ModelBox Type failed, unsupport type "};
  }
  modelbox_type = iter->second;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status ConvertModelBoxTypeToTFType(
    modelbox::ModelBoxDataType modelbox_type, TF_DataType &tf_type) {
  auto iter = mbtype_tftype_map.find(modelbox_type);
  if (iter == mbtype_tftype_map.end()) {
    return {modelbox::STATUS_NOTSUPPORT,
            "covert ModelBox Type to Tensorflow Type failed, unsupport type " +
                std::to_string(modelbox_type)};
  }
  tf_type = iter->second;
  return modelbox::STATUS_SUCCESS;
}

void DeleteTensor(TF_Tensor *tensor) {
  if (tensor == nullptr) {
    return;
  }
  TF_DeleteTensor(tensor);
}

modelbox::Status InferenceTensorflowFlowUnit::ClearTensor(
    std::vector<TF_Tensor *> &input_tensor_list,
    std::vector<TF_Tensor *> &output_tensor_list) {
  for (auto &t : input_tensor_list) {
    DeleteTensor(t);
  }

  for (auto &t : output_tensor_list) {
    DeleteTensor(t);
  }

  input_tensor_list.clear();
  output_tensor_list.clear();
  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowParams::Clear() {
  input_name_list_.clear();
  output_name_list_.clear();
  input_type_list_.clear();
  output_type_list_.clear();
  input_op_list.clear();
  output_op_list.clear();

  if (nullptr != options) {
    TF_DeleteSessionOptions(options);
    options = nullptr;
  }

  if (nullptr != session && nullptr != status) {
    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
      auto err_msg = "close session failed: " + std::string(TF_Message(status));
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
      auto err_msg =
          "delete session failed: " + std::string(TF_Message(status));
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    session = nullptr;
  }

  if (nullptr != status) {
    TF_DeleteStatus(status);
    status = nullptr;
  }

  if (graph != nullptr) {
    TF_DeleteGraph(graph);
    graph = nullptr;
  }

  return modelbox::STATUS_OK;
}

InferenceTensorflowFlowUnit::InferenceTensorflowFlowUnit() = default;
InferenceTensorflowFlowUnit::~InferenceTensorflowFlowUnit() {
  pre_process_ = nullptr;
  post_process_ = nullptr;
  inference_plugin_ = nullptr;

  if (driver_handler_ != nullptr) {
    dlclose(driver_handler_);
    driver_handler_ = nullptr;
  }
};

modelbox::Status InferenceTensorflowFlowUnit::ReadBufferFromFile(
    const std::string &file, TF_Buffer *buf) {
  int64_t model_len = 0;
  auto config = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
                    this->GetFlowUnitDesc())
                    ->GetConfiguration();
  ModelDecryption model_decrypt;

  if (modelbox::STATUS_SUCCESS !=
      model_decrypt.Init(
          file, GetBindDevice()->GetDeviceManager()->GetDrivers(), config)) {
    return {modelbox::STATUS_INVALID, "int model failed."};
  }
  uint8_t *modelBuf = model_decrypt.GetModelBuffer(model_len);
  if (!modelBuf) {
    return {modelbox::STATUS_INVALID, "decrypt model data failed."};
  }
  buf->data = modelBuf;
  buf->length = model_len;
  buf->data_deallocator = [](void *data, size_t length) { free(data); };

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::LoadGraph(
    const std::string &model_path) {
  modelbox::Status status;
  MBLOG_INFO << "model path: " << model_path;
  if (model_path.empty()) {
    return {modelbox::STATUS_INVALID, "model path is empty."};
  }

  TF_Buffer *buffer = TF_NewBuffer();
  if (buffer == nullptr) {
    return {modelbox::STATUS_NOMEM, "create tf buffer failed."};
  }
  Defer { TF_DeleteBuffer(buffer); };

  status = ReadBufferFromFile(model_path, buffer);
  if (status != modelbox::STATUS_OK) {
    return {status, "load model failed."};
  }

  params_.graph = TF_NewGraph();
  if (nullptr == params_.graph) {
    return {modelbox::STATUS_FAULT, "TF_NewGraph() failed."};
  }

  auto *opts = TF_NewImportGraphDefOptions();
  if (nullptr == opts) {
    TF_DeleteGraph(params_.graph);
    auto err_msg = "TF_NewImportGraphDefOptions() failed: " +
                   std::string(TF_Message(params_.status));
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  TF_GraphImportGraphDef(params_.graph, buffer, opts, params_.status);
  if (TF_GetCode(params_.status) != TF_OK) {
    TF_DeleteGraph(params_.graph);
    auto err_msg = "TF_GraphImportGraphDef failed: " +
                   std::string(TF_Message(params_.status));
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  TF_DeleteImportGraphDefOptions(opts);

  if (TF_GetCode(params_.status) != TF_OK) {
    TF_DeleteGraph(params_.graph);
    auto err_msg =
        "loadGraph failed: " + std::string(TF_Message(params_.status));
    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::GetTFOperation(
    const std::string &name, TF_Output &op) {
  auto port_info = modelbox::StringSplit(name, ':');
  int index = 0;
  try {
    if (port_info.size() == 2) {
      index = std::stoi(port_info[1]);
    }
  } catch (const std::exception &e) {
    MBLOG_WARN << "Convert id " << port_info[1] << " failed, err " << e.what()
               << "; use index 0 as default.";
  }

  op = TF_Output{TF_GraphOperationByName(params_.graph, port_info[0].c_str()),
                 index};
  if (nullptr == op.oper) {
    auto err_msg = "can't init op " + name + ":" + std::to_string(index);
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::FillInput(
    const std::vector<modelbox::FlowUnitInput> &flowunit_input_list) {
  for (auto const &input_item : flowunit_input_list) {
    auto input_name = input_item.GetPortName();
    auto input_type = input_item.GetPortType();
    params_.input_name_list_.push_back(input_name);
    params_.input_type_list_.push_back(input_type);
    TF_Output input_op;
    auto status = GetTFOperation(input_name, input_op);
    if (status != modelbox::STATUS_OK) {
      return status;
    }

    params_.input_op_list.push_back(input_op);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::FillOutput(
    const std::vector<modelbox::FlowUnitOutput> &flowunit_output_list) {
  for (auto const &output_item : flowunit_output_list) {
    auto output_name = output_item.GetPortName();
    auto output_type = output_item.GetPortType();
    params_.output_name_list_.push_back(output_name);
    params_.output_type_list_.push_back(output_type);
    TF_Output output_op;
    auto status = GetTFOperation(output_name, output_op);
    if (status != modelbox::STATUS_OK) {
      return status;
    }
    params_.output_op_list.push_back(output_op);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::NewSession(
    bool is_save_model, const std::string &model_entry) {
  params_.status = TF_NewStatus();
  if (nullptr == params_.status) {
    return {modelbox::STATUS_FAULT, "TF_NewStatus failed."};
  }

  params_.options = TF_NewSessionOptions();
  if (nullptr == params_.options) {
    return {modelbox::STATUS_FAULT, "TF_NewSessionOptions failed."};
  }

  TF_SetConfig(params_.options, (void *)params_.config_proto_binary_.data(),
               params_.config_proto_binary_.size(), params_.status);
  if (TF_GetCode(params_.status) != TF_OK) {
    auto err_msg =
        "TF_SetConfig failed: " + std::string(TF_Message(params_.status));
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (is_save_model) {
    TF_Buffer *metagraph = TF_NewBuffer();
    if (metagraph == nullptr) {
      const auto *err_msg = "TF_NewBuffer metagraph failed.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    params_.graph = TF_NewGraph();
    if (params_.graph == nullptr) {
      const auto *err_msg = "TF_NewGraph graph failed.";
      return {modelbox::STATUS_FAULT, err_msg};
    }

    params_.session = TF_LoadSessionFromSavedModel(
        params_.options, nullptr, model_entry.c_str(), &TAGS, 1, params_.graph,
        metagraph, params_.status);
    Defer { TF_DeleteBuffer(metagraph); };

    if (TF_GetCode(params_.status) != TF_OK) {
      TF_DeleteGraph(params_.graph);
      auto err_msg = "TF_LoadSessionFromSavedModel failed: " +
                     std::string(TF_Message(params_.status));
      return {modelbox::STATUS_FAULT, err_msg};
    }

    return modelbox::STATUS_OK;
  }

  params_.session =
      TF_NewSession(params_.graph, params_.options, params_.status);

  if (TF_GetCode(params_.status) != TF_OK) {
    auto err_msg =
        "TF_NewSession failed: " + std::string(TF_Message(params_.status));
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

bool InferenceTensorflowFlowUnit::IsSaveModelType(
    const std::string &model_path) {
  size_t found = model_path.find(".pb");
  if (found == std::string::npos) {
    return true;
  }

  return false;
}

static void StringHex2Hex(const std::vector<std::string> &string_vector,
                          std::vector<uint8_t> &uint8_vector) {
  if (string_vector.empty()) {
    uint8_vector.clear();
  }

  for (const auto &str : string_vector) {
    auto num = std::stoul(str, nullptr, 16);
    uint8_vector.push_back((uint8_t)num);
  }
}

modelbox::Status InferenceTensorflowFlowUnit::InitConfig(
    const std::shared_ptr<modelbox::Configuration> &fu_config) {
  auto inference_desc_ =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
          this->GetFlowUnitDesc());
  auto flowunit_input_list = inference_desc_->GetFlowUnitInput();
  auto flowunit_output_list = inference_desc_->GetFlowUnitOutput();

  std::string model_path = inference_desc_->GetModelEntry();
  params_.status = TF_NewStatus();
  if (params_.status == nullptr) {
    return {modelbox::STATUS_FAULT, "TF_NewStatus failed."};
  }

  if (fu_config->Contain("config.config_proto")) {
    auto config_strings = fu_config->GetStrings("config.config_proto");
    StringHex2Hex(config_strings, params_.config_proto_binary_);
  }

  bool is_save_model = IsSaveModelType(model_path);
  MBLOG_INFO << "is_save_model:\t" << is_save_model;
  modelbox::Status status = modelbox::STATUS_OK;
  if (!is_save_model) {
    status = LoadGraph(model_path);
    if (modelbox::STATUS_OK != status) {
      auto err_msg =
          "could not load inference graph, err: " + status.WrapErrormsgs();
      MBLOG_ERROR << err_msg;
      return {status, err_msg};
    }
  }

  status = NewSession(is_save_model, model_path);
  if (modelbox::STATUS_OK != status) {
    auto err_msg = "new session failed, err: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  status = FillInput(flowunit_input_list);
  if (modelbox::STATUS_OK != status) {
    auto err_msg = "fill input failed, err: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  status = FillOutput(flowunit_output_list);
  if (modelbox::STATUS_OK != status) {
    auto err_msg = "fill output failed, err: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  if (setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1) == -1) {
    MBLOG_WARN << "set tensorflow cpp log level failed.";
  };

  auto inference_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  inference_desc->SetResourceNice(false);
  auto config = inference_desc->GetConfiguration();
  if (config == nullptr) {
    return {modelbox::STATUS_BADCONF, "inference config is invalid."};
  }

  auto merge_config = std::make_shared<modelbox::Configuration>();
  // opts override python_desc_ config
  merge_config->Add(*config);
  merge_config->Add(*opts);
  modelbox::Status status = InitConfig(merge_config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "init config failed: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  plugin_ = merge_config->GetString("config.plugin");
  status = SetUpInferencePlugin(merge_config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "setup preprocess and postprocess failed: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::SetUpInferencePlugin(
    const std::shared_ptr<modelbox::Configuration> &config) {
  if (plugin_.empty()) {
    pre_process_ = std::bind(&InferenceTensorflowFlowUnit::PreProcess, this,
                             std::placeholders::_1, std::placeholders::_2);
    post_process_ = std::bind(&InferenceTensorflowFlowUnit::PostProcess, this,
                              std::placeholders::_1, std::placeholders::_2);
    return modelbox::STATUS_OK;
  }

  if (!modelbox::IsAbsolutePath(plugin_)) {
    auto relpath = modelbox::GetDirName(plugin_);
    plugin_ = relpath + "/" + plugin_;
  }

  return SetUpDynamicLibrary(config);
}

modelbox::Status InferenceTensorflowFlowUnit::PreProcess(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<TF_Tensor *> &input_tf_tensor_list) {
  int index = 0;
  modelbox::Status status;
  for (const auto &input_name : params_.input_name_list_) {
    const auto input_buf = data_ctx->Input(input_name);

    std::string type = params_.input_type_list_[index++];

    TF_DataType tf_type;
    if (type.empty()) {
      // Get type form buffer meta when model input type is not set
      modelbox::ModelBoxDataType buffer_type;
      status = input_buf->At(0)->Get("type", buffer_type);
      if (!status) {
        auto err_msg =
            "input type is not set ,please set it in inference toml file or "
            "buffer meta . error: " +
            status.WrapErrormsgs();
        return {modelbox::STATUS_FAULT, err_msg};
      }
      status = ConvertModelBoxTypeToTFType(buffer_type, tf_type);
      if (!status) {
        auto err_msg =
            "input type convert failed, error: " + status.WrapErrormsgs();
        return {modelbox::STATUS_FAULT, err_msg};
      }
    } else {
      std::transform(type.begin(), type.end(), type.begin(), ::toupper);
      status = ConvertType(type, tf_type);
      if (status != modelbox::STATUS_OK) {
        return {status, "input type convert failed."};
      }
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

    auto *buf_list_ptr = new std::shared_ptr<modelbox::BufferList>(input_buf);
    TF_Tensor *input_tensor = TF_NewTensor(
        tf_type, tf_dims.data(), tf_dims.size(),
        const_cast<void *>(input_buf->ConstData()), input_buf->GetBytes(),
        [](void *data, size_t length, void *arg) {
          delete (std::shared_ptr<modelbox::BufferList> *)(arg);
        },
        buf_list_ptr);
    if (nullptr == input_tensor) {
      auto err_msg = "TF_NewTensor " + std::string(input_name) + " failed. ";
      err_msg += "please check the input type and shape.";
      MBLOG_ERROR << err_msg;
      return {modelbox::STATUS_FAULT, err_msg};
    }
    input_tf_tensor_list.push_back(input_tensor);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::PostProcess(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<TF_Tensor *> &output_tf_tensor_list) {
  int index = 0;
  for (const auto &output_name : params_.output_name_list_) {
    auto tensor_byte = TF_TensorByteSize(output_tf_tensor_list[index]);
    auto *tensor_data = TF_TensorData(output_tf_tensor_list[index]);
    auto tensor_type = TF_TensorType(output_tf_tensor_list[index]);
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
      return {modelbox::STATUS_FAULT, "dim is zero"};
    }

    auto output_buf = data_ctx->Output(output_name);
    auto single_bytes = tensor_byte / num;
    std::vector<size_t> shape_vector(num, single_bytes);
    auto status = CreateOutputBufferList(output_buf, shape_vector, tensor_data,
                                         tensor_byte, tensor_type, index);
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

modelbox::Status InferenceTensorflowFlowUnit::SetUpDynamicLibrary(
    const std::shared_ptr<modelbox::Configuration> &config) {
  typedef std::shared_ptr<InferencePlugin> (*PluginObject)();
  auto status = modelbox::STATUS_OK;
  void *driver_handler = dlopen(plugin_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    auto *dl_errmsg = dlerror();
    auto err_msg = "dlopen " + plugin_ + " failed";
    if (dl_errmsg) {
      err_msg += ", error: " + std::string(dl_errmsg);
    }

    MBLOG_ERROR << err_msg;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  DeferCond { return !status; };
  DeferCondAdd {
    if (driver_handler != nullptr) {
      dlclose(driver_handler);
      driver_handler = nullptr;
    }
  };

  auto create_plugin =
      reinterpret_cast<PluginObject>(dlsym(driver_handler, "CreatePlugin"));
  if (create_plugin == nullptr) {
    auto *dlerr_msg = dlerror();
    std::string err_msg = "dlsym CreatePlugin failed";
    if (dlerr_msg) {
      err_msg += " error: ";
      err_msg += dlerr_msg;
    }

    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  std::shared_ptr<InferencePlugin> inference_plugin = create_plugin();
  if (inference_plugin == nullptr) {
    const auto *err_msg = "CreatePlugin failed";
    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  status = inference_plugin->PluginInit(config);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "plugin init failed, error: " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    status = {modelbox::STATUS_FAULT, err_msg};
    return status;
  }

  driver_handler_ = driver_handler;
  inference_plugin_ = inference_plugin;

  pre_process_ = std::bind(&InferencePlugin::PreProcess, inference_plugin_,
                           std::placeholders::_1, std::placeholders::_2);
  post_process_ = std::bind(&InferencePlugin::PostProcess, inference_plugin_,
                            std::placeholders::_1, std::placeholders::_2);

  return status;
}

modelbox::Status InferenceTensorflowFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  // TODO consider without N model and nhwc check

  std::vector<TF_Tensor *> input_tf_tensor_list;
  std::vector<TF_Tensor *> output_tf_tensor_list(
      params_.output_name_list_.size(), nullptr);

  Defer { ClearTensor(input_tf_tensor_list, output_tf_tensor_list); };

  auto status = pre_process_(data_ctx, input_tf_tensor_list);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "tensorflow flowunit preprocess failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  status = Inference(input_tf_tensor_list, output_tf_tensor_list);
  if (modelbox::STATUS_OK != status) {
    auto err_msg =
        "tensorflow flowunit inference failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  status = post_process_(data_ctx, output_tf_tensor_list);
  if (modelbox::STATUS_OK != status) {
    auto err_msg =
        "tensorflow flowunit postprocess failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {status, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::CreateOutputBufferList(
    std::shared_ptr<modelbox::BufferList> &output_buffer_list,
    const std::vector<size_t> &shape_vector, void *tensor_data,
    size_t tensor_byte, TF_DataType tensor_type, int index) {
  auto status =
      output_buffer_list->BuildFromHost(shape_vector, tensor_data, tensor_byte);
  if (!status) {
    auto err_msg = "output buffer list builds error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }

  modelbox::ModelBoxDataType modelbox_type = modelbox::MODELBOX_TYPE_INVALID;
  status = ConvertTFTypeToModelBoxType(tensor_type, modelbox_type);
  if (!status) {
    auto err_msg =
        "output type convert failed ,error: " + status.WrapErrormsgs();
    return {modelbox::STATUS_FAULT, err_msg};
  }
  output_buffer_list->Set("type", modelbox_type);
  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::ConvertType(
    const std::string &type, TF_DataType &TFType) {
  if (type_map.find(type) == type_map.end()) {
    return {modelbox::STATUS_FAULT, "unsupported type " + type};
  }

  TFType = type_map[type];
  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::Inference(
    const std::vector<TF_Tensor *> &input_tensor_list,
    std::vector<TF_Tensor *> &output_tensor_list) {
  TF_SessionRun(params_.session, nullptr, params_.input_op_list.data(),
                input_tensor_list.data(), params_.input_name_list_.size(),
                params_.output_op_list.data(), output_tensor_list.data(),
                params_.output_name_list_.size(), nullptr, 0, nullptr,
                params_.status);
  if (TF_GetCode(params_.status) != TF_OK) {
    auto err_msg =
        "doInference failed: " + std::string(TF_Message(params_.status));
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status InferenceTensorflowFlowUnit::Close() {
  return params_.Clear();
}

void InferenceTensorflowFlowUnitDesc::SetModelEntry(std::string model_entry) {
  model_entry_ = std::move(model_entry);
}

std::string InferenceTensorflowFlowUnitDesc::GetModelEntry() {
  return model_entry_;
}
