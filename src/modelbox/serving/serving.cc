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

#include "serving.h"

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>

#include <fstream>
#include <unistd.h>

const std::string copyright_content = R"(
# Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
)";

const std::string import_content = R"(
import _flowunit as modelbox
import numpy as np
import json
)";

const std::string template_content = R"(
    def __init__(self):
         super().__init__()
    def open(self, config):
        return modelbox.Status()
    def close(self):
        return modelbox.Status()
    def data_pre(self, data_context):
        return modelbox.Status()
    def data_post(self, data_context):
        return modelbox.Status()
    def data_group_pre(self, data_context):
        return modelbox.Status()
    def data_group_post(self, data_context):
        return modelbox.Status()
)";

modelbox::Status ModelServing::GenerateTemplate(const std::string &model_name,
                                                const std::string &model_path,
                                                int port) {
  auto status = CheckConfigFiles(model_path);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "check path failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }


  status = ParseModelToml();
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "parse model toml failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::string default_file_path =
      std::string(DEFAULT_FLOWUNIT_PATH) + "/" + model_name;

  bool generate_failed = false;
  DeferCond { return generate_failed; };
  DeferCondAdd {
    if (!graph_toml_file_.empty()) {
      remove(graph_toml_file_.c_str());
    }

    if (access(default_file_path.c_str(), R_OK) == 0) {
      modelbox::RemoveDirectory(default_file_path);
    }
  };

  status = GenerateModelServingTemplate(model_name, port);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "generate model-serving template failed, err: " +
                   status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    generate_failed = true;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (custom_service_) {
    fprintf(stdout, "generate custom custom_service\n");
    status = GeneratePrePostFlowUnit(default_file_path, "preprocess");
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "generate default custom_service flowunit failed, err: " +
                     status.WrapErrormsgs();
      fprintf(stderr, "%s\n", err_msg.c_str());
      generate_failed = true;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    status = GeneratePrePostFlowUnit(default_file_path, "postprocess");
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "generate default custom_service flowunit failed, err: " +
                     status.WrapErrormsgs();
      fprintf(stderr, "%s\n", err_msg.c_str());
      generate_failed = true;
      return {modelbox::STATUS_FAULT, err_msg};
    }

  } else {
    fprintf(stdout, "generate default custom_service\n");
    status = GenerateDefaultPrePostFlowUnit(default_file_path, "preprocess");
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "generate default custom_service flowunit failed, err: " +
                     status.WrapErrormsgs();
      fprintf(stderr, "%s\n", err_msg.c_str());
      generate_failed = true;
      return {modelbox::STATUS_FAULT, err_msg};
    }

    status = GenerateDefaultPrePostFlowUnit(default_file_path, "postprocess");
    if (status != modelbox::STATUS_OK) {
      auto err_msg = "generate default custom_service flowunit failed, err: " +
                     status.WrapErrormsgs();
      fprintf(stderr, "%s\n", err_msg.c_str());
      generate_failed = true;
      return {modelbox::STATUS_FAULT, err_msg};
    }
  }

  status = UpdateGraphTemplateByToml(model_name);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "update graph template failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    generate_failed = true;
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
};

modelbox::Status ModelServing::CheckConfigFiles(const std::string &model_path) {
  std::vector<std::string> files;
  auto status = modelbox::ListFiles(model_path, "*", &files);
  if (status != modelbox::STATUS_OK) {
    return status;
  }

  for (auto &file : files) {
    auto str_vec = modelbox::StringSplit(file, '/');
    auto file_name = str_vec[str_vec.size() - 1];

    if (file_name == "model.toml") {
      model_toml_ = modelbox::PathCanonicalize(file);
      continue;
    }

    if (file_name == "custom_service.py") {
      model_custom_service_file_ = file;
      custom_service_ = true;
      continue;
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::FillModelItem(const std::string &type) {
  auto item = config_->GetSubKeys(type);
  if (item.empty()) {
    fprintf(stderr, "the key %s is not found in the config file.\n",
            type.c_str());
    return modelbox::STATUS_BADCONF;
  }

  std::vector<std::string> item_names;
  std::vector<std::string> item_types;
  for (unsigned int i = 1; i <= item.size(); ++i) {
    std::string item_name;
    std::string item_type;
    auto key = type + "." + type + std::to_string(i);
    auto item_table = config_->GetSubKeys(key);
    if (item_table.empty()) {
      fprintf(stderr, "the key %s is not found in the config file.\n",
              key.c_str());
      return modelbox::STATUS_BADCONF;
    }

    for (const auto &inner_item : item_table) {
      auto item_index = key + "." + inner_item;
      if (inner_item == "name") {
        item_name = config_->GetString(item_index);
        if (item_name.empty()) {
          fprintf(stderr, "the key %s should have key name.\n", key.c_str());
          return modelbox::STATUS_BADCONF;
        }

        item_names.emplace_back(item_name);
        continue;
      }

      if (inner_item == "type") {
        item_type = config_->GetString(item_index);
        if (item_type.empty()) {
          fprintf(stderr, "the key %s should have key type.\n", key.c_str());
          return modelbox::STATUS_BADCONF;
        }

        item_types.emplace_back(item_type);
        continue;
      }
    }
  }

  if (type == "input") {
    model_serving_config_.SetInputNames(item_names);
    model_serving_config_.SetInputTypes(item_types);
  } else {
    model_serving_config_.SetOutputNames(item_names);
    model_serving_config_.SetOutputTypes(item_types);
  }
  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::ParseModelToml() {
  modelbox::ConfigurationBuilder configbuilder;
  config_ = configbuilder.Build(model_toml_);
  if (config_ == nullptr) {
    std::string err_msg = "parse model toml failed, err: " +
                          modelbox::StatusError.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  auto base_config = config_->GetSubConfig("base");
  model_serving_config_.SetModelEntry(base_config->GetString("entry", ""));
  model_serving_config_.SetMaxBatchSize(
      base_config->GetInt64("max_batch_size", 1));
  model_serving_config_.SetDevices(base_config->GetStrings("device", {"cpu"}));
  model_serving_config_.SetModelEngine(
      base_config->GetString("engine", "tensorflow"));
  model_serving_config_.SetMode(base_config->GetString("mode", "model"));

  auto status = FillModelItem("input");
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "fille model input config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = FillModelItem("output");
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "fille model output config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::GenerateModelServingTemplate(
    const std::string &model_name, int port) {
  auto status = GenerateDefaultGraphConfig(model_name, port);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "generate default graph config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::string default_flowunit_path =
      std::string(DEFAULT_FLOWUNIT_PATH) + "/" + model_name;
  status = modelbox::CreateDirectory(default_flowunit_path);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "create directory /tmp" + model_name +
                   " failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = GenerateInferConfig(default_flowunit_path, model_name);
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "generate infer config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = GeneratePrePostConfig(default_flowunit_path, "preprocess");
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "generate preprocess config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = GeneratePrePostConfig(default_flowunit_path, "postprocess");
  if (status != modelbox::STATUS_OK) {
    auto err_msg =
        "generate postprocess config failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

std::string ModelServing::GetDeviceType(const std::string &model_engine) {
  std::string device{"cpu"};
  if (model_engine == "tensorrt") {
    device = "cuda";
  } else if (model_engine == "mindspore") {
    device = "ascend";
  } else if (model_engine == "acl") {
    device = "ascend";
  } else {
    auto devices = model_serving_config_.GetDevices();
    for (auto &device : devices) {
      if (device.size() < 4) {
        continue;
      }

      auto sub = device.substr(0, 4);
      if (sub == "cuda") {
        device = "cuda";
        break;
      }
    }
  }

  return device;
}

modelbox::Status ModelServing::GenerateInferConfig(
    const std::string &default_flowunit_path, const std::string &model_name) {
  std::string infer_flowunit_path = default_flowunit_path + "/" + model_name;
  auto status = modelbox::CreateDirectory(infer_flowunit_path);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "create directory " + infer_flowunit_path +
                   "failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::string infer_toml = infer_flowunit_path + "/" + model_name + ".toml";
  std::ofstream file{infer_toml};
  if (!file.is_open()) {
    auto err_msg = "Open failed, path " + infer_toml;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { file.close(); };

  std::string device = GetDeviceType(model_serving_config_.GetModelEngine());
  std::stringstream ss;
  std::string base_content = R"([base]
name = ")" + model_name + R"("
device = ")" + device + R"("
version = "1.0.0"
description = "model-serving template description."
entry = ")" + model_serving_config_.GetModelEntry() +
                             R"("
type = "inference"
virtual_type = ")" + model_serving_config_.GetModelEngine() +
                             R"("

[input])";

  ss << base_content;
  std::string input_content;
  auto input_names = model_serving_config_.GetInputNames();
  auto input_types = model_serving_config_.GetInputTypes();
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::string input_content = R"(
[input.input)" + std::to_string(i + 1) +
                                R"(]
name = ")" + input_names[i] +
                                R"("
type = ")" + input_types[i] +
                                R"("
    )";
    ss << input_content;
  }
  ss << std::endl;
  ss << R"([output])";
  auto output_names = model_serving_config_.GetOutputNames();
  auto output_types = model_serving_config_.GetOutputTypes();
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::string output_content = R"(
[output.output)" + std::to_string(i + 1) +
                                 R"(]
name = ")" + output_names[i] +
                                 R"("
type = ")" + output_types[i] +
                                 R"(")";
    ss << output_content;
  }

  file << ss.str();
  if (!file.good()) {
    std::string err_msg = "write infer config failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::GeneratePrePostConfig(
    const std::string &default_flowunit_path, const std::string &type) {
  std::string python_flowunit_path = default_flowunit_path + "/" + type;
  auto status = modelbox::CreateDirectory(python_flowunit_path);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "create directory " + python_flowunit_path +
                   "failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::string python_toml = python_flowunit_path + "/" + type + ".toml";
  std::ofstream file{python_toml};
  if (!file.is_open()) {
    auto err_msg = "Open failed, path " + python_toml;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { file.close(); };

  std::stringstream ss;
  std::string class_name;
  std::string input_output_content;
  if (type == "preprocess") {
    class_name = R"(Preprocess")";
    input_output_content = R"([input]
[input.input1]
name = "in_data"
type = "string")";
  } else {
    class_name = R"(Postprocess")";
    input_output_content = R"([output]
[output.output1]
name = "out_data"
type = "string")";
  }

  std::string content = R"([base]
name = ")" + type + R"("
device = "cpu"
version = "1.0.0"
description = "model-serving pre/post template description."
entry = ")" + type + R"(@)" +
                        class_name + R"(
type = "python" 

)" + input_output_content;

  ss << content;
  file << ss.str();
  if (!file.good()) {
    std::string err_msg = "write python config failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::GenerateDefaultGraphConfig(
    const std::string &model_name, int port) {
  graph_toml_file_ =
      std::string(DEFAULT_GRAPTH_PATH) + "/" + model_name + "_origin.toml";
  std::ofstream file{graph_toml_file_};
  if (!file.is_open()) {
    auto err_msg = "Open failed, path " + graph_toml_file_;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { file.close(); };

  std::stringstream ss;
  std::string content = R"([driver]
skip-default=false
dir="/tmp/)" + model_name +
                        R"("
[graph]
format = "graphviz"
graphconf = '''digraph demo {
      httpserver_sync_receive[type=flowunit, flowunit=httpserver_sync_receive, device=cpu, time_out_ms=5000, endpoint="http://0.0.0.0:)" +
                    std::to_string(port) + R"(", max_requests=100]
      preprocess[type=flowunit, flowunit=preprocess, device=cpu]
)" +
"     " + model_name + R"([type=flowunit, flowunit=)" +
                    model_name + R"(, device=infer_device, deviceid=0]
      postprocess[type=flowunit, flowunit=postprocess, device=cpu]
      httpserver_sync_reply[type=flowunit, flowunit=httpserver_sync_reply, device=cpu]
      
      httpserver_sync_receive:out_request_info -> preprocess:in_data
      preprocess:out_data -> )" +
                    model_name + R"(:input
      )" + model_name +
                    R"(:output -> postprocess:in_data
      postprocess:out_data -> httpserver_sync_reply:in_reply_info
}''')";
  ss << content;
  file << ss.str();
  if (!file.good()) {
    std::string err_msg = "write default graph config failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::GeneratePrePostFlowUnit(
    const std::string &default_file_path, const std::string &type) {
  std::string copy_custom_python = default_file_path + "/" + type + "/custom_service.py";
  auto status = modelbox::CopyFile(model_custom_service_file_, copy_custom_python, 0, true);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "copy custom service file failed, err: " + status.WrapErrormsgs();
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  std::string flowunit_path =
      default_file_path + "/" + type + "/" + type + ".py";

  std::ofstream file{flowunit_path};
  if (!file.is_open()) {
    auto err_msg = "Open failed, path " + flowunit_path;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { file.close(); };

  std::stringstream ss;
  auto input_names = model_serving_config_.GetInputNames();
  auto input_types = model_serving_config_.GetInputTypes();
  auto output_names = model_serving_config_.GetOutputNames();
  ss << copyright_content << "\n";
  ss << import_content << "\n";

  ss << "import custom_service\n\n";

  if (type == "preprocess") {
    ss << "class Preprocess(modelbox.FlowUnit):\n";
  } else {
    ss << "class Postprocess(modelbox.FlowUnit):\n";
  }

  ss << template_content << "\n";

  ss << "    def process(self, data_context):\n";
  if (type == "preprocess") {
    ss << "        in_data = data_context.input(\"in_data\")\n";
    for (auto &input_name : input_names) {
      ss << "        " << input_name << " = data_context.output(\""
         << input_name << "\")\n";
    }
    ss << "\n";
    ss << "        for buffer in in_data:\n";
    ss << "            # get data from json\n";
    ss << "            request_body = "
          "json.loads(buffer.as_object().strip(chr(0)))\n";
    ss << "\n";
    ss << "            for item in dir(custom_service):\n";
    ss << "                class_name = getattr(custom_service, item)\n";
    ss << "                if isinstance(class_name, type):\n";
    ss << "                    try:\n";
    ss << "                        instance = class_name()\n";
    ss << "                        preprocess = getattr(instance, "
          "\"_preprocess\")\n";
    ss << "                        result = preprocess(request_body)\n";
    for (size_t i = 0; i < input_names.size(); ++i) {
      ss << "                        data_" << input_names[i]
         << " = np.asarray(result[\"" << input_names[i] << "\"])";
      if (input_types[i] == "float") {
         ss << ".astype(np.float32)\n";
      } else if (input_types[i] == "double") {
         ss << ".astype(np.float64)\n";
      }
      ss << "                        add_buffer_" << input_names[i]
         << " = self.create_buffer(data_" << input_names[i] << ")\n";
      ss << "                        " << input_names[i] << ".push_back(add_buffer_"
         << input_names[i] << ")\n";
    }
    ss << "                    except Exception as e:\n";
    ss << "                        print(\"custom preprocess failed, \", "
          "e)\n";
    ss << "                        return "
          "modelbox.Status.StatusCode.STATUS_FAULT\n";
    ss << "        return modelbox.Status()\n";
  } else {
    for (auto &output_name : output_names) {
      ss << "        " << output_name << " = data_context.input(\""
         << output_name << "\")\n";
    }
    ss << "        out_data = data_context.output(\"out_data\")\n";
    ss << "\n";
    ss << "        for i in range(" << output_names[0] << ".size()):\n";
    ss << "            postprocess_result = {}\n";
    for (auto &output_name : output_names) {
      ss << "            postprocess_result[\"" << output_name
         << "\"] = " << output_name << "[i].as_object()\n";
    }
    ss << "            for item in dir(custom_service):\n";
    ss << "                class_name = getattr(custom_service, item)\n";
    ss << "                if isinstance(class_name, type):\n";
    ss << "                    try:\n";
    ss << "                        instance = class_name()\n";
    ss << "                        postprocess = getattr(instance, "
          "\"_postprocess\")\n";
    ss << "                        result = postprocess(postprocess_result)\n";
    ss << "                        result_str = (json.dumps(result) + "
          "chr(0)).encode('utf-8').strip()\n";
    ss << "                        add_buffer = "
          "self.create_buffer(result_str)\n";
    ss << "                        out_data.push_back(add_buffer)\n";
    ss << "                    except Exception as e:\n";
    ss << "                        print(\"custom postprocess failed, \", "
          "e)\n";
    ss << "                        return "
          "modelbox.Status.StatusCode.STATUS_FAULT\n";
    ss << "        return modelbox.Status()\n";
  }

  file << ss.str();
  if (!file.good()) {
    std::string err_msg = "write pre/post flowunit failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::GenerateDefaultPrePostFlowUnit(
    const std::string &default_file_path, const std::string &type) {
  std::string flowunit_path =
      default_file_path + "/" + type + "/" + type + ".py";
  std::ofstream file{flowunit_path};
  if (!file.is_open()) {
    auto err_msg = "Open failed, path " + flowunit_path;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { file.close(); };

  std::stringstream ss;
  auto input_names = model_serving_config_.GetInputNames();
  auto input_types = model_serving_config_.GetInputTypes();
  auto output_names = model_serving_config_.GetOutputNames();
  ss << copyright_content << "\n";
  ss << import_content << "\n";
  ss << "import cv2\n";
  ss << "import base64\n";
  ss << "\n";

  if (type == "preprocess") {
    ss << "class Preprocess(modelbox.FlowUnit):\n";
  } else {
    ss << "class Postprocess(modelbox.FlowUnit):\n";
  }

  ss << template_content << "\n";
  ss << "    def process(self, data_context):\n";
  if (type == "preprocess") {
    ss << "        in_data = data_context.input(\"in_data\")\n";
    for (auto &input_name : input_names) {
      ss << "        " << input_name << " = data_context.output(\""
         << input_name << "\")\n";
    }
    ss << "\n";
    ss << "        for buffer in in_data:\n";
    ss << "            # get data from json\n";
    ss << "            request_body = "
          "json.loads(buffer.as_object().strip(chr(0)))\n";
    ss << "\n";
    for (size_t i = 0; i < input_names.size(); ++i) {
      ss << "            if request_body.get(\"" << input_names[i] << "\"):\n";
      ss << "                data = np.asarray(request_body[\"" << input_names[i]
         << "\"])";
      if (input_types[i] == "float") {
         ss << ".astype(np.float32)\n";
      } else if (input_types[i] == "double") {
         ss << ".astypd(np.float64)\n";
      }
      ss << "                add_buffer = self.create_buffer(data)\n";
      ss << "                " << input_names[i] << ".push_back(add_buffer)\n";
      ss << "            else:\n";
      ss << "                print(\"wrong key of request_body\")\n";
      ss << "                return modelbox.Status.StatusCode.STATUS_FAULT\n";
      ss << "\n";
    }

    ss << "        return modelbox.Status()\n";
  } else {
    for (auto &output_name : output_names) {
      ss << "        " << output_name << " = data_context.input(\""
         << output_name << "\")\n";
    }
    ss << "        out_data = data_context.output(\"out_data\")\n";
    ss << "\n";
    ss << "        for i in range(" << output_names[0] << ".size()):\n";
    ss << "            result = {}\n";
    for (auto &output_name : output_names) {
      ss << "            data = " << output_name << "[i].as_object()\n";
      ss << "            result[\"" << output_name << "\"] = data.tolist()\n";
    }

    ss << "            result_str = (json.dumps(result) + "
          "chr(0)).encode('utf-8').strip()\n";
    ss << "            add_buffer = self.create_buffer(result_str)\n";
    ss << "            out_data.push_back(add_buffer)\n";
    ss << "        return modelbox.Status()\n";
  }
  file << ss.str();
  if (!file.good()) {
    std::string err_msg = "write default pre/post flowunit failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  file.close();
  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::UpdatePreFlowUnit(
    const std::string &model_name) {
  std::string pre_writer_toml_path = std::string(DEFAULT_FLOWUNIT_PATH) + "/" +
                                     model_name + "/preprocess/preprocess.toml";
  std::ofstream pre_writer(pre_writer_toml_path, std::ios::app);
  if (!pre_writer.is_open()) {
    auto err_msg = "Open failed, path " + pre_writer_toml_path;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { pre_writer.close(); };

  std::stringstream ss;
  auto input_names = model_serving_config_.GetInputNames();
  auto input_types = model_serving_config_.GetInputTypes();
  ss << "\n";
  ss << "[output]\n";
  for (size_t i = 0; i < input_names.size(); ++i) {
    ss << "[output.output" << std::to_string(i + 1) << "]\n";
    ss << "name = \"" << input_names[i] << "\"\n";
    ss << "type = \"" << input_types[i] << "\"\n";
  }

  pre_writer << ss.str();
  if (!pre_writer.good()) {
    std::string err_msg = "update pre flowunit config failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::UpdatePostFlowUnit(
    const std::string &model_name) {
  std::string post_writer_toml_path = std::string(DEFAULT_FLOWUNIT_PATH) + "/" +
                                      model_name +
                                      "/postprocess/postprocess.toml";
  std::ofstream post_writer(post_writer_toml_path, std::ios::app);
  if (!post_writer.is_open()) {
    auto err_msg = "Open failed, path " + post_writer_toml_path;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { post_writer.close(); };
  std::stringstream ss;
  auto output_names = model_serving_config_.GetOutputNames();
  auto output_types = model_serving_config_.GetOutputTypes();
  ss << "\n";
  ss << "[input]\n";
  for (size_t i = 0; i < output_names.size(); ++i) {
    ss << "[input.input" << std::to_string(i + 1) << "]\n";
    ss << "name = \"" << output_names[i] << "\"\n";
    ss << "type = \"" << output_types[i] << "\"\n";
  }
  post_writer << ss.str();
  if (!post_writer.good()) {
    std::string err_msg = "update post flowunit config failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::UpdateGraphToml(const std::string &model_name) {
  std::ifstream graph_reader(graph_toml_file_);
  if (!graph_reader.is_open()) {
    auto err_msg = "Open failed, path " + graph_toml_file_;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { graph_reader.close(); };

  std::string update_grah_toml_file =
      std::string(DEFAULT_GRAPTH_PATH) + "/" + model_name + ".toml";
  std::ofstream new_graph_writer(update_grah_toml_file);
  if (!new_graph_writer.is_open()) {
    auto err_msg = "Open failed, path " + update_grah_toml_file;
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  Defer { new_graph_writer.close(); };

  std::stringstream ss;
  std::string content;
  while (std::getline(graph_reader, content)) {
    size_t pos;
    pos = content.find("infer_device");
    if (pos != std::string::npos) {
      auto config_device = config_->GetString("base.device");
      bool single_device = true;
      if (config_device.find(":") != std::string::npos) {
        modelbox::StringReplaceAll(config_device, "~", ";");
        single_device = false;
      }

      ss << "      " << model_name << "[type=flowunit, flowunit=" << model_name
         << ", device=\"";
      if (single_device) {
        ss << config_device << "\", deviceid=0]\n";
      } else {
        ss << config_device << "\"]\n";
      }

      continue;
    }

    pos = content.find("preprocess:out_data");
    auto input_names = model_serving_config_.GetInputNames();
    if (pos != std::string::npos) {
      for (auto &port_name : input_names) {
        ss << "      preprocess:" << port_name << " -> " << model_name << ":"
           << port_name << "\n";
      }
      continue;
    }

    pos = content.find("postprocess:in_data");
    auto output_names = model_serving_config_.GetOutputNames();
    if (pos != std::string::npos) {
      for (auto &port_name : output_names) {
        ss << "      " << model_name << ":" << port_name << " -> "
           << "postprocess:" << port_name << "\n";
      }
      continue;
    }

    ss << content << "\n";
  }

  new_graph_writer << ss.str();
  if (!new_graph_writer.good()) {
    std::string err_msg = "update graph failed";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  if (remove(graph_toml_file_.c_str()) == -1) {
    fprintf(stderr, "remove origin template graph toml failed.\n");
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ModelServing::UpdateGraphTemplateByToml(
    const std::string &model_name) {
  auto status = UpdatePreFlowUnit(model_name);
  if (status != modelbox::STATUS_OK) {
    std::string err_msg = "update preprocess flowunit failed.";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = UpdatePostFlowUnit(model_name);
  if (status != modelbox::STATUS_OK) {
    std::string err_msg = "update postprocess flowunit failed.";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  status = UpdateGraphToml(model_name);
  if (status != modelbox::STATUS_OK) {
    std::string err_msg = "update graph toml failed.";
    fprintf(stderr, "%s\n", err_msg.c_str());
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}
